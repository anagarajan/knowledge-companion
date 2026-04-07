"""
patient_extractor.py — Structured medical record extraction (Track 1).

For every patient folder, this module reads all of the patient's documents
and produces a structured PatientRecord:

  full_name, date_of_birth, gender, city, state,
  insurance_provider, insurance_id,
  icd10_codes[], diagnoses[], medications[],
  medical_history (short narrative)

It uses a map-reduce strategy with the configured PATIENT_EXTRACTION_MODEL:

  MAP    one extraction call per document → partial PatientRecord
  REDUCE one merge call combining the per-doc results → final PatientRecord

Why map-reduce?
  A patient with 25 docs × 5 pages averages ~125 pages of text. That blows
  past any single-prompt context budget. Per-doc extraction keeps each call
  small, parallelizable, and individually cacheable.

Why not just one giant call per patient?
  Even when the context fits, models pay attention worse on long inputs.
  Small focused calls are more accurate and cheaper to retry on failure.

Every extracted field carries provenance: which file and which page it
came from, and the model's self-reported confidence (0–1). The provenance
is what powers "DOB: 1958-03-12 (from intake.pdf, p2)" in the UI and what
lets a query layer skip low-confidence values.

This module is pure Python — it does NOT touch the database. The caller
(ingest.py) is responsible for persisting the result via PatientStore.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import httpx

from config import (
    OLLAMA_BASE_URL,
    PATIENT_EXTRACTION_MAX_CHARS,
    PATIENT_EXTRACTION_MODEL,
    PATIENT_EXTRACTION_TIMEOUT_S,
)

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class DocumentText:
    """One document's extracted text, ready for an extraction call."""
    filename: str
    pages: list[tuple[int, str]]   # [(page_number, text), ...]

    def combined(self, max_chars: int = PATIENT_EXTRACTION_MAX_CHARS) -> str:
        """Concatenate pages with markers, truncated to fit the prompt budget."""
        parts: list[str] = []
        used = 0
        for page_number, text in self.pages:
            marker = f"\n[Page {page_number}]\n"
            chunk = marker + text
            if used + len(chunk) > max_chars:
                remaining = max(0, max_chars - used)
                if remaining > 0:
                    parts.append(chunk[:remaining])
                break
            parts.append(chunk)
            used += len(chunk)
        return "".join(parts)


@dataclass
class FieldProvenance:
    """Records where a single extracted field value came from."""
    field_name: str
    field_value: str
    source_file: str
    source_page: int | None
    confidence: float            # 0.0–1.0


@dataclass
class PatientRecord:
    """A structured medical record for one patient."""
    patient_id: str
    patient_id_source: str       # 'folder_name' | 'extracted' | 'hash'
    folder_path: str
    full_name: str | None = None
    date_of_birth: date | None = None
    gender: str | None = None
    city: str | None = None
    state: str | None = None
    insurance_provider: str | None = None
    insurance_id: str | None = None
    icd10_codes: list[str] = field(default_factory=list)
    diagnoses: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    medical_history: str = ""
    extraction_model: str = ""
    source_doc_hash: str = ""
    last_extracted_at: datetime | None = None
    provenance: list[FieldProvenance] = field(default_factory=list)


# ── Public API ────────────────────────────────────────────────────────────────


def extract_patient_record(
    folder_path: str,
    documents: list[DocumentText],
) -> PatientRecord:
    """
    Run map-reduce extraction over a patient's documents.

    Args:
        folder_path: absolute path to the patient's folder
        documents:   list of the patient's parsed documents

    Returns:
        A fully populated PatientRecord (fields may still be None where the
        documents do not mention them — null is intentional, never guess).
    """
    folder_name = Path(folder_path).name

    if not documents:
        logger.warning(f"No documents for patient at {folder_path}")
        return _empty_record(folder_path, folder_name)

    # ── MAP: one extraction per document ────────────────────────────────────
    per_doc_results: list[dict] = []
    for doc in documents:
        try:
            result = _extract_from_document(doc)
            if result:
                per_doc_results.append({"filename": doc.filename, "fields": result})
        except Exception as e:
            logger.warning(f"  Per-doc extraction failed on {doc.filename}: {e}")

    if not per_doc_results:
        logger.warning(f"All per-doc extractions failed for {folder_name}")
        return _empty_record(folder_path, folder_name)

    # ── REDUCE: merge per-doc results into a single record ─────────────────
    merged = _merge_results(per_doc_results)

    # ── Identify the patient (folder name → extracted name → hash) ─────────
    patient_id, id_source = _resolve_patient_id(
        folder_name=folder_name,
        folder_path=folder_path,
        extracted_name=merged.get("full_name"),
        extracted_dob=merged.get("date_of_birth"),
    )

    # ── Build the record ───────────────────────────────────────────────────
    record = PatientRecord(
        patient_id=patient_id,
        patient_id_source=id_source,
        folder_path=folder_path,
        full_name=merged.get("full_name"),
        date_of_birth=_parse_date(merged.get("date_of_birth")),
        gender=_normalize_gender(merged.get("gender")),
        city=merged.get("city"),
        state=merged.get("state"),
        insurance_provider=merged.get("insurance_provider"),
        insurance_id=merged.get("insurance_id"),
        icd10_codes=_dedupe_list(merged.get("icd10_codes", [])),
        diagnoses=_dedupe_list(merged.get("diagnoses", [])),
        medications=_dedupe_list(merged.get("medications", [])),
        medical_history=merged.get("medical_history", "") or "",
        extraction_model=PATIENT_EXTRACTION_MODEL,
        source_doc_hash=_doc_hash(documents),
        last_extracted_at=datetime.utcnow(),
        provenance=_build_provenance(per_doc_results, merged),
    )
    return record


# ── Map step: extract from one document ──────────────────────────────────────


def _extract_from_document(doc: DocumentText) -> dict | None:
    """
    Single LLM call. Returns parsed JSON dict or None on failure.
    The prompt is strict about returning null for missing fields.
    """
    text = doc.combined()
    if not text.strip():
        return None

    prompt = _build_extraction_prompt(text)
    raw = _call_ollama_json(prompt)
    if not raw:
        return None
    return _parse_json_safely(raw)


def _build_extraction_prompt(document_text: str) -> str:
    return (
        "You are a medical records extraction assistant. Read the document and "
        "return a JSON object with the structured fields below. Follow these rules:\n"
        "  • Return null for any field you cannot find. NEVER guess.\n"
        "  • icd10_codes: list of ICD-10 codes EXACTLY as they appear in the document.\n"
        "  • diagnoses: list of diagnosis names in plain English.\n"
        "  • medications: list of medication names with dose if available.\n"
        "  • date_of_birth: ISO format YYYY-MM-DD if present, else null.\n"
        "  • gender: 'M', 'F', or 'O' for other; null if not stated.\n"
        "  • Output JSON only, no commentary.\n\n"
        "Schema:\n"
        "{\n"
        '  "full_name": string|null,\n'
        '  "date_of_birth": string|null,\n'
        '  "gender": string|null,\n'
        '  "city": string|null,\n'
        '  "state": string|null,\n'
        '  "insurance_provider": string|null,\n'
        '  "insurance_id": string|null,\n'
        '  "icd10_codes": [string],\n'
        '  "diagnoses": [string],\n'
        '  "medications": [string],\n'
        '  "medical_history": string|null\n'
        "}\n\n"
        "Document:\n"
        f"{document_text}\n\n"
        "JSON:"
    )


# ── Reduce step: merge per-document results ─────────────────────────────────


def _merge_results(per_doc_results: list[dict]) -> dict:
    """
    Merge a list of per-document extraction dicts into one final record.

    Strategy:
      • Scalar fields (name, DOB, gender, city, state, insurance) — first
        non-null value wins. Documents are processed in folder order, which
        usually puts intake forms first.
      • List fields (icd10_codes, diagnoses, medications) — union across
        all documents, deduplicated case-insensitively.
      • medical_history — concatenate all non-empty values, deduplicated.
    """
    SCALAR_FIELDS = (
        "full_name", "date_of_birth", "gender", "city", "state",
        "insurance_provider", "insurance_id",
    )
    LIST_FIELDS = ("icd10_codes", "diagnoses", "medications")

    merged: dict = {f: None for f in SCALAR_FIELDS}
    for f in LIST_FIELDS:
        merged[f] = []
    merged["medical_history"] = ""

    history_parts: list[str] = []

    for entry in per_doc_results:
        fields = entry.get("fields") or {}
        for f in SCALAR_FIELDS:
            if merged[f] is None and fields.get(f):
                merged[f] = fields[f]
        for f in LIST_FIELDS:
            value = fields.get(f) or []
            if isinstance(value, list):
                merged[f].extend(str(v) for v in value if v)
        history = fields.get("medical_history")
        if history:
            history_parts.append(str(history).strip())

    if history_parts:
        merged["medical_history"] = " ".join(_dedupe_list(history_parts))

    return merged


def _build_provenance(
    per_doc_results: list[dict],
    merged: dict,
) -> list[FieldProvenance]:
    """
    Build the provenance trail by walking the per-doc results in order
    and recording the first document that supplied each scalar field, plus
    every document that contributed any item to a list field.
    """
    provenance: list[FieldProvenance] = []
    SCALAR_FIELDS = (
        "full_name", "date_of_birth", "gender", "city", "state",
        "insurance_provider", "insurance_id",
    )
    LIST_FIELDS = ("icd10_codes", "diagnoses", "medications")

    seen_scalar: set[str] = set()
    for entry in per_doc_results:
        filename = entry["filename"]
        fields = entry.get("fields") or {}
        for f in SCALAR_FIELDS:
            if f in seen_scalar:
                continue
            value = fields.get(f)
            if value:
                provenance.append(FieldProvenance(
                    field_name=f,
                    field_value=str(value),
                    source_file=filename,
                    source_page=None,
                    confidence=1.0,  # placeholder until per-page lookup added
                ))
                seen_scalar.add(f)
        for f in LIST_FIELDS:
            for item in fields.get(f) or []:
                if not item:
                    continue
                provenance.append(FieldProvenance(
                    field_name=f,
                    field_value=str(item),
                    source_file=filename,
                    source_page=None,
                    confidence=1.0,
                ))
    return provenance


# ── Helpers ───────────────────────────────────────────────────────────────────


def _empty_record(folder_path: str, folder_name: str) -> PatientRecord:
    return PatientRecord(
        patient_id=folder_name or _hash_id(folder_path),
        patient_id_source="folder_name" if folder_name else "hash",
        folder_path=folder_path,
        extraction_model=PATIENT_EXTRACTION_MODEL,
        last_extracted_at=datetime.utcnow(),
    )


def _resolve_patient_id(
    folder_name: str,
    folder_path: str,
    extracted_name: str | None,
    extracted_dob: str | None,
) -> tuple[str, str]:
    """
    Decide on a stable patient_id. Priority:
      1. folder name if it looks like an ID (alphanumeric, ≥4 chars)
      2. <lastname>_<firstname>_<dob> if both extracted
      3. SHA-1 hash of folder path (always works, never collides)
    """
    if folder_name and re.match(r"^[A-Za-z0-9_\-]{4,}$", folder_name):
        return folder_name, "folder_name"

    if extracted_name and extracted_dob:
        cleaned_name = re.sub(r"[^A-Za-z0-9]+", "_", extracted_name).strip("_").lower()
        cleaned_dob = re.sub(r"[^0-9]", "", extracted_dob)
        if cleaned_name and cleaned_dob:
            return f"{cleaned_name}_{cleaned_dob}", "extracted"

    return _hash_id(folder_path), "hash"


def _hash_id(s: str) -> str:
    return "p_" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _doc_hash(documents: list[DocumentText]) -> str:
    """Stable hash of all docs combined — used for skip-if-unchanged logic."""
    h = hashlib.sha256()
    for doc in sorted(documents, key=lambda d: d.filename):
        h.update(doc.filename.encode("utf-8"))
        for page_number, text in doc.pages:
            h.update(str(page_number).encode("utf-8"))
            h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def _parse_date(value: object) -> date | None:
    if not value:
        return None
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _normalize_gender(value: object) -> str | None:
    if not value:
        return None
    s = str(value).strip().upper()
    if s in {"M", "MALE"}:
        return "M"
    if s in {"F", "FEMALE"}:
        return "F"
    if s in {"O", "OTHER", "X"}:
        return "O"
    return None


def _dedupe_list(values: list) -> list[str]:
    """Case-insensitive dedupe preserving original casing of first occurrence."""
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


def _parse_json_safely(raw: str) -> dict | None:
    """Try to extract a JSON object from the model output."""
    if not raw:
        return None
    raw = raw.strip()
    # Direct parse
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    # Find first {...} block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group())
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def _call_ollama_json(prompt: str) -> str:
    """
    Call the configured patient extraction model with format=json.
    Returns the raw response string, or empty string on failure.
    """
    payload = {
        "model": PATIENT_EXTRACTION_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",   # forces well-formed JSON output
        "options": {"temperature": 0},
    }
    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=PATIENT_EXTRACTION_TIMEOUT_S,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as e:
        logger.warning(f"  Patient extraction call failed: {e}")
        return ""
