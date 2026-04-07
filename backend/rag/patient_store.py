"""
patient_store.py — DB writes for the patients / provenance / folders tables.

Companion to patient_extractor.py. The extractor produces a PatientRecord
in pure Python; this module persists it to PostgreSQL.

Idempotency contract:
  upsert_patient() is safe to call repeatedly. If a patient with the same
  patient_id already exists, the row is updated and old provenance entries
  are replaced. The source_doc_hash on the record is what callers use to
  decide whether re-extraction is needed at all (skip if unchanged).
"""

import json
import logging
from dataclasses import asdict

from db.connection import db_conn
from rag.patient_extractor import PatientRecord

logger = logging.getLogger(__name__)


# ── Folder type registry ──────────────────────────────────────────────────────


def register_folder(name: str, path: str, folder_type: str) -> None:
    """
    Record a folder's type. UPSERT — safe to call on every ingest.
    folder_type must be one of: 'patient' | 'policy' | 'general'.
    """
    if folder_type not in {"patient", "policy", "general"}:
        raise ValueError(f"invalid folder_type: {folder_type}")

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO folders (name, path, folder_type)
                VALUES (%s, %s, %s)
                ON CONFLICT (name) DO UPDATE
                  SET path = EXCLUDED.path,
                      folder_type = EXCLUDED.folder_type
                """,
                (name, path, folder_type),
            )


def get_folder_type(name: str) -> str | None:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT folder_type FROM folders WHERE name = %s", (name,))
            row = cur.fetchone()
            return row["folder_type"] if row else None


# ── Patient upsert ────────────────────────────────────────────────────────────


def get_existing_doc_hash(patient_id: str) -> str | None:
    """Return the previously stored source_doc_hash, or None if not present."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source_doc_hash FROM patients WHERE patient_id = %s",
                (patient_id,),
            )
            row = cur.fetchone()
            return row["source_doc_hash"] if row else None


def upsert_patient(record: PatientRecord) -> None:
    """
    Insert or update a patient row, then replace its provenance entries.
    """
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO patients (
                    patient_id, folder_path, patient_id_source,
                    full_name, date_of_birth, gender, city, state,
                    insurance_provider, insurance_id,
                    icd10_codes, diagnoses, medications,
                    medical_history, last_extracted_at,
                    source_doc_hash, extraction_model
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s,
                    %s, %s, %s,
                    %s, %s,
                    %s, %s
                )
                ON CONFLICT (patient_id) DO UPDATE SET
                    folder_path        = EXCLUDED.folder_path,
                    patient_id_source  = EXCLUDED.patient_id_source,
                    full_name          = EXCLUDED.full_name,
                    date_of_birth      = EXCLUDED.date_of_birth,
                    gender             = EXCLUDED.gender,
                    city               = EXCLUDED.city,
                    state              = EXCLUDED.state,
                    insurance_provider = EXCLUDED.insurance_provider,
                    insurance_id       = EXCLUDED.insurance_id,
                    icd10_codes        = EXCLUDED.icd10_codes,
                    diagnoses          = EXCLUDED.diagnoses,
                    medications        = EXCLUDED.medications,
                    medical_history    = EXCLUDED.medical_history,
                    last_extracted_at  = EXCLUDED.last_extracted_at,
                    source_doc_hash    = EXCLUDED.source_doc_hash,
                    extraction_model   = EXCLUDED.extraction_model
                """,
                (
                    record.patient_id, record.folder_path, record.patient_id_source,
                    record.full_name, record.date_of_birth, record.gender,
                    record.city, record.state,
                    record.insurance_provider, record.insurance_id,
                    json.dumps(record.icd10_codes),
                    json.dumps(record.diagnoses),
                    json.dumps(record.medications),
                    record.medical_history, record.last_extracted_at,
                    record.source_doc_hash, record.extraction_model,
                ),
            )

            # Replace provenance for this patient
            cur.execute(
                "DELETE FROM patient_field_provenance WHERE patient_id = %s",
                (record.patient_id,),
            )
            for entry in record.provenance:
                cur.execute(
                    """
                    INSERT INTO patient_field_provenance
                        (patient_id, field_name, field_value, source_file, source_page, confidence)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record.patient_id,
                        entry.field_name,
                        entry.field_value,
                        entry.source_file,
                        entry.source_page,
                        entry.confidence,
                    ),
                )


# ── Read API ──────────────────────────────────────────────────────────────────


def get_patient(patient_id: str) -> dict | None:
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM patients WHERE patient_id = %s", (patient_id,))
            row = cur.fetchone()
            if not row:
                return None
            patient = dict(row)
            cur.execute(
                """
                SELECT field_name, field_value, source_file, source_page, confidence
                FROM patient_field_provenance
                WHERE patient_id = %s
                ORDER BY field_name
                """,
                (patient_id,),
            )
            patient["provenance"] = [dict(r) for r in cur.fetchall()]
            return patient


def patient_stats() -> dict:
    """High-level extraction quality stats for the validation API."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM patients")
            total = cur.fetchone()["n"]

            stats: dict = {"total_patients": total, "field_completeness": {}}
            if total == 0:
                return stats

            for field_name in (
                "full_name", "date_of_birth", "gender", "city", "state",
                "insurance_provider", "insurance_id", "medical_history",
            ):
                cur.execute(
                    f"SELECT COUNT(*) AS n FROM patients "
                    f"WHERE {field_name} IS NOT NULL AND {field_name}::text <> ''"
                )
                stats["field_completeness"][field_name] = cur.fetchone()["n"]

            for field_name in ("icd10_codes", "diagnoses", "medications"):
                cur.execute(
                    f"SELECT COUNT(*) AS n FROM patients "
                    f"WHERE jsonb_array_length({field_name}) > 0"
                )
                stats["field_completeness"][field_name] = cur.fetchone()["n"]

            return stats
