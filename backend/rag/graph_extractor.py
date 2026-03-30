"""
graph_extractor.py — LLM-based entity and relationship extraction

Reads parent chunks and asks llama3.2:3b to identify entities (nouns)
and relationships (verbs connecting them).  Results feed into the
knowledge graph stored in PostgreSQL.

Design decisions:
  - Batches parent chunks (5 per call) to minimise LLM round-trips
  - Uses few-shot prompting for consistent output from a 3b model
  - Regex-extracts JSON from LLM responses (tolerates prose wrapping)
  - Deterministic IDs via hashing for idempotent re-ingestion
  - Graceful fallback: extraction failure → empty lists (non-critical)
"""

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field

import httpx

from config import (
    GRAPH_ENTITY_TYPES,
    GRAPH_EXTRACTION_BATCH_SIZE,
    GRAPH_MAX_ENTITIES_PER_CHUNK,
    GRAPH_MAX_RELS_PER_CHUNK,
    GRAPH_RELATION_TYPES,
    OLLAMA_BASE_URL,
    WORKER_MODEL,
)

logger = logging.getLogger(__name__)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Entity:
    """A noun extracted from a document chunk."""
    id:              str
    name:            str
    name_normalized: str
    entity_type:     str
    doc_id:          str
    source:          str
    folder:          str
    chunk_id:        str
    page:            int
    description:     str = ""
    properties:      dict = field(default_factory=dict)


@dataclass(frozen=True)
class Relationship:
    """A directed edge between two entities."""
    id:               str
    source_entity_id: str
    target_entity_id: str
    relation_type:    str
    description:      str
    confidence:       float
    doc_id:           str
    chunk_id:         str


@dataclass(frozen=True)
class DocumentMetadata:
    """Summary card for an ingested PDF."""
    doc_id:           str
    source:           str
    folder:           str
    title:            str = ""
    doc_type:         str = ""
    version:          str = ""
    effective_date:   str = ""
    summary:          str = ""
    cross_references: tuple[str, ...] = ()


# ── Public API ───────────────────────────────────────────────────────────────

def extract_entities_and_relationships(
    chunks: list,
    doc_id: str,
    source: str,
    folder: str,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Extract entities and relationships from document chunks.

    Groups chunks by parent text (deduped), batches them, and sends
    each batch to the LLM.  Returns all entities and relationships
    found across the entire document.
    """
    # Deduplicate parent texts — multiple child chunks share a parent
    seen_parents: set[str] = set()
    parent_chunk_map: list[tuple[str, str, int]] = []  # (parent_text, chunk_id, page)

    for chunk in chunks:
        if chunk.parent_text not in seen_parents:
            seen_parents.add(chunk.parent_text)
            parent_chunk_map.append((chunk.parent_text, chunk.chunk_id, chunk.page))

    logger.info(
        f"  Graph extraction: {len(parent_chunk_map)} unique parents "
        f"(from {len(chunks)} child chunks)"
    )

    all_entities: list[Entity] = []
    all_relationships: list[Relationship] = []

    # Process in batches
    for i in range(0, len(parent_chunk_map), GRAPH_EXTRACTION_BATCH_SIZE):
        batch = parent_chunk_map[i : i + GRAPH_EXTRACTION_BATCH_SIZE]
        batch_num = i // GRAPH_EXTRACTION_BATCH_SIZE + 1

        try:
            entities, rels = _extract_batch(batch, doc_id, source, folder)
            all_entities.extend(entities)
            all_relationships.extend(rels)
            logger.info(
                f"    Batch {batch_num}: {len(entities)} entities, {len(rels)} relationships"
            )
        except Exception as e:
            logger.warning(f"    Batch {batch_num} extraction failed (non-critical): {e}")

    # Deduplicate entities by ID (same entity found in multiple parents)
    unique_entities = list({e.id: e for e in all_entities}.values())

    logger.info(
        f"  Graph totals: {len(unique_entities)} entities, "
        f"{len(all_relationships)} relationships"
    )

    return unique_entities, all_relationships


def extract_document_metadata(
    chunks: list,
    doc_id: str,
    source: str,
    folder: str,
) -> DocumentMetadata:
    """
    Extract document-level metadata (title, type, version, summary)
    from the first few parent chunks — enough to capture the title page
    and introduction.
    """
    # Take up to 3 unique parent texts from the start of the document
    seen: set[str] = set()
    intro_texts: list[str] = []
    for chunk in chunks:
        if chunk.parent_text not in seen and len(intro_texts) < 3:
            seen.add(chunk.parent_text)
            intro_texts.append(chunk.parent_text)

    combined = "\n\n---\n\n".join(intro_texts)

    prompt = (
        "Read this document excerpt and extract metadata. "
        "Reply with JSON only, no other text.\n\n"
        "Return exactly:\n"
        '{"title": "<document title>", '
        '"doc_type": "<policy|manual|report|legal|other>", '
        '"version": "<version string if found, else empty>", '
        '"effective_date": "<date if found, else empty>", '
        '"summary": "<2-3 sentence summary of what this document covers>", '
        '"cross_references": ["<other document names referenced>"]}\n\n'
        f"Excerpt:\n{combined[:3000]}"
    )

    try:
        raw = _call_llm(prompt)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            data = json.loads(match.group())
            return DocumentMetadata(
                doc_id=doc_id,
                source=source,
                folder=folder,
                title=data.get("title", ""),
                doc_type=data.get("doc_type", ""),
                version=data.get("version", ""),
                effective_date=data.get("effective_date", ""),
                summary=data.get("summary", ""),
                cross_references=tuple(data.get("cross_references", [])),
            )
    except Exception as e:
        logger.warning(f"  Document metadata extraction failed (non-critical): {e}")

    return DocumentMetadata(doc_id=doc_id, source=source, folder=folder)


# ── Batch extraction ─────────────────────────────────────────────────────────

def _extract_batch(
    batch: list[tuple[str, str, int]],
    doc_id: str,
    source: str,
    folder: str,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Send a batch of parent chunks to the LLM and parse the response
    into Entity and Relationship objects.
    """
    prompt = _build_extraction_prompt([text for text, _, _ in batch])
    raw = _call_llm(prompt)
    return _parse_extraction_response(raw, batch, doc_id, source, folder)


def _build_extraction_prompt(parent_texts: list[str]) -> str:
    """
    Construct the extraction prompt with few-shot examples.

    Why few-shot?  A 3b model follows examples far better than
    abstract instructions.  One concrete example teaches it the
    output format more reliably than a page of rules.
    """
    entity_types = ", ".join(GRAPH_ENTITY_TYPES)
    relation_types = ", ".join(GRAPH_RELATION_TYPES)

    # Number each text block so the model can reference them
    text_blocks = []
    for i, text in enumerate(parent_texts, 1):
        text_blocks.append(f"[BLOCK {i}]\n{text[:600]}")

    texts_section = "\n\n".join(text_blocks)

    return f"""Extract all named entities and relationships from the text blocks below.

ENTITY TYPES (use ONLY these): {entity_types}
RELATIONSHIP TYPES (use ONLY these): {relation_types}

Reply with JSON only. No explanation, no markdown fences.

Format:
{{"entities": [
    {{"name": "entity name", "type": "ENTITY_TYPE", "description": "one-line description"}}
  ],
  "relationships": [
    {{"source": "entity name", "target": "entity name", "type": "RELATION_TYPE", "description": "what this relationship means"}}
  ]
}}

Example — given text about "The Data Protection Policy applies to the IT Department and was written by James Lee":
{{"entities": [
    {{"name": "Data Protection Policy", "type": "POLICY", "description": "Policy governing data protection practices"}},
    {{"name": "IT Department", "type": "DEPARTMENT", "description": "Department responsible for information technology"}},
    {{"name": "James Lee", "type": "PERSON", "description": "Author of the Data Protection Policy"}}
  ],
  "relationships": [
    {{"source": "Data Protection Policy", "target": "IT Department", "type": "APPLIES_TO", "description": "Policy applies to IT Department"}},
    {{"source": "Data Protection Policy", "target": "James Lee", "type": "AUTHORED_BY", "description": "James Lee wrote the policy"}}
  ]
}}

Now extract from these text blocks:

{texts_section}"""


# ── Response parsing ─────────────────────────────────────────────────────────

def _parse_extraction_response(
    raw: str,
    batch: list[tuple[str, str, int]],
    doc_id: str,
    source: str,
    folder: str,
) -> tuple[list[Entity], list[Relationship]]:
    """
    Parse the LLM's JSON response into typed objects.

    Tolerates: prose wrapping, unknown types (mapped to OTHER/RELATED_TO),
    malformed entries (skipped individually).
    """
    # Extract JSON from response — model often wraps it in text
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        logger.warning("  No JSON found in extraction response")
        return [], []

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as e:
        logger.warning(f"  JSON parse failed: {e}")
        return [], []

    # Use the first chunk in the batch as the reference for chunk_id and page
    # (entities span the batch, we attribute to the first chunk for simplicity)
    ref_chunk_id = batch[0][1]
    ref_page = batch[0][2]

    raw_entities = data.get("entities", [])[:GRAPH_MAX_ENTITIES_PER_CHUNK]
    raw_rels = data.get("relationships", [])[:GRAPH_MAX_RELS_PER_CHUNK]

    # Build entities
    entities: list[Entity] = []
    entity_id_by_name: dict[str, str] = {}  # name_normalized → entity_id

    for raw_ent in raw_entities:
        name = raw_ent.get("name", "").strip()
        if not name:
            continue

        name_normalized = name.lower().strip()
        entity_type = raw_ent.get("type", "OTHER").upper()
        if entity_type not in GRAPH_ENTITY_TYPES:
            entity_type = "OTHER"

        entity_id = _make_id(name_normalized, entity_type, doc_id)
        entity_id_by_name[name_normalized] = entity_id

        entities.append(Entity(
            id=entity_id,
            name=name,
            name_normalized=name_normalized,
            entity_type=entity_type,
            doc_id=doc_id,
            source=source,
            folder=folder,
            chunk_id=ref_chunk_id,
            page=ref_page,
            description=raw_ent.get("description", ""),
        ))

    # Build relationships — only if both source and target entities exist
    relationships: list[Relationship] = []

    for raw_rel in raw_rels:
        src_name = raw_rel.get("source", "").strip().lower()
        tgt_name = raw_rel.get("target", "").strip().lower()

        src_id = entity_id_by_name.get(src_name)
        tgt_id = entity_id_by_name.get(tgt_name)

        if not src_id or not tgt_id:
            continue  # skip if either entity wasn't extracted

        rel_type = raw_rel.get("type", "RELATED_TO").upper()
        if rel_type not in GRAPH_RELATION_TYPES:
            rel_type = "RELATED_TO"

        rel_id = _make_id(src_id, tgt_id, rel_type)

        relationships.append(Relationship(
            id=rel_id,
            source_entity_id=src_id,
            target_entity_id=tgt_id,
            relation_type=rel_type,
            description=raw_rel.get("description", ""),
            confidence=0.7,  # default; could be LLM-assessed later
            doc_id=doc_id,
            chunk_id=ref_chunk_id,
        ))

    return entities, relationships


# ── Helpers ──────────────────────────────────────────────────────────────────

# ── Query-time extraction ────────────────────────────────────────────────────
#
# These functions run in the query pipeline (Phase 3), not during ingestion.
# They extract entities from the USER'S QUESTION — much simpler than
# extracting from a full document chunk.
#
# Why separate from ingestion extraction?
#   Ingestion: thorough, batched, can be slow (offline process)
#   Query-time: fast, single question, must complete in <2 seconds

def extract_entities_from_question(question: str) -> list[dict]:
    """
    Extract named entities from a user question.

    Returns a lightweight list of {name, type} dicts — no IDs,
    no relationships, just enough to look up in the graph.

    Uses a simpler, shorter prompt than ingestion extraction
    because speed matters here (query hot path).
    """
    entity_types = ", ".join(GRAPH_ENTITY_TYPES)

    prompt = (
        "Extract all named entities from this question. "
        "Reply with JSON only, no other text.\n\n"
        f"Entity types: {entity_types}\n\n"
        'Format: [{{"name": "entity name", "type": "TYPE"}}]\n\n'
        "If no entities found, return []\n\n"
        f"Question: {question}"
    )

    try:
        raw = _call_llm(prompt)
        # Find JSON array in response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            entities = json.loads(match.group())
            return [
                {"name": e.get("name", "").strip(), "type": e.get("type", "OTHER")}
                for e in entities
                if e.get("name", "").strip()
            ]
    except Exception as e:
        logger.warning(f"Question entity extraction failed: {e}")

    return []


def resolve_question_entities(
    extracted: list[dict],
    graph_store,
    folders: list[str] | None = None,
) -> list[dict]:
    """
    Match extracted entity names against the stored graph.

    Fuzzy matching via ILIKE — "leave policy" matches "Annual Leave Policy".
    Returns the actual stored Entity rows for matched names.

    Why fuzzy?  Users rarely type exact entity names.  "the leave policy"
    should match "Annual Leave Policy v3".
    """
    matched: list[dict] = []

    for entity in extracted:
        results = graph_store.find_entities_by_name(
            name=entity["name"],
            entity_type=entity["type"] if entity["type"] != "OTHER" else None,
            folders=folders,
        )
        matched.extend(results)

    # Deduplicate by entity ID
    seen: set[str] = set()
    unique: list[dict] = []
    for m in matched:
        eid = m["id"]
        if eid not in seen:
            seen.add(eid)
            unique.append(m)

    return unique


def _call_llm(prompt: str) -> str:
    """Send a prompt to llama3.2:3b and return the raw text response."""
    response = httpx.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": WORKER_MODEL, "prompt": prompt, "stream": False},
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def _make_id(*parts: str) -> str:
    """
    Deterministic ID from component strings.

    Why deterministic?  Re-ingesting the same document produces the
    same IDs, so INSERT ... ON CONFLICT DO NOTHING naturally deduplicates.
    With random UUIDs, every re-ingestion creates duplicates.
    """
    raw = "|".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
