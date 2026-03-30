"""
ingest.py — Document ingestion CLI

Usage:
  python ingest.py                          # process all new PDFs in documents/
  python ingest.py --folder path/to/folder  # process a specific folder
  python ingest.py --force file.pdf         # re-ingest a specific file
  python ingest.py --remove file.pdf        # remove a file from the knowledge base

Coordinates: ocr.py -> chunker.py -> ollama embeddings -> vectorstore.py
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx  # for calling Ollama embedding API

from db.connection import init_pool
from rag.ocr import extract_text_from_pdf
from rag.chunker import build_chunks, compute_doc_id
from rag.vectorstore import VectorStore
from rag.graph_extractor import extract_entities_and_relationships, extract_document_metadata
from rag.graph_store import GraphStore

# ── Configuration ─────────────────────────────────────────────────────────────

DOCUMENTS_DIR   = Path(__file__).parent.parent / "documents"
INGESTION_LOG   = Path(__file__).parent / "ingestion_log.json"
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL     = "nomic-embed-text"
WORKER_MODEL    = "llama3.2:3b"
BATCH_SIZE      = 32   # embed this many chunks per Ollama API call

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Knowledge Companion — Document Ingestion")
    parser.add_argument("--folder", type=str, default=None,   help="Process a specific folder")
    parser.add_argument("--force",  type=str, default=None,   help="Re-ingest a specific PDF (filename)")
    parser.add_argument("--remove", type=str, default=None,   help="Remove a PDF from the knowledge base")
    args = parser.parse_args()

    init_pool()
    store       = VectorStore()
    graph_store = GraphStore()
    log         = _load_log()

    # ── Remove mode ───────────────────────────────────────────────────────────
    if args.remove:
        _remove_document(args.remove, store, graph_store, log)
        return

    # ── Collect PDFs to process ───────────────────────────────────────────────
    search_root = Path(args.folder) if args.folder else DOCUMENTS_DIR
    if not search_root.exists():
        logger.error(f"Folder not found: {search_root}")
        sys.exit(1)

    pdf_files = sorted(search_root.rglob("*.pdf"))
    if not pdf_files:
        logger.info(f"No PDF files found in {search_root}")
        return

    logger.info(f"Found {len(pdf_files)} PDF(s) in {search_root}")

    # ── Process each PDF ──────────────────────────────────────────────────────
    stats = {"processed": 0, "skipped": 0, "failed": 0}

    for pdf_path in pdf_files:
        try:
            _process_pdf(pdf_path, store, graph_store, log, force=(args.force == pdf_path.name))
            stats["processed"] += 1
        except AlreadyIngestedException:
            stats["skipped"] += 1
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")
            stats["failed"] += 1

    _save_log(log)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        f"\nDone — {stats['processed']} processed, "
        f"{stats['skipped']} skipped, "
        f"{stats['failed']} failed"
    )
    logger.info(f"Total chunks in knowledge base: {store.count()}")


# ── Core processing ───────────────────────────────────────────────────────────

def _process_pdf(
    pdf_path: Path,
    store: VectorStore,
    graph_store: GraphStore,
    log: dict,
    force: bool = False,
) -> None:
    """Full pipeline for a single PDF file."""
    from config import GRAPH_EXTRACTION_ENABLED

    doc_id = compute_doc_id(str(pdf_path))
    filename = pdf_path.name
    folder = pdf_path.parent.name  # just the folder name, e.g. "RAG Techniques"

    # Skip if already ingested (unless forced)
    if not force and doc_id in log:
        logger.info(f"  Skipping {filename} — already ingested")
        raise AlreadyIngestedException()

    # If forcing re-ingest, remove old data first
    if force and doc_id in log:
        logger.info(f"  Force re-ingesting {filename} — removing old data")
        store.delete_by_doc_id(doc_id)
        graph_store.delete_by_doc_id(doc_id)

    logger.info(f"Processing: {filename}")
    start = time.time()

    # Step 1 — Extract text (with OCR fallback)
    pages = extract_text_from_pdf(str(pdf_path))
    if not pages:
        raise ValueError(f"No pages extracted from {filename}")

    # Step 2 — Build hierarchical chunks
    chunks = build_chunks(pages, doc_id=doc_id, source=filename, folder=folder)
    if not chunks:
        raise ValueError(f"No chunks produced from {filename}")

    # Step 3 — Extract metadata with llama3.2:3b
    chunks = _enrich_metadata(chunks)

    # Step 3b — Extract knowledge graph entities and relationships
    #   Why here?  We need parent chunks (Step 2) but not embeddings (Step 4).
    #   Wrapped in try/except: graph extraction is supplementary.
    #   If it fails, the document is still ingested normally (fail-open).
    entity_count = 0
    rel_count = 0

    if GRAPH_EXTRACTION_ENABLED:
        try:
            entities, relationships = extract_entities_and_relationships(
                chunks, doc_id, filename, folder,
            )
            doc_meta = extract_document_metadata(chunks, doc_id, filename, folder)
            entity_count = len(entities)
            rel_count = len(relationships)
        except Exception as e:
            logger.warning(f"  Graph extraction failed (non-critical): {e}")
            entities, relationships, doc_meta = [], [], None

    # Step 4 — Embed all child chunks via Ollama
    texts = [c.text for c in chunks]
    embeddings = _embed_batch(texts)

    # Step 5 — Store chunks in vectorstore
    store.add_chunks(chunks, embeddings)

    # Step 5b — Store graph data
    if GRAPH_EXTRACTION_ENABLED and (entities or doc_meta):
        try:
            graph_store.store_entities(entities)
            graph_store.store_relationships(relationships)
            if doc_meta:
                graph_store.store_document_metadata(doc_meta)
        except Exception as e:
            logger.warning(f"  Graph storage failed (non-critical): {e}")

    elapsed = round(time.time() - start, 1)

    # Step 6 — Update ingestion log
    log[doc_id] = {
        "filename":      filename,
        "folder":        folder,
        "chunks":        len(chunks),
        "pages":         len(pages),
        "entities":      entity_count,
        "relationships": rel_count,
        "ingested_at":   datetime.utcnow().isoformat(),
        "elapsed_s":     elapsed,
    }

    logger.info(
        f"  Done: {len(chunks)} chunks, {entity_count} entities, "
        f"{rel_count} relationships in {elapsed}s"
    )


# ── Metadata enrichment ───────────────────────────────────────────────────────

def _enrich_metadata(chunks):
    """
    Ask llama3.2:3b to identify the section heading and document type
    from the first parent chunk. Apply to all chunks from that document.
    Fast — only one LLM call per document, not per chunk.
    """
    if not chunks:
        return chunks

    sample_text = chunks[0].parent_text[:800]  # first 800 chars is enough

    prompt = (
        "Read this document excerpt and reply in JSON only.\n"
        "Return exactly: {\"section\": \"<main section heading or topic>\", "
        "\"doc_type\": \"<policy|manual|report|legal|other>\"}\n\n"
        f"Excerpt:\n{sample_text}"
    )

    try:
        response = httpx.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": WORKER_MODEL, "prompt": prompt, "stream": False},
            timeout=30,
        )
        response.raise_for_status()
        raw = response.json().get("response", "{}")
        # Extract JSON from response (model may add extra text)
        import re
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            meta = json.loads(match.group())
            for chunk in chunks:
                chunk.metadata["section"]  = meta.get("section", "")
                chunk.metadata["doc_type"] = meta.get("doc_type", "other")
    except Exception as e:
        logger.warning(f"Metadata extraction failed (non-critical): {e}")

    return chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed_batch(texts: list[str]) -> list[list[float]]:
    """
    Call Ollama's embedding API in batches.
    Returns one 768-dim vector per text.
    """
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/embed",
                json={"model": EMBED_MODEL, "input": batch},
                timeout=120,
            )
            response.raise_for_status()
            all_embeddings.extend(response.json()["embeddings"])
        except Exception as e:
            logger.error(f"Embedding batch {i // BATCH_SIZE + 1} failed: {e}")
            raise

    return all_embeddings


# ── Remove document ───────────────────────────────────────────────────────────

def _remove_document(filename: str, store: VectorStore, graph_store: GraphStore, log: dict) -> None:
    """Remove all chunks and graph data for a document by filename."""
    # Find doc_id by filename in the log
    doc_id = next(
        (k for k, v in log.items() if v["filename"] == filename), None
    )
    if not doc_id:
        logger.error(f"'{filename}' not found in ingestion log")
        return

    store.delete_by_doc_id(doc_id)
    graph_store.delete_by_doc_id(doc_id)
    del log[doc_id]
    _save_log(log)
    logger.info(f"Removed '{filename}' from knowledge base")


# ── Log helpers ───────────────────────────────────────────────────────────────

def _load_log() -> dict:
    if INGESTION_LOG.exists():
        with open(INGESTION_LOG) as f:
            return json.load(f)
    return {}


def _save_log(log: dict) -> None:
    with open(INGESTION_LOG, "w") as f:
        json.dump(log, f, indent=2)


# ── Custom exceptions ─────────────────────────────────────────────────────────

class AlreadyIngestedException(Exception):
    pass


if __name__ == "__main__":
    main()
