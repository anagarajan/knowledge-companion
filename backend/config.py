"""
config.py — Central configuration for the Knowledge Companion backend.
All environment-driven settings live here.
"""

import os
from pathlib import Path

# ── Database ──────────────────────────────────────────────────────────────────

# Full PostgreSQL connection URL
# Format: postgresql://user:password@host:port/dbname
# Override via environment variable in production
DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/knowledge_companion",
)

# ── Ollama ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL     = "nomic-embed-text"
WORKER_MODEL    = "llama3.2:3b"
REASONER_MODEL  = "llama3.1:8b"

# ── Documents ─────────────────────────────────────────────────────────────────

DOCUMENTS_DIR = Path(__file__).parent.parent / "documents"

# ── Ingestion ─────────────────────────────────────────────────────────────────

EMBED_BATCH_SIZE = 32     # chunks per Ollama embedding call
MIN_TEXT_CHARS   = 50     # below this → treat page as scanned
OCR_DPI          = 300    # render DPI for scanned pages

# ── Retrieval ─────────────────────────────────────────────────────────────────

SEMANTIC_TOP_K  = 20      # candidates from pgvector search
BM25_TOP_K      = 20      # candidates from BM25
FINAL_TOP_K     = 5       # survivors after re-ranking
SCORE_THRESHOLD = 0.45    # minimum re-rank score to proceed

# ── Pipeline ──────────────────────────────────────────────────────────────────

MAX_HISTORY_TURNS    = 12   # sliding window of chat history
PARENT_CHUNK_SIZE    = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE     = 400
CHILD_CHUNK_OVERLAP  = 40
