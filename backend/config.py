"""
config.py — Central configuration for the Knowledge Companion backend.
All environment-driven settings live here.
"""

import os

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

# ── Knowledge Graph ──────────────────────────────────────────────────────────

GRAPH_EXTRACTION_ENABLED = True   # kill switch — set False to skip entirely

# Fixed vocabulary forces the 3b model to categorize consistently.
# Free-form extraction with small models produces chaos ("THING", "CONCEPT", etc.)
GRAPH_ENTITY_TYPES = [
    "PERSON", "POLICY", "DEPARTMENT", "DATE", "AMOUNT",
    "DOCUMENT", "ORGANIZATION", "ROLE", "LOCATION", "OTHER",
]

GRAPH_RELATION_TYPES = [
    "SUPERSEDES", "REFERENCES", "APPLIES_TO", "AUTHORED_BY",
    "BELONGS_TO", "GOVERNS", "REQUIRES", "CONFLICTS_WITH", "RELATED_TO",
]

GRAPH_MAX_ENTITIES_PER_CHUNK  = 10   # cap per parent chunk
GRAPH_MAX_RELS_PER_CHUNK      = 15
GRAPH_EXTRACTION_BATCH_SIZE   = 5    # parent chunks per LLM call
GRAPH_TRAVERSAL_MAX_DEPTH     = 3    # hops in recursive CTE
GRAPH_TRAVERSAL_MAX_NODES     = 50   # total nodes returned from traversal
