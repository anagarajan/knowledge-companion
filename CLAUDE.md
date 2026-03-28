# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

A fully local, private RAG (Retrieval-Augmented Generation) system for chatting with thousands of PDFs — no data leaves the machine. Runs on a Mac Mini with Ollama (Apple Metal) and PostgreSQL + pgvector.

## Commands

### Backend

```bash
# Install Python dependencies
cd backend && pip install -r requirements.txt

# Start the API server (development, auto-reload)
cd backend && uvicorn main:app --reload --port 8000

# Ingest a folder of PDFs
cd backend && python ingest.py --folder ../documents/HR
cd backend && python ingest.py --folder ../documents/HR --force   # re-ingest even if already stored
cd backend && python ingest.py --folder ../documents/HR --remove  # remove from DB

# Health check (confirms DB is up and chunks are indexed)
curl http://localhost:8000/api/health
```

### Frontend

```bash
cd frontend && npm install
cd frontend && npm run dev        # dev server on :5173
cd frontend && npm run build      # production build
```

### PostgreSQL + pgvector

```bash
# Requires PostgreSQL running locally with pgvector extension
# Default connection: postgresql://postgres:postgres@localhost:5432/knowledge_companion
# Override via: export DATABASE_URL=postgresql://...

# Create the database once
psql -U postgres -c "CREATE DATABASE knowledge_companion;"

# Tables and indexes are created automatically on first server start (init_db in db/connection.py)
```

### Ollama models required

```bash
ollama pull nomic-embed-text   # embeddings
ollama pull llama3.2:3b        # fast worker (HyDE, re-ranking, routing, confidence)
ollama pull llama3.2:8b        # reasoner (complex queries only)
```

## Architecture

```
User question
     │
     ▼
FastAPI (main.py) — SSE streaming
     │
     ▼
Pipeline (rag/pipeline.py)
  1. classify_complexity      → SIMPLE | COMPLEX
  2. HyDE query expansion     → hypothetical answer → embedding vector
  3. Hybrid retrieval         → semantic (pgvector) + BM25, merged via Reciprocal Rank Fusion
  4. Re-ranking               → llama3.2:3b scores each chunk 1–10
  5. Threshold gate           → below 0.45 → fallback response
  6. Prompt construction      → parent chunks (2000 chars) injected into system prompt
  7. Streaming answer         → llama3.2:3b (SIMPLE) or llama3.2:8b (COMPLEX)
  8. Confidence scoring       → HIGH | MEDIUM | LOW with reason
     │
     ▼
React frontend — SSE event stream → token-by-token display
```

## Key Files

| File | Purpose |
|---|---|
| `backend/main.py` | FastAPI app, all HTTP endpoints, SSE streaming |
| `backend/rag/pipeline.py` | Full RAG orchestration — entry point for every question |
| `backend/rag/retriever.py` | Hybrid search (pgvector + BM25) + re-ranking |
| `backend/rag/vectorstore.py` | pgvector read/write (chunks table) |
| `backend/rag/hyde.py` | HyDE query expansion |
| `backend/rag/chunker.py` | Hierarchical chunking (parent 2000 / child 400 chars) |
| `backend/rag/ocr.py` | PDF text extraction + pytesseract OCR fallback |
| `backend/models/ollama.py` | Single Ollama interface (embed, generate, stream) |
| `backend/db/connection.py` | PostgreSQL connection pool + schema creation |
| `backend/db/sessions.py` | Session and message CRUD |
| `backend/ingest.py` | CLI: PDF → chunks → embeddings → PostgreSQL |
| `backend/config.py` | All constants (DB_URL, model names, chunk sizes, thresholds) |

## Data Model

**chunks table** — one row per child chunk
- `embedding vector(768)` — nomic-embed-text output, HNSW index for fast cosine search
- `parent_text` — the larger surrounding context sent to the LLM
- `content` — the smaller child chunk used for matching

**sessions table** — one row per chat tab
**messages table** — one row per user question or assistant answer

JSONB is used for `folders`, `sources`, and `confidence` — parsed automatically by psycopg2.

## Three Ollama Models Only

No external ML libraries. All intelligence via Ollama:

| Model | Used for |
|---|---|
| `nomic-embed-text` | Embedding queries and chunks at ingest time |
| `llama3.2:3b` | HyDE, complexity routing, re-ranking, confidence scoring, topic switch detection |
| `llama3.2:8b` | Answering complex questions only |

## Anti-Hallucination Layers

1. Score threshold (0.45) — no answer if nothing relevant found
2. Strict system prompt — "answer ONLY from provided context"
3. Temperature = 0 — no creativity, pure retrieval
4. Multi-chunk agreement — conflict detection across top chunks
5. Source citations — every answer cites filename + page

## Config Tuning (backend/config.py)

- `SEMANTIC_TOP_K / BM25_TOP_K` — candidates before re-ranking (default 20 each)
- `FINAL_TOP_K` — survivors after re-ranking (default 5)
- `SCORE_THRESHOLD` — minimum re-rank score (default 0.45)
- `MAX_HISTORY_TURNS` — sliding window for chat history (default 12, drops to 4 on topic switch)
- `PARENT_CHUNK_SIZE / CHILD_CHUNK_SIZE` — chunk sizes at ingest time
