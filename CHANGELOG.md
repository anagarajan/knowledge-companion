# Changelog

## [2.3.0] - 2026-04-20

### Added — Track 2: query routing (text → SQL vs RAG)

Questions about structured patient data now bypass the RAG pipeline and
query the patients table directly. Aggregation questions like "how many
female patients are on Metformin?" now return instant, accurate answers.

- **Intent classifier** (`backend/rag/intent_classifier.py`)
  - One llama3.2:3b call classifies every question as SQL / HYBRID / RAG
  - SQL: structured queries answered from the patients table
  - HYBRID: SQL to find patients, then RAG scoped to their folders
  - RAG: existing pipeline, unchanged
  - Falls back to RAG on any classification error

- **Patient query module** (`backend/rag/patient_query.py`)
  - LLM outputs a JSON filter spec, not raw SQL — injection-proof by design
  - Whitelist of allowed columns and aggregation functions
  - Parameterised SQL built from the validated filter spec
  - `statement_timeout` and row cap (`PATIENT_QUERY_MAX_ROWS=100`)
  - Uses GIN indexes on `icd10_codes`, `diagnoses`, `medications`
  - Result formatting via llama3.2:3b — markdown tables, age calculation

- **Pipeline routing** (`backend/rag/pipeline.py`)
  - New intent classification step before complexity routing
  - RAG pipeline extracted into `_run_rag_pipeline()` — shared by RAG and HYBRID
  - `query_route` field on QueryResult surfaced in SSE done event

- **JSON generation helper** (`backend/models/ollama.py`)
  - `generate_worker_json()` — Ollama `format=json` for structured outputs

### Configuration (`backend/config.py`)

- `QUERY_ROUTING_ENABLED` — kill switch, set False to always use RAG
- `PATIENT_QUERY_MAX_ROWS` — cap rows returned (default 100)
- `PATIENT_QUERY_TIMEOUT_MS` — per-query statement_timeout (default 5000)

### Notes

- Track 3 (patient browser UI) deferred to a future release.
- The `query_route` field in the SSE done event lets the frontend display
  a badge or icon showing which path answered the question.

## [2.2.0] - 2026-04-07

### Added — Track 1: structured patient extraction layer

Foundation for healthcare use cases (Noctrix) where users ask aggregation
questions like "list all patients above 65" or "how many female patients had
Metformin". Pure RAG cannot answer these — they need a structured table.

- **OCR pipeline upgrades** (`backend/rag/ocr.py`)
  - File-hash-keyed cache so re-ingestion never re-OCRs the same file
    (cache lives at `backend/ocr_cache/`, gitignored)
  - `force_ocr=True` parameter bypasses the embedded-text heuristic — used
    automatically for `--type patient` because scanned medical PDFs often
    have unreliable text layers (page numbers / headers only)
  - Image preprocessing: grayscale + autocontrast + binarization
  - Per-page OCR confidence scoring; pages below 60 are logged to
    `logs/ocr_low_confidence.log` for review
  - Thread-parallel page-level OCR (Tesseract releases the GIL)
  - Tesseract `--psm 6` for medical forms

- **New schema** (`backend/db/connection.py`)
  - `patients` table with GIN indexes on `icd10_codes`, `diagnoses`,
    `medications` for fast multi-value filters
  - `patient_field_provenance` table — every extracted field links back
    to its source file and page
  - `folders` table — registers each ingested folder as
    patient | policy | general

- **Map-reduce extractor** (`backend/rag/patient_extractor.py`)
  - One LLM call per document (MAP) plus a merge step (REDUCE)
  - Strict JSON output via Ollama `format=json`
  - Configurable model via `PATIENT_EXTRACTION_MODEL` (default `llama3.1:8b`)
  - Patient ID resolution: folder name → extracted name+DOB → hash fallback
  - Prompt instructs the model to return null for missing fields, never guess

- **Persistence layer** (`backend/rag/patient_store.py`)
  - Idempotent UPSERT keyed on `patient_id`
  - `source_doc_hash` enables skip-if-unchanged for fast re-runs

- **Ingest CLI changes** (`backend/ingest.py`)
  - `--type patient|policy|general` (default `general`)
  - `--no-extract` to skip extraction (chunks only, fast for testing)
  - `--extract-only` to re-run extraction without re-chunking
  - `--force-ocr` to bypass the OCR heuristic
  - `--extract-patient PATIENT_ID` to re-extract a single patient
  - Patient mode groups PDFs by parent folder and runs extraction per patient
  - Resumable: skips patients whose source documents have not changed
  - Errors logged to `logs/extraction_errors.log` so long batches can continue

- **Validation API**
  - `GET /api/patients/stats` — field completeness across all patients
  - `GET /api/patients/{patient_id}` — full record + provenance trail

### Configuration (`backend/config.py`)

New section for OCR (`OCR_CACHE_ENABLED`, `OCR_PARALLEL_PAGES`,
`OCR_MAX_WORKERS`, `OCR_PREPROCESS`, `OCR_TESSERACT_PSM`,
`OCR_LOW_CONFIDENCE_THRESHOLD`, `OCR_ENGINE`) and patient extraction
(`PATIENT_EXTRACTION_ENABLED`, `PATIENT_EXTRACTION_MODEL`,
`PATIENT_EXTRACTION_MAX_CHARS`, `PATIENT_EXTRACTION_TIMEOUT_S`).

### Notes

- Track 1 only — query routing (text → SQL vs RAG) is Track 3, frontend
  patient browser is Track 4. Use the validation API endpoints for now.
- Tesseract is the only OCR engine wired in this release. The `OCR_ENGINE`
  config flag is in place so PaddleOCR / docTR / Surya can be added later
  without code changes elsewhere.

## [2.1.0] - 2026-04-05

### Fixed
- Backend port mismatch: `start.sh` launched backend on port 8001 but Vite proxy targeted port 8000 — unified to port 8000
- Empty state flash on startup: frontend now retries API connection with a loading spinner instead of showing "No conversations yet" while backend starts

### Changed
- Frontend dev server port changed from 5173 to 5457
- Added re-ingest instruction to `start.sh` startup message for when entity/relation types are changed in config.py

## [2.0.0] - 2026-03-31

- Initial release with PDF ingestion, hybrid RAG pipeline, knowledge graph, and entity browser
