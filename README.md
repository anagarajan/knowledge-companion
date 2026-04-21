# Knowledge Companion

A fully private, local AI assistant for your documents. Ask questions about thousands of PDFs — no data leaves your machine.

Built for Apple Silicon Mac. All AI runs locally via Ollama.

---

## Quick Start

```bash
git clone <this-repo>
cd knowledge-companion
./start.sh
```

First run takes 10–15 minutes (downloads and installs everything automatically). Every run after starts in seconds.

The app opens at **http://localhost:5457**

### What gets installed automatically

- **PostgreSQL 16** + pgvector (vector database)
- **Ollama** + 3 AI models (embedding, fast worker, reasoner)
- **Python 3.12** + virtual environment with all dependencies
- **Node.js** + frontend packages

No manual setup required — `./start.sh` handles everything.

---

## Adding Documents

Ingest PDFs from **any folder on your system** — they don't need to be inside the project directory.

### Using the ingest script (recommended)

Make sure the app is running (`./start.sh`), then:

```bash
./ingest.sh /path/to/any/folder
```

Examples:
```bash
# Ingest from anywhere on disk
./ingest.sh ~/Documents/HR
./ingest.sh /Users/me/Desktop/Legal
./ingest.sh ~/Dropbox/Company-Policies

# Re-ingest a folder (updates existing documents)
./ingest.sh ~/Documents/HR --force

# Remove a specific file from the knowledge base
./ingest.sh --remove "Annual Leave Policy.pdf"
```

The folder name (e.g. `HR`, `Legal`) appears in the app's sidebar. Select it to search those documents.

### Ingesting patient records (healthcare mode)

For folders containing one subfolder per patient (each with their own scanned
PDFs of intake forms, labs, prescriptions, etc.), use `--type patient`:

```bash
./ingest.sh /path/to/patients --type patient
```

This enables:
- **Forced OCR** on every page (scanned medical PDFs often have unreliable
  embedded text — page numbers and headers only)
- **Structured extraction** of name, DOB, gender, city, state, insurance,
  ICD-10 codes, diagnoses, and medications into the `patients` table
- **Provenance tracking** — every extracted field links back to the
  source file and page

Useful flags during testing:
```bash
./ingest.sh /path/to/patients --type patient --no-extract   # chunks only, fast
./ingest.sh /path/to/patients --type patient --extract-only # re-run extraction
```

Verify the results without a UI:
```bash
curl http://localhost:8000/api/patients/stats              # field completeness
curl http://localhost:8000/api/patients/<patient_id>       # full record + provenance
```

### Using Python directly (manual setup)

If you're running the backend yourself instead of using `./start.sh`, you must use the project's virtual environment. Bare `python3` will fail with `ModuleNotFoundError` because the dependencies are installed in the venv, not system-wide.

**Option A — Activate the venv first:**

```bash
cd backend
source .venv/bin/activate
python ingest.py --folder /path/to/any/folder
```

**Option B — Call the venv Python directly (no activation needed):**

```bash
cd backend
.venv/bin/python ingest.py --folder /path/to/any/folder
```

Both options are equivalent. Option B is useful for one-off commands or scripts.

**Common mistake:**
```bash
# This will NOT work — system Python doesn't have the project's packages
python3 ingest.py --folder ~/Documents/HR
# → ModuleNotFoundError: No module named 'psycopg2'

# This WILL work
.venv/bin/python ingest.py --folder ~/Documents/HR
```

### Setting up the venv from scratch

If you cloned the repo without running `./start.sh`:

```bash
cd backend
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Then use `.venv/bin/python` for all commands.

### Starting the backend manually

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

Or without activation:
```bash
cd backend
.venv/bin/uvicorn main:app --reload --port 8000
```

---

## Updating

When a new version is available, just run:

```bash
./update.sh
```

This handles everything automatically:
- Stops the running app
- Pulls the latest code
- Reinstalls Python/Node dependencies (only if they changed)
- Shows what's new from the changelog
- Restarts the app

No manual steps needed.

---

## Stopping the App

```bash
./start.sh --stop
```

---

## Logs

```
logs/backend.log    — FastAPI server
logs/frontend.log   — React dev server
logs/ollama.log     — Ollama AI runtime
```

---

## How It Works

1. **Ingest** — PDFs are split into chunks, embedded as vectors, and stored in PostgreSQL. A knowledge graph extracts entities and relationships. Patient folders get structured extraction (name, DOB, diagnoses, medications, etc.) into a queryable table.
2. **Route** — Each question is classified as **SQL** (structured patient queries like "how many patients over 65?"), **HYBRID** (patient lookup + document context), or **RAG** (open-ended document questions). This ensures each question type gets the fastest, most accurate answer path.
3. **Ask** — RAG questions are expanded via AI, searched using semantic search, keyword search, and graph traversal, then re-ranked. SQL questions are converted into safe, parameterised database queries.
4. **Answer** — RAG answers come from document chunks sent to a local LLM. SQL answers come directly from the structured patients table, formatted by the LLM.
5. **Cite** — Every answer shows which document and page it came from.

All answers come only from your documents. The system is designed to say "I could not find this" rather than guess.

---

## Requirements

- macOS (Apple Silicon or Intel)
- Internet connection on first run (to download dependencies and AI models)

Everything else is installed automatically by `./start.sh`.
