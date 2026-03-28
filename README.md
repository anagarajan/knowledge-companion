# Knowledge Companion

A fully private, local AI assistant for your documents. Ask questions about thousands of PDFs — no data leaves your machine.

Built for Apple Silicon Mac. All AI runs locally via Ollama.

---

## Requirements

- macOS (Apple Silicon or Intel)
- Internet connection on first run (to download dependencies and AI models)

Everything else — PostgreSQL, Ollama, Python, Node.js — is installed automatically.

---

## Start

```bash
git clone <this-repo>
cd knowledge-companion
./start.sh
```

That's it. First run takes 10–15 minutes (downloading AI models). Every run after that starts in seconds.

The app opens automatically at **http://localhost:5173**

---

## Add documents

Drop PDFs into a folder inside `documents/`, then run:

```bash
./ingest.sh <folder-name>
```

Example — if you created `documents/HR/`:
```bash
./ingest.sh HR
```

Then select the **HR** folder in the app's sidebar to search it.

---

## Stop

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

## How it works

1. **Ingest** — PDFs are split into chunks, embedded as vectors, and stored in PostgreSQL
2. **Ask** — Your question is expanded via AI, searched using both semantic and keyword search, and re-ranked
3. **Answer** — The best matching chunks are sent to a local LLM which answers using only your documents
4. **Cite** — Every answer shows which document and page it came from

All answers come only from your documents. The system is explicitly designed to say "I could not find this" rather than guess.
