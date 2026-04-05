# Changelog

## [2.1.0] - 2026-04-05

### Fixed
- Backend port mismatch: `start.sh` launched backend on port 8001 but Vite proxy targeted port 8000 — unified to port 8000
- Empty state flash on startup: frontend now retries API connection with a loading spinner instead of showing "No conversations yet" while backend starts

### Changed
- Frontend dev server port changed from 5173 to 5457
- Added re-ingest instruction to `start.sh` startup message for when entity/relation types are changed in config.py

## [2.0.0] - 2026-03-31

- Initial release with PDF ingestion, hybrid RAG pipeline, knowledge graph, and entity browser
