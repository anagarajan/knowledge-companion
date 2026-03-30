"""
main.py — FastAPI application: the bridge between the React UI and the Python RAG pipeline.

Endpoints:
  POST /api/chat                         — stream a question (Server-Sent Events)
  POST /api/sessions                     — create a new session
  GET  /api/sessions                     — list all sessions (sidebar)
  GET  /api/sessions/{id}                — get one session
  DELETE /api/sessions/{id}              — soft-delete a session
  POST /api/sessions/{id}/clear          — clear messages in a session
  GET  /api/sessions/{id}/messages       — full message history for a session
  PATCH /api/sessions/{id}/folders       — update folder scope for a session
  GET  /api/folders                      — list ingested folders for the folder picker

Start with:
  uvicorn main:app --reload --port 8000
"""

import json
import logging
import os
from collections.abc import AsyncGenerator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from db.connection import db_conn, init_pool
from db.sessions import SessionStore
from rag.pipeline import Pipeline
from rag.vectorstore import VectorStore
from rag.graph_store import GraphStore

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="Knowledge Companion API", version="1.0.0")

# Allow the React dev server (port 5173) and the production build to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singletons (created once at startup) ──────────────────────────────────────

_vector_store: VectorStore | None = None
_pipeline: Pipeline | None = None
_session_store: SessionStore | None = None
_graph_store: GraphStore | None = None


@app.on_event("startup")
def startup() -> None:
    """
    Runs once when the server starts.
    1. Opens the PostgreSQL connection pool.
    2. Creates/migrates all tables (idempotent).
    3. Instantiates the shared VectorStore, Pipeline, and SessionStore.
    """
    global _vector_store, _pipeline, _session_store, _graph_store

    logger.info("Starting Knowledge Companion API...")
    init_pool()                           # opens the Postgres pool, creates tables

    _vector_store = VectorStore()
    _graph_store = GraphStore()
    _pipeline = Pipeline(_vector_store, _graph_store)
    _session_store = SessionStore()

    logger.info(
        f"Ready — {_vector_store.count()} chunks, "
        f"{_graph_store.entity_count()} entities, "
        f"{_graph_store.relationship_count()} relationships"
    )


def get_pipeline() -> Pipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised")
    return _pipeline


def get_store() -> SessionStore:
    if _session_store is None:
        raise RuntimeError("SessionStore not initialised")
    return _session_store


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str
    question:   str
    folders:    list[str] = []


class CreateSessionRequest(BaseModel):
    folders: list[str] = []
    title:   str = "New Chat"


class UpdateFoldersRequest(BaseModel):
    folders: list[str]


# ── Chat endpoint (SSE streaming) ─────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """
    The main chat endpoint. Returns a Server-Sent Events stream.

    Event types the client should handle:
      {"type": "token",  "content": "..."}          — one piece of the answer
      {"type": "done",   "sources": [...],
                         "confidence": {...},
                         "model_used": "..."}        — stream complete, includes citations
      {"type": "error",  "message": "..."}           — something went wrong

    Design note: pipeline.query() returns a lazy generator — tokens are streamed
    to the client in real time as the LLM generates them. Confidence is computed
    from retrieval scores (pre-generation) so no buffering is needed.
    """
    store    = get_store()
    pipeline = get_pipeline()

    # Validate session exists
    session = store.get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Persist the user's question
    store.add_message(
        session_id=req.session_id,
        role="user",
        content=req.question,
    )

    # Auto-title the session from the first question (if still "New Chat")
    if session.title == "New Chat":
        store.update_session_title(req.session_id, req.question)

    # Load conversation history for the pipeline
    history = store.get_recent_messages(req.session_id)

    # Run the RAG pipeline (HyDE + retrieval + reranking are sync;
    # token_stream in the result is a lazy generator consumed below)
    folders = req.folders or session.folders or None
    result  = pipeline.query(
        question=req.question,
        history=history,
        folders=folders if folders else None,
    )

    # Collect sources and confidence for the done event
    sources_payload = [
        {
            "filename": s.filename,
            "page":     s.page,
            "score":    s.score,
            "was_ocr":  s.was_ocr,
            "folder":   s.folder,
        }
        for s in result.sources
    ]

    # Buffer the full answer text so we can persist it after streaming
    full_tokens: list[str] = []

    # Phrases the LLM uses when it correctly refuses an out-of-domain question.
    # If the generated answer contains any of these, it's a self-refusal —
    # override the pre-computed confidence to LOW so the UI badge is accurate.
    _REFUSAL_PHRASES = (
        "could not find this in the available documents",
        "could not find relevant information in the available documents",
        "doesn't look like a question i can help with",
    )

    async def event_stream() -> AsyncGenerator[str, None]:
        """
        Yield SSE-formatted events.
        Each event is:   data: <json>\n\n
        """
        try:
            for token in result.token_stream:
                full_tokens.append(token)
                event = json.dumps({"type": "token", "content": token})
                yield f"data: {event}\n\n"

            full_answer = "".join(full_tokens).lower()

            # If the LLM itself refused (despite chunks passing threshold),
            # downgrade confidence to LOW so the badge reflects reality.
            confidence = result.confidence
            if any(phrase in full_answer for phrase in _REFUSAL_PHRASES):
                confidence = {"level": "LOW", "reason": "No relevant information found for this question"}
                sources_payload.clear()   # don't cite sources for a refusal

            # Persist the full assistant answer once streaming is complete
            store.add_message(
                session_id=req.session_id,
                role="assistant",
                content="".join(full_tokens),
                sources=sources_payload,
                confidence=confidence,
                model_used=result.model_used,
            )

            # Final event — sends citations and confidence to the UI
            done_event = json.dumps({
                "type":       "done",
                "sources":    sources_payload,
                "confidence": confidence,
                "model_used": result.model_used,
            })
            yield f"data: {done_event}\n\n"

        except Exception as exc:
            logger.exception("Error during streaming")
            error_event = json.dumps({"type": "error", "message": str(exc)})
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",    # prevents nginx from buffering SSE
        },
    )


# ── Session endpoints ─────────────────────────────────────────────────────────

@app.post("/api/sessions")
def create_session(req: CreateSessionRequest) -> dict:
    """Create a new chat session. Called when user clicks '+ New Chat'."""
    session = get_store().create_session(folders=req.folders, title=req.title)
    return _session_to_dict(session)


@app.get("/api/sessions")
def list_sessions() -> list[dict]:
    """Return all active sessions for the sidebar, newest first."""
    sessions = get_store().list_sessions(include_inactive=False)
    return [_session_to_dict(s) for s in sessions]


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str) -> dict:
    """Fetch one session by ID."""
    session = get_store().get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return _session_to_dict(session)


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str) -> dict:
    """Soft-delete a session — hides it from the sidebar."""
    store = get_store()
    if not store.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    store.delete_session(session_id)
    return {"ok": True}


@app.post("/api/sessions/{session_id}/clear")
def clear_session(session_id: str) -> dict:
    """Delete all messages in a session ('Clear chat' button)."""
    store = get_store()
    if not store.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    store.clear_session(session_id)
    return {"ok": True}


@app.get("/api/sessions/{session_id}/messages")
def get_messages(session_id: str) -> list[dict]:
    """Return full message history for a session (used when reopening a tab)."""
    store = get_store()
    if not store.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    messages = store.get_messages(session_id)
    return [_message_to_dict(m) for m in messages]


@app.patch("/api/sessions/{session_id}/folders")
def update_folders(session_id: str, req: UpdateFoldersRequest) -> dict:
    """Update the folder scope for a session (folder picker in the sidebar)."""
    store = get_store()
    if not store.get_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    store.update_session_folders(session_id, req.folders)
    return {"ok": True}


# ── Folder discovery endpoint ──────────────────────────────────────────────────

@app.get("/api/folders")
def list_folders() -> list[str]:
    """
    Return all folder names that have been ingested into the knowledge base.
    Queries distinct folder values from the chunks table — works regardless
    of where the source PDFs lived on disk.
    """
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT folder FROM chunks ORDER BY folder")
            return [row["folder"] for row in cur.fetchall()]


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/api/health")
def health() -> dict:
    """Quick liveness check. Also reports how many chunks are indexed."""
    vs = _vector_store
    return {
        "status": "ok",
        "chunks_indexed": vs.count() if vs else 0,
    }


# ── Knowledge Graph endpoints ────────────────────────────────────────────────
#
# These endpoints expose the graph for inspection and debugging.
# They're also used by the frontend's EntityBrowser component (Phase 4).
#
# Why separate from the chat endpoint?
#   The chat endpoint streams answers.  These return structured data
#   for browsing entities, relationships, and graph statistics.
#   Different consumers, different response shapes.

def get_graph_store() -> GraphStore:
    if _graph_store is None:
        raise RuntimeError("GraphStore not initialised")
    return _graph_store


@app.get("/api/graph/entities")
def list_entities(
    search: str | None = None,
    type: str | None = None,
    folder: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """
    Search and filter entities in the knowledge graph.

    Query params:
      search  — substring match on entity name (case-insensitive)
      type    — filter by entity type (PERSON, POLICY, DEPARTMENT, etc.)
      folder  — filter by source folder
      limit   — max results (default 50)
      offset  — pagination offset
    """
    gs = get_graph_store()

    if search:
        results = gs.find_entities_by_name(
            name=search,
            entity_type=type,
            folders=[folder] if folder else None,
        )
    elif type:
        results = gs.find_entities_by_type(
            entity_type=type,
            folders=[folder] if folder else None,
        )
    else:
        # No filter — return all entities (paginated)
        with db_conn() as conn:
            with conn.cursor() as cur:
                if folder:
                    cur.execute(
                        "SELECT * FROM entities WHERE folder = %s ORDER BY name LIMIT %s OFFSET %s",
                        (folder, limit, offset),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM entities ORDER BY name LIMIT %s OFFSET %s",
                        (limit, offset),
                    )
                results = cur.fetchall()

    return [dict(r) for r in results]


@app.get("/api/graph/entities/{entity_id}")
def get_entity(entity_id: str) -> dict:
    """
    Get one entity with all its relationships (both incoming and outgoing).

    Returns the entity plus a list of relationships with connected
    entity names resolved — so the frontend doesn't need extra calls.
    """
    gs = get_graph_store()

    # Find the entity itself
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM entities WHERE id = %s", (entity_id,))
            entity = cur.fetchone()

    if not entity:
        raise HTTPException(status_code=404, detail="Entity not found")

    # Find all relationships involving this entity
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT r.*,
                       se.name AS source_name, se.entity_type AS source_type,
                       te.name AS target_name, te.entity_type AS target_type
                FROM relationships r
                JOIN entities se ON se.id = r.source_entity_id
                JOIN entities te ON te.id = r.target_entity_id
                WHERE r.source_entity_id = %s OR r.target_entity_id = %s
            """, (entity_id, entity_id))
            relationships = cur.fetchall()

    return {
        "entity": dict(entity),
        "relationships": [dict(r) for r in relationships],
    }


@app.get("/api/graph/entities/{entity_id}/related")
def get_related_entities(
    entity_id: str,
    depth: int = 2,
    max_nodes: int = 30,
) -> list[dict]:
    """
    Traverse the graph from an entity.

    This uses the recursive CTE — walks outgoing AND incoming edges
    up to `depth` hops.  Returns connected entities with their
    relationship type and distance.

    Example: starting from "Leave Policy v3" at depth 2:
      depth 1: Engineering (APPLIES_TO), Sarah Chen (AUTHORED_BY)
      depth 2: Engineering Handbook (REFERENCES), HR Department (BELONGS_TO)
    """
    gs = get_graph_store()
    # Cap depth at 3 to prevent expensive traversals
    depth = min(depth, 3)
    return gs.get_related_entities(entity_id, max_depth=depth, max_nodes=max_nodes)


@app.get("/api/graph/documents/{doc_id}/metadata")
def get_doc_metadata(doc_id: str) -> dict:
    """Get the metadata summary for a document (title, type, version, summary)."""
    gs = get_graph_store()
    meta = gs.get_document_metadata(doc_id)
    if not meta:
        raise HTTPException(status_code=404, detail="Document metadata not found")
    return dict(meta)


@app.get("/api/graph/documents/{doc_id}/entities")
def get_doc_entities(doc_id: str) -> list[dict]:
    """List all entities extracted from a specific document."""
    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM entities WHERE doc_id = %s ORDER BY entity_type, name",
                (doc_id,),
            )
            return [dict(r) for r in cur.fetchall()]


@app.get("/api/graph/stats")
def graph_stats() -> dict:
    """
    Graph-wide statistics — useful for verifying ingestion quality.

    After ingesting a batch of documents, hit this endpoint to confirm:
    - entities were extracted (total > 0)
    - types are distributed (not everything is OTHER)
    - relationships exist (connections were found)
    """
    gs = get_graph_store()

    with db_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT entity_type, COUNT(*) AS count
                FROM entities GROUP BY entity_type ORDER BY count DESC
            """)
            entities_by_type = {r["entity_type"]: r["count"] for r in cur.fetchall()}

            cur.execute("""
                SELECT relation_type, COUNT(*) AS count
                FROM relationships GROUP BY relation_type ORDER BY count DESC
            """)
            rels_by_type = {r["relation_type"]: r["count"] for r in cur.fetchall()}

            cur.execute("SELECT COUNT(*) AS count FROM document_metadata")
            doc_count = cur.fetchone()["count"]

    return {
        "total_entities": gs.entity_count(),
        "total_relationships": gs.relationship_count(),
        "total_documents_with_metadata": doc_count,
        "entities_by_type": entities_by_type,
        "relationships_by_type": rels_by_type,
    }


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _session_to_dict(session) -> dict:
    return {
        "id":         session.id,
        "title":      session.title,
        "folders":    session.folders,
        "created_at": session.created_at.isoformat(),
        "updated_at": session.updated_at.isoformat(),
        "is_active":  session.is_active,
    }


def _message_to_dict(message) -> dict:
    return {
        "id":         message.id,
        "session_id": message.session_id,
        "role":       message.role,
        "content":    message.content,
        "sources":    message.sources,
        "confidence": message.confidence,
        "model_used": message.model_used,
        "timestamp":  message.timestamp.isoformat(),
    }
