"""
Query API endpoints — synchronous and streaming RAG queries for AEGIS v3.0.

Authenticated + session-scoped: retrieval only uses embeddings from the
current chat's uploads.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.app.pipelines.query_pipeline import QueryPipeline
from backend.app.api.deps import get_current_user
from backend.app.api.chats import add_chat_message
from backend.app.db.database import get_db, SessionLocal
from backend.app.db.models import ChatRole, User, Session as ChatSession
from backend.app.services.query_scope import (
    ensure_query_ready,
    resolve_session_artifact_ids,
)

router = APIRouter()
pipeline = QueryPipeline()


class QueryRequest(BaseModel):
    """Query request scoped to a chat session."""
    query: str = Field(..., min_length=2, max_length=2000)
    session_id: int = Field(..., description="Chat session whose uploads may be searched")
    top_k: int = Field(default=3, ge=1, le=20)
    evaluate: bool = Field(default=False, description="Run inline RAGAS evaluation (adds latency)")


class QueryResponse(BaseModel):
    """Structured query response with confidence metrics and monitoring ID."""
    answer:              str
    context_used:        List[Any] = []
    warnings:            List[str] = []
    confidence:          float     = Field(ge=0.0, le=1.0)
    grounded:            bool      = False
    has_hallucination:   bool      = False
    error:               bool      = False
    run_id:              Optional[str] = None
    retrieval_metadata:  Optional[Dict[str, Any]] = None
    latency_ms:          Optional[Dict[str, float]] = None
    eval_scores:         Optional[Dict[str, Any]] = None
    structured_answer:   Optional[Dict[str, Any]] = None


def _scope(db: Session, user: User, session_id: int) -> List[int]:
    ids = resolve_session_artifact_ids(db, user.id, session_id)
    blocked = ensure_query_ready(db, ids)
    if blocked:
        raise HTTPException(status_code=409, detail=blocked)
    return ids


@router.post("/query", response_model=QueryResponse)
def query_data(
    request: QueryRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> QueryResponse:
    try:
        artifact_ids = _scope(db, user, request.session_id)
        add_chat_message(db, request.session_id, ChatRole.USER, request.query)
        result = pipeline.answer(
            request.query,
            request.top_k,
            evaluate=request.evaluate,
            artifact_ids=artifact_ids,
        )
        answer = result.get("answer") or ""
        add_chat_message(db, request.session_id, ChatRole.ASSISTANT, answer)
        return QueryResponse(**{k: v for k, v in result.items() if k in QueryResponse.model_fields})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/query-stream")
def query_stream(
    request: QueryRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    artifact_ids = _scope(db, user, request.session_id)
    add_chat_message(db, request.session_id, ChatRole.USER, request.query)

    session_id = request.session_id
    query_text = request.query

    def _sse_data(payload: str) -> str:
        # SSE: each event line is "data: ..."; blank line terminates the event
        lines = payload.split("\n")
        return "".join(f"data: {line}\n" for line in lines) + "\n"

    def token_generator():
        parts: List[str] = []
        try:
            for token in pipeline.stream_answer(query_text, artifact_ids=artifact_ids):
                parts.append(token)
                yield _sse_data(token)
            yield _sse_data("[DONE]")
        except Exception as e:
            err = f"\n\n❌ Error: {str(e)}\n"
            parts.append(err)
            yield _sse_data(err)
            yield _sse_data("[DONE]")
        finally:
            # Persist assistant reply (strip UI metadata lines lightly)
            raw = "".join(parts)
            cleaned_lines = []
            for line in raw.split("\n"):
                s = line.strip()
                if s.startswith("[Retrieved:") or s.startswith("[Response confidence:"):
                    continue
                if s.startswith("⚠️") or s.startswith("ℹ️") or s.startswith("❌"):
                    continue
                cleaned_lines.append(line)
            answer = "\n".join(cleaned_lines).strip() or raw.strip()
            try:
                sdb = SessionLocal()
                try:
                    add_chat_message(sdb, session_id, ChatRole.ASSISTANT, answer)
                    chat = sdb.query(ChatSession).filter(ChatSession.id == session_id).first()
                    if chat:
                        chat.updated_at = datetime.now(timezone.utc)
                        sdb.commit()
                finally:
                    sdb.close()
            except Exception:
                pass

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
