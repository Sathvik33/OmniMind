"""Chat (session) CRUD — sidebar history + per-chat artifact scope."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.app.api.deps import get_current_user
from backend.app.db.database import get_db
from backend.app.db.models import (
    Artifact,
    ChatHistory,
    ChatRole,
    IngestionJob,
    Metadata,
    ProcessingStatus,
    Session as ChatSession,
    User,
    VectorEmbedding,
)

router = APIRouter(prefix="/chats", tags=["chats"])

DOC_PREFIX = "__aegis_doc__:"


class ChatSummary(BaseModel):
    id: int
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class ChatCreate(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)


class ChatRename(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    created_at: Optional[datetime] = None
    document: Optional[Dict[str, Any]] = None


class ArtifactOut(BaseModel):
    id: int
    filename: str
    modality: str
    status: str


class ChatDetail(BaseModel):
    id: int
    title: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    messages: List[MessageOut]
    artifacts: List[ArtifactOut]


def _owned_chat(db: Session, chat_id: int, user: User) -> ChatSession:
    chat = (
        db.query(ChatSession)
        .filter(ChatSession.id == chat_id, ChatSession.user_id == user.id)
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


def _touch(chat: ChatSession) -> None:
    chat.updated_at = datetime.now(timezone.utc)


def encode_document_message(payload: Dict[str, Any]) -> str:
    return DOC_PREFIX + json.dumps(payload)


def decode_document_message(content: str) -> Optional[Dict[str, Any]]:
    if not content.startswith(DOC_PREFIX):
        return None
    try:
        return json.loads(content[len(DOC_PREFIX) :])
    except json.JSONDecodeError:
        return None


@router.get("", response_model=List[ChatSummary])
def list_chats(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(ChatSession)
        .filter(ChatSession.user_id == user.id)
        .order_by(ChatSession.updated_at.desc().nullslast(), ChatSession.created_at.desc())
        .all()
    )
    return [
        ChatSummary(
            id=r.id,
            title=r.title or "New chat",
            created_at=r.created_at,
            updated_at=r.updated_at or r.created_at,
        )
        for r in rows
    ]


@router.post("", response_model=ChatSummary)
def create_chat(
    body: ChatCreate = ChatCreate(),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    chat = ChatSession(user_id=user.id, title=body.title or "New chat")
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return ChatSummary(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at or chat.created_at,
    )


@router.get("/{chat_id}", response_model=ChatDetail)
def get_chat(chat_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    chat = _owned_chat(db, chat_id, user)
    messages: List[MessageOut] = []
    for m in (
        db.query(ChatHistory)
        .filter(ChatHistory.session_id == chat.id)
        .order_by(ChatHistory.created_at.asc(), ChatHistory.id.asc())
        .all()
    ):
        role = m.role.value if hasattr(m.role, "value") else str(m.role)
        doc = decode_document_message(m.content)
        if doc is not None:
            messages.append(
                MessageOut(
                    id=m.id,
                    role="document",
                    content="",
                    created_at=m.created_at,
                    document=doc,
                )
            )
            continue
        messages.append(
            MessageOut(
                id=m.id,
                role=role,
                content=m.content,
                created_at=m.created_at,
                document=None,
            )
        )

    arts = (
        db.query(Artifact)
        .filter(Artifact.session_id == chat.id, Artifact.user_id == user.id)
        .order_by(Artifact.created_at.asc())
        .all()
    )
    artifacts = [
        ArtifactOut(
            id=a.id,
            filename=a.filename,
            modality=a.modality,
            status=a.processing_status.value
            if hasattr(a.processing_status, "value")
            else str(a.processing_status),
        )
        for a in arts
    ]
    return ChatDetail(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at or chat.created_at,
        messages=messages,
        artifacts=artifacts,
    )


@router.patch("/{chat_id}", response_model=ChatSummary)
def rename_chat(
    chat_id: int,
    body: ChatRename,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    chat = _owned_chat(db, chat_id, user)
    chat.title = body.title.strip()
    _touch(chat)
    db.commit()
    db.refresh(chat)
    return ChatSummary(
        id=chat.id,
        title=chat.title,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
    )


@router.delete("/{chat_id}")
def delete_chat(chat_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """
    Delete a chat and all of its data:
      messages, feedback, artifacts, embeddings, metadata, ingestion jobs, MinIO blobs.
    BM25 index is rebuilt afterward so retrieval cannot hit purged chunks.
    """
    chat = _owned_chat(db, chat_id, user)
    arts = (
        db.query(Artifact)
        .filter(Artifact.session_id == chat.id, Artifact.user_id == user.id)
        .all()
    )
    minio_paths = [a.file_path for a in arts if a.file_path]
    artifact_ids = [a.id for a in arts]

    # Purge MinIO objects first (before DB rows disappear)
    removed_blobs = 0
    if minio_paths:
        from backend.app.storage.minio_client import minio_client

        for path in minio_paths:
            try:
                minio_client.delete_file_path(path)
                removed_blobs += 1
            except Exception:
                # Continue DB purge even if one blob fails
                pass

    for art in arts:
        db.query(IngestionJob).filter(IngestionJob.artifact_id == art.id).delete(
            synchronize_session=False
        )
        db.query(VectorEmbedding).filter(VectorEmbedding.artifact_id == art.id).delete(
            synchronize_session=False
        )
        db.query(Metadata).filter(Metadata.artifact_id == art.id).delete(
            synchronize_session=False
        )
        db.delete(art)

    # Messages (+ feedback) cascade via Session.messages relationship
    db.delete(chat)
    db.commit()

    # Drop purged docs from the sparse index
    try:
        from backend.app.retrieval.bm25_store import BM25Store

        BM25Store().rebuild_from_postgres(db)
    except Exception:
        pass

    return {
        "ok": True,
        "deleted": chat_id,
        "artifacts_removed": len(artifact_ids),
        "minio_removed": removed_blobs,
    }


def add_chat_message(
    db: Session,
    session_id: int,
    role: ChatRole,
    content: str,
    *,
    touch: bool = True,
) -> ChatHistory:
    msg = ChatHistory(session_id=session_id, role=role, content=content)
    db.add(msg)
    if touch:
        chat = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if chat:
            _touch(chat)
            if (not chat.title or chat.title == "New chat") and role == ChatRole.USER:
                chat.title = (content.strip()[:60] + ("…" if len(content.strip()) > 60 else ""))
    db.commit()
    db.refresh(msg)
    return msg
