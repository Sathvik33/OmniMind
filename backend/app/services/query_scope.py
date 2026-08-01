"""Query readiness + session-scoped artifact resolution."""

from __future__ import annotations

from typing import List, Optional, Sequence

from fastapi import HTTPException
from sqlalchemy.orm import Session

from backend.app.db.models import (
    Artifact,
    IngestionJob,
    ProcessingStatus,
    Session as ChatSession,
    VectorEmbedding,
)

_IN_FLIGHT = {
    ProcessingStatus.QUEUED,
    ProcessingStatus.RUNNING,
    ProcessingStatus.PARSING,
    ProcessingStatus.VISION_CAPTIONING,
    ProcessingStatus.CHUNKING_EMBEDDING,
    ProcessingStatus.STORING,
}


def get_owned_session(db: Session, session_id: int, user_id: int) -> ChatSession:
    chat = (
        db.query(ChatSession)
        .filter(ChatSession.id == session_id, ChatSession.user_id == user_id)
        .first()
    )
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


def resolve_session_artifact_ids(db: Session, user_id: int, session_id: int) -> List[int]:
    """
    Ready artifact IDs for this chat only — never global latest, never other users.
    """
    get_owned_session(db, session_id, user_id)
    rows = (
        db.query(Artifact.id)
        .join(VectorEmbedding, VectorEmbedding.artifact_id == Artifact.id)
        .filter(
            Artifact.session_id == session_id,
            Artifact.user_id == user_id,
            Artifact.processing_status == ProcessingStatus.COMPLETED,
        )
        .distinct()
        .order_by(Artifact.id.asc())
        .all()
    )
    return [r[0] for r in rows]


def resolve_artifact_ids(
    db: Session, requested: Optional[Sequence[int]] = None
) -> List[int]:
    """
    Legacy helper (evaluate / unscoped tools). Prefer resolve_session_artifact_ids
    for authenticated chat queries.
    """
    if requested:
        return [int(a) for a in requested if a is not None]

    row = (
        db.query(Artifact.id)
        .join(VectorEmbedding, VectorEmbedding.artifact_id == Artifact.id)
        .filter(Artifact.processing_status == ProcessingStatus.COMPLETED)
        .order_by(Artifact.created_at.desc())
        .first()
    )
    return [row[0]] if row else []


def ensure_query_ready(db: Session, artifact_ids: Sequence[int]) -> Optional[str]:
    """
    Return an error message if retrieval must not run yet; None if ready.
    """
    if not artifact_ids:
        return (
            "No embedded documents in this chat yet. Upload a file and wait until "
            "ingestion finishes before asking questions."
        )

    ids = list(artifact_ids)

    in_flight = (
        db.query(IngestionJob)
        .filter(
            IngestionJob.artifact_id.in_(ids),
            IngestionJob.status.in_(list(_IN_FLIGHT)),
        )
        .first()
    )
    if in_flight:
        status = (
            in_flight.status.value
            if hasattr(in_flight.status, "value")
            else str(in_flight.status)
        )
        return (
            f"Still converting your upload into embeddings (status: {status}). "
            "Please wait until ingestion completes, then ask again."
        )

    for aid in ids:
        art = db.query(Artifact).filter(Artifact.id == aid).first()
        if not art:
            return f"Artifact {aid} was not found."
        if art.processing_status != ProcessingStatus.COMPLETED:
            status = (
                art.processing_status.value
                if hasattr(art.processing_status, "value")
                else str(art.processing_status)
            )
            return (
                f"'{art.filename}' is not ready yet (status: {status}). "
                "Wait for embedding to finish before querying."
            )
        has_vecs = (
            db.query(VectorEmbedding.id)
            .filter(VectorEmbedding.artifact_id == aid)
            .first()
        )
        if not has_vecs:
            return (
                f"'{art.filename}' has no embeddings yet. "
                "Wait for ingestion to finish before querying."
            )

    return None
