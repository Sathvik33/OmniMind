"""
Query readiness + artifact scoping.

Blocks retrieval while ingestion/embedding is still running, and resolves
which artifact IDs answers may use (session uploads or latest completed).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from sqlalchemy.orm import Session

from backend.app.db.models import (
    Artifact,
    IngestionJob,
    ProcessingStatus,
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


def resolve_artifact_ids(
    db: Session, requested: Optional[Sequence[int]] = None
) -> List[int]:
    """
    Use explicit artifact_ids when provided; otherwise the single most recently
    completed artifact that already has embeddings.
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
            "No embedded documents yet. Upload a file and wait until ingestion "
            "finishes before asking questions."
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

    # Also block if anything else is still embedding globally and caller
    # asked about "latest" without explicit ids — already scoped to completed.
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
