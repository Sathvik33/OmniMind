from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
import tempfile
import os
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from backend.app.db.database import get_db
from backend.app.db.models import (
    Artifact,
    ChatHistory,
    ChatRole,
    ProcessingStatus,
    IngestionJob,
    User,
    Session as ChatSession,
)
from backend.app.storage.minio_client import minio_client
from backend.app.guardrails.ingestion_guard import IngestionGuard
from backend.app.tasks.ingestion_tasks import process_ingestion_task
from backend.app.api.deps import get_current_user
from backend.app.api.chats import encode_document_message
from backend.app.services.query_scope import get_owned_session

router = APIRouter()

def determine_modality(filename: str) -> str:
    ext = filename.lower().split('.')[-1]
    if ext in ['pdf', 'docx', 'pptx', 'xlsx', 'txt', 'md']:
        return "document"
    elif ext in ['jpg', 'jpeg', 'png']:
        return "image"
    elif ext in ['mp4', 'avi', 'mov']:
        return "video"
    elif ext in ['mp3', 'wav', 'm4a']:
        return "audio"
    return "unknown"

@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: int = Form(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    chat = get_owned_session(db, session_id, user.id)

    content = await file.read()
    guard = IngestionGuard.validate_file(file.filename, len(content), file.content_type)
    if not guard["ok"]:
        raise HTTPException(status_code=400, detail=guard["reason"])
    
    modality = determine_modality(file.filename)

    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name

        minio_path = minio_client.upload_file(
            object_name=file.filename,
            file_path=temp_path,
            content_type=file.content_type
        )
        os.unlink(temp_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage error: {str(e)}")

    try:
        new_artifact = Artifact(
            filename=file.filename,
            file_path=minio_path,
            modality=modality,
            user_id=user.id,
            session_id=chat.id,
            upload_status=ProcessingStatus.COMPLETED,
            processing_status=ProcessingStatus.QUEUED
        )
        db.add(new_artifact)
        db.flush()

        new_job = IngestionJob(
            artifact_id=new_artifact.id,
            status=ProcessingStatus.QUEUED
        )
        db.add(new_job)

        # Document card in chat history
        db.add(
            ChatHistory(
                session_id=chat.id,
                role=ChatRole.SYSTEM,
                content=encode_document_message(
                    {
                        "name": file.filename,
                        "ext": (file.filename.rsplit(".", 1)[-1][:4].upper()
                                if "." in file.filename else "FILE"),
                        "status": "processing",
                        "statusLabel": "Queued — preparing for search…",
                        "artifact_id": new_artifact.id,
                        "sizeLabel": None,
                    }
                ),
            )
        )
        chat.updated_at = datetime.now(timezone.utc)
        if not chat.title or chat.title == "New chat":
            chat.title = f"Upload: {file.filename[:40]}"

        db.commit()
        
        db.refresh(new_artifact)
        db.refresh(new_job)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    process_ingestion_task.delay(new_job.id)

    return {
        "message": "File uploaded and task queued.", 
        "artifact_id": new_artifact.id,
        "job_id": new_job.id,
        "session_id": chat.id,
        "minio_path": minio_path,
        "filename": file.filename,
        "modality": modality,
    }

@router.get("/jobs/{job_id}")
async def get_job_status(
    job_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    art = db.query(Artifact).filter(Artifact.id == job.artifact_id).first()
    if not art or art.user_id != user.id:
        raise HTTPException(status_code=404, detail="Job not found")

    # Keep document system message status in sync when completed/failed
    if art.session_id and job.status in (
        ProcessingStatus.COMPLETED,
        ProcessingStatus.FAILED,
        ProcessingStatus.DEAD_LETTER,
    ):
        from backend.app.api.chats import DOC_PREFIX, encode_document_message
        import json

        msgs = (
            db.query(ChatHistory)
            .filter(
                ChatHistory.session_id == art.session_id,
                ChatHistory.role == ChatRole.SYSTEM,
                ChatHistory.content.like(DOC_PREFIX + "%"),
            )
            .all()
        )
        for m in msgs:
            try:
                payload = json.loads(m.content[len(DOC_PREFIX):])
            except Exception:
                continue
            if payload.get("artifact_id") != art.id:
                continue
            if job.status == ProcessingStatus.COMPLETED:
                payload["status"] = "ready"
                payload["statusLabel"] = "Ready — ask away"
            else:
                payload["status"] = "failed"
                payload["statusLabel"] = job.error_message or "Processing failed"
            m.content = encode_document_message(payload)
        db.commit()
        
    return {
        "job_id": job.id,
        "artifact_id": job.artifact_id,
        "status": job.status.value,
        "retry_count": job.retry_count,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "updated_at": job.updated_at
    }
