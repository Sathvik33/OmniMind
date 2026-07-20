from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pathlib import Path
import tempfile
import os

from sqlalchemy.orm import Session
from backend.app.db.database import get_db
from backend.app.db.models import Artifact, ProcessingStatus, IngestionJob
from backend.app.storage.minio_client import minio_client
from backend.app.guardrails.ingestion_guard import IngestionGuard
from backend.app.tasks.ingestion_tasks import process_ingestion_task

router = APIRouter()

def determine_modality(filename: str) -> str:
    ext = filename.lower().split('.')[-1]
    if ext in ['pdf', 'docx', 'pptx', 'xlsx', 'txt']:
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
    db: Session = Depends(get_db)
):
    # ── Security Guard ────────────────────────────────────────────────────────
    # Apply file size and type validation
    content = await file.read()
    guard = IngestionGuard.validate_file(file.filename, len(content), file.content_type)
    if not guard["ok"]:
        raise HTTPException(status_code=400, detail=guard["reason"])
    
    modality = determine_modality(file.filename)

    # ── Save file to MinIO ─────────────────────────────────────────────────────
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

    # ── Save Artifact and Job to PostgreSQL ─────────────────────────────────────
    try:
        new_artifact = Artifact(
            filename=file.filename,
            file_path=minio_path,
            modality=modality,
            upload_status=ProcessingStatus.COMPLETED,
            processing_status=ProcessingStatus.QUEUED
        )
        db.add(new_artifact)
        db.flush() # flush to get artifact id

        new_job = IngestionJob(
            artifact_id=new_artifact.id,
            status=ProcessingStatus.QUEUED
        )
        db.add(new_job)
        db.commit()
        
        db.refresh(new_artifact)
        db.refresh(new_job)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

    # ── Dispatch Celery Task ───────────────────────────────────────────────────
    process_ingestion_task.delay(new_job.id)

    return {
        "message": "File uploaded and task queued.", 
        "artifact_id": new_artifact.id,
        "job_id": new_job.id,
        "minio_path": minio_path
    }

@router.get("/jobs/{job_id}")
async def get_job_status(job_id: int, db: Session = Depends(get_db)):
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
        
    return {
        "job_id": job.id,
        "artifact_id": job.artifact_id,
        "status": job.status.value,
        "retry_count": job.retry_count,
        "error_message": job.error_message,
        "created_at": job.created_at,
        "updated_at": job.updated_at
    }