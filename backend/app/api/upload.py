from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pathlib import Path
import shutil

from backend.app.core.config import DATA_DIR
from backend.app.pipelines.ingestion_pipeline import IngestionPipeline
from backend.app.guardrails.ingestion_guard import IngestionGuard

router = APIRouter()


def process_ingestion(file_path: str):
    pipeline = IngestionPipeline()
    pipeline.ingest_file(file_path)


@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # ── Ingestion Guard ────────────────────────────────────────────────────────
    content = await file.read()
    guard = IngestionGuard.validate_file(file.filename, len(content))
    if not guard["ok"]:
        raise HTTPException(status_code=400, detail=guard["reason"])

    # ── Save file ──────────────────────────────────────────────────────────────
    file_path = DATA_DIR / file.filename
    try:
        with open(file_path, "wb") as buffer:
            buffer.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # ── Background ingestion ───────────────────────────────────────────────────
    background_tasks.add_task(process_ingestion, str(file_path))

    return {"message": "File uploaded. Ingestion started.", "filename": file.filename}