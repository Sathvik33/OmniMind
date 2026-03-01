from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from pathlib import Path
import shutil
from backend.app.core.config import DATA_DIR
from backend.app.pipelines.ingestion_pipeline import IngestionPipeline

router = APIRouter()


def process_ingestion(file_path: str):
    pipeline = IngestionPipeline()
    pipeline.ingest_file(file_path)


@router.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):

    file_path = DATA_DIR / file.filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        background_tasks.add_task(process_ingestion, str(file_path))

        return {
            "message": "File uploaded. Ingestion started."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))