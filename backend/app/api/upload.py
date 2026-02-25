from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
from backend.app.core.config import DATA_DIR
from backend.app.pipelines.ingestion_pipeline import IngestionPipeline

router = APIRouter()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    file_path = DATA_DIR / file.filename

    try:
        with open(file_path, "wb") as f:
            f.write(await file.read())

        pipeline = IngestionPipeline()
        result = pipeline.ingest_file(str(file_path))

        return {
            "message": "File ingested successfully",
            "details": result
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))