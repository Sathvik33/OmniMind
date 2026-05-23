from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Request, HTTPException
import uuid

from backend.app.core.config import DATA_DIR
from backend.app.vectorstore.collection_manager import CollectionManager
from backend.app.services.video_service import VideoService
from backend.app.guardrails.ingestion_guard import IngestionGuard
from backend.app.retrieval.bm25_store import BM25Store

router = APIRouter()
collection_manager = CollectionManager()


@router.post("/ingest-video")
async def ingest_video(
    request: Request,
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
    with open(file_path, "wb") as f:
        f.write(content)

    job_id = str(uuid.uuid4())
    request.app.state.video_jobs[job_id] = "started"

    vision_service = request.app.state.vision_service
    video_service  = VideoService(vision_service, collection_manager)

    background_tasks.add_task(
        video_service.process,
        str(file_path),
        file.filename,
        job_id,
        request.app,
    )

    return {"message": "Video processing started", "job_id": job_id}


@router.get("/video-status/{job_id}")
def video_status(request: Request, job_id: str):
    status = request.app.state.video_jobs.get(job_id, "not_found")
    return {"job_id": job_id, "status": status}


@router.delete("/clear-memory")
def clear_memory():
    """Wipe both text and multimodal ChromaDB collections + BM25 index."""
    collection_manager.clear_all()
    BM25Store().clear()
    return {"message": "Memory cleared successfully (text + multimodal + BM25 index)."}