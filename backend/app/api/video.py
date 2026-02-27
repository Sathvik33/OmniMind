from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Request
from pathlib import Path
from backend.app.core.config import DATA_DIR
from backend.app.vectorstore.collection_manager import CollectionManager
from backend.app.services.video_service import VideoService

router = APIRouter()
collection_manager = CollectionManager()

import uuid

@router.post("/ingest-video")
async def ingest_video(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = DATA_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    job_id = str(uuid.uuid4())

    request.app.state.video_jobs[job_id] = "started"

    vision_service = request.app.state.vision_service
    video_service = VideoService(vision_service, collection_manager)

    background_tasks.add_task(
        video_service.process,
        str(file_path),
        file.filename,
        job_id,
        request.app
    )

    return {
        "message": "Video processing started",
        "job_id": job_id
    }

@router.get("/video-status/{job_id}")
def video_status(request: Request, job_id: str):
    status = request.app.state.video_jobs.get(job_id, "not_found")
    return {"status": status}


@router.delete("/clear-memory")
def clear_memory():
    collection_name = collection_manager.collection.name

    collection_manager.client.delete_collection(collection_name)

    collection_manager.collection = collection_manager.client.get_or_create_collection(
        name=collection_name
    )

    return {"message": "Memory cleared successfully"}