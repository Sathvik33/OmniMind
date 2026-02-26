from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Request
from pathlib import Path
from backend.app.core.config import DATA_DIR
from backend.app.vectorstore.collection_manager import CollectionManager
from backend.app.services.video_service import VideoService

router = APIRouter()
collection_manager = CollectionManager()


@router.post("/ingest-video")
async def ingest_video(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    file_path = DATA_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    vision_service = request.app.state.vision_service

    video_service = VideoService(vision_service, collection_manager)

    background_tasks.add_task(
        video_service.process,
        str(file_path),
        file.filename
    )

    return {"message": "Video processing started"}