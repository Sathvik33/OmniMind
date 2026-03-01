from fastapi import APIRouter, UploadFile, File, Request, BackgroundTasks
from pathlib import Path
import uuid
from backend.app.vectorstore.collection_manager import CollectionManager
from backend.app.core.config import DATA_DIR

router = APIRouter()
collection_manager = CollectionManager()


def process_image(file_path: str, file_name: str, app):
    vision_service = app.state.vision_service
    description = vision_service.describe(file_path)

    document = [description]
    ids = [str(uuid.uuid4())]
    metadata = [{
        "source": file_name,
        "modality": "image"
    }]

    collection_manager.add_documents(document, ids, metadata)

@router.post("/ingest-image")
async def ingest_image(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):

    file_path = DATA_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    job_id = str(uuid.uuid4())
    request.app.state.image_jobs[job_id] = "started"

    def process_image():
        request.app.state.image_jobs[job_id] = "processing"

        vision_service = request.app.state.vision_service
        description = vision_service.describe(str(file_path))

        document = [description]
        ids = [str(uuid.uuid4())]
        metadata = [{
            "source": file.filename,
            "modality": "image"
        }]

        collection_manager.add_documents(document, ids, metadata)

        request.app.state.image_jobs[job_id] = "completed"

    background_tasks.add_task(process_image)

    return {
        "message": "Image processing started",
        "job_id": job_id
    }

@router.get("/image-status/{job_id}")
def image_status(request: Request, job_id: str):
    status = request.app.state.image_jobs.get(job_id, "not_found")
    return {"status": status}