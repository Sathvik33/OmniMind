from fastapi import APIRouter, UploadFile, File, Request, BackgroundTasks, HTTPException
import uuid

from backend.app.vectorstore.collection_manager import CollectionManager
from backend.app.core.config import DATA_DIR
from backend.app.guardrails.ingestion_guard import IngestionGuard

router = APIRouter()
collection_manager = CollectionManager()


@router.post("/ingest-image")
async def ingest_image(
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
    request.app.state.image_jobs[job_id] = "started"

    def process_image():
        request.app.state.image_jobs[job_id] = "processing"
        try:
            vision_service = request.app.state.vision_service

            # LLaVA generates a rich text description of the image
            description = vision_service.describe(str(file_path))

            # Store in MultimodalCollection with CLIP embedding (512-dim)
            # routing handled by CollectionManager based on modality="image"
            collection_manager.add_documents(
                documents=[description],
                ids=[str(uuid.uuid4())],
                metadata=[{"source": file.filename, "modality": "image"}],
            )

            request.app.state.image_jobs[job_id] = "completed"
        except Exception as exc:
            request.app.state.image_jobs[job_id] = f"failed: {exc}"

    background_tasks.add_task(process_image)

    return {"message": "Image processing started", "job_id": job_id}


@router.get("/image-status/{job_id}")
def image_status(request: Request, job_id: str):
    status = request.app.state.image_jobs.get(job_id, "not_found")
    return {"job_id": job_id, "status": status}