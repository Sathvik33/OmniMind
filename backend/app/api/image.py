from fastapi import APIRouter, UploadFile, File, Request
from pathlib import Path
import uuid
from backend.app.vectorstore.collection_manager import CollectionManager
from backend.app.core.config import DATA_DIR

router = APIRouter()
collection_manager = CollectionManager()


@router.post("/ingest-image")
async def ingest_image(request: Request, file: UploadFile = File(...)):
    file_path = DATA_DIR / file.filename

    with open(file_path, "wb") as f:
        f.write(await file.read())

    vision_service = request.app.state.vision_service
    description = vision_service.describe(str(file_path))

    document = [description]
    ids = [str(uuid.uuid4())]
    metadata = [{
        "source": file.filename,
        "modality": "image"
    }]

    collection_manager.add_documents(document, ids, metadata)

    return {
        "message": "Image ingested successfully",
        "description": description
    }