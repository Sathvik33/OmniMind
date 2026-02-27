from dotenv import load_dotenv
import os
load_dotenv()
from fastapi import FastAPI
from backend.app.api.health import router as health_router
from backend.app.api.upload import router as upload_rourter
from backend.app.api.query import router as query_router
from backend.app.api.image import router as image_router
from backend.app.services.vision_service import VisionService
from backend.app.api.video import router as video_router
from langsmith import Client

app=FastAPI(
    title="Aegis",
    description="Multi-Model RAG Backend System",
    version="1.0.0"
)
app.state.video_jobs = {}
@app.on_event("startup")
def load_models():
    app.state.vision_service = VisionService()

app.include_router(health_router)
app.include_router(upload_rourter)
app.include_router(query_router)
app.include_router(image_router)
app.include_router(video_router)


@app.get("/")
def home():
    return {
        "message": "Welcome to Aegis"
    }
