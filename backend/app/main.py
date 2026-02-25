from fastapi import FastAPI
from backend.app.api.health import router as health_router
from backend.app.api.upload import router as upload_rourter

app=FastAPI(
    title="Aegis",
    description="Multi-Model RAG Backend System",
    version="1.0.0"
)

app.include_router(health_router)
app.include_router(upload_rourter)

@app.get("/")
def home():
    return {
        "message": "Welcome to Aegis 🛡"
    }
