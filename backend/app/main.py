from fastapi import FastAPI
from app.api.health import router as health_router

app=FastAPI(
    title="Aegis",
    description="Multi-Model RAG Backend System",
    version="1.0.0"
)

app.include_router(health_router)


@app.get("/")
def home():
    return {
        "message": "Welcome to Aegis 🛡"
    }
