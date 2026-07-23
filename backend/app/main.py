from dotenv import load_dotenv
import os

load_dotenv()

os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

from fastapi import FastAPI
from backend.app.api.health import router as health_router
from backend.app.api.upload import router as upload_router
from backend.app.api.query import router as query_router
from backend.app.api.evaluate import router as evaluate_router
from backend.app.api.monitor import router as monitor_router
from backend.app.api.evaluate import set_pipeline
from backend.app.api.feedback import router as feedback_router
from backend.app.services.groq_vision_service import GroqVisionService
from backend.app.retrieval.bm25_store import BM25Store

from backend.app.monitoring.langsmith_logger import tracer as aegis_tracer
from backend.app.core.middleware import ObservabilityMiddleware

app = FastAPI(
    title="AEGIS",
    description=(
        "AEGIS v3.0 — Multi-Modal RAG Engine\n\n"
        "Features: Hybrid Search (BM25 + Dense + CLIP) · Cross-Encoder Reranking · "
        "Input/Output Guardrails · LangSmith Monitoring · RAGAS Evaluation (Groq LLM)"
    ),
    version="3.0.0",
)

app.add_middleware(ObservabilityMiddleware)

app.state.video_jobs = {}
app.state.image_jobs = {}


@app.on_event("startup")
def startup():
    # 1. Initialize Groq Vision Service (cloud — no GPU load at startup)
    app.state.vision_service = GroqVisionService()

    # 2. Load / rebuild BM25 index from persisted Postgres vector_embeddings
    bm25 = BM25Store()
    if not bm25._docs:
        from backend.app.db.database import SessionLocal
        db = SessionLocal()
        try:
            bm25.rebuild_from_postgres(db)
        finally:
            db.close()

    app.state.bm25 = bm25

    # 3. Initialize LangSmith tracer
    app.state.tracer = aegis_tracer
    if aegis_tracer.is_active():
        print("LangSmith monitoring active — project: Aegis")
    else:
        print("LangSmith monitoring in no-op mode (check LANGSMITH_API_KEY)")

    # 4. Wire pipeline into evaluate router for live evaluation
    from backend.app.api.query import pipeline as query_pipeline
    set_pipeline(query_pipeline)


app.include_router(health_router)
app.include_router(upload_router)
app.include_router(query_router)
app.include_router(monitor_router)
app.include_router(evaluate_router)
app.include_router(feedback_router)


@app.get("/")
def home():
    return {
        "message": "Welcome to AEGIS v3.0",
        "features": {
            "hybrid_search":  "BM25 + Dense Semantic + CLIP Multimodal + RRF Fusion",
            "reranking":      "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "guardrails":     "Input (injection/SQL/harmful) + Output (PII/hallucination/grounding)",
            "monitoring":     "LangSmith — full run tracing + per-node latency",
            "evaluation":     "RAGAS — faithfulness, answer_relevancy, context_precision, context_recall",
        },
        "models": {
            "llm":       "llama3-70b-8192 (Groq cloud)",
            "vision":    "llama-4-scout-17b (Groq vision API)",
            "embed":     "all-MiniLM-L6-v2 (text) + clip-ViT-B-32 (multimodal)",
            "rerank":    "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "eval_llm":  "llama3-8b-8192 (Groq)",
        },
        "endpoints": {
            "query":         "POST /query",
            "query_stream":  "POST /query-stream",
            "evaluate":      "POST /evaluate",
            "monitor_stats": "GET /monitor/stats",
            "monitor_runs":  "GET /monitor/runs",
            "feedback":      "POST /monitor/feedback",
            "docs":          "/docs",
        },
    }
