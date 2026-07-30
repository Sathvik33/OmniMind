import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Directory Paths ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR   = BASE_DIR / "data"
INDEX_DIR  = BASE_DIR / "storage"

DATA_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# ── BM25 Index ─────────────────────────────────────────────────────────────────
BM25_INDEX_PATH = INDEX_DIR / "bm25_index.json"


# ── Embedding Dimensions & Models ──────────────────────────────────────────────
TEXT_EMBEDDING_MODEL     = "BAAI/bge-m3"
TEXT_EMBEDDING_DIM       = 1024  # BAAI/bge-m3
MULTIMODAL_EMBEDDING_DIM = 768   # ViT-B-16-SigLIP


# ── Chunking ───────────────────────────────────────────────────────────────────
SEMANTIC_THRESHOLD  = 0.75   # cosine similarity below this → new chunk
MIN_CHUNK_SENTENCES = 2
MAX_CHUNK_SENTENCES = 15

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K  = 20   # candidates before reranking
RERANKER_TOP_K   = 5    # final results after reranking
RRF_K            = 60   # Reciprocal Rank Fusion constant

# ── Environment & Mode (development vs production) ────────────────────────────

APP_ENV       = os.getenv("APP_ENV", "development").lower()
USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "true").lower() == "true"

# ── LLM (Ollama — primary) ────────────────────────────────────────────────────
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TEMP     = float(os.getenv("OLLAMA_TEMP", "0.2"))
OLLAMA_MAX_PRED = int(os.getenv("OLLAMA_MAX_PRED", "1500"))


# ── Groq LLM & Vision API Keys (Separate keys for LLM vs Vision) ─────────────
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
GROQ_LLM_API_KEY    = os.getenv("GROQ_LLM_API_KEY") or os.getenv("GROQ_GENERATION_API_KEY") or GROQ_API_KEY
GROQ_VISION_API_KEY = os.getenv("GROQ_VISION_API_KEY") or GROQ_API_KEY

GROQ_MODEL            = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_GENERATION_MODEL = os.getenv("GROQ_GENERATION_MODEL", "llama-3.3-70b-versatile")
GROQ_VISION_MODEL     = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")



# ── Video Ingestion ───────────────────────────────────────────────────────────
VIDEO_FRAME_INTERVAL_SEC    = int(os.getenv("VIDEO_FRAME_INTERVAL_SEC", "10"))   # sample 1 frame every N seconds
VIDEO_SCENE_DIFF_THRESHOLD  = float(os.getenv("VIDEO_SCENE_DIFF_THRESHOLD", "8")) # skip near-duplicate frames

# ── LangSmith Monitoring ───────────────────────────────────────────────────────
LANGSMITH_API_KEY      = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT      = os.getenv("LANGCHAIN_PROJECT", "Aegis")
LANGSMITH_TRACING      = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGSMITH_ENDPOINT     = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_SAMPLING_RATE = float(os.getenv("LANGSMITH_SAMPLING_RATE", "1.0"))

# ── Redis Broker & Embedding Cache Separation ──────────────────────────────────
REDIS_URL               = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_BROKER_URL        = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
REDIS_CACHE_URL         = os.getenv("REDIS_CACHE_URL", "redis://localhost:6379/2")
EMBEDDING_CACHE_VERSION = os.getenv("EMBEDDING_CACHE_VERSION", "v1")

# ── RAGAS Evaluation ──────────────────────────────────────────────────────────
RAGAS_EVALUATION_THRESHOLD = float(os.getenv("RAGAS_EVALUATION_THRESHOLD", "0.5"))
RAGAS_ASYNC_MODE           = os.getenv("RAGAS_ASYNC_MODE", "true").lower() == "true"
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]