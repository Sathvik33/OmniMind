import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Directory Paths ────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR   = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"

DATA_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# ── BM25 Index ─────────────────────────────────────────────────────────────────
BM25_INDEX_PATH = CHROMA_DIR / "bm25_index.pkl"

# ── Embedding Dimensions ───────────────────────────────────────────────────────
TEXT_EMBEDDING_DIM       = 384   # all-MiniLM-L6-v2
MULTIMODAL_EMBEDDING_DIM = 512  # clip-ViT-B-32

# ── Chunking ───────────────────────────────────────────────────────────────────
SEMANTIC_THRESHOLD  = 0.75   # cosine similarity below this → new chunk
MIN_CHUNK_SENTENCES = 2
MAX_CHUNK_SENTENCES = 15

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_TOP_K  = 20   # candidates before reranking
RERANKER_TOP_K   = 5    # final results after reranking
RRF_K            = 60   # Reciprocal Rank Fusion constant

# ── LLM (Ollama — primary) ────────────────────────────────────────────────────
OLLAMA_MODEL    = "qwen2.5:7b"
OLLAMA_TEMP     = 0.2
OLLAMA_MAX_PRED = 1500

# ── Groq LLM (for RAGAS Evaluation) ──────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# ── LangSmith Monitoring ───────────────────────────────────────────────────────
LANGSMITH_API_KEY      = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "")
LANGSMITH_PROJECT      = os.getenv("LANGCHAIN_PROJECT", "Aegis")
LANGSMITH_TRACING      = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
LANGSMITH_ENDPOINT     = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_SAMPLING_RATE = float(os.getenv("LANGSMITH_SAMPLING_RATE", "1.0"))

# ── RAGAS Evaluation ──────────────────────────────────────────────────────────
RAGAS_EVALUATION_THRESHOLD = float(os.getenv("RAGAS_EVALUATION_THRESHOLD", "0.5"))
RAGAS_ASYNC_MODE           = os.getenv("RAGAS_ASYNC_MODE", "true").lower() == "true"
RAGAS_METRICS = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]