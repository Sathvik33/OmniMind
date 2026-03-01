<div align="center">

# 🧠 OmniMind

### Multi-Modal Retrieval Augmented Generation (RAG) Engine

**Ground your AI answers in real data — not hallucinations.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-LLaMA3-black?logo=ollama)](https://ollama.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#-features) • [Architecture](#-system-architecture) • [Quick Start](#-quick-start) • [API Reference](#-api-reference) • [Roadmap](#-roadmap) • [Contributing](#-contributing)

</div>

---

## 📖 Overview

**OmniMind** is a production-grade, multi-modal RAG system that grounds every answer strictly in user-uploaded content — documents, images, and videos. Unlike general-purpose chatbots, OmniMind never fabricates information from pretrained knowledge alone. It retrieves semantically relevant context from your private data before generating a response.

```
User Query
    ↓
Semantic Retrieval  ←  ChromaDB (PDFs · Images · Video Segments)
    ↓
Context Assembly
    ↓
LLM Generation  ←  LLaMA3 via Ollama (local, offline)
    ↓
Streaming Response
```

**Why OmniMind?**

| Capability | General LLM | OmniMind |
|---|---|---|
| Answers from your private data | ❌ | ✅ |
| Multi-modal input (docs + images + video) | ❌ | ✅ |
| Timestamp-aware video retrieval | ❌ | ✅ |
| Runs fully offline | ❌ | ✅ |
| Reduced hallucination via RAG | ❌ | ✅ |
| Token streaming | Varies | ✅ |

---

## ✨ Features

- **Multi-Modal Ingestion** — Ingest PDFs, DOCX, PPTX, XLSX, TXT, images, and videos through a single unified pipeline
- **Vision Understanding** — Images are processed by LLaVA, converted to rich textual descriptions, then embedded and indexed
- **Video Intelligence** — Videos are split into frames every 2 seconds; visually similar frames are skipped via difference detection; remaining frames are captioned and stored with `start_time` / `end_time` metadata
- **Timestamp-Aware Queries** — Ask "what happened in the first 30 seconds?" and get answers grounded in exact video segments
- **Cross-Modal Retrieval** — A single query retrieves relevant text chunks, image descriptions, and video segments simultaneously
- **Streaming Responses** — Token-by-token streaming via FastAPI `StreamingResponse` for a ChatGPT-style UX
- **GPU-Accelerated** — Embeddings and vision inference run on CUDA; 4-bit / 8-bit quantization keeps VRAM usage lean
- **Background Ingestion** — Uploads return immediately; heavy processing runs asynchronously via FastAPI `BackgroundTasks`
- **Offline by Default** — LLaMA3 runs locally via Ollama; no cloud API keys required

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Frontend (Streamlit)                  │
│   ChatGPT-style UI · Streaming Display · Job Status Polling  │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP / StreamingResponse
┌──────────────────────────▼──────────────────────────────────┐
│                      API Layer (FastAPI)                      │
│  /upload  /ingest-image  /ingest-video  /query  /query-stream │
└──────┬──────────────────────────────────────────┬───────────┘
       │                                          │
┌──────▼──────────┐                    ┌──────────▼──────────┐
│  Ingestion Layer │                    │    RAG Pipeline      │
│  ┌────────────┐ │                    │  ┌───────────────┐  │
│  │  Document  │ │                    │  │   Retriever   │  │
│  │  Loaders   │ │                    │  │  (top-k + time│  │
│  └────────────┘ │                    │  │   filtering)  │  │
│  ┌────────────┐ │                    │  └───────┬───────┘  │
│  │  LLaVA     │ │                    │  ┌───────▼───────┐  │
│  │  Vision    │ │                    │  │Context Builder│  │
│  └────────────┘ │                    │  └───────┬───────┘  │
│  ┌────────────┐ │                    │  ┌───────▼───────┐  │
│  │  Video     │ │                    │  │   Generator   │  │
│  │  Service   │ │                    │  │  (LLaMA3 +    │  │
│  └────────────┘ │                    │  │  Streaming)   │  │
└──────┬──────────┘                    └──────────┬──────────┘
       │                                          │
┌──────▼──────────────────────────────────────────▼──────────┐
│                   Embedding Layer                            │
│             all-MiniLM-L6-v2  ·  GPU Batch Encoding         │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│                   Vector Store (ChromaDB)                     │
│   Document Chunks · Image Descriptions · Video Segments      │
│          Metadata: modality · source · timestamps            │
└─────────────────────────────────────────────────────────────┘
```

### Layer Breakdown

#### 1. API Layer — `backend/app/api/`

Thin HTTP interface built on FastAPI. Receives file uploads, triggers background ingestion jobs, polls job status, and streams query responses. Key endpoints:

| Endpoint | Method | Purpose |
|---|---|---|
| `/upload` | POST | Ingest documents |
| `/ingest-image` | POST | Queue image ingestion |
| `/image-status/{job_id}` | GET | Poll image job status |
| `/ingest-video` | POST | Queue video ingestion |
| `/video-status/{job_id}` | GET | Poll video job status |
| `/query` | POST | Blocking RAG query |
| `/query-stream` | POST | Streaming RAG query |
| `/clear-memory` | POST | Wipe vector store |

#### 2. Ingestion Layer — `backend/app/ingestion/`

Converts raw files into structured, embeddable content.

- **Documents** → loaded by type-specific loaders → chunked via LangChain → embedded → stored in ChromaDB
- **Images** → passed to LLaVA → natural language description generated → embedded → stored as document
- **Videos** → frames extracted every 2s via OpenCV → frame diff detection skips similar frames → LLaVA captions remaining frames → temporally adjacent similar captions merged → segments stored with `start_time` / `end_time`

#### 3. Vision Service — `backend/app/services/`

Handles all vision-model operations: frame extraction, adaptive frame skipping (`cv2.absdiff`), temporal merging, and caption generation. Uses `BackgroundTasks` to keep the API non-blocking.

#### 4. Embedding Layer — `backend/app/embeddings/`

```python
SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
```

Batch GPU encoding for fast, high-quality dense vector representations.

#### 5. Vector Store — `backend/app/vectorstore/`

ChromaDB with persistent storage. Holds all modalities in a unified collection with rich metadata. Lazy collection recreation ensures stability after memory clearing.

#### 6. Retrieval Layer — `backend/app/rag/retriever.py`

Semantic top-k retrieval with time-based filtering for video content. Supports natural language time queries:

- `"first 30 seconds"` → filters `start_time ≤ 30`
- `"between 1:00 and 2:30"` → filters by timestamp range
- `"at 45 seconds"` → nearest segment lookup

#### 7. Generation Layer — `backend/app/rag/generator.py`

```python
ChatOllama(
    model="llama3",
    temperature=0.2,
    num_predict=1500,
    streaming=True
)
```

Prompt-engineered to answer strictly from retrieved context. Explicitly states when no relevant context was found.

---

## 🤖 Models

### Vision — LLaVA 1.5 7B

```python
LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    quantization_config=quant_config,  # 4-bit or 8-bit via bitsandbytes
    device_map="auto"
)
```

Quantization reduces VRAM consumption significantly, enabling deployment on consumer GPUs (RTX 4060 8GB tested).

### Language Model — LLaMA3 via Ollama

Runs as a local server (`ollama serve`). No internet required after initial pull. Streaming enabled for low-latency token generation.

---

## ⚡ Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: RTX 4060 8GB+)
- [Ollama](https://ollama.ai/) installed

### 1. Clone the Repository

```bash
git clone https://github.com/Sathvik33/OmniMind.git
cd OmniMind
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Pull and Start the LLM

```bash
ollama pull llama3
ollama serve
```

### 5. Start the Backend

```bash
uvicorn backend.app.main:app --reload --reload-dir backend
```

### 6. Start the Frontend

```bash
streamlit run frontend/user.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💻 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | Any CUDA GPU | NVIDIA RTX 4060 (8GB VRAM) |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB | 20 GB+ |
| CUDA | 11.8+ | 12.x |

> **CPU Mode:** Supported but significantly slower for vision inference. Set `device_map="cpu"` in vision service config.

---

## 📁 Project Structure

```
OmniMind/
├── backend/
│   └── app/
│       ├── api/                  # HTTP endpoints
│       │   ├── upload.py
│       │   ├── image.py
│       │   ├── video.py
│       │   └── query.py
│       ├── ingestion/
│       │   ├── loaders/          # Per-format file loaders
│       │   │   ├── pdf_loader.py
│       │   │   ├── docx_loader.py
│       │   │   ├── ppt_loader.py
│       │   │   ├── excel_loader.py
│       │   │   └── text_loader.py
│       │   ├── chunking/         # LangChain text splitters
│       │   └── experimental/     # Semantic chunking research
│       ├── embeddings/           # SentenceTransformer GPU encoding
│       ├── vectorstore/          # ChromaDB client + collection manager
│       ├── rag/
│       │   ├── retriever.py      # Semantic + time-based retrieval
│       │   ├── context_builder.py
│       │   └── generator.py      # LLaMA3 streaming generation
│       ├── services/
│       │   ├── vision_service.py # LLaVA image captioning
│       │   └── video_service.py  # Frame extraction + temporal merging
│       ├── pipelines/            # End-to-end orchestration
│       ├── models/               # Ollama model wrappers
│       ├── monitoring/           # LangSmith logging
│       └── utils/                # Time parsing utilities
├── frontend/
│   └── user.py                   # Streamlit chat interface
├── chroma_db/                    # Persistent vector storage
├── data/                         # Uploaded file storage
├── tests/
│   ├── test_ingestion_router.py
│   └── test_vectorstore.py
└── requirements.txt
```

---

## 🔌 API Reference

### Upload Document

```http
POST /upload
Content-Type: multipart/form-data

file: <PDF | DOCX | PPTX | XLSX | TXT>
```

**Response:**
```json
{ "status": "ingested", "chunks": 42, "source": "report.pdf" }
```

---

### Ingest Image

```http
POST /ingest-image
Content-Type: multipart/form-data

file: <PNG | JPG>
```

**Response:**
```json
{ "job_id": "abc123", "status": "processing" }
```

---

### Stream Query

```http
POST /query-stream
Content-Type: application/json

{
  "query": "What does the document say about Q3 revenue?"
}
```

**Response:** `text/event-stream` — token-by-token SSE stream.

---

### Clear Memory

```http
POST /clear-memory
```

Wipes all vectors from ChromaDB. Lazy recreation ensures the system is immediately ready for new ingestion.

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Test coverage includes:
- Vector store operations (`test_vectorstore.py`)
- Ingestion routing logic (`test_ingestion_router.py`)

---

## ⚙️ Performance Optimizations

- **Frame difference detection** — `cv2.absdiff` skips visually redundant video frames, reducing LLaVA inference calls by ~60–80%
- **Temporal segment merging** — consecutive frames with similar captions are merged into a single timestamped segment
- **4-bit / 8-bit quantization** — LLaVA VRAM footprint reduced from ~14GB to ~5GB
- **GPU batch embeddings** — all-MiniLM-L6-v2 processes chunks in batches on CUDA
- **Reduced `max_new_tokens`** for vision inference — faster per-frame captioning
- **Streaming generation** — first token appears in <1s; no waiting for full response
- **Lazy Chroma collection recreation** — avoids race conditions after memory clearing
- **Background ingestion** — uploads are non-blocking; frontend polls job status

---

## 🗺 Roadmap

- [ ] Per-user isolated memory / namespaced collections
- [ ] JWT authentication and multi-tenant support
- [ ] WebSocket-based streaming (replace SSE)
- [ ] Hybrid dense + sparse retrieval (BM25 + embeddings)
- [ ] Semantic frame merging via embedding similarity
- [ ] Redis caching layer for frequent queries
- [ ] Async ingestion queues (Celery / ARQ)
- [ ] Distributed embedding workers
- [ ] Docker Compose deployment
- [ ] Kubernetes horizontal scaling
- [ ] Model selector interface (swap LLMs / vision models at runtime)
- [ ] Persistent conversational memory across sessions
- [ ] Ingestion progress percentage tracking
- [ ] LangSmith tracing dashboard

---

## 🤝 Contributing

Contributions are welcome. Please follow these guidelines to maintain code quality:

1. **Fork** the repository
2. **Create a feature branch** (`git checkout -b feature/your-feature`)
3. **Commit your changes** with clear messages
4. **Open a Pull Request** against `main`

**Code Standards:**

- Maintain modular architecture — one responsibility per module
- No hardcoded paths — use `config.py` and environment variables
- GPU-safe model loading — always use `device_map="auto"` or explicit CUDA checks
- Clean separation of concerns between API, service, and RAG layers
- Add tests for new ingestion paths or retrieval logic

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Sathvik**
GitHub: [@Sathvik33](https://github.com/Sathvik33)
Repository: [github.com/Sathvik33/OmniMind](https://github.com/Sathvik33/OmniMind)

---

<div align="center">

**OmniMind** — *A unified multimodal knowledge engine for grounded, real-time AI answers.*

⭐ Star this repo if you find it useful!

</div>