<div align="center">

# ⚡ AEGIS v3.0

### Production Multi-Modal Hybrid RAG Engine

**Ground your AI answers in real data — hybrid search, guardrails, and enterprise storage.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-pgvector-blue?logo=postgresql)](https://github.com/pgvector/pgvector)
[![MinIO](https://img.shields.io/badge/MinIO-Object%20Store-red?logo=minio)](https://min.io/)
[![Redis](https://img.shields.io/badge/Redis-Cache%20%26%20Queue-red?logo=redis)](https://redis.io/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA3--70B-orange)](https://groq.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

[Overview](#-overview) · [Features](#-features) · [Architecture](#-system-architecture) · [Quick Start](#-quick-start) · [API Reference](#-api-reference)

</div>

---

## 📖 Overview

**AEGIS** is an enterprise-grade multi-modal Retrieval Augmented Generation (RAG) engine that grounds every answer strictly in user-uploaded content — documents, images, and videos. Driven by a **PostgreSQL 15 + pgvector** vector store, **MinIO** object storage, **Redis** job queuing/caching, and **LangGraph** orchestration, AEGIS retrieves semantically relevant context before generating answers via **Groq Cloud (LLaMA3-70B & LLaMA-4 Vision)**.

```
User Query ──► Input Guardrails ──► Hybrid Retrieval (BM25 + pgvector + CLIP)
           ──► RRF Fusion ──► Cross-Encoder Reranking ──► LangGraph Assembly
           ──► Groq Streaming ──► Output Guardrails ──► Streamlit UI
```


**Why OmniMind?**

| Capability | General LLM | OmniMind |
|---|---|---|
| Answers from your private data | ❌ | ✅ |
| Multi-modal input (docs + images + video) | ❌ | ✅ |
| Timestamp-aware video retrieval | ❌ | ✅ |
| Runs fully offline | ❌ | ✅ |
| Reduced hallucination via RAG | ❌ | ✅ |
| Token-by-token streaming | Varies | ✅ |

### Real-World Example

> A user uploads a PDF report on India's economy, a video lecture on Indian history, and a geographic image of India.
> They ask: *"What is mentioned about India's economy in the first minute of the video?"*

OmniMind:
1. Parses `"first minute"` → filters video segments with `start_time ≤ 60`
2. Retrieves those video segments + related PDF sections + image descriptions
3. Assembles a structured, grounded context prompt
4. Streams a source-backed answer token-by-token via LLaMA3

If no relevant data is uploaded, OmniMind explicitly responds: *"No relevant context found in your uploaded data."*

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Document RAG** | Ingest PDF, DOCX, PPTX, XLSX, TXT and query against them semantically |
| 🖼️ **Image Understanding** | LLaVA vision model converts images to rich text descriptions for embedding and retrieval |
| 🎥 **Video Intelligence** | Frame extraction + frame-diff deduplication + LLaVA captioning + timestamped segment storage |
| ⏱️ **Temporal Queries** | Natural language time queries: *"first 30 seconds"*, *"between 1:00 and 2:30"*, *"at 45 seconds"* |
| 🔀 **Cross-Modal Retrieval** | A single query retrieves text chunks, image descriptions, and video segments simultaneously |
| 🌊 **Streaming Responses** | Token-by-token streaming via FastAPI `StreamingResponse` for a real-time UX |
| ⚡ **GPU-Accelerated** | CUDA-based batch embeddings + 4-bit/8-bit quantized vision model inference |
| 🔄 **Background Ingestion** | Non-blocking uploads via FastAPI `BackgroundTasks` with real-time job status polling |
| 🧹 **Memory Management** | Wipe all indexed data instantly; lazy Chroma recreation ensures immediate readiness |
| 🖥️ **ChatGPT-Style UI** | Streamlit frontend with inline file upload, streaming display, and ingestion status tracking |
| 🔒 **Fully Offline** | LLaMA3 via Ollama — no cloud API keys, no data leaves your machine |

---

## 🏗️ System Architecture

### High-Level Pipeline

```
Upload
  │
  ▼
Ingestion Layer ──── Documents  (PDF / DOCX / PPTX / XLSX / TXT)
  │              ──── Images    (LLaVA vision → text description)
  │              ──── Videos    (frame extract → diff detection → caption → timestamped segments)
  ▼
Chunking & Embedding  (all-MiniLM-L6-v2 · GPU batch encoding)
  │
  ▼
Vector Storage (ChromaDB)  ── metadata: modality · source · start_time · end_time
  │
  ▼
Retrieval Layer  ── semantic top-k  +  time-based filtering
  │
  ▼
Context Builder  ── merges chunks · formats grounded prompt
  │
  ▼
LLM Generation  (LLaMA3 via Ollama · streaming · temperature 0.2)
  │
  ▼
FastAPI StreamingResponse  →  Streamlit Frontend
```

### Detailed Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Frontend  (Streamlit)                        │
│   ChatGPT-style UI · Token Streaming · Job Status Polling        │
└────────────────────────────┬────────────────────────────────────┘
                             │  HTTP / StreamingResponse
┌────────────────────────────▼────────────────────────────────────┐
│                      API Layer  (FastAPI)                         │
│  /upload · /ingest-image · /ingest-video · /query-stream · ...   │
└──────┬──────────────────────────────────────────┬───────────────┘
       │                                          │
┌──────▼──────────────┐                ┌──────────▼──────────────┐
│   Ingestion Layer   │                │      RAG Pipeline        │
│  ┌───────────────┐  │                │  ┌───────────────────┐  │
│  │Document Loaders│ │                │  │     Retriever     │  │
│  │PDF·DOCX·PPTX  │ │                │  │ top-k + temporal  │  │
│  │XLSX·TXT       │ │                │  │    filtering      │  │
│  └───────────────┘  │                │  └────────┬──────────┘  │
│  ┌───────────────┐  │                │  ┌────────▼──────────┐  │
│  │ LLaVA Vision  │  │                │  │  Context Builder  │  │
│  │ Image Caption │  │                │  │  Prompt Assembly  │  │
│  └───────────────┘  │                │  └────────┬──────────┘  │
│  ┌───────────────┐  │                │  ┌────────▼──────────┐  │
│  │ Video Service │  │                │  │    Generator      │  │
│  │Frame Extract  │  │                │  │ LLaMA3 · Ollama   │  │
│  │Diff Detection │  │                │  │ Streaming · 0.2°  │  │
│  │Temporal Merge │  │                │  └───────────────────┘  │
│  └───────────────┘  │                └─────────────────────────┘
└──────┬──────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────┐
│                      Embedding Layer                            │
│           all-MiniLM-L6-v2  ·  GPU Batch Encoding (CUDA)        │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                     Vector Store  (ChromaDB)                    │
│   Document Chunks · Image Descriptions · Video Segments         │
│         Metadata:  modality · source · start_time · end_time    │
└─────────────────────────────────────────────────────────────────┘
```

---

### Module Breakdown

#### 1. API Layer — `backend/app/api/`

Thin HTTP interface built on FastAPI. Receives uploads, triggers background jobs, polls status, and streams responses.

| Endpoint | Method | Purpose |
|---|---|---|
| `/upload` | POST | Ingest document files |
| `/ingest-image` | POST | Queue background image ingestion |
| `/image-status/{job_id}` | GET | Poll image ingestion status |
| `/ingest-video` | POST | Queue background video ingestion |
| `/video-status/{job_id}` | GET | Poll video ingestion status |
| `/query` | POST | Blocking synchronous RAG query |
| `/query-stream` | POST | Streaming RAG query (SSE) |
| `/clear-memory` | DELETE | Wipe all vector store data |

#### 2. Ingestion Layer — `backend/app/ingestion/`

Converts raw multimodal files into structured, embeddable content. Each modality follows a dedicated processing path:

- **Documents** → format-specific loader → LangChain chunker → GPU embeddings → ChromaDB
- **Images** → LLaVA vision model → natural language description → GPU embeddings → ChromaDB
- **Videos** → OpenCV frame extraction (2s intervals) → `cv2.absdiff` frame deduplication → LLaVA captioning → temporal segment merging → stored with `start_time` / `end_time` metadata

#### 3. Vision Service — `backend/app/services/`

Manages all vision-model operations end-to-end:

| Component | Implementation |
|---|---|
| Frame Extraction | OpenCV at 2-second intervals |
| Frame Deduplication | `cv2.absdiff` pixel-level difference threshold |
| Caption Generation | LLaVA 1.5-7B (4-bit / 8-bit quantized via `bitsandbytes`) |
| Temporal Merging | Adjacent segments with similar captions fused into single entries |
| Async Execution | FastAPI `BackgroundTasks` — never blocks the API thread |

#### 4. Embedding Layer — `backend/app/embeddings/`

```python
SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
# Batch encoding enabled for maximum throughput
```

All modalities — text chunks, image descriptions, video captions — are encoded into the same 384-dimensional embedding space, enabling unified cross-modal retrieval from a single query.

#### 5. Vector Store — `backend/app/vectorstore/`

- **Database:** ChromaDB with persistent on-disk storage
- **Stores:** Document chunks, image descriptions, and timestamped video segments
- **Metadata per entry:** `modality`, `source`, `start_time`, `end_time`
- **Stability:** Lazy collection recreation prevents race conditions after memory clearing

#### 6. Retrieval Layer — `backend/app/rag/retriever.py`

Supports two retrieval modes operating in tandem:

- **Semantic search** — top-k cosine similarity over dense embeddings
- **Temporal filtering** — parses natural language time expressions into metadata filters:
  - `"first 30 seconds"` → `start_time ≤ 30`
  - `"between 1:00 and 2:30"` → timestamp range filter
  - `"at 45 seconds"` → nearest segment lookup
  - `"last X seconds"` → tail-end time filter

#### 7. Generation Layer — `backend/app/rag/generator.py`

```python
ChatOllama(
    model="llama3",
    temperature=0.2,
    num_predict=1500,
    streaming=True
)
```

Prompt-engineered to answer strictly from retrieved context. Temperature 0.2 keeps responses factual and deterministic. Runs entirely locally — zero cloud dependency after the initial `ollama pull`.

---

## 🤖 Model Details

### Vision Model — LLaVA 1.5-7B

| Property | Value |
|---|---|
| Model ID | `llava-hf/llava-1.5-7b-hf` |
| Quantization | 4-bit or 8-bit via `bitsandbytes` |
| Device Mapping | `device_map="auto"` (auto GPU/CPU split) |
| VRAM — 4-bit mode | ~5 GB |
| VRAM — full precision | ~14 GB |
| Tested On | NVIDIA RTX 4060 8GB |
| Purpose | Image understanding + video frame captioning |

```python
LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    quantization_config=quant_config,   # BitsAndBytesConfig: 4-bit or 8-bit
    device_map="auto"
)
```

Quantization reduces VRAM footprint from ~14GB to ~5GB, enabling deployment on consumer GPUs.

### Language Model — LLaMA3 via Ollama

| Property | Value |
|---|---|
| Model | `llama3` |
| Interface | `ChatOllama` (LangChain) |
| Streaming | Token-by-token via `StreamingResponse` |
| Temperature | 0.2 (factual, low-creativity) |
| Deployment | Fully local via `ollama serve` |
| Internet Required | ❌ After initial pull |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: NVIDIA RTX 4060 8GB+)
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

### 5. Configure Environment

Create a `.env` file inside `backend/`:

```env
# Optional: Enable LangSmith tracing for observability
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
```

### 6. Start the Backend

```bash
uvicorn backend.app.main:app --reload --reload-dir backend
```

Backend available at: `http://localhost:8000`

### 7. Start the Frontend

```bash
streamlit run frontend/user.py
```

Frontend available at: `http://localhost:8501`

---

## 🖥️ Frontend Overview

The Streamlit frontend provides a ChatGPT-style experience built for multimodal interaction:

- **Inline upload button** — attach files directly inside the chat input
- **Real-time token streaming** — responses appear token-by-token with no wait
- **Query blocking during ingestion** — prevents retrieval race conditions
- **Background job polling** — live status updates during video/image processing
- **Memory clear** — wipe the entire vector store from the sidebar in one click

---

## 📁 Project Structure

```
OmniMind/
├── backend/
│   └── app/
│       ├── api/                    # FastAPI HTTP route handlers
│       │   ├── upload.py
│       │   ├── image.py
│       │   ├── video.py
│       │   ├── query.py
│       │   └── health.py
│       ├── ingestion/
│       │   ├── loaders/            # Per-format file loaders
│       │   │   ├── base_loader.py
│       │   │   ├── pdf_loader.py
│       │   │   ├── docx_loader.py
│       │   │   ├── ppt_loader.py
│       │   │   ├── excel_loader.py
│       │   │   ├── text_loader.py
│       │   │   └── file_router.py
│       │   ├── chunking/           # LangChain text splitters
│       │   │   └── langchain_chunker.py
│       │   └── experimental/       # Semantic & recursive chunking R&D
│       │       ├── semantic.py
│       │       ├── recursive.py
│       │       └── factory.py
│       ├── embeddings/             # SentenceTransformer GPU encoding
│       │   └── embedding_model.py
│       ├── vectorstore/            # ChromaDB client + collection manager
│       │   ├── chroma_client.py
│       │   └── collection_manager.py
│       ├── rag/                    # Core RAG components
│       │   ├── retriever.py        # Semantic + time-based retrieval
│       │   ├── context_builder.py  # Prompt assembly
│       │   └── generator.py        # LLaMA3 streaming generation
│       ├── services/
│       │   ├── vision_service.py   # LLaVA image captioning
│       │   └── video_service.py    # Frame extraction + temporal merging
│       ├── pipelines/              # End-to-end orchestration
│       │   ├── ingestion_pipeline.py
│       │   ├── query_pipeline.py
│       │   └── multimodal_pipeline.py
│       ├── models/                 # Ollama model wrappers
│       │   └── ollama_model.py
│       ├── monitoring/             # LangSmith observability
│       │   └── langsmith_logger.py
│       ├── core/                   # App config & settings
│       │   └── config.py
│       └── utils/                  # Shared utilities
│           └── time_utils.py       # Natural language time parsing
├── frontend/
│   └── user.py                     # Streamlit chat interface
├── chroma_db/                      # Persistent ChromaDB storage
├── data/                           # Uploaded file storage
├── tests/
│   ├── test_ingestion_router.py
│   └── test_vectorstore.py
├── requirements.txt
└── README.md
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
{
  "status": "ingested",
  "chunks": 42,
  "source": "report.pdf"
}
```

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

### Ingest Video

```http
POST /ingest-video
Content-Type: multipart/form-data

file: <MP4 | AVI | MOV>
```

**Response:**
```json
{ "job_id": "xyz789", "status": "processing" }
```

### Poll Ingestion Status

```http
GET /video-status/{job_id}
GET /image-status/{job_id}
```

**Response:**
```json
{ "job_id": "xyz789", "status": "complete", "segments": 18 }
```

### Streaming Query

```http
POST /query-stream
Content-Type: application/json

{
  "query": "What does the first minute of the video say about revenue?"
}
```

**Response:** `text/event-stream` — server-sent events, token-by-token.

### Clear Memory

```http
DELETE /clear-memory
```

Wipes all vectors from ChromaDB. Lazy collection recreation ensures the system is immediately ready for new ingestion.

---

## ⚙️ Performance Optimizations

| Optimization | Mechanism | Impact |
|---|---|---|
| Frame Difference Detection | `cv2.absdiff` pixel threshold | Reduces LLaVA calls by ~60–80% |
| Temporal Segment Merging | Adjacent similar captions fused | Compact storage, faster retrieval |
| 4-bit / 8-bit Quantization | `bitsandbytes` on LLaVA | VRAM: ~14 GB → ~5 GB |
| GPU Batch Embeddings | CUDA + `encode(batch)` | 5–10× faster than CPU |
| Reduced `max_new_tokens` | Vision inference config | Faster per-frame captioning |
| Streaming Generation | FastAPI `StreamingResponse` | First token in < 1s |
| Lazy Chroma Recreation | Deferred collection init | Stable post-clear behavior |
| Query Blocking | Frontend ingestion lock | Zero retrieval race conditions |
| Background Ingestion | FastAPI `BackgroundTasks` | Non-blocking API responses |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Current test coverage:

- `test_vectorstore.py` — ChromaDB collection creation, insertion, deletion
- `test_ingestion_router.py` — API-level ingestion endpoint validation

---

## 🔧 Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | Any CUDA GPU (4GB VRAM) | NVIDIA RTX 4060 8GB |
| CUDA Version | 11.8+ | 12.x |
| RAM | 8 GB | 16 GB |
| Storage | 10 GB free | 20 GB+ |
| CPU Mode | ✅ Supported (slower) | — |

> **Note:** CPU-only mode is supported but vision inference will be significantly slower. Set `device_map="cpu"` in the vision service config for CPU-only deployments.

---

## 🗺️ Roadmap

- [ ] Per-user isolated memory / namespaced ChromaDB collections
- [ ] JWT authentication and multi-tenant session management
- [ ] WebSocket-based streaming (replace SSE)
- [ ] Hybrid dense + sparse retrieval (BM25 + embeddings)
- [ ] Semantic frame merging via embedding similarity
- [ ] Redis caching layer for repeated queries
- [ ] Async ingestion queues (Celery / ARQ)
- [ ] Distributed embedding workers
- [ ] Docker & Docker Compose deployment
- [ ] Kubernetes horizontal scaling
- [ ] Model selector interface (swap LLMs / vision models at runtime)
- [ ] Persistent conversational memory across sessions
- [ ] Ingestion progress percentage tracking
- [ ] LangSmith tracing dashboard integration
- [ ] Auto-scaling model inference

---

## 🤝 Contributing

Contributions are welcome. Please follow these guidelines to maintain code quality and architectural integrity:

1. **Fork** the repository
2. **Create a feature branch:** `git checkout -b feature/your-feature-name`
3. **Commit your changes:** `git commit -m "feat: describe your change"`
4. **Push to your branch:** `git push origin feature/your-feature-name`
5. **Open a Pull Request** against `main`

**Code Standards:**

- One responsibility per module — maintain clean separation between API, ingestion, service, and RAG layers
- No hardcoded paths — use `config.py` and environment variables for all configuration
- GPU-safe model loading — always use `device_map="auto"` or explicit CUDA availability checks
- Add tests for any new ingestion paths or retrieval logic
- Document new endpoints in the API Reference section above

---

## 📄 License

This project is licensed under the **Apache License 2.0**.

```
Copyright 2024 Sathvik

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

See the full [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Sathvik**
- GitHub: [@Sathvik33](https://github.com/Sathvik33)
- Repository: [github.com/Sathvik33/OmniMind](https://github.com/Sathvik33/OmniMind)

---

<div align="center">

**OmniMind** — *A unified multimodal knowledge engine for grounded, real-time AI answers.*

*Your data. Your answers. No hallucination.*

⭐ Star this repo if you find it useful!

</div>