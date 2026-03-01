<div align="center">

# 🧠 OmniMind

### Multi-Modal Retrieval Augmented Generation (RAG) Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-orange)](https://www.trychroma.com/)
[![Ollama](https://img.shields.io/badge/Ollama-LLaMA3-black)](https://ollama.com/)
[![License](https://img.shields.io/badge/License-Apache-green)](LICENSE)

**OmniMind** is a production-grade, multi-modal RAG engine that grounds every answer strictly in your uploaded data — documents, images, and videos — with no hallucination from pretrained knowledge.

[Features](#-features) · [Architecture](#-system-architecture) · [Quick Start](#-quick-start) · [API Reference](#-api-reference) · [Tech Stack](#-tech-stack) · [Roadmap](#-roadmap)

</div>

---

## 🎯 What is OmniMind?

Unlike a general-purpose chatbot, OmniMind **does not rely on pretrained knowledge** to answer your questions. It retrieves semantically relevant content from your own uploaded data and uses that context to generate grounded, accurate responses.

```
Traditional LLM:
  User Query → Model generates from training data → ⚠️ May hallucinate

OmniMind (RAG):
  User Query → Semantic Retrieval (ChromaDB) → Context Assembly
             → LLM generates grounded in YOUR data → ✅ Source-backed answer
```

### Real-World Example

> User uploads a PDF about India, a video lecture on Indian history, and an image of Indian geography.
> User asks: *"Tell me about India's economy in the first minute of the video."*

OmniMind:
1. Searches the vector database semantically
2. Retrieves relevant video segments (0–60s), PDF sections, and image descriptions
3. Assembles a grounded context prompt
4. Streams a structured, source-backed answer via LLaMA3

If no relevant data is uploaded, OmniMind explicitly states: *"No relevant context found."*

---

## ✨ Features

| Feature | Description |
|---|---|
| 📄 **Document RAG** | Ingest PDF, DOCX, PPTX, XLSX, TXT and query against them |
| 🖼️ **Image Understanding** | Vision model converts images to text descriptions for semantic retrieval |
| 🎥 **Video RAG** | Frame extraction + captioning with timestamp-aware retrieval |
| ⏱️ **Temporal Queries** | Query by time ranges: *"first 30 seconds"*, *"between 1:00 and 2:30"* |
| 🌊 **Streaming Responses** | Token-by-token streaming via FastAPI `StreamingResponse` |
| ⚡ **GPU-Accelerated** | CUDA-based embeddings and quantized vision model inference |
| 🔄 **Background Ingestion** | Non-blocking uploads with real-time job status polling |
| 🧹 **Memory Management** | Clear all indexed data and start fresh via a single API call |
| 🖥️ **ChatGPT-Style UI** | Streamlit frontend with inline upload, streaming, and status tracking |

---

## 🏗️ System Architecture

### High-Level Pipeline

```
Upload
  │
  ▼
Ingestion Layer ──── Documents (PDF/DOCX/PPTX/XLSX/TXT)
  │              ──── Images   (LLaVA vision → text description)
  │              ──── Videos   (frame extract → caption → timestamped segments)
  ▼
Chunking & Embedding (all-MiniLM-L6-v2, GPU)
  │
  ▼
Vector Storage (ChromaDB) ── metadata: modality, source, timestamps
  │
  ▼
Retrieval Layer ── semantic top-k + time-based filtering
  │
  ▼
Context Builder ── merges chunks, formats prompt
  │
  ▼
LLM Generation (LLaMA3 via Ollama, streaming)
  │
  ▼
FastAPI StreamingResponse → Streamlit Frontend
```

### Module Breakdown

#### 1. API Layer — `backend/app/api/`

| Endpoint | Method | Purpose |
|---|---|---|
| `/upload` | POST | Upload documents for ingestion |
| `/ingest-image` | POST | Trigger background image ingestion |
| `/image-status/{job_id}` | GET | Poll image ingestion status |
| `/ingest-video` | POST | Trigger background video ingestion |
| `/video-status/{job_id}` | GET | Poll video ingestion status |
| `/query` | POST | Synchronous RAG query |
| `/query-stream` | POST | Streaming RAG query |
| `/clear-memory` | DELETE | Wipe all vector store data |

#### 2. Ingestion Layer — `backend/app/ingestion/`

Converts raw multimodal inputs into structured, embeddable content.

- **Documents** → Loaded by format-specific loaders → chunked via LangChain → embedded → stored in Chroma
- **Images** → LLaVA vision model generates textual description → embedded as document → stored in Chroma
- **Videos** → OpenCV frame extraction every 2s → frame-diff deduplication → LLaVA captioning → temporal segment merging → stored with `start_time`/`end_time` metadata

#### 3. Vision Service — `backend/app/services/video_service.py`

| Component | Implementation |
|---|---|
| Frame Extraction | OpenCV at 2s intervals |
| Deduplication | `cv2.absdiff` for frame similarity detection |
| Captioning | LLaVA 1.5-7B (4-bit/8-bit quantized) |
| Temporal Merging | Combines adjacent segments with similar captions |
| Async Processing | FastAPI `BackgroundTasks` for non-blocking API |

#### 4. Embedding Layer — `backend/app/embeddings/`

```python
SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
# Batch encoding enabled for throughput
```

- Model: `all-MiniLM-L6-v2`
- Library: `sentence-transformers`
- Mode: GPU-accelerated batch encoding

#### 5. Vector Store — `backend/app/vectorstore/`

- Database: **ChromaDB** (persistent local storage)
- Stores document chunks, image descriptions, and video segments
- All entries tagged with metadata: `modality`, `source`, `start_time`, `end_time`
- Lazy collection recreation ensures stability after memory clearing

#### 6. Retrieval Layer — `backend/app/rag/retriever.py`

Supports:
- **Semantic search** — top-k cosine similarity retrieval
- **Temporal queries** — parses natural language time expressions:
  - `"first 30 seconds"`
  - `"at 1:20"`
  - `"between 0:45 and 2:10"`
  - `"last X seconds"`

#### 7. Generation Layer — `backend/app/rag/generator.py`

```python
ChatOllama(
    model="llama3",
    temperature=0.2,
    num_predict=1500,
    streaming=True
)
```

Runs fully locally via Ollama — no cloud dependency, no data leaving your machine.

---

## 🤖 Model Details

### Vision Model — LLaVA 1.5-7B

| Property | Value |
|---|---|
| Model | `llava-hf/llava-1.5-7b-hf` |
| Quantization | 4-bit or 8-bit via `bitsandbytes` |
| Device Mapping | `device_map="auto"` (auto GPU/CPU split) |
| Purpose | Image and video frame captioning |
| VRAM Optimized for | RTX 4060 8GB |

```python
LlavaForConditionalGeneration.from_pretrained(
    "llava-hf/llava-1.5-7b-hf",
    quantization_config=quant_config,
    device_map="auto"
)
```

### Text LLM — LLaMA3 via Ollama

| Property | Value |
|---|---|
| Model | `llama3` |
| Interface | `ChatOllama` (LangChain) |
| Streaming | Token-by-token via `StreamingResponse` |
| Temperature | 0.2 (factual, low-creativity) |
| Runs | Fully local, offline |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-enabled GPU (recommended: RTX 4060 8GB+)
- [Ollama](https://ollama.com/) installed

### 1. Clone the Repository

```bash
git clone https://github.com/Sathvik33/OmniMind.git
cd OmniMind
```

### 2. Create Virtual Environment

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

### 4. Set Up Ollama

```bash
ollama pull llama3
ollama serve
```

### 5. Configure Environment

Create a `.env` file in `backend/`:

```env
# Optional: LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_key_here
```

### 6. Start the Backend

```bash
uvicorn backend.app.main:app --reload --reload-dir backend
```

Backend runs at: `http://localhost:8000`

### 7. Start the Frontend

```bash
streamlit run frontend/user.py
```

Frontend runs at: `http://localhost:8501`

---

## 🖥️ Frontend Overview

The Streamlit frontend provides a ChatGPT-style interface with:

- **Inline + upload button** — attach files directly in the chat input
- **Real-time streaming** — responses display token-by-token
- **Query blocking during ingestion** — prevents race conditions
- **Background job polling** — shows ingestion progress
- **Memory clear button** — wipe all indexed data from sidebar

---

## 📁 Project Structure

```
OmniMind/
├── backend/
│   └── app/
│       ├── api/              # FastAPI route handlers
│       │   ├── upload.py
│       │   ├── image.py
│       │   ├── video.py
│       │   └── query.py
│       ├── ingestion/        # Document, image, video ingestion
│       │   ├── loaders/      # Format-specific file loaders
│       │   ├── chunking/     # LangChain text chunking
│       │   └── experimental/ # Semantic & recursive chunkers
│       ├── embeddings/       # SentenceTransformer GPU embeddings
│       ├── vectorstore/      # ChromaDB client & collection manager
│       ├── rag/              # Retriever, context builder, generator
│       ├── services/         # Vision & video processing services
│       ├── pipelines/        # Ingestion & query orchestration
│       ├── models/           # Ollama model interface
│       ├── monitoring/       # LangSmith logging
│       └── utils/            # Time parsing utilities
├── frontend/
│   └── user.py               # Streamlit chat interface
├── chroma_db/                # Persistent vector storage
├── data/                     # Uploaded file storage
├── tests/                    # pytest test suite
├── requirements.txt
└── README.md
```

---

## ⚙️ Performance Optimizations

| Optimization | Description |
|---|---|
| Frame Difference Detection | Skips visually similar video frames using `cv2.absdiff` |
| Temporal Segment Merging | Combines adjacent video captions for compact storage |
| GPU Embeddings | `all-MiniLM-L6-v2` runs on CUDA for fast batch encoding |
| Batched Encoding | Multiple chunks embedded in a single forward pass |
| Reduced Vision Tokens | Lower `max_new_tokens` for faster caption generation |
| Streaming Responses | FastAPI `StreamingResponse` for real-time UX |
| Lazy Chroma Recreation | Collection rebuilt only when needed after memory clear |
| Query Blocking | Frontend prevents queries during active ingestion |
| BackgroundTasks | Image/video ingestion never blocks the API thread |

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

Test coverage includes:
- `test_vectorstore.py` — ChromaDB collection operations
- `test_ingestion_router.py` — API ingestion endpoints

---

## 🔧 Hardware Requirements

| Component | Recommended | Minimum |
|---|---|---|
| GPU | RTX 4060 8GB (CUDA) | Any CUDA GPU |
| RAM | 16 GB | 8 GB |
| Storage | 20 GB free | 10 GB free |
| CPU Mode | Supported (slower) | — |

> **Note:** CPU-only mode works but vision inference will be significantly slower. GPU is strongly recommended for real-time video processing.

---

## 🗺️ Roadmap

- [ ] Per-user isolated memory layers
- [ ] JWT authentication & user sessions
- [ ] Redis caching layer for repeated queries
- [ ] WebSocket-based streaming
- [ ] Hybrid dense + sparse retrieval (BM25 + vector)
- [ ] Embedding-based semantic frame merging
- [ ] Distributed embedding workers
- [ ] Docker & Docker Compose deployment
- [ ] Kubernetes horizontal scaling
- [ ] Model selector interface (swap LLMs at runtime)
- [ ] Persistent conversational memory across sessions
- [ ] Ingestion progress percentage tracking
- [ ] Async ingestion queues (Celery/ARQ)
- [ ] Auto-scaling model inference

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`
3. **Commit** your changes: `git commit -m "feat: add your feature"`
4. **Push** to the branch: `git push origin feature/your-feature-name`
5. **Open** a Pull Request

### Development Guidelines

- Maintain modular architecture — one concern per module
- No hardcoded paths — use `config.py` for all configurable values
- GPU-safe model loading — always use `device_map="auto"` or explicit device handling
- Clean separation of concerns — API, ingestion, RAG, and generation layers must remain independent

---

## 📄 License

This project is licensed under the Apache License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Sathvik**
- GitHub: [@Sathvik33](https://github.com/Sathvik33)
- Repository: [OmniMind](https://github.com/Sathvik33/OmniMind)

---

<div align="center">

**OmniMind** — *A unified multimodal knowledge engine. Your data. Your answers. No hallucination.*

⭐ Star this repo if you find it useful!

</div>