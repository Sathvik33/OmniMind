# LLM model wrappers

**Path:** `backend/app/models/`

Thin clients with a shared `.generate()` / `.stream()` surface so `QueryPipeline` can swap local vs cloud via `USE_LOCAL_LLM`.

| File | Backend | Role |
|------|---------|------|
| `ollama_model.py` | ChatOllama | Local gen; falls back to Groq on failure |
| `groq_model.py` | ChatGroq | Cloud gen (`GROQ_GENERATION_MODEL`), streaming on |
| `vision_llm.py` | Groq vision | Structured caption dict for image ingest |

---

## Why wrappers?

| Without wrappers | With wrappers |
|------------------|---------------|
| Pipeline imports LangChain classes directly | One config flag switches providers |
| Stream/generate APIs diverge | Same methods on both |
| Vision caption shape drifts | `VisionLLM` normalizes dict fields |

---

## Generation defaults

| Setting | Typical value | Why |
|---------|---------------|-----|
| Temperature | ~0.2 | Factual RAG |
| `max_tokens` / `num_predict` | ~1500 | Long enough for structured sections |
| Streaming | enabled on ChatGroq / ChatOllama | UI token stream |

Vision captions are **not** these LLMs — see `services/groq_vision_service.py`.

---

## Design rules

1. Keep provider SDKs out of `pipelines/` and `api/`.  
2. Stream path must tolerate `chunk.content is None`.  
3. Prefer failing closed to Groq only when keys exist (Ollama fallback path).
