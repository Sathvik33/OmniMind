# LLM model wrappers

**Path:** `backend/app/models/`

Thin clients with a shared `.generate()` / `.stream()` surface. `QueryPipeline` always uses `FailoverLLM`, which builds a **cost-aware chain**.

| File | Backend | Role |
|------|---------|------|
| `failover_llm.py` | Chain orchestrator | Local → Groq → OpenRouter |
| `ollama_model.py` | ChatOllama | Local Qwen (`qwen2.5:7b`) — preferred for daily dev |
| `groq_model.py` | ChatGroq | Cloud gen (`GROQ_GENERATION_MODEL`), streaming |
| `openrouter_model.py` | OpenRouter HTTP | Free-tier `:free` models only |
| `vision_llm.py` | Groq vision | Structured caption dict for image ingest |

Vision / ASR for **ingest** live under `services/` (not these answer LLMs).

---

## Failover priority

```text
USE_LOCAL_LLM=true  (default for local development)
  1. Ollama qwen2.5:7b     ← cost-free daily path
  2. Groq LLaMA            ← if Ollama OOM / down
  3. OpenRouter :free      ← if Groq rate-limited / 404

USE_LOCAL_LLM=false  (demos / low RAM)
  1. Groq → 2. OpenRouter :free
```

---

## Why wrappers?

| Without wrappers | With wrappers |
|------------------|---------------|
| Pipeline imports LangChain classes directly | One config flag reorders the chain |
| Stream/generate APIs diverge | Same methods on every backend |
| Fallbacks scattered | Single `FailoverLLM` policy |

---

## Generation defaults

| Setting | Typical value | Why |
|---------|---------------|-----|
| Temperature | ~0.2 | Factual RAG |
| `max_tokens` / `num_predict` | ~1500 | Long enough for structured sections |
| Streaming | enabled | UI SSE token stream |

---

## Design rules

1. Keep provider SDKs out of `pipelines/` and `api/`.
2. Stream path must tolerate `chunk.content is None`.
3. OpenRouter is **fallback only** and must use `:free` model IDs.
4. Prefer local Qwen for development cost control.
