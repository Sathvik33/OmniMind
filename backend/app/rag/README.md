# RAG generation

**Path:** `backend/app/rag/`

Turns retrieved strings into grounded answers.

| File | Role |
|------|------|
| `context_builder.py` | Join chunks with blank lines |
| `generator.py` | Structured generate + **token stream** |
| `schemas.py` | `StructuredAnswer` Pydantic model |
| `text_format.py` | Markdown → plain text |
| `retriever.py` | Legacy surface — live path uses `retrieval/` |

---

## Why two generation modes?

| Mode | Prompt style | Used by | Why |
|------|--------------|---------|-----|
| `generate` / structured | JSON schema (`answer`, `sections`, `no_context`, …) | `/query`, LangGraph | Reliable downstream eval + plain-text flatten |
| `stream_generate` | Plain-text rules, no JSON | `/query-stream` | Tokens must be user-readable as they arrive |

Streaming uses the LLM’s native `.stream()`. Fallback: full generate then slice — only if native stream fails.

---

## Grounding rules (stream prompt)

- Use **only** Context  
- `Spoken` and `Visual` lines both count as evidence  
- Prefer concrete objects/times from context  
- Refuse only when context is unrelated — not when captions are merely partial  

**Why soften refuse?** Earlier strict wording caused “could not find…” when vision mentioned vehicles but not the word “cars”.

---

## ContextBuilder

Intentionally minimal: `"\n\n".join(chunks)`.

**Why not fancy packing yet?** Reranker already selected top-k; packing heuristics can wait until context overflow is measured. Keep this layer dumb so retrieval bugs stay visible.

---

## Design rules

1. Never stream JSON to the React UI.  
2. Strip markdown in structured answers before return.  
3. Temperature stays low (model wrappers ~0.2) for factual tone.
