# Embeddings package

**Path:** `backend/app/embeddings/`

Thin historical wrappers around SentenceTransformers / CLIP.  
**Production ingest and retrieval use `services/embedding_service.py`**, not these classes directly.

| File | Model | Dim | Notes |
|------|-------|-----|-------|
| `embedding_model.py` | `BAAI/bge-m3` | 1024 | Batch encode helper (`batch_size=32`) |
| `clip_embedding_model.py` | `clip-ViT-B-32` | 512 | Legacy; superseded by SigLIP in `EmbeddingService` |

---

## Why both packages exist

```text
embeddings/          → early standalone wrappers / tests
services/embedding_service.py  → live singleton + Redis cache + SigLIP + registry
```

Live path adds:

1. Redis embedding cache (DB 2)  
2. Optional `AIModelRegistry` lookup  
3. SigLIP vision + `embed_query_for_vision` for cross-modal search  
4. Process-local model load (avoid reloading per Celery task chunk)

---

## Encoding contract (live)

| Method | Input | Output space |
|--------|-------|--------------|
| `embed_text(str)` | chunk / caption / query | BGE 1024-d, normalized |
| `embed_image(path)` | image file | SigLIP 768-d, L2-normalized |
| `embed_query_for_vision(str)` | natural language | SigLIP text tower → same 768-d space as images |

**Why normalize?** Cosine distance in pgvector (`<=>`) assumes unit-ish geometry; L2 norm keeps scores comparable.

---

## Design rule

Do **not** add a third embedding entrypoint. Extend `EmbeddingService` so Celery workers, API process, and retrievers share one cache key scheme (`emb:{version}:{model}:{md5}`).
