# API layer

**Path:** `backend/app/api/`

Thin HTTP adapters. Business logic stays in pipelines / services.

| Router | Endpoints | Notes |
|--------|-----------|-------|
| `upload.py` | `POST /upload`, `GET /jobs/{id}` | MinIO + job row + Celery `.delay` |
| `query.py` | `POST /query`, `POST /query-stream` | Optional `artifact_ids`, `evaluate` |
| `health.py` | `GET /health` | Liveness |
| `monitor.py` | `/monitor/*` | LangSmith-backed ops |
| `evaluate.py` | `/evaluate/*` | RAGAS-style eval |
| `feedback.py` | `POST /feedback/` | Message feedback |

---

## Why thin controllers?

| Concern | Owner |
|---------|-------|
| MIME / size validation | `IngestionGuard` |
| Embed readiness | `query_scope` |
| Token streaming | `QueryPipeline.stream_answer` + `StreamingResponse` |
| Retries | Celery task |

The API should remain swappable (gRPC, queue consumers) without rewriting RAG.

---

## Upload contract

```text
multipart file
  → guard (ext, 100MB, safe name)
  → MinIO put
  → Artifact + IngestionJob(queued)
  → process_ingestion_task.delay(job_id)
  → { artifact_id, job_id, minio_path }
```

**Why not ingest in-request?** Video captioning and embedding can take minutes; HTTP timeouts and worker isolation matter.

---

## Query-stream contract

- Body: `{ query, artifact_ids?, top_k?, evaluate? }` (`top_k`/`evaluate` unused on stream today)  
- Media type: `text/plain; charset=utf-8` (raw tokens, **not** SSE frames)  
- Headers: `Cache-Control: no-cache`, `X-Accel-Buffering: no`  

**Why plain text not SSE?** UI already consumes a byte stream; SSE framing adds complexity without EventSource usage. Roadmap may add true SSE later.

---

## Design rules

1. No embedding or OpenCV inside route handlers.  
2. Always return `job_id` for async work.  
3. Document response shapes in OpenAPI descriptions when you add routes.
