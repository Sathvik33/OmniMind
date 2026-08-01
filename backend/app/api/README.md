# API layer

**Path:** `backend/app/api/`

Thin HTTP adapters. Business logic stays in pipelines / services.

| Router | Endpoints | Notes |
|--------|-----------|-------|
| `auth.py` | `POST /auth/signup`, `/login`, `GET /auth/me` | Email + password, JWT |
| `chats.py` | `GET/POST /chats`, `GET/PATCH/DELETE /chats/{id}` | Sidebar history; `DELETE` purges messages, embeddings, jobs, **and MinIO blobs**, then rebuilds BM25 |
| `upload.py` | `POST /upload`, `GET /jobs/{id}` | Auth + `session_id`; ownership checks |
| `query.py` | `POST /query`, `POST /query-stream` | Auth + `session_id`; server-side artifact scope |
| `health.py` | `GET /health` | Liveness |
| `monitor.py` | `/monitor/*` | LangSmith-backed ops |
| `evaluate.py` | `/evaluate/*` | RAGAS-style eval |
| `feedback.py` | `POST /feedback/` | Message feedback |
| `deps.py` | `get_current_user` | Bearer JWT dependency |

---

## Why thin controllers?

| Concern | Owner |
|---------|-------|
| MIME / size validation | `IngestionGuard` |
| Embed readiness | `query_scope` |
| Token streaming | `QueryPipeline.stream_answer` + `StreamingResponse` |
| Retries | Celery task |
| Chat ownership | `get_owned_session` / `get_current_user` |

---

## Upload contract

```text
Bearer JWT + multipart file + session_id
  → guard (ext, 100MB, safe name)
  → ownership check on session
  → MinIO put
  → Artifact(user_id, session_id) + IngestionJob
  → system ChatHistory document card
  → process_ingestion_task.delay(job_id)
```

---

## Query-stream contract

```json
{ "query": "…", "session_id": 12 }
```

Server resolves `artifact_ids` for that chat only (never client-supplied foreign IDs). Persists user + assistant `ChatHistory` rows.

---

## Design rules

1. No embedding or OpenCV inside route handlers.  
2. Always return `job_id` for async work.  
3. Never fall back to global “latest artifact” on authenticated chat queries.
