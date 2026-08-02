# Docs index

| Doc | Audience | Contents |
|-----|----------|----------|
| [../readme.md](../readme.md) | Everyone | Product overview, architecture, quick start, API |
| [architecture.md](./architecture.md) | Everyone | Combined system design + LLM/monitoring map |
| [../backend/README.md](../backend/README.md) | Backend | Design principles, per-modality chunk/embed, module map |
| [GUIDE.md](./GUIDE.md) | Operators / demos | Boot checklist, curl lab, temporal playground, failure matrix |
| [vector_storage_migration_path.md](./vector_storage_migration_path.md) | Backend / infra | pgvector scaling phases |
| [../frontend/README.md](../frontend/README.md) | Frontend | React workspace contracts |

### Backend module READMEs

| Module | Path |
|--------|------|
| Services (embed / video / ASR / vision) | [`backend/app/services/README.md`](../backend/app/services/README.md) |
| Ingestion & chunking | [`backend/app/ingestion/README.md`](../backend/app/ingestion/README.md) |
| Retrieval | [`backend/app/retrieval/README.md`](../backend/app/retrieval/README.md) |
| Pipelines | [`backend/app/pipelines/README.md`](../backend/app/pipelines/README.md) |
| RAG / workflow / API / … | See table inside [`backend/README.md`](../backend/README.md) |

### Live surfaces

- OpenAPI: `http://127.0.0.1:8000/docs`
- UI: `http://localhost:5173`
- MinIO console: `http://localhost:9001`

### Architecture diagrams

| Image | What it shows |
|-------|----------------|
| [**System architecture (full stack)**](./assets/aegis-system-architecture.png) | Frontend → FastAPI → Celery → stores → retrieval → LLM failover → LangSmith |
| [Ingestion reference](./assets/aegis-architecture-reference.png) | Modality lanes → embedding service → storage |

Narrative: [`architecture.md`](./architecture.md) · ingest lanes: [`../backend/app/ingestion/README.md`](../backend/app/ingestion/README.md)
