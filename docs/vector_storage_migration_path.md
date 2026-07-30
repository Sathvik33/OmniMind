# 🗺️ Vector Storage Scaling & Migration Blueprint

This document outlines the architectural scaling signals and 3-phase migration path for AEGIS vector storage.

---

## 📊 1. Contention Signals & Thresholds

Instrumentation in `PgVectorStore` (`read_latency_ms`) and `IngestionPipeline` (`oltp_write_latency_ms`) provides metrics to detect database contention before user impact occurs.

| Metric | Target Baseline | Alert Threshold | Action Required |
| :--- | :--- | :--- | :--- |
| **`pgvector_read_latency`** | < 25 ms | > 150 ms (P95) | Add HNSW index or provision Read Replica |
| **`oltp_write_latency`** | < 10 ms | > 100 ms | Separate vector writes from OLTP transaction log |
| **Vector DB Size** | < 500,000 vectors | > 1,000,000 vectors | Prepare Phase 3 dedicated cluster migration |
| **Lock Contention** | 0 deadlocks/min | > 5 lock waits/min | Split ingestion writes to isolated slave/instance |

---

## 🚀 2. Three-Phase Scaling Path

### **Phase 1: Current Architecture (PostgreSQL 15 + pgvector on Single Instance)**
- **Setup**: Single PostgreSQL container handling both relational tables (`artifacts`, `ingestion_jobs`) and vector rows (`vector_embeddings`).
- **Index**: IVFFlat or HNSW (`CREATE INDEX ON vector_embeddings USING hnsw (embedding vector_cosine_ops)`).
- **Capacity**: Suitable up to **500,000 embeddings** and **50 concurrent QPS**.

---

### **Phase 2: PostgreSQL Read Replica Isolation**
- **Trigger**: `pgvector_read_latency` exceeds 100 ms or high concurrent search traffic degrades ingestion database transactions.
- **Architecture**:
  - **Primary Node**: Handles OLTP transactions (`Artifact`, `IngestionJob`, `ChatHistory`) and vector write insertions.
  - **Read Replica Node**: Asynchronously replicates data; dedicated solely to executing `<=>` vector distance searches.
- **Capacity**: Suitable up to **2,000,000 embeddings** and **300 concurrent QPS**.

---

### **Phase 3: Dedicated Cloud Vector Database (Qdrant / Weaviate / Milvus)**
- **Trigger**: Vector count exceeds **2,000,000** rows or sub-10ms ANN latency is required under heavy parallel ingestion.
- **Migration Blueprint**:
  1. Implement a unified `BaseVectorStore` interface with methods: `add()`, `search()`, `delete()`.
  2. Deploy Qdrant or Weaviate cluster alongside PostgreSQL.
  3. Relational data (`artifacts`, `ingestion_jobs`, `metadata`) remains in PostgreSQL.
  4. Vector payload & embedding vectors move to Qdrant collection.
  5. Backfill script migrates existing `vector_embeddings` rows into Qdrant using `artifact_id` as the payload point reference.
