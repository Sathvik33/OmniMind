from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import bindparam, text
from backend.app.db.models import VectorEmbedding, Artifact, Metadata
from backend.app.services.embedding_service import embedding_service

from backend.app.db.database import SessionLocal

class PgVectorStore:
    def __init__(self, db: Optional[Session] = None):
        self.db = db

    def search(
        self, 
        query: str, 
        modality: Optional[str] = "document", 
        embedding_type: str = "text", 
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10,
        artifact_ids: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes a dense vector search with SQL metadata pre-filtering.
        When artifact_ids is set, only those uploads are searchable.
        Pass modality=None to search all modalities for the scoped artifacts.
        """
        if embedding_type == "text":
            query_embedding = embedding_service.embed_text(query)
        elif embedding_type == "vision":
            query_embedding = embedding_service.embed_query_for_vision(query)
        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

        # 2. Build Base Query String
        # We use pgvector's cosine distance operator <=> 
        sql = """
            SELECT 
                ve.id,
                ve.artifact_id,
                ve.content,
                1 - (ve.embedding <=> :query_embedding) AS similarity
            FROM vector_embeddings ve
            JOIN artifacts a ON ve.artifact_id = a.id
            WHERE ve.embedding_type = :embedding_type
        """
        
        params: Dict[str, Any] = {
            "query_embedding": str(query_embedding),
            "embedding_type": embedding_type,
        }
        bindparams = []

        if modality:
            sql += " AND a.modality = :modality"
            params["modality"] = modality

        if artifact_ids:
            sql += " AND ve.artifact_id IN :artifact_ids"
            params["artifact_ids"] = list(artifact_ids)
            bindparams.append(bindparam("artifact_ids", expanding=True))

        # 3. Add Metadata Pre-Filtering
        # This is basic; in a real scenario we might join the artifact_metadata table
        if metadata_filters:
            # Example logic for simple exact match filtering if it was flat
            # To query the JSON metadata table properly requires JSONB querying 
            pass

        sql += " ORDER BY ve.embedding <=> :query_embedding LIMIT :top_k"
        params["top_k"] = top_k

        # 4. Execute Query & Instrument Read Latency
        session = self.db or SessionLocal()
        import time, logging
        logger = logging.getLogger(__name__)
        t0 = time.perf_counter()
        try:
            stmt = text(sql)
            if bindparams:
                stmt = stmt.bindparams(*bindparams)
            results = session.execute(stmt, params).fetchall()
            read_latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            logger.info(f"📊 pgvector read latency: {read_latency_ms} ms (modality={modality}, top_k={top_k})")
        finally:
            if not self.db:
                session.close()


        # 5. Format Results
        formatted = []
        for row in results:
            formatted.append({
                "id": row.id,
                "artifact_id": row.artifact_id,
                "content": row.content,
                "score": row.similarity
            })
            
        return formatted

    def query_time_range(
        self,
        start_time: int,
        end_time: int,
        top_k: int = 10,
        artifact_ids: Optional[List[int]] = None,
    ) -> List[str]:
        """
        Retrieves video segments matching a temporal range from PostgreSQL metadata.
        Prefers content stored on temporal metadata rows (one row per segment).
        """
        sql = """
            SELECT COALESCE(am.value->>'content', '') AS content
            FROM artifact_metadata am
            WHERE am.key = 'temporal'
              AND CAST(am.value->>'start_time' AS INTEGER) <= :end_time
              AND CAST(am.value->>'end_time' AS INTEGER) >= :start_time
              AND COALESCE(am.value->>'content', '') <> ''
        """
        params: Dict[str, Any] = {
            "start_time": start_time,
            "end_time": end_time,
            "top_k": top_k,
        }
        bindparams = []
        if artifact_ids:
            sql += " AND am.artifact_id IN :artifact_ids"
            params["artifact_ids"] = list(artifact_ids)
            bindparams.append(bindparam("artifact_ids", expanding=True))
        sql += """
            ORDER BY CAST(am.value->>'start_time' AS INTEGER)
            LIMIT :top_k
        """
        session = self.db or SessionLocal()
        try:
            stmt = text(sql)
            if bindparams:
                stmt = stmt.bindparams(*bindparams)
            results = session.execute(stmt, params).fetchall()
            return [row.content for row in results if row.content]
        except Exception:
            return []
        finally:
            if not self.db:
                session.close()
