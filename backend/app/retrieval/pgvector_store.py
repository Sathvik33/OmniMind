from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.app.db.models import VectorEmbedding, Artifact, Metadata
from backend.app.services.embedding_service import embedding_service

class PgVectorStore:
    def __init__(self, db: Session):
        self.db = db

    def search(
        self, 
        query: str, 
        modality: str = "document", 
        embedding_type: str = "text", 
        metadata_filters: Optional[Dict[str, Any]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Executes a dense vector search with SQL metadata pre-filtering.
        """
        # 1. Embed Query
        if embedding_type == "text":
            query_embedding = embedding_service.embed_text(query)
        else:
            query_embedding = embedding_service.embed_image(query) # Path for image-to-image or text-to-image

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
              AND a.modality = :modality
        """
        
        params = {
            "query_embedding": str(query_embedding),
            "embedding_type": embedding_type,
            "modality": modality
        }

        # 3. Add Metadata Pre-Filtering
        # This is basic; in a real scenario we might join the artifact_metadata table
        if metadata_filters:
            # Example logic for simple exact match filtering if it was flat
            # To query the JSON metadata table properly requires JSONB querying 
            pass

        sql += " ORDER BY ve.embedding <=> :query_embedding LIMIT :top_k"
        params["top_k"] = top_k

        # 4. Execute Query
        results = self.db.execute(text(sql), params).fetchall()

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
