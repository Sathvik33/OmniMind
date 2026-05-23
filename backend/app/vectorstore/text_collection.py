from typing import List, Dict, Any

from backend.app.embeddings.embedding_model import TextEmbeddingModel
from backend.app.vectorstore.chroma_client import get_chroma_client

COLLECTION_NAME = "omnimind_text"


class TextCollection:
    """
    ChromaDB collection for text document chunks.
    Embedding: all-MiniLM-L6-v2 → 384-dim
    Modalities stored: document (PDF / DOCX / PPTX / XLSX / TXT chunks)
    """

    def __init__(self):
        self.client = get_chroma_client()
        self.collection_name = COLLECTION_NAME
        self.embedding_model = TextEmbeddingModel()

    def _get_collection(self):
        return self.client.get_or_create_collection(name=self.collection_name)

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadata: List[Dict[str, Any]],
    ) -> None:
        collection = self._get_collection()
        embeddings = self.embedding_model.embed_documents(documents)
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata,
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    def semantic_query(self, query: str, n_results: int = 20) -> Dict[str, Any]:
        """Dense vector similarity search."""
        collection = self._get_collection()
        query_embedding = self.embedding_model.embed_query(query)
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def get_all_documents(self) -> Dict[str, Any]:
        """Retrieve all stored documents (used to rebuild BM25 index)."""
        collection = self._get_collection()
        return collection.get(include=["documents", "metadatas"])

    def count(self) -> int:
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    # ── Delete ─────────────────────────────────────────────────────────────────

    def delete(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
