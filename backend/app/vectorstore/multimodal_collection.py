from typing import List, Dict, Any, Optional

from backend.app.embeddings.clip_embedding_model import CLIPEmbeddingModel
from backend.app.vectorstore.chroma_client import get_chroma_client

COLLECTION_NAME = "omnimind_multimodal"


class MultimodalCollection:
    """
    ChromaDB collection for image and video content.
    Embedding: clip-ViT-B-32 → 512-dim (shared text-image space)
    Modalities stored: image (LLaVA captions), video (timestamped segments)

    CLIP's unified embedding space means a text query can retrieve
    semantically similar images/video frames without needing a separate
    image-to-text conversion at query time.
    """

    def __init__(self):
        self.client = get_chroma_client()
        self.collection_name = COLLECTION_NAME
        self.embedding_model = CLIPEmbeddingModel()

    def _get_collection(self):
        return self.client.get_or_create_collection(name=self.collection_name)

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_image(
        self,
        description: str,
        doc_id: str,
        metadata: Dict[str, Any],
    ) -> None:
        """
        Store an image caption as a CLIP text embedding.
        The LLaVA description is encoded in CLIP text space so it can be
        retrieved by CLIP-encoded text queries.
        """
        collection = self._get_collection()
        embedding = self.embedding_model.embed_text(description)
        collection.add(
            documents=[description],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[metadata],
        )

    def add_video_segments(
        self,
        documents: List[str],
        ids: List[str],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """Batch-store video segment captions with temporal metadata."""
        collection = self._get_collection()
        embeddings = self.embedding_model.embed_texts(documents)
        collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata,
        )

    # ── Read ───────────────────────────────────────────────────────────────────

    def query_by_text(self, query: str, n_results: int = 10) -> Dict[str, Any]:
        """Cross-modal retrieval: text query → similar image/video content."""
        collection = self._get_collection()
        query_embedding = self.embedding_model.embed_text(query)
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    def query_time_range(self, start_time: int, end_time: int) -> List[str]:
        """Retrieve video segments overlapping with [start_time, end_time]."""
        collection = self._get_collection()
        results = collection.get(where={"modality": "video"})

        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        return [
            doc
            for doc, meta in zip(documents, metadatas)
            if meta.get("start_time") is not None
            and meta["start_time"] <= end_time
            and meta["end_time"] >= start_time
        ]

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
