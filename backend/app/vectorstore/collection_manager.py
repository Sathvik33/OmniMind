"""
CollectionManager — facade over TextCollection + MultimodalCollection.

Routes documents to the correct collection based on modality metadata:
  - modality == "image" | "video"  → MultimodalCollection (CLIP 512-dim)
  - everything else (documents)    → TextCollection (all-MiniLM 384-dim)

All external code that used to import CollectionManager continues to work;
only the routing internals have changed.
"""

from typing import List, Dict, Any

from backend.app.vectorstore.text_collection import TextCollection
from backend.app.vectorstore.multimodal_collection import MultimodalCollection


class CollectionManager:
    """Unified interface over both vector store collections."""

    def __init__(self):
        self.text = TextCollection()
        self.multimodal = MultimodalCollection()

        # Expose names for backwards compat (e.g. clear-memory endpoint)
        self.collection_name = "omnimind_text"

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[str],
        ids: List[str],
        metadata: List[Dict[str, Any]],
    ) -> None:
        """
        Route documents to the correct collection.
        Modality is read from metadata[0] (all items in a batch share modality).
        """
        if not documents:
            return

        modality = (metadata[0] if metadata else {}).get("modality", "document")

        if modality in ("image", "video"):
            if modality == "image":
                # Single image at a time
                self.multimodal.add_image(documents[0], ids[0], metadata[0])
            else:
                # Video segments batch
                self.multimodal.add_video_segments(documents, ids, metadata)
        else:
            self.text.add_documents(documents, ids, metadata)

    # ── Read ───────────────────────────────────────────────────────────────────

    def query(self, query_text: str, n_results: int = 20) -> Dict[str, Any]:
        """Dense semantic search over text collection (primary retrieval)."""
        return self.text.semantic_query(query_text, n_results)

    def query_time_range(self, start_time: int, end_time: int) -> List[str]:
        """Temporal video segment retrieval from multimodal collection."""
        return self.multimodal.query_time_range(start_time, end_time)

    def get_all_text_documents(self) -> Dict[str, Any]:
        """Return all text docs for BM25 index reconstruction."""
        return self.text.get_all_documents()

    # ── Delete ─────────────────────────────────────────────────────────────────

    def clear_all(self) -> None:
        """Wipe both collections."""
        self.text.delete()
        self.multimodal.delete()

    # Legacy helper kept for the /clear-memory endpoint
    @property
    def client(self):
        return self.text.client