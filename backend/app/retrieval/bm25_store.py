"""
BM25Store — wraps LangChain's BM25Retriever for sparse lexical retrieval.

Uses: langchain_community.retrievers.BM25Retriever
      (which wraps rank_bm25 under the hood)

Why LangChain instead of raw rank_bm25?
- Fits the LangChain Document ecosystem (used by EnsembleRetriever)
- Consistent interface across all retrievers
- Persistence: index rebuilt from PostgreSQL snapshot at startup

RRF fusion is handled by LangChain's EnsembleRetriever (see hybrid_retriever.py).
"""

import pickle
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from backend.app.core.config import BM25_INDEX_PATH


class BM25Store:
    """
    Thin wrapper around LangChain's BM25Retriever with disk persistence.

    Rebuilds from PostgreSQL text collection documents on first startup
    or after a clear-memory operation.
    """

    def __init__(self):
        self._retriever: BM25Retriever | None = None
        self._docs: List[Document] = []
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            texts = [d.page_content for d in self._docs]
            metas = [d.metadata for d in self._docs]
            with open(BM25_INDEX_PATH, "wb") as f:
                pickle.dump({"texts": texts, "metas": metas}, f)
        except Exception:
            pass

    def _load(self) -> None:
        path = Path(BM25_INDEX_PATH)
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            texts = data.get("texts", [])
            metas = data.get("metas", [{}] * len(texts))
            self._docs = [
                Document(page_content=t, metadata=m)
                for t, m in zip(texts, metas)
            ]
            if self._docs:
                self._retriever = BM25Retriever.from_documents(self._docs)
        except Exception:
            self._docs, self._retriever = [], None

    # ── Build / Update ─────────────────────────────────────────────────────────

    def add(self, documents: List[str], metadatas: List[Dict] | None = None) -> None:
        """Incrementally add documents and rebuild the BM25 index."""
        metas = metadatas or [{} for _ in documents]
        new_docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(documents, metas)
        ]
        self._docs.extend(new_docs)
        if self._docs:
            self._retriever = BM25Retriever.from_documents(self._docs)
        self._save()

    def rebuild_from_postgres(self, db: Any) -> None:
        """
        Full rebuild from PostgreSQL snapshot (called on startup or after clear).
        """
        from sqlalchemy import text
        results = db.execute(text("SELECT content FROM vector_embeddings WHERE embedding_type = 'text'")).fetchall()
        texts = [row.content for row in results if row.content]
        
        if not texts:
            return

        self._docs = [
            Document(page_content=t, metadata={})
            for t in texts
        ]
        self._retriever = BM25Retriever.from_documents(self._docs)
        self._save()

    def clear(self) -> None:
        self._docs, self._retriever = [], None
        path = Path(BM25_INDEX_PATH)
        if path.exists():
            path.unlink()

    # ── Retrieval ──────────────────────────────────────────────────────────────

    @property
    def retriever(self) -> BM25Retriever | None:
        """Returns the underlying LangChain BM25Retriever (for EnsembleRetriever)."""
        return self._retriever

    def search(self, query: str, top_k: int = 20) -> List[str]:
        """Direct text search; returns list of document strings."""
        if self._retriever is None:
            return []
        self._retriever.k = top_k
        results = self._retriever.invoke(query)
        return [doc.page_content for doc in results]
