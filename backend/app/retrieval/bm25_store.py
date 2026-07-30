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

import json
import logging
import threading
from pathlib import Path
from typing import List, Dict, Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from backend.app.core.config import BM25_INDEX_PATH

logger = logging.getLogger(__name__)


class BM25Store:
    """
    Thin wrapper around LangChain's BM25Retriever with disk persistence and
    automatic reload/rebuild when new documents are ingested by background workers.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._retriever: BM25Retriever | None = None
        self._docs: List[Document] = []
        self._last_mtime: float | None = None
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            texts = [d.page_content for d in self._docs]
            metas = [d.metadata for d in self._docs]
            with open(BM25_INDEX_PATH, "w", encoding="utf-8") as f:
                json.dump({"texts": texts, "metas": metas}, f, ensure_ascii=False)
            self._last_mtime = Path(BM25_INDEX_PATH).stat().st_mtime
        except Exception as e:
            logger.warning(f"Failed to save BM25 index to {BM25_INDEX_PATH}: {e}")

    def _load(self) -> None:
        path = Path(BM25_INDEX_PATH)
        if not path.exists():
            return
        try:
            mtime = path.stat().st_mtime
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            texts = data.get("texts", [])
            metas = data.get("metas", [{}] * len(texts))
            self._docs = [
                Document(page_content=t, metadata=m)
                for t, m in zip(texts, metas)
            ]
            if self._docs:
                self._retriever = BM25Retriever.from_documents(self._docs)
            self._last_mtime = mtime
        except Exception as e:
            logger.warning(f"Failed to load BM25 index from {BM25_INDEX_PATH}: {e}")
            self._docs, self._retriever = [], None

    def _check_reload(self) -> None:
        """Reload index if file on disk was updated by a background worker."""
        with self._lock:
            path = Path(BM25_INDEX_PATH)
            if path.exists():
                current_mtime = path.stat().st_mtime
                if self._last_mtime is None or current_mtime > self._last_mtime:
                    self._load()
            elif self._retriever is None:
                # If no index file exists on disk, auto-rebuild from Postgres
                try:
                    from backend.app.db.database import SessionLocal
                    db = SessionLocal()
                    try:
                        self.rebuild_from_postgres(db)
                    finally:
                        db.close()
                except Exception as e:
                    logger.debug(f"Auto-rebuild from Postgres skipped: {e}")


    # ── Build / Update ─────────────────────────────────────────────────────────

    def add(self, documents: List[str], metadatas: List[Dict] | None = None) -> None:
        """Incrementally add documents and rebuild the BM25 index."""
        metas = metadatas or [{} for _ in documents]
        new_docs = [
            Document(page_content=t, metadata=m)
            for t, m in zip(documents, metas)
        ]
        with self._lock:
            self._docs.extend(new_docs)
            if self._docs:
                self._retriever = BM25Retriever.from_documents(self._docs)
            self._save()

    def rebuild_from_postgres(self, db: Any) -> None:
        """
        Full rebuild from PostgreSQL snapshot (called on startup or after clear).
        """
        from sqlalchemy import text
        query = text("""
            SELECT ve.content, a.filename, a.modality
            FROM vector_embeddings ve
            LEFT JOIN artifacts a ON ve.artifact_id = a.id
            WHERE ve.embedding_type = 'text'
        """)
        results = db.execute(query).fetchall()
        
        if not results:
            return

        with self._lock:
            self._docs = [
                Document(
                    page_content=row.content,
                    metadata={"source": row.filename or "unknown", "modality": row.modality or "text"}
                )
                for row in results if row.content
            ]
            if self._docs:
                self._retriever = BM25Retriever.from_documents(self._docs)
            self._save()

    def clear(self) -> None:
        with self._lock:
            self._docs, self._retriever = [], None
            path = Path(BM25_INDEX_PATH)
            if path.exists():
                path.unlink()


    # ── Retrieval ──────────────────────────────────────────────────────────────

    @property
    def is_empty(self) -> bool:
        """Whether the BM25 index has any documents."""
        return len(self._docs) == 0

    @property
    def retriever(self) -> BM25Retriever | None:
        """Returns the underlying LangChain BM25Retriever (for EnsembleRetriever)."""
        self._check_reload()
        return self._retriever

    def search(self, query: str, top_k: int = 20) -> List[str]:
        """Direct text search; returns list of document strings."""
        self._check_reload()
        if self._retriever is None:
            return []
        self._retriever.k = top_k
        results = self._retriever.invoke(query)
        return [doc.page_content for doc in results]

