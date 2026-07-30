"""
HybridRetriever — unified multi-strategy retrieval with fusion and reranking.

Combines:
  1. BM25 (sparse, keyword-based) - fast, good for exact matches
  2. Dense semantic (embeddings) - captures semantic similarity
  3. Multimodal (CLIP) - cross-modal retrieval (images↔text, video↔text)
  4. Cross-encoder reranking - joint (query, doc) scoring
  5. Reciprocal Rank Fusion (RRF) - intelligent score normalization

No LangChain dependencies - pure Python + sentence-transformers.
"""

import logging
import re
from typing import List, Dict, Any

from backend.app.retrieval.pgvector_store import PgVectorStore
from backend.app.retrieval.bm25_store import BM25Store
from backend.app.retrieval.reranker import CrossEncoderReranker
from backend.app.core.config import RETRIEVAL_TOP_K, RERANKER_TOP_K, RRF_K

logger = logging.getLogger(__name__)

_META_PREFIX = re.compile(
    r"^(?:\[(?:Document|Image Document|Image File|Hierarchy|Contains Table|Contains Image):[^\]]*\]\s*)+",
    re.MULTILINE,
)


def _body_text(doc: str) -> str:
    """Strip retrieval metadata headers to judge real content length."""
    text = _META_PREFIX.sub("", doc).strip()
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", text).strip()


def _filter_weak_docs(docs: List[str], min_chars: int = 100) -> List[str]:
    """Drop tiny date/header-only chunks that poison dense retrieval."""
    kept = [d for d in docs if len(_body_text(d)) >= min_chars]
    return kept if kept else docs


# ── RRF (Reciprocal Rank Fusion) Implementation ────────────────────────────────

def _rrf_score(rank: int, k: int = RRF_K) -> float:
    """
    Calculate RRF score for a given rank position.
    Formula: 1 / (k + rank)

    Args:
        rank: 1-indexed position (1 = first place)
        k: RRF constant (default 60)

    Returns:
        Score in range (0, 1/(k+1)]
    """
    return 1.0 / (k + rank)


def _fuse_results_rrf(
    bm25_results: List[str],
    dense_results: List[str],
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6,
) -> List[tuple[str, float]]:
    """
    Fuse BM25 and dense results using Reciprocal Rank Fusion.

    Returns: [(doc, fused_score), ...] sorted by score descending
    """
    # Create rank dictionaries
    bm25_scores = {doc: _rrf_score(i + 1) * bm25_weight for i, doc in enumerate(bm25_results)}
    dense_scores = {doc: _rrf_score(i + 1) * dense_weight for i, doc in enumerate(dense_results)}

    # Merge with additive fusion
    fused = {}
    for doc, score in bm25_scores.items():
        fused[doc] = fused.get(doc, 0) + score

    for doc, score in dense_scores.items():
        fused[doc] = fused.get(doc, 0) + score

    # Sort by fused score descending
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return ranked


# ── HybridRetriever ────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Multi-strategy retrieval combining BM25, semantic, multimodal, and reranking.

    Pure Python implementation with no LangChain dependencies.
    Uses sentence-transformers cross-encoder for final reranking.
    """

    def __init__(
        self,
        bm25_store: BM25Store,
        reranker: CrossEncoderReranker,
    ):
        self.pg_store = PgVectorStore()
        self.reranker = reranker
        self.bm25_store = bm25_store

    def retrieve(
        self,
        query: str,
        top_k: int = RETRIEVAL_TOP_K,
        final_k: int = RERANKER_TOP_K,
        include_scores: bool = True,
        adaptive_weights: bool = True,
    ) -> List[str] | List[Dict[str, Any]]:
        """
        Full hybrid retrieval pipeline with fusion and reranking.

        Pipeline:
          1. BM25 search (keyword-based)
          2. Dense semantic search (embedding-based)
          3. RRF fusion (combine rank positions)
          4. Cross-encoder reranking (joint scoring)
          5. Multimodal fusion (image/video cross-modal)
          6. Final deduplication

        Args:
            query: Search query
            top_k: Candidate pool before reranking
            final_k: Final results returned
            include_scores: Return with relevance scores
            adaptive_weights: Adjust BM25/dense weights by query type

        Returns:
            List of document texts (or dicts with metadata if include_scores=True)
        """
        # Get adaptive weights
        bm25_weight, dense_weight = (
            self._get_adaptive_weights(query) if adaptive_weights else (0.4, 0.6)
        )

        # 1. BM25 Search (sparse, keyword-based)
        bm25_results = []
        if self.bm25_store.retriever is not None:
            try:
                bm25_results = _filter_weak_docs(self.bm25_store.search(query, top_k=top_k))
            except Exception as e:
                logger.warning(f"BM25 search failed (falling back to dense only): {e}")
                bm25_results = []

        # 2. Dense Semantic Search (embedding-based via pgvector)
        dense_results = []
        try:
            # Fetch extra candidates so filtering weak micro-chunks still leaves enough
            pg_res = self.pg_store.search(query, top_k=max(top_k * 2, top_k + 10))
            dense_results = _filter_weak_docs([r["content"] for r in pg_res])[:top_k]
        except Exception as e:
            logger.warning(f"Dense semantic search failed (falling back to BM25 only): {e}")

        # 3. RRF Fusion (combine rank positions)
        if bm25_results and dense_results:
            fused = _fuse_results_rrf(bm25_results, dense_results, bm25_weight, dense_weight)
            candidate_texts = [doc for doc, _ in fused]
        elif dense_results:
            candidate_texts = dense_results
        elif bm25_results:
            candidate_texts = bm25_results
        else:
            candidate_texts = []

        # 4. Cross-encoder reranking (with scores)
        reranked_with_scores = (
            self.reranker.rerank(query, candidate_texts, return_scores=True)
            if candidate_texts
            else []
        )

        # 5. CLIP multimodal cross-modal retrieval via pgvector
        modal_results = []
        try:
            modal_res = self.pg_store.search(query, modality="image", embedding_type="vision", top_k=max(2, final_k // 2))
            modal_results = [(r["content"], r.get("score", 0.5)) for r in modal_res if r.get("content")]
        except Exception as e:
            logger.warning(f"Multimodal CLIP search failed: {e}")

        # 6. Merge results with deduplication
        seen_texts = {doc for doc, _ in reranked_with_scores}
        merged_results = list(reranked_with_scores)

        for modal_text, modal_score in modal_results:
            if modal_text not in seen_texts and len(merged_results) < final_k + len(modal_results):
                merged_results.append((modal_text, float(modal_score)))
                seen_texts.add(modal_text)

        # Limit to final_k
        final_results = merged_results[:final_k]

        if include_scores:
            return [
                {
                    "text": doc,
                    "relevance_score": float(score),
                    "source": "multimodal" if any(doc == mt for mt, _ in modal_results) else "text",
                }
                for doc, score in final_results
            ]
        else:
            return [doc for doc, _ in final_results]

    def retrieve_with_confidence(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve documents with detailed confidence information.

        Returns documents with:
          - text: the document content
          - relevance_score: 0.0-1.0 relevance score
          - source: 'text' or 'multimodal'
          - explanation: why this result was chosen
        """
        results = self.retrieve(query, include_scores=True)

        for result in results:
            score = result["relevance_score"]
            if score >= 0.8:
                result["explanation"] = "Highly relevant"
            elif score >= 0.6:
                result["explanation"] = "Moderately relevant"
            elif score >= 0.4:
                result["explanation"] = "Potentially relevant"
            else:
                result["explanation"] = "Low confidence match"

        return results

    def _get_adaptive_weights(self, query: str) -> tuple[float, float]:
        """
        Adjust BM25/dense weights based on query characteristics.

        Heuristics:
          - Exact phrases (quoted) → increase BM25
          - Complex questions → increase semantic
          - Temporal queries → increase dense
          - Short queries → increase BM25
        """
        # Quoted text suggests exact phrase matching
        if '"' in query:
            return 0.55, 0.45

        # Question words suggest semantic understanding needed
        if any(qword in query.lower() for qword in ["why", "how", "what does", "explain"]):
            return 0.35, 0.65

        # Temporal expressions suggest time-aware retrieval
        if any(time_word in query.lower() for time_word in ["time", "when", "date", "period"]):
            return 0.3, 0.7

        # Short queries often benefit from keyword matching
        if len(query.split()) <= 3:
            return 0.5, 0.5

        # Default adaptive
        return 0.4, 0.6

    def get_stats(self) -> Dict[str, Any]:
        """Return retriever statistics."""
        return {
            "bm25_active": self.bm25_store.retriever is not None,
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "reranker_device": self.reranker.get_device(),
            "top_k_candidates": RETRIEVAL_TOP_K,
            "final_k_results": RERANKER_TOP_K,
            "rrf_constant": RRF_K,
        }
