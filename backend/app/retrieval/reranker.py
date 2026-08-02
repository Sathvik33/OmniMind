"""
CrossEncoderReranker — intelligent document re-ranking for relevance.

Uses cross-encoder models to jointly score (query, document) pairs,
providing more accurate relevance scores than embedding-based retrieval alone.

Features:
  - GPU-accelerated scoring (CUDA when available)
  - Batch processing for efficiency
  - Confidence scoring + metadata preservation
  - Top-k selection with score thresholding
"""

import torch
from typing import List, Tuple, Dict, Any, Optional
from sentence_transformers import CrossEncoder

from backend.app.core.config import RERANKER_TOP_K

_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


class CrossEncoderReranker:
    """
    Scores (query, document) pairs jointly and returns ranked results
    with confidence scores. Runs on GPU when available.
    """

    def __init__(self, model_name: str = _MODEL_NAME, top_k: int = RERANKER_TOP_K):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)
        self.top_k = top_k
        self.device = device

    def rerank(
        self,
        query: str,
        candidates: List[str],
        threshold: float = 0.0,
        return_scores: bool = True,
    ) -> List[str] | List[Tuple[str, float]]:
        """
        Rerank candidates by relevance to query.

        Args:
            query: User query
            candidates: List of document chunks to rerank
            threshold: Minimum score to include (0.0-1.0, where 0.5 is neutral)
            return_scores: If True, return (document, score) tuples

        Returns:
            Top-k documents (with scores if return_scores=True)
        """
        if not candidates:
            return []

        from backend.app.monitoring.langsmith_logger import tracer

        with tracer.model_call(
            name=f"reranker:{_MODEL_NAME}",
            tags=["reranker", "cross-encoder", "bge-reranker"],
            provider="sentence-transformers",
            model=_MODEL_NAME,
            inputs={
                "query": (query or "")[:1000],
                "candidate_count": len(candidates),
                "threshold": threshold,
            },
            run_type="chain",
        ) as span:
            pairs = [(query, doc) for doc in candidates]
            scores = self._score_pairs(pairs)

            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

            filtered = [(doc, score) for score, doc in ranked if score >= threshold]
            results = filtered[: self.top_k]

            span["outputs"] = {
                "output_count": len(results),
                "top_scores": (
                    [round(float(s), 4) for _, s in results[:8]]
                    if return_scores
                    else [round(float(s), 4) for s, _ in ranked[:8]]
                ),
            }
            span["output"] = (
                f"reranked {len(candidates)} → {len(results)} "
                f"(device={self.device})"
            )

            return results if return_scores else [doc for doc, _ in results]

    def rerank_with_metadata(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates that include metadata (source, modality, timestamp).

        Args:
            query: User query
            candidates: List of dicts with 'text' and optional metadata
            threshold: Minimum relevance score

        Returns:
            Reranked candidates with scores added
        """
        if not candidates:
            return []

        texts = [c.get("text", "") for c in candidates]
        pairs = [(query, text) for text in texts]
        scores = self._score_pairs(pairs)

        # Attach scores to original candidates
        scored = [
            {**c, "relevance_score": float(score)}
            for c, score in zip(candidates, scores)
        ]

        # Sort by score, apply threshold, limit to top_k
        ranked = sorted(scored, key=lambda x: x["relevance_score"], reverse=True)
        return [c for c in ranked if c["relevance_score"] >= threshold][: self.top_k]

    def batch_rerank(
        self,
        query: str,
        candidate_batches: List[List[str]],
        return_scores: bool = True,
    ) -> List[List[str]] | List[List[Tuple[str, float]]]:
        """
        Rerank multiple batches of candidates efficiently.

        Useful for parallel processing or multi-stage retrieval.
        """
        return [
            self.rerank(query, batch, return_scores=return_scores)
            for batch in candidate_batches
        ]

    def _score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score all (query, document) pairs efficiently.

        Returns: List of scores (0.0-1.0, roughly)
        """
        if not pairs:
            return []

        # Get raw scores (typically -10 to +10 range for this model)
        raw_scores = self.model.predict(pairs, show_progress_bar=False)

        # Normalize to 0.0-1.0 range using sigmoid
        # This model's scores centered around 0, so we map:
        # -inf → 0.0, 0 → 0.5, +inf → 1.0
        normalized = 1.0 / (1.0 + torch.exp(-torch.tensor(raw_scores, dtype=torch.float32)))

        return normalized.numpy().tolist()

    def get_device(self) -> str:
        """Return device being used (cuda or cpu)."""
        return self.device

    def get_top_k(self) -> int:
        """Return current top_k setting."""
        return self.top_k

    def set_top_k(self, k: int) -> None:
        """Update top_k threshold."""
        self.top_k = max(1, k)
