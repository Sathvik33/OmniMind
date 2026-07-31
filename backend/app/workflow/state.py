"""
OmniMindState — shared state schema for the AEGIS v3.0 LangGraph RAG workflow.

Every node receives the full state dict and returns a partial update.
LangGraph merges updates automatically.
"""

from typing import Any, Dict, List, Optional, TypedDict


class OmniMindState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────────
    query: str                       # raw user query
    artifact_ids: List[int]          # only answer from these uploaded artifacts

    # ── Tracing ────────────────────────────────────────────────────────────────
    run_id: str                      # LangSmith run ID for end-to-end tracing

    # ── Guard ──────────────────────────────────────────────────────────────────
    guard_result: Dict[str, Any]     # {\"ok\": bool, \"reason\": str}

    # ── Classification ─────────────────────────────────────────────────────────
    query_type: str                  # \"semantic\" | \"temporal\"
    time_range: Optional[Dict[str, int]]  # {\"start\": int, \"end\": int} | None

    # ── Retrieval ──────────────────────────────────────────────────────────────
    candidates: List[str]            # raw retrieved chunks (pre-rerank)
    reranked: List[str]              # reranked final chunks
    retrieval_metadata: Dict[str, Any]  # bm25_hits, dense_hits, fused_count, latency_ms

    # ── Generation ─────────────────────────────────────────────────────────────
    context: str                     # assembled context string
    answer: str                      # raw LLM answer
    structured_answer: Dict[str, Any]  # Pydantic StructuredAnswer dump

    # ── Post-guard ─────────────────────────────────────────────────────────────
    final_answer: str                # PII-masked, grounding-checked answer
    warnings: List[str]              # non-blocking output guard warnings
    confidence: float                # 0.0-1.0 answer confidence
    grounded: bool                   # whether answer is grounded in context
    has_hallucination: bool          # whether hallucination markers detected

    # ── Evaluation ─────────────────────────────────────────────────────────────
    eval_scores: Dict[str, Any]      # RAGAS scores (populated when evaluate=True)

    # ── Latency Tracking ───────────────────────────────────────────────────────
    latency_ms: Dict[str, float]     # per-node latency: {node_name: ms}

    # ── Error propagation ──────────────────────────────────────────────────────
    error: Optional[str]             # set on any unrecoverable failure
