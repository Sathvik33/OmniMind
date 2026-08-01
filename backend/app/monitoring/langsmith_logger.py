"""
AegisTracer — production-grade LangSmith monitoring for AEGIS v3.0.

Provides:
  - End-to-end RAG run tracing (query → retrieval → rerank → generate → guard)
  - Per-node latency tracking
  - Retrieval metrics logging (BM25 vs dense hit counts, RRF weights)
  - RAGAS score logging tied to LangSmith runs
  - User feedback submission (thumbs up/down)
  - Run statistics dashboard helpers
  - Graceful no-op fallback when LANGSMITH_API_KEY is missing
"""

from __future__ import annotations

import os
import time
import uuid
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ── Optional LangSmith import (graceful fallback) ─────────────────────────────
try:
    from langsmith import Client as LangSmithClient, traceable
    from langsmith.run_helpers import get_current_run_tree
    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False
    logger.warning("langsmith not installed — monitoring running in no-op mode")


# ── Config ────────────────────────────────────────────────────────────────────
_API_KEY  = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY", "")
_PROJECT  = os.getenv("LANGCHAIN_PROJECT", "Aegis")
_ENDPOINT = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
_TRACING  = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"


def _utc_now() -> datetime:
    """LangSmith expects timezone-aware datetime, not time.time() floats."""
    return datetime.now(timezone.utc)


def _finish_run(client: "LangSmithClient", run_id: str, **kwargs: Any) -> None:
    """Always close a run; log failures loudly so traces don't stay 'running'."""
    payload = {"end_time": _utc_now(), **kwargs}
    try:
        client.update_run(run_id, **payload)
    except Exception as e:
        logger.warning("LangSmith update_run failed for %s: %s", run_id, e)
        # Retry with minimal close so the parent does not hang forever
        try:
            client.update_run(run_id, end_time=_utc_now(), error=str(e))
        except Exception as e2:
            logger.error("LangSmith forced close also failed for %s: %s", run_id, e2)


def _preview(text: str, n: int = 280) -> str:
    t = (text or "").replace("\n", " ").strip()
    return t if len(t) <= n else t[: n - 3] + "..."


# ── Singleton Tracer ──────────────────────────────────────────────────────────

class AegisTracer:
    """
    Central monitoring hub for AEGIS.

    Usage:
        tracer = AegisTracer()

        run_id = tracer.start_run(query="What is AEGIS?")
        tracer.log_retrieval(run_id, bm25_hits=5, dense_hits=8, fused_count=12)
        tracer.log_rerank(run_id, input_count=12, output_count=5, scores=[0.9, 0.7])
        tracer.log_generation(run_id, answer="...", latency_ms=320)
        tracer.log_guardrails(run_id, input_ok=True, output_ok=True, confidence=0.85)
        tracer.end_run(run_id, final_answer="...", total_latency_ms=900)
    """

    _instance: Optional["AegisTracer"] = None

    def __new__(cls) -> "AegisTracer":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._client: Optional[LangSmithClient] = None
        self._active_runs: Dict[str, Dict[str, Any]] = {}

        if _LANGSMITH_AVAILABLE and _API_KEY:
            try:
                self._client = LangSmithClient(
                    api_url=_ENDPOINT,
                    api_key=_API_KEY,
                )
                logger.info(f"✅ AegisTracer connected to LangSmith project='{_PROJECT}'")
            except Exception as e:
                logger.warning(f"⚠️ LangSmith connection failed: {e} — running in no-op mode")
                self._client = None
        else:
            reason = "langsmith not installed" if not _LANGSMITH_AVAILABLE else "no API key"
            logger.info(f"ℹ️ AegisTracer in no-op mode ({reason})")

    # ── Public: Active Status ─────────────────────────────────────────────────

    def is_active(self) -> bool:
        """True if connected to LangSmith."""
        return self._client is not None

    # ── Public: Run Lifecycle ─────────────────────────────────────────────────

    def start_run(self, query: str, metadata: Optional[Dict] = None) -> str:
        """
        Start a new RAG run trace. Returns a run_id to pass through the pipeline.
        """
        run_id = str(uuid.uuid4())
        self._active_runs[run_id] = {
            "run_id": run_id,
            "query": query,
            "start_time": time.perf_counter(),
            "metadata": metadata or {},
            "steps": [],
        }

        if self._client:
            try:
                self._client.create_run(
                    name="aegis_rag_query",
                    run_type="chain",
                    project_name=_PROJECT,
                    id=run_id,
                    inputs={"query": query, **(metadata or {})},
                    tags=["aegis", "rag", "v3"],
                    start_time=_utc_now(),
                )
            except Exception as e:
                logger.warning(f"LangSmith start_run failed: {e}")

        return run_id

    def end_run(
        self,
        run_id: str,
        final_answer: str,
        total_latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """Complete a RAG run trace."""
        run_data = self._active_runs.pop(run_id, {})

        if total_latency_ms is None and run_data.get("start_time"):
            elapsed = time.perf_counter() - run_data["start_time"]
            total_latency_ms = round(elapsed * 1000, 2)

        outputs = {
            "answer": final_answer,
            "total_latency_ms": total_latency_ms,
            "steps_count": len(run_data.get("steps", [])),
        }

        if self._client:
            _finish_run(
                self._client,
                run_id,
                outputs=outputs,
                error=error,
            )

    # ── Public: Ingest lifecycle ──────────────────────────────────────────────

    def start_ingest_run(
        self,
        filename: str,
        artifact_id: int,
        metadata: Optional[Dict] = None,
    ) -> str:
        """Start an ingestion trace (chunking / embedding)."""
        run_id = str(uuid.uuid4())
        self._active_runs[run_id] = {
            "run_id": run_id,
            "filename": filename,
            "artifact_id": artifact_id,
            "start_time": time.perf_counter(),
            "metadata": metadata or {},
            "steps": [],
        }
        if self._client:
            try:
                self._client.create_run(
                    name="aegis_ingest",
                    run_type="chain",
                    project_name=_PROJECT,
                    id=run_id,
                    inputs={
                        "filename": filename,
                        "artifact_id": artifact_id,
                        **(metadata or {}),
                    },
                    tags=["aegis", "ingest", "chunking", "v3"],
                    start_time=_utc_now(),
                )
            except Exception as e:
                logger.warning("LangSmith start_ingest_run failed: %s", e)
        return run_id

    def end_ingest_run(
        self,
        run_id: str,
        chunk_count: int = 0,
        error: Optional[str] = None,
    ) -> None:
        run_data = self._active_runs.pop(run_id, {})
        latency_ms = None
        if run_data.get("start_time"):
            latency_ms = round((time.perf_counter() - run_data["start_time"]) * 1000, 2)
        if self._client:
            _finish_run(
                self._client,
                run_id,
                outputs={
                    "chunk_count": chunk_count,
                    "latency_ms": latency_ms,
                    "steps_count": len(run_data.get("steps", [])),
                },
                error=error,
            )

    def log_chunking(
        self,
        run_id: str,
        filename: str,
        chunk_count: int,
        truncated_to: Optional[int] = None,
        markdown_chars: int = 0,
        sample_hierarchies: Optional[List[str]] = None,
        sample_previews: Optional[List[str]] = None,
        latency_ms: Optional[float] = None,
        chunker: str = "MarkdownHierarchicalChunker",
    ) -> None:
        """Log document chunking under an ingest (or query) parent run."""
        step = {
            "step": "chunking",
            "filename": filename,
            "chunk_count": chunk_count,
            "truncated_to": truncated_to,
            "latency_ms": latency_ms,
        }
        self._append_step(run_id, step)
        self._child_run(
            parent_run_id=run_id,
            name="document_chunking",
            run_type="chain",
            inputs={
                "filename": filename,
                "chunker": chunker,
                "markdown_chars": markdown_chars,
            },
            outputs={
                "chunk_count": chunk_count,
                "truncated_to": truncated_to,
                "sample_hierarchies": (sample_hierarchies or [])[:8],
                "sample_previews": (sample_previews or [])[:5],
                "latency_ms": latency_ms,
            },
            tags=["chunking", "ingest", "markdown"],
        )

    # ── Public: Step Logging ──────────────────────────────────────────────────

    def log_retrieval(
        self,
        run_id: str,
        bm25_hits: int,
        dense_hits: int,
        fused_count: int,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        latency_ms: Optional[float] = None,
        bm25_previews: Optional[List[str]] = None,
        dense_previews: Optional[List[str]] = None,
        fused_previews: Optional[List[str]] = None,
        artifact_ids: Optional[List[int]] = None,
        query: Optional[str] = None,
    ) -> None:
        """Log hybrid retrieval metrics (+ optional chunk previews)."""
        step = {
            "step": "retrieval",
            "bm25_hits": bm25_hits,
            "dense_hits": dense_hits,
            "fused_count": fused_count,
            "bm25_weight": bm25_weight,
            "dense_weight": dense_weight,
            "latency_ms": latency_ms,
        }
        self._append_step(run_id, step)

        self._child_run(
            parent_run_id=run_id,
            name="bm25_retrieve",
            run_type="retriever",
            inputs={"query": query or "", "artifact_ids": artifact_ids or []},
            outputs={
                "hits": bm25_hits,
                "previews": [_preview(p) for p in (bm25_previews or [])[:8]],
            },
            tags=["retrieval", "bm25"],
        )
        self._child_run(
            parent_run_id=run_id,
            name="dense_retrieve",
            run_type="retriever",
            inputs={"query": query or "", "artifact_ids": artifact_ids or []},
            outputs={
                "hits": dense_hits,
                "previews": [_preview(p) for p in (dense_previews or [])[:8]],
            },
            tags=["retrieval", "dense", "pgvector"],
        )
        self._child_run(
            parent_run_id=run_id,
            name="hybrid_retrieval",
            run_type="retriever",
            inputs={
                "query": query or "",
                "bm25_weight": bm25_weight,
                "dense_weight": dense_weight,
                "artifact_ids": artifact_ids or [],
            },
            outputs={
                "bm25_hits": bm25_hits,
                "dense_hits": dense_hits,
                "fused_count": fused_count,
                "fused_previews": [_preview(p) for p in (fused_previews or [])[:8]],
                "latency_ms": latency_ms,
            },
            tags=["retrieval", "hybrid", "rrf"],
        )

    def log_rerank(
        self,
        run_id: str,
        input_count: int,
        output_count: int,
        scores: List[float],
        model: str = "BAAI/bge-reranker-v2-m3",
        latency_ms: Optional[float] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        query: Optional[str] = None,
    ) -> None:
        """Log reranker metrics (+ ranked chunk previews)."""
        step = {
            "step": "rerank",
            "input_count": input_count,
            "output_count": output_count,
            "scores": scores,
            "model": model,
            "latency_ms": latency_ms,
        }
        self._append_step(run_id, step)

        ranked = []
        for i, doc in enumerate(documents or []):
            if isinstance(doc, dict):
                ranked.append(
                    {
                        "rank": i + 1,
                        "score": doc.get("score"),
                        "preview": _preview(str(doc.get("text") or doc.get("preview") or "")),
                    }
                )
            else:
                ranked.append({"rank": i + 1, "preview": _preview(str(doc))})

        self._child_run(
            parent_run_id=run_id,
            name="cross_encoder_rerank",
            run_type="chain",
            inputs={
                "query": query or "",
                "input_count": input_count,
                "model": model,
            },
            outputs={
                "output_count": output_count,
                "score_mean": round(sum(scores) / len(scores), 4) if scores else 0.0,
                "score_min": round(min(scores), 4) if scores else 0.0,
                "score_max": round(max(scores), 4) if scores else 0.0,
                "ranked": ranked[:10],
                "latency_ms": latency_ms,
            },
            tags=["reranking", "cross-encoder"],
        )

    def log_generation(
        self,
        run_id: str,
        answer: str,
        prompt_tokens: Optional[int] = None,
        latency_ms: Optional[float] = None,
        model: str = "qwen2.5:7b",
    ) -> None:
        """Log LLM generation metrics."""
        step = {
            "step": "generation",
            "answer_length": len(answer),
            "prompt_tokens": prompt_tokens,
            "latency_ms": latency_ms,
            "model": model,
        }
        self._append_step(run_id, step)

        if self._client:
            try:
                child_id = str(uuid.uuid4())
                self._client.create_run(
                    name="llm_generation",
                    run_type="llm",
                    project_name=_PROJECT,
                    id=child_id,
                    parent_run_id=run_id,
                    inputs={"model": model, "prompt_tokens": prompt_tokens},
                    outputs={
                        "answer_length": len(answer),
                        "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                        "latency_ms": latency_ms,
                    },
                    tags=["generation", "llm"],
                )
                self._client.update_run(run_id=child_id, end_time=_utc_now())
            except Exception as e:
                logger.debug(f"LangSmith log_generation failed: {e}")

    def log_guardrails(
        self,
        run_id: str,
        input_ok: bool,
        output_ok: bool,
        confidence: float,
        grounded: bool,
        has_hallucination: bool,
        input_warnings: Optional[List[str]] = None,
        output_warnings: Optional[List[str]] = None,
    ) -> None:
        """Log guardrail decisions."""
        step = {
            "step": "guardrails",
            "input_ok": input_ok,
            "output_ok": output_ok,
            "confidence": confidence,
            "grounded": grounded,
            "has_hallucination": has_hallucination,
        }
        self._append_step(run_id, step)

        if self._client:
            try:
                child_id = str(uuid.uuid4())
                self._client.create_run(
                    name="aegis_guardrails",
                    run_type="chain",
                    project_name=_PROJECT,
                    id=child_id,
                    parent_run_id=run_id,
                    inputs={},
                    outputs={
                        "input_ok": input_ok,
                        "output_ok": output_ok,
                        "confidence": round(confidence, 4),
                        "grounded": grounded,
                        "has_hallucination": has_hallucination,
                        "input_warnings": input_warnings or [],
                        "output_warnings": output_warnings or [],
                    },
                    tags=["guardrails", "safety"],
                )
                self._client.update_run(run_id=child_id, end_time=_utc_now())
            except Exception as e:
                logger.debug(f"LangSmith log_guardrails failed: {e}")

    def log_evaluation(
        self,
        run_id: str,
        eval_scores: Dict[str, float],
        composite_score: float,
    ) -> None:
        """Log RAGAS evaluation scores tied to a run."""
        step = {
            "step": "evaluation",
            "eval_scores": eval_scores,
            "composite_score": composite_score,
        }
        self._append_step(run_id, step)

        if self._client:
            try:
                child_id = str(uuid.uuid4())
                self._client.create_run(
                    name="ragas_evaluation",
                    run_type="chain",
                    project_name=_PROJECT,
                    id=child_id,
                    parent_run_id=run_id,
                    inputs={},
                    outputs={
                        **eval_scores,
                        "composite_score": round(composite_score, 4),
                    },
                    tags=["evaluation", "ragas"],
                )
                self._client.update_run(run_id=child_id, end_time=_utc_now())
            except Exception as e:
                logger.debug(f"LangSmith log_evaluation failed: {e}")

    # ── Public: User Feedback ─────────────────────────────────────────────────

    def submit_feedback(
        self,
        run_id: str,
        score: float,
        comment: Optional[str] = None,
        key: str = "user_feedback",
    ) -> bool:
        """
        Submit user feedback (thumbs up/down) for a run.
        score: 1.0 = positive, 0.0 = negative
        Returns True if submitted successfully.
        """
        if not self._client:
            logger.debug("No LangSmith client — feedback not submitted")
            return False

        try:
            self._client.create_feedback(
                run_id=run_id,
                key=key,
                score=score,
                comment=comment or "",
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to submit feedback: {e}")
            return False

    # ── Public: Stats & Dashboard ─────────────────────────────────────────────

    def get_recent_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Fetch recent runs from LangSmith for the dashboard."""
        if not self._client:
            return []

        try:
            runs = list(self._client.list_runs(
                project_name=_PROJECT,
                run_type="chain",
                limit=limit,
                filter='eq(name, "aegis_rag_query")',
            ))
            return [
                {
                    "run_id": str(r.id),
                    "name": r.name,
                    "status": r.status,
                    "start_time": r.start_time.isoformat() if r.start_time else None,
                    "latency_ms": round(r.latency * 1000, 2) if r.latency else None,
                    "query": (r.inputs or {}).get("query", ""),
                    "error": r.error,
                }
                for r in runs
            ]
        except Exception as e:
            logger.warning(f"Failed to fetch runs: {e}")
            return []

    def get_run_stats(self, limit: int = 100) -> Dict[str, Any]:
        """Aggregate statistics from recent runs."""
        if not self._client:
            return {"active": False, "project": _PROJECT}

        runs = self.get_recent_runs(limit=limit)
        if not runs:
            return {"active": True, "project": _PROJECT, "total_runs": 0}

        latencies = [r["latency_ms"] for r in runs if r.get("latency_ms")]
        errors    = [r for r in runs if r.get("error")]

        return {
            "active": True,
            "project": _PROJECT,
            "total_runs": len(runs),
            "error_count": len(errors),
            "error_rate": round(len(errors) / len(runs), 3) if runs else 0.0,
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else None,
            "min_latency_ms": round(min(latencies), 2) if latencies else None,
            "max_latency_ms": round(max(latencies), 2) if latencies else None,
            "langsmith_url": f"https://smith.langchain.com/o/public/projects/p/{_PROJECT}",
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _child_run(
        self,
        parent_run_id: str,
        name: str,
        run_type: str,
        inputs: Dict[str, Any],
        outputs: Dict[str, Any],
        tags: Optional[List[str]] = None,
    ) -> None:
        if not self._client or not parent_run_id:
            return
        try:
            child_id = str(uuid.uuid4())
            self._client.create_run(
                name=name,
                run_type=run_type,
                project_name=_PROJECT,
                id=child_id,
                parent_run_id=parent_run_id,
                inputs=inputs,
                outputs=outputs,
                tags=tags or [],
                start_time=_utc_now(),
            )
            self._client.update_run(run_id=child_id, end_time=_utc_now())
        except Exception as e:
            logger.debug("LangSmith child run %s failed: %s", name, e)

    def _append_step(self, run_id: str, step: Dict[str, Any]) -> None:
        if run_id in self._active_runs:
            self._active_runs[run_id]["steps"].append(step)


# ── Module-level singleton ────────────────────────────────────────────────────
tracer = AegisTracer()
