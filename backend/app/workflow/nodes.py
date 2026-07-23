"""
LangGraph node functions for the AEGIS v3.0 RAG workflow.

Each function:
  - Accepts  OmniMindState  (full state dict)
  - Returns  dict           (partial state update — only changed keys)

Dependencies (hybrid_retriever, reranker, etc.) are injected at graph
compile time via a shared AppServices container so nodes stay pure.

v3.0 additions:
  - Per-node latency tracking
  - LangSmith tracing via AegisTracer
  - retrieval_metadata propagation
  - Confidence + grounding in state
"""

import re
import time
import uuid
from typing import Any, Dict, List, Optional

from backend.app.workflow.state import OmniMindState
from backend.app.guardrails.input_guard import InputGuard
from backend.app.guardrails.output_guard import OutputGuard
from backend.app.utils.time_utils import timestamp_to_seconds


# ── Time detection helpers (shared with classify + temporal nodes) ─────────────

_HHMMSS_RE    = re.compile(r"\b(\d{2}:\d{2}:\d{2})\b")
_SECONDS_RE   = re.compile(r"\b(\d+)\s*seconds?\b", re.IGNORECASE)
_FIRST_RE     = re.compile(r"first\s+(\d+)\s*seconds?", re.IGNORECASE)
_BETWEEN_RE   = re.compile(r"between\s+(\d+)\s+and\s+(\d+)\s*seconds?", re.IGNORECASE)
_LAST_RE      = re.compile(r"last\s+(\d+)\s*seconds?", re.IGNORECASE)


def _detect_time(query: str) -> Optional[Dict[str, int]]:
    """
    Parse natural-language time references in the query.
    Returns {\"start\": int, \"end\": int} or None.
    """
    q = query.lower()

    m = _FIRST_RE.search(q)
    if m:
        return {"start": 0, "end": int(m.group(1))}

    m = _BETWEEN_RE.search(q)
    if m:
        return {"start": int(m.group(1)), "end": int(m.group(2))}

    # HH:MM:SS timestamps
    stamps = _HHMMSS_RE.findall(query)
    if stamps:
        times = [timestamp_to_seconds(t) for t in stamps]
        return {"start": min(times), "end": max(times)}

    # Plain "N seconds"
    secs = _SECONDS_RE.findall(q)
    if secs:
        times = [int(s) for s in secs]
        return {"start": min(times), "end": max(times)}

    return None


def _ms(start: float) -> float:
    """Milliseconds since start time."""
    return round((time.perf_counter() - start) * 1000, 2)


def _get_tracer():
    """Lazy import to avoid circular deps."""
    try:
        from backend.app.monitoring.langsmith_logger import tracer
        return tracer
    except Exception:
        return None


# ── Node: guard_input ──────────────────────────────────────────────────────────

def guard_input(state: OmniMindState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    query = state.get("query", "")

    # Generate a run_id if not already set
    run_id = state.get("run_id") or str(uuid.uuid4())

    # Start LangSmith run
    tracer = _get_tracer()
    if tracer:
        tracer.start_run(query=query, metadata={"run_id": run_id})

    result = InputGuard.validate(query)

    latency = state.get("latency_ms", {})
    latency["guard_input"] = _ms(t0)

    return {
        "run_id": run_id,
        "guard_result": result,
        "latency_ms": latency,
    }


# ── Node: classify_query ───────────────────────────────────────────────────────

def classify_query(state: OmniMindState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    time_range = _detect_time(state["query"])

    latency = state.get("latency_ms", {})
    latency["classify_query"] = _ms(t0)

    return {
        "query_type": "temporal" if time_range else "semantic",
        "time_range": time_range,
        "latency_ms": latency,
    }


# ── Node: temporal_retrieve ────────────────────────────────────────────────────

def make_temporal_retrieve(collection_manager):
    """Factory: binds collection_manager into the node closure."""

    def temporal_retrieve(state: OmniMindState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        tr = state.get("time_range") or {"start": 0, "end": 0}
        segments = []
        if collection_manager:
            try:
                segments = collection_manager.query_time_range(tr["start"], tr["end"])
            except Exception:
                pass

        latency = state.get("latency_ms", {})
        latency["temporal_retrieve"] = _ms(t0)

        return {
            "candidates": segments,
            "reranked": segments,   # temporal results skip reranking
            "context": "\n\n".join(segments) if segments else "",
            "retrieval_metadata": {
                "type": "temporal",
                "time_range": tr,
                "results_count": len(segments),
                "latency_ms": latency["temporal_retrieve"],
            },
            "latency_ms": latency,
        }

    return temporal_retrieve


# ── Node: hybrid_retrieve ──────────────────────────────────────────────────────

def make_hybrid_retrieve(hybrid_retriever):
    """Factory: binds hybrid_retriever into the node closure."""

    def hybrid_retrieve(state: OmniMindState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        results = hybrid_retriever.retrieve(state["query"], include_scores=True)

        # Separate text and metadata
        if results and isinstance(results[0], dict):
            candidates = [r["text"] for r in results]
            bm25_count  = sum(1 for r in results if r.get("source") == "text")
            dense_count = sum(1 for r in results if r.get("source") != "multimodal")
        else:
            candidates = results
            bm25_count = dense_count = len(candidates) // 2

        latency = state.get("latency_ms", {})
        latency["hybrid_retrieve"] = _ms(t0)

        retrieval_meta = {
            "type": "hybrid",
            "bm25_hits": bm25_count,
            "dense_hits": dense_count,
            "fused_count": len(candidates),
            "latency_ms": latency["hybrid_retrieve"],
        }

        # Log to LangSmith
        tracer = _get_tracer()
        if tracer and state.get("run_id"):
            tracer.log_retrieval(
                run_id=state["run_id"],
                bm25_hits=bm25_count,
                dense_hits=dense_count,
                fused_count=len(candidates),
                latency_ms=latency["hybrid_retrieve"],
            )

        return {
            "candidates": candidates,
            "retrieval_metadata": retrieval_meta,
            "latency_ms": latency,
        }

    return hybrid_retrieve


# ── Node: rerank ───────────────────────────────────────────────────────────────

def make_rerank(reranker):
    """Factory: binds reranker into the node closure."""

    def rerank(state: OmniMindState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        candidates = state.get("candidates", [])

        if not candidates:
            return {"reranked": [], "latency_ms": state.get("latency_ms", {})}

        reranked_with_scores = reranker.rerank(state["query"], candidates, return_scores=True)
        reranked = [doc for doc, _ in reranked_with_scores]
        scores   = [float(score) for _, score in reranked_with_scores]

        latency = state.get("latency_ms", {})
        latency["rerank"] = _ms(t0)

        # Log to LangSmith
        tracer = _get_tracer()
        if tracer and state.get("run_id"):
            tracer.log_rerank(
                run_id=state["run_id"],
                input_count=len(candidates),
                output_count=len(reranked),
                scores=scores,
                latency_ms=latency["rerank"],
            )

        return {"reranked": reranked, "latency_ms": latency}

    return rerank


# ── Node: build_context ────────────────────────────────────────────────────────

def build_context(state: OmniMindState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    chunks = state.get("reranked") or state.get("candidates", [])
    if not chunks:
        return {"context": ""}
    context = "\n\n".join(chunks)

    latency = state.get("latency_ms", {})
    latency["build_context"] = _ms(t0)

    return {"context": context, "latency_ms": latency}


# ── Node: generate ─────────────────────────────────────────────────────────────

def make_generate(generator):
    """Factory: binds generator into the node closure."""

    def generate(state: OmniMindState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        context = state.get("context", "")
        query   = state.get("query", "")

        if not context:
            return {"answer": "No relevant context found in your uploaded data."}

        answer = generator.generate(query, context)

        latency = state.get("latency_ms", {})
        latency["generate"] = _ms(t0)

        # Log to LangSmith
        tracer = _get_tracer()
        if tracer and state.get("run_id"):
            tracer.log_generation(
                run_id=state["run_id"],
                answer=answer,
                latency_ms=latency["generate"],
            )

        return {"answer": answer, "latency_ms": latency}

    return generate


# ── Node: guard_output ─────────────────────────────────────────────────────────

def guard_output(state: OmniMindState) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = OutputGuard.validate(
        answer=state.get("answer", ""),
        context=state.get("context", ""),
    )

    latency = state.get("latency_ms", {})
    latency["guard_output"] = _ms(t0)

    # Log guardrails to LangSmith
    tracer = _get_tracer()
    if tracer and state.get("run_id"):
        tracer.log_guardrails(
            run_id=state["run_id"],
            input_ok=(state.get("guard_result") or {}).get("ok", True),
            output_ok=result.get("ok", True),
            confidence=result.get("confidence", 0.5),
            grounded=result.get("grounded", False),
            has_hallucination=result.get("has_hallucination", False),
            output_warnings=result.get("warnings", []),
        )
        # End the LangSmith run
        tracer.end_run(
            run_id=state["run_id"],
            final_answer=result.get("answer", ""),
        )

    return {
        "final_answer":     result["answer"],
        "warnings":         result.get("warnings", []),
        "confidence":       result.get("confidence", 0.5),
        "grounded":         result.get("grounded", False),
        "has_hallucination": result.get("has_hallucination", False),
        "latency_ms":       latency,
    }


# ── Node: reject ──────────────────────────────────────────────────────────────

def reject(state: OmniMindState) -> Dict[str, Any]:
    reason = (state.get("guard_result") or {}).get("reason", "Request was rejected.")

    # End the LangSmith run with error
    tracer = _get_tracer()
    if tracer and state.get("run_id"):
        tracer.end_run(
            run_id=state["run_id"],
            final_answer="",
            error=f"Blocked by InputGuard: {reason}",
        )

    return {
        "final_answer": f"⚠️ Your query was blocked by the safety guard: {reason}",
        "warnings":     ["Query blocked by InputGuard"],
        "confidence":   0.0,
        "grounded":     False,
        "has_hallucination": False,
    }


# ── Routing functions (used as conditional edges) ──────────────────────────────

def route_after_guard(state: OmniMindState) -> str:
    return "ok" if (state.get("guard_result") or {}).get("ok") else "blocked"


def route_query_type(state: OmniMindState) -> str:
    return state.get("query_type", "semantic")
