"""
Monitor API — LangSmith monitoring dashboard endpoints for AEGIS v3.0.

Endpoints:
  GET  /monitor/stats           — aggregate run statistics
  GET  /monitor/runs            — list recent traced runs
  POST /monitor/feedback        — submit user feedback on a run
  GET  /monitor/health          — tracer connection health check
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional

from backend.app.monitoring.langsmith_logger import tracer

router = APIRouter(prefix="/monitor", tags=["Monitoring"])


# ── Request / Response Models ─────────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    run_id: str = Field(..., description="LangSmith run ID to attach feedback to")
    score: float = Field(..., ge=0.0, le=1.0, description="1.0 = positive, 0.0 = negative")
    comment: Optional[str] = Field(None, max_length=1000, description="Optional text feedback")
    key: str = Field(default="user_feedback", description="Feedback key/category")


class FeedbackResponse(BaseModel):
    success: bool
    message: str


class RunSummary(BaseModel):
    run_id: str
    name: str
    status: Optional[str]
    start_time: Optional[str]
    latency_ms: Optional[float]
    query: Optional[str]
    error: Optional[str]


class StatsResponse(BaseModel):
    active: bool
    project: str
    total_runs: Optional[int] = None
    error_count: Optional[int] = None
    error_rate: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    min_latency_ms: Optional[float] = None
    max_latency_ms: Optional[float] = None
    langsmith_url: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
def monitor_health():
    """Check LangSmith tracer connection status."""
    return {
        "tracer_active": tracer.is_active(),
        "project": "Aegis",
        "message": "LangSmith connected" if tracer.is_active() else "Running in no-op mode",
    }


@router.get("/stats", response_model=StatsResponse)
def get_monitor_stats(limit: int = 100):
    """
    Aggregate statistics from recent LangSmith runs.

    Returns avg/min/max latency, error rate, and total run count.
    """
    try:
        stats = tracer.get_run_stats(limit=limit)
        return StatsResponse(**{k: v for k, v in stats.items() if k in StatsResponse.model_fields})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


@router.get("/runs")
def get_recent_runs(limit: int = 20):
    """
    List recent traced RAG runs from LangSmith.

    Returns run IDs, queries, statuses, and latencies.
    """
    try:
        runs = tracer.get_recent_runs(limit=limit)
        return {"runs": runs, "count": len(runs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch runs: {str(e)}")


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback on a specific RAG run.

    Use score=1.0 for positive feedback (thumbs up),
    score=0.0 for negative feedback (thumbs down).
    """
    success = tracer.submit_feedback(
        run_id=request.run_id,
        score=request.score,
        comment=request.comment,
        key=request.key,
    )
    if success:
        return FeedbackResponse(success=True, message="Feedback submitted to LangSmith")
    else:
        return FeedbackResponse(
            success=False,
            message="Feedback not submitted (LangSmith not connected or run_id invalid)"
        )
