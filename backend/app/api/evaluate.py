"""
Evaluate API — RAGAS evaluation endpoints for AEGIS v3.0.

Endpoints:
  POST /evaluate              — evaluate a single Q&A pair
  POST /evaluate/batch        — batch evaluate multiple Q&A pairs
  POST /evaluate/live         — run questions through live pipeline + evaluate
  GET  /evaluate/health       — evaluator status check
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

from backend.app.evaluation.ragas_evaluator import AegisEvaluator
from backend.app.evaluation.eval_dataset import EvalDataset
from backend.app.evaluation.eval_runner import EvalRunner

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])

# Singletons (constructed once)
_evaluator = AegisEvaluator()
_runner    = EvalRunner()   # pipeline injected later via set_pipeline()


def set_pipeline(pipeline) -> None:
    """Called from main.py startup to inject pipeline into eval runner."""
    _runner.pipeline = pipeline


# ── Request / Response Models ─────────────────────────────────────────────────

class EvalRequest(BaseModel):
    query: str           = Field(..., min_length=2, max_length=2000)
    answer: str          = Field(..., min_length=1)
    contexts: List[str]  = Field(..., min_items=1)
    ground_truth: Optional[str] = Field(None)
    run_id: Optional[str] = Field(None, description="LangSmith run_id to attach scores to")


class EvalResponse(BaseModel):
    query: str
    answer: str
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]
    composite_score: float
    passed: bool
    threshold: float
    error: Optional[str]


class BatchEvalRequest(BaseModel):
    samples: List[Dict[str, Any]] = Field(
        ...,
        description="List of {question, answer, contexts, ground_truth?} dicts"
    )


class LiveEvalRequest(BaseModel):
    questions: List[str]     = Field(..., min_items=1, max_items=50)
    ground_truths: Optional[List[str]] = Field(None)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/health")
def eval_health():
    """Check RAGAS evaluator status."""
    return {
        "evaluator_active": _evaluator._llm is not None,
        "groq_model": "llama3-8b-8192",
        "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        "message": (
            "RAGAS evaluator ready with Groq LLM"
            if _evaluator._llm is not None
            else "Running in fallback mode (Groq not configured)"
        ),
    }


@router.post("/", response_model=EvalResponse)
def evaluate_single(request: EvalRequest):
    """
    Evaluate a single Q&A pair using RAGAS metrics.

    - faithfulness: Is the answer faithful to retrieved context?
    - answer_relevancy: Is the answer relevant to the question?
    - context_precision: Are retrieved chunks useful?
    - context_recall: Was all needed context retrieved? (requires ground_truth)
    - composite_score: Weighted average of all available metrics
    """
    try:
        from backend.app.core.config import RAGAS_EVALUATION_THRESHOLD
        result = _evaluator.evaluate_single(
            query=request.query,
            answer=request.answer,
            contexts=request.contexts,
            ground_truth=request.ground_truth,
            run_id=request.run_id,
        )
        return EvalResponse(
            query=result.query,
            answer=result.answer,
            faithfulness=result.faithfulness,
            answer_relevancy=result.answer_relevancy,
            context_precision=result.context_precision,
            context_recall=result.context_recall,
            composite_score=result.composite_score,
            passed=result.passed,
            threshold=RAGAS_EVALUATION_THRESHOLD,
            error=result.error,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


@router.post("/batch")
def evaluate_batch(request: BatchEvalRequest):
    """
    Batch evaluate multiple Q&A pairs using RAGAS.

    Returns aggregate scores and per-sample breakdowns.
    """
    try:
        return _evaluator.evaluate_batch_ragas(request.samples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")


@router.post("/live")
def evaluate_live(request: LiveEvalRequest, background_tasks: BackgroundTasks):
    """
    Run questions through the live AEGIS pipeline and evaluate with RAGAS.

    NOTE: This requires the pipeline to be available (set at startup).
    For large question sets, consider running in background.
    """
    if not _runner.pipeline:
        raise HTTPException(
            status_code=503,
            detail="Live evaluation unavailable: pipeline not initialized"
        )

    if len(request.questions) > 10:
        raise HTTPException(
            status_code=422,
            detail="Live evaluation limited to 10 questions per request to avoid timeout. Use /evaluate/batch for larger sets."
        )

    try:
        report = _runner.run_live(
            questions=request.questions,
            ground_truths=request.ground_truths,
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live evaluation failed: {str(e)}")
