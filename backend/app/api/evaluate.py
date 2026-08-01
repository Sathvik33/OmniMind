"""
Evaluate API — RAGAS evaluation endpoints for AEGIS v3.0.

Opt-in only: nothing is constructed at import/startup. The judge LLM and
MiniLM embeddings load on the first POST to /evaluate* (or via CLI scripts).

Endpoints:
  POST /evaluate              — evaluate a single Q&A pair
  POST /evaluate/batch        — batch evaluate multiple Q&A pairs
  POST /evaluate/live         — run questions through live pipeline + evaluate
  GET  /evaluate/health       — status (does not start the judge)
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])

_evaluator = None
_runner = None
_pipeline = None


def set_pipeline(pipeline) -> None:
    """Optional: inject QueryPipeline for /evaluate/live (no RAGAS init)."""
    global _pipeline, _runner
    _pipeline = pipeline
    if _runner is not None:
        _runner.pipeline = pipeline


def _get_evaluator():
    global _evaluator
    if _evaluator is None:
        from backend.app.evaluation.ragas_evaluator import AegisEvaluator

        _evaluator = AegisEvaluator()
    return _evaluator


def _get_runner():
    global _runner
    if _runner is None:
        from backend.app.evaluation.eval_runner import EvalRunner

        _runner = EvalRunner(pipeline=_pipeline)
    elif _pipeline is not None and _runner.pipeline is None:
        _runner.pipeline = _pipeline
    return _runner


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
    """Report eval availability without starting the judge LLM."""
    return {
        "evaluator_loaded": _evaluator is not None and getattr(_evaluator, "_llm", None) is not None,
        "opt_in": True,
        "message": (
            "RAGAS runs only on POST /evaluate*, CLI scripts, or query with evaluate=true. "
            "Not started at API boot."
        ),
        "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
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

        result = _get_evaluator().evaluate_single(
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
        return _get_evaluator().evaluate_batch_ragas(request.samples)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch evaluation failed: {str(e)}")


@router.post("/live")
def evaluate_live(request: LiveEvalRequest, background_tasks: BackgroundTasks):
    """
    Run questions through the live AEGIS pipeline and evaluate with RAGAS.
    """
    runner = _get_runner()
    if not runner.pipeline:
        from backend.app.api.query import pipeline as query_pipeline

        runner.pipeline = query_pipeline
        set_pipeline(query_pipeline)

    if not runner.pipeline:
        raise HTTPException(
            status_code=503,
            detail="Live evaluation unavailable: pipeline not initialized",
        )

    if len(request.questions) > 10:
        raise HTTPException(
            status_code=422,
            detail="Live evaluation limited to 10 questions per request to avoid timeout. Use /evaluate/batch for larger sets.",
        )

    try:
        report = runner.run_live(
            questions=request.questions,
            ground_truths=request.ground_truths,
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Live evaluation failed: {str(e)}")
