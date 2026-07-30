"""
Query API endpoints — synchronous and streaming RAG queries for AEGIS v3.0.

Endpoints:
  POST /query        — Blocking query with full response data + run_id
  POST /query-stream — Streaming query with SSE (Server-Sent Events)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from fastapi.responses import StreamingResponse
from typing import Any, Dict, List, Optional

from backend.app.pipelines.query_pipeline import QueryPipeline

router = APIRouter()
pipeline = QueryPipeline()


class QueryRequest(BaseModel):
    """Query request with optional parameters."""
    query:    str   = Field(..., min_length=2, max_length=2000)
    top_k:    int   = Field(default=3, ge=1, le=20)
    evaluate: bool  = Field(default=False, description="Run inline RAGAS evaluation (adds latency)")


class QueryResponse(BaseModel):
    """Structured query response with confidence metrics and monitoring ID."""
    answer:              str
    context_used:        List[Any] = []
    warnings:            List[str] = []
    confidence:          float     = Field(ge=0.0, le=1.0)
    grounded:            bool      = False
    has_hallucination:   bool      = False
    error:               bool      = False
    run_id:              Optional[str] = None
    retrieval_metadata:  Optional[Dict[str, Any]] = None
    latency_ms:          Optional[Dict[str, float]] = None
    eval_scores:         Optional[Dict[str, Any]] = None
    structured_answer:   Optional[Dict[str, Any]] = None


@router.post("/query", response_model=QueryResponse)
def query_data(request: QueryRequest) -> QueryResponse:
    """
    Synchronous RAG query with full guardrails and LangSmith tracing.

    Returns:
      - answer: LLM response (PII-masked, grounding-validated)
      - context_used: Retrieved documents used in generation
      - confidence: 0.0-1.0 confidence score from OutputGuard
      - grounded: Whether answer is grounded in retrieved context
      - has_hallucination: Whether hallucination markers detected
      - run_id: LangSmith run ID — use for /monitor/feedback
      - retrieval_metadata: BM25/dense hit counts and latency
      - latency_ms: Per-node latency breakdown
      - eval_scores: RAGAS scores (only when evaluate=True)
    """
    try:
        result = pipeline.answer(request.query, request.top_k, evaluate=request.evaluate)
        return QueryResponse(**{k: v for k, v in result.items() if k in QueryResponse.model_fields})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/query-stream")
def query_stream(request: QueryRequest):
    """
    Streaming RAG query with token-by-token generation.

    Returns: text/plain with tokens streamed one-by-one.
    Includes metadata comments: [Retrieved: source (confidence)], etc.
    """
    def token_generator():
        try:
            for token in pipeline.stream_answer(request.query):
                yield token
        except Exception as e:
            yield f"\n\n❌ Error: {str(e)}\n"

    return StreamingResponse(token_generator(), media_type="text/plain")