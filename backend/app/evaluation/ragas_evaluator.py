"""
AegisEvaluator — RAGAS-powered evaluation engine for AEGIS v3.0.

Uses Groq (llama3-8b-8192) as the evaluation LLM for:
  - faithfulness       : Is the answer faithful to the retrieved context?
  - answer_relevancy   : How relevant is the answer to the question?
  - context_precision  : Are the retrieved chunks actually useful?
  - context_recall     : Was important context retrieved? (needs ground_truth)

Architecture:
  AegisEvaluator.evaluate_single()  → single Q&A evaluation
  AegisEvaluator.evaluate_batch()   → batch evaluation returning DataFrame
  AegisEvaluator.quick_score()      → fast faithfulness-only check

All scores are logged to LangSmith via AegisTracer.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama3-8b-8192")
RAGAS_THRESHOLD = float(os.getenv("RAGAS_EVALUATION_THRESHOLD", "0.5"))

# ── Optional imports (graceful fallback) ──────────────────────────────────────
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
    from ragas.llms import LangchainLLMWrapper
    from datasets import Dataset
    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False
    logger.warning("ragas not installed — evaluation running in fallback mode")

try:
    from langchain_groq import ChatGroq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
    logger.warning("langchain-groq not installed")


# ── Result Dataclass ──────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Structured result from a RAGAS evaluation."""
    query: str
    answer: str
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    composite_score: float = 0.0
    passed: bool = False
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "composite_score": round(self.composite_score, 4),
            "passed": self.passed,
            "error": self.error,
        }


# ── AegisEvaluator ────────────────────────────────────────────────────────────

class AegisEvaluator:
    """
    RAGAS evaluation engine powered by Groq LLM.

    Usage:
        evaluator = AegisEvaluator()
        result = evaluator.evaluate_single(
            query="What is AEGIS?",
            answer="AEGIS is a multi-modal RAG system...",
            contexts=["AEGIS stands for..."],
            ground_truth="AEGIS is a multi-modal RAG engine"   # optional
        )
        print(result.composite_score)   # 0.0 to 1.0
    """

    def __init__(self):
        self._llm = None
        self._metrics_with_gt  = []   # metrics that need ground_truth
        self._metrics_no_gt    = []   # metrics that work without ground_truth
        self._setup()

    def _setup(self) -> None:
        """Initialize Groq LLM and RAGAS metrics."""
        if not _RAGAS_AVAILABLE:
            logger.warning("RAGAS not available — evaluation will return fallback scores")
            return

        if not _GROQ_AVAILABLE or not GROQ_API_KEY:
            logger.warning("Groq not available/configured — evaluation will return fallback scores")
            return

        try:
            groq_chat = ChatGroq(
                api_key=GROQ_API_KEY,
                model_name=GROQ_MODEL,
                temperature=0.0,   # deterministic evaluation
                max_tokens=1024,
            )
            self._llm = LangchainLLMWrapper(groq_chat)

            # Wire the LLM into RAGAS metrics
            faithfulness.llm        = self._llm
            answer_relevancy.llm    = self._llm
            context_precision.llm   = self._llm
            context_recall.llm      = self._llm

            self._metrics_no_gt   = [faithfulness, answer_relevancy, context_precision]
            self._metrics_with_gt = [faithfulness, answer_relevancy, context_precision, context_recall]

            logger.info(f"✅ AegisEvaluator initialized with Groq model '{GROQ_MODEL}'")
        except Exception as e:
            logger.error(f"AegisEvaluator setup failed: {e}")
            self._llm = None

    # ── Public: Single Evaluation ─────────────────────────────────────────────

    def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> EvalResult:
        """
        Evaluate a single Q&A pair using RAGAS metrics.

        Args:
            query:        The user question
            answer:       The LLM-generated answer
            contexts:     List of retrieved context chunks
            ground_truth: Optional reference answer for context_recall
            run_id:       LangSmith run ID (for logging)

        Returns:
            EvalResult with per-metric scores and composite score
        """
        result = EvalResult(query=query, answer=answer)

        if not self._llm or not _RAGAS_AVAILABLE:
            return self._fallback_score(result, contexts)

        try:
            # Build RAGAS dataset
            dataset_dict: Dict[str, List] = {
                "question":  [query],
                "answer":    [answer],
                "contexts":  [contexts],
            }
            metrics = self._metrics_no_gt
            if ground_truth:
                dataset_dict["ground_truth"] = [ground_truth]
                metrics = self._metrics_with_gt

            dataset = Dataset.from_dict(dataset_dict)
            eval_result = ragas_evaluate(dataset=dataset, metrics=metrics)

            # Extract scores from the result DataFrame
            scores_df = eval_result.to_pandas()
            row = scores_df.iloc[0]

            result.faithfulness      = self._safe_float(row, "faithfulness")
            result.answer_relevancy  = self._safe_float(row, "answer_relevancy")
            result.context_precision = self._safe_float(row, "context_precision")
            result.context_recall    = self._safe_float(row, "context_recall") if ground_truth else None

            result.composite_score = self._compute_composite(
                result.faithfulness,
                result.answer_relevancy,
                result.context_precision,
                result.context_recall,
            )
            result.passed = result.composite_score >= RAGAS_THRESHOLD

            # Log to LangSmith
            if run_id:
                self._log_to_langsmith(run_id, result)

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            result.error = str(e)
            return self._fallback_score(result, contexts)

        return result

    # ── Public: Batch Evaluation ──────────────────────────────────────────────

    def evaluate_batch(
        self,
        samples: List[Dict[str, Any]],
    ) -> List[EvalResult]:
        """
        Evaluate a list of Q&A pairs.

        Each sample dict must have: question, answer, contexts
        Optionally: ground_truth

        Returns list of EvalResult objects.
        """
        results = []
        for i, sample in enumerate(samples):
            logger.info(f"Evaluating sample {i+1}/{len(samples)}")
            result = self.evaluate_single(
                query=sample.get("question", ""),
                answer=sample.get("answer", ""),
                contexts=sample.get("contexts", []),
                ground_truth=sample.get("ground_truth"),
            )
            results.append(result)

        return results

    def evaluate_batch_ragas(
        self,
        samples: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Batch evaluation using a single RAGAS call (more efficient).
        Returns aggregate scores + per-sample results.
        """
        if not self._llm or not _RAGAS_AVAILABLE:
            results = self.evaluate_batch(samples)
            return self._aggregate_results(results)

        has_gt = all("ground_truth" in s for s in samples)
        dataset_dict = {
            "question": [s.get("question", "") for s in samples],
            "answer":   [s.get("answer", "") for s in samples],
            "contexts": [s.get("contexts", []) for s in samples],
        }
        if has_gt:
            dataset_dict["ground_truth"] = [s.get("ground_truth", "") for s in samples]

        metrics = self._metrics_with_gt if has_gt else self._metrics_no_gt

        try:
            dataset = Dataset.from_dict(dataset_dict)
            eval_result = ragas_evaluate(dataset=dataset, metrics=metrics)
            df = eval_result.to_pandas()

            per_sample = []
            for i, row in df.iterrows():
                r = EvalResult(
                    query=samples[i].get("question", ""),
                    answer=samples[i].get("answer", ""),
                    faithfulness=self._safe_float(row, "faithfulness"),
                    answer_relevancy=self._safe_float(row, "answer_relevancy"),
                    context_precision=self._safe_float(row, "context_precision"),
                    context_recall=self._safe_float(row, "context_recall") if has_gt else None,
                )
                r.composite_score = self._compute_composite(
                    r.faithfulness, r.answer_relevancy, r.context_precision, r.context_recall
                )
                r.passed = r.composite_score >= RAGAS_THRESHOLD
                per_sample.append(r)

            return self._aggregate_results(per_sample)

        except Exception as e:
            logger.error(f"Batch RAGAS evaluation failed: {e}")
            results = self.evaluate_batch(samples)
            return self._aggregate_results(results)

    # ── Public: Quick Score ───────────────────────────────────────────────────

    def quick_score(self, query: str, answer: str, contexts: List[str]) -> float:
        """
        Fast composite score without full RAGAS overhead.
        Uses token overlap as a proxy for faithfulness.
        Returns 0.0 to 1.0.
        """
        if not contexts or not answer:
            return 0.0

        # Token overlap proxy
        answer_tokens = set(answer.lower().split())
        context_tokens = set(" ".join(contexts).lower().split())
        if not answer_tokens:
            return 0.0

        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)

        # Length quality proxy (answers between 50–500 words are ideal)
        words = len(answer.split())
        length_score = 1.0 if 10 <= words <= 300 else 0.7

        return round((overlap * 0.7 + length_score * 0.3), 3)

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _compute_composite(
        self,
        faithfulness: Optional[float],
        answer_relevancy: Optional[float],
        context_precision: Optional[float],
        context_recall: Optional[float],
    ) -> float:
        """Weighted composite score from RAGAS metrics."""
        scores = []
        weights = []

        if faithfulness is not None:
            scores.append(faithfulness);      weights.append(0.35)
        if answer_relevancy is not None:
            scores.append(answer_relevancy);  weights.append(0.30)
        if context_precision is not None:
            scores.append(context_precision); weights.append(0.20)
        if context_recall is not None:
            scores.append(context_recall);    weights.append(0.15)

        if not scores:
            return 0.0

        total_weight = sum(weights)
        composite = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return round(composite, 4)

    def _safe_float(self, row: Any, col: str) -> Optional[float]:
        """Safely extract a float from a DataFrame row."""
        try:
            v = row.get(col) if hasattr(row, "get") else getattr(row, col, None)
            return round(float(v), 4) if v is not None and not (v != v) else None  # NaN check
        except Exception:
            return None

    def _fallback_score(self, result: EvalResult, contexts: List[str]) -> EvalResult:
        """Return a proxy score when RAGAS is not available."""
        proxy = self.quick_score(result.query, result.answer, contexts)
        result.composite_score = proxy
        result.passed = proxy >= RAGAS_THRESHOLD
        result.metadata["fallback"] = True
        return result

    def _aggregate_results(self, results: List[EvalResult]) -> Dict[str, Any]:
        """Compute aggregate statistics across all results."""
        if not results:
            return {"total": 0}

        def avg(vals): return round(sum(v for v in vals if v is not None) / max(1, sum(1 for v in vals if v is not None)), 4)

        return {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "pass_rate": round(sum(1 for r in results if r.passed) / len(results), 3),
            "avg_composite": avg([r.composite_score for r in results]),
            "avg_faithfulness": avg([r.faithfulness for r in results]),
            "avg_answer_relevancy": avg([r.answer_relevancy for r in results]),
            "avg_context_precision": avg([r.context_precision for r in results]),
            "avg_context_recall": avg([r.context_recall for r in results]),
            "per_sample": [r.to_dict() for r in results],
        }

    def _log_to_langsmith(self, run_id: str, result: EvalResult) -> None:
        """Log RAGAS scores to LangSmith."""
        try:
            from backend.app.monitoring.langsmith_logger import tracer
            tracer.log_evaluation(
                run_id=run_id,
                eval_scores={
                    k: v for k, v in {
                        "faithfulness": result.faithfulness,
                        "answer_relevancy": result.answer_relevancy,
                        "context_precision": result.context_precision,
                        "context_recall": result.context_recall,
                    }.items() if v is not None
                },
                composite_score=result.composite_score,
            )
        except Exception as e:
            logger.debug(f"Failed to log eval to LangSmith: {e}")
