"""
AegisEvaluator — RAGAS-powered evaluation engine for AEGIS v3.0.

Judge LLM (configurable via RAGAS_LLM_PROVIDER):
  - ollama (default): local Qwen via ChatOllama — preferred when Groq is rate-limited
  - groq: cloud llama for faster remote judging

Metrics (scored against retrieved contexts):
  - faithfulness       : Is the answer faithful to the retrieved context?
  - answer_relevancy   : How relevant is the answer to the question?
  - context_precision  : Are the retrieved chunks actually useful?
  - context_recall     : Was important context retrieved? (needs ground_truth)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
RAGAS_THRESHOLD = float(os.getenv("RAGAS_EVALUATION_THRESHOLD", "0.5"))

try:
    from backend.app.core.config import RAGAS_LLM_PROVIDER, RAGAS_OLLAMA_MODEL, OLLAMA_MODEL
except Exception:
    RAGAS_LLM_PROVIDER = os.getenv("RAGAS_LLM_PROVIDER", "ollama").lower()
    RAGAS_OLLAMA_MODEL = os.getenv("RAGAS_OLLAMA_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5:7b"))
    OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

try:
    from ragas import evaluate as ragas_evaluate
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

try:
    from langchain_ollama import ChatOllama

    _OLLAMA_AVAILABLE = True
except ImportError:
    _OLLAMA_AVAILABLE = False
    logger.warning("langchain-ollama not installed")


@dataclass
class EvalResult:
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
            "judge": self.metadata.get("judge"),
        }


class AegisEvaluator:
    """RAGAS evaluation with local Ollama Qwen as the default judge.

    Lazy: judge LLM / MiniLM are not loaded until the first evaluate_* call.
    """

    def __init__(self, provider: Optional[str] = None):
        self._llm = None
        self._judge_name = "none"
        self._metrics_with_gt = []
        self._metrics_no_gt = []
        self._provider = (provider or RAGAS_LLM_PROVIDER or "ollama").lower()
        self._ready = False

    @property
    def judge_name(self) -> str:
        return self._judge_name

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        self._setup()
        self._ready = True

    def _setup(self) -> None:
        if not _RAGAS_AVAILABLE:
            logger.warning("RAGAS not available — fallback scores only")
            return

        chat = None
        if self._provider == "ollama":
            # Stay local — do not fall back to Groq (rate limits / bad keys).
            chat = self._make_ollama()
        elif self._provider == "groq":
            chat = self._make_groq()
            if chat is None:
                logger.warning("Groq judge unavailable — falling back to Ollama")
                chat = self._make_ollama()
        else:
            chat = self._make_ollama() or self._make_groq()

        if chat is None:
            logger.warning("No judge LLM available — fallback scores only")
            return

        try:
            self._llm = LangchainLLMWrapper(chat)
            from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

            _faithfulness = Faithfulness()
            _answer_relevancy = AnswerRelevancy()
            _context_precision = ContextPrecision()
            _context_recall = ContextRecall()
            for m in (_faithfulness, _answer_relevancy, _context_precision, _context_recall):
                m.llm = self._llm

            try:
                # Lightweight local embedder for AnswerRelevancy — avoid loading BGE-M3
                # (keeps RAM free for Ollama Qwen judge).
                from ragas.embeddings import LangchainEmbeddingsWrapper

                try:
                    from langchain_huggingface import HuggingFaceEmbeddings
                except ImportError:
                    from langchain_community.embeddings import HuggingFaceEmbeddings

                hf = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                emb = LangchainEmbeddingsWrapper(hf)
                for m in (_faithfulness, _answer_relevancy, _context_precision, _context_recall):
                    if hasattr(m, "embeddings"):
                        m.embeddings = emb
                logger.info("RAGAS embeddings wired to all-MiniLM-L6-v2")
            except Exception as emb_err:
                logger.warning("Could not wire MiniLM RAGAS embeddings: %s", emb_err)

            self._metrics_no_gt = [_faithfulness, _answer_relevancy]
            self._metrics_with_gt = [
                _faithfulness,
                _answer_relevancy,
                _context_precision,
                _context_recall,
            ]
            logger.info("AegisEvaluator ready — judge=%s", self._judge_name)
        except Exception as e:
            logger.error("AegisEvaluator setup failed: %s", e)
            self._llm = None

    def _make_ollama(self):
        if not _OLLAMA_AVAILABLE:
            return None
        model = RAGAS_OLLAMA_MODEL or OLLAMA_MODEL or "qwen2.5:7b"
        try:
            # Keep context short — RAGAS prompts are long; leave RAM for the model weights.
            chat = ChatOllama(
                model=model,
                temperature=0.0,
                num_predict=512,
                num_ctx=4096,
            )
            # No smoke invoke — keep boot cheap; first RAGAS call warms the model.
            self._judge_name = f"ollama:{model}"
            return chat
        except Exception as e:
            logger.warning("Ollama judge init failed (%s): %s", model, e)
            return None

    def _make_groq(self):
        if not _GROQ_AVAILABLE or not GROQ_API_KEY:
            return None
        try:
            chat = ChatGroq(
                api_key=GROQ_API_KEY,
                model_name=GROQ_MODEL,
                temperature=0.0,
                max_tokens=1024,
            )
            self._judge_name = f"groq:{GROQ_MODEL}"
            return chat
        except Exception as e:
            logger.warning("Groq judge init failed: %s", e)
            return None

    def evaluate_single(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> EvalResult:
        self._ensure_ready()
        result = EvalResult(query=query, answer=answer)
        result.metadata["judge"] = self._judge_name

        if not self._llm or not _RAGAS_AVAILABLE:
            return self._fallback_score(result, contexts)

        try:
            ref_val = ground_truth if ground_truth else answer
            dataset_dict: Dict[str, List] = {
                "question": [query],
                "user_input": [query],
                "answer": [answer],
                "response": [answer],
                "contexts": [contexts],
                "retrieved_contexts": [contexts],
                "reference": [ref_val],
                "ground_truth": [ref_val],
            }
            metrics = self._metrics_with_gt if ground_truth else self._metrics_no_gt
            dataset = Dataset.from_dict(dataset_dict)
            eval_result = ragas_evaluate(dataset=dataset, metrics=metrics)
            scores_df = eval_result.to_pandas()
            row = scores_df.iloc[0]

            result.faithfulness = self._safe_float(row, "faithfulness")
            result.answer_relevancy = self._safe_float(row, "answer_relevancy")
            result.context_precision = self._safe_float(row, "context_precision")
            result.context_recall = (
                self._safe_float(row, "context_recall") if ground_truth else None
            )
            result.composite_score = self._compute_composite(
                result.faithfulness,
                result.answer_relevancy,
                result.context_precision,
                result.context_recall,
            )
            result.passed = result.composite_score >= RAGAS_THRESHOLD
            if run_id:
                self._log_to_langsmith(run_id, result)
        except Exception as e:
            logger.error("RAGAS evaluation failed: %s", e)
            result.error = str(e)
            return self._fallback_score(result, contexts)

        return result

    def evaluate_batch(self, samples: List[Dict[str, Any]]) -> List[EvalResult]:
        results = []
        for i, sample in enumerate(samples):
            logger.info("Evaluating sample %s/%s", i + 1, len(samples))
            results.append(
                self.evaluate_single(
                    query=sample.get("question", ""),
                    answer=sample.get("answer", ""),
                    contexts=sample.get("contexts", []),
                    ground_truth=sample.get("ground_truth"),
                )
            )
        return results

    def evaluate_batch_ragas(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Prefer per-sample loop with local Ollama — more stable than one giant batch
        results = self.evaluate_batch(samples)
        agg = self._aggregate_results(results)
        agg["judge"] = self._judge_name
        return agg

    def quick_score(self, query: str, answer: str, contexts: List[str]) -> float:
        if not contexts or not answer:
            return 0.0
        answer_tokens = set(answer.lower().split())
        context_tokens = set(" ".join(contexts).lower().split())
        if not answer_tokens:
            return 0.0
        overlap = len(answer_tokens & context_tokens) / len(answer_tokens)
        words = len(answer.split())
        length_score = 1.0 if 10 <= words <= 300 else 0.7
        return round((overlap * 0.7 + length_score * 0.3), 3)

    def _compute_composite(
        self,
        faithfulness: Optional[float],
        answer_relevancy: Optional[float],
        context_precision: Optional[float],
        context_recall: Optional[float],
    ) -> float:
        scores, weights = [], []
        if faithfulness is not None:
            scores.append(faithfulness)
            weights.append(0.35)
        if answer_relevancy is not None:
            scores.append(answer_relevancy)
            weights.append(0.30)
        if context_precision is not None:
            scores.append(context_precision)
            weights.append(0.20)
        if context_recall is not None:
            scores.append(context_recall)
            weights.append(0.15)
        if not scores:
            return 0.0
        return round(sum(s * w for s, w in zip(scores, weights)) / sum(weights), 4)

    def _safe_float(self, row: Any, col: str) -> Optional[float]:
        try:
            v = row.get(col) if hasattr(row, "get") else getattr(row, col, None)
            return round(float(v), 4) if v is not None and not (v != v) else None
        except Exception:
            return None

    def _fallback_score(self, result: EvalResult, contexts: List[str]) -> EvalResult:
        proxy = self.quick_score(result.query, result.answer, contexts)
        result.faithfulness = proxy
        result.answer_relevancy = proxy
        result.composite_score = proxy
        result.passed = proxy >= RAGAS_THRESHOLD
        result.metadata["fallback"] = True
        result.metadata["judge"] = self._judge_name
        return result

    def _aggregate_results(self, results: List[EvalResult]) -> Dict[str, Any]:
        if not results:
            return {"total": 0, "judge": self._judge_name}

        def avg(vals):
            nums = [v for v in vals if v is not None]
            return round(sum(nums) / max(1, len(nums)), 4) if nums else None

        return {
            "total": len(results),
            "passed": sum(1 for r in results if r.passed),
            "pass_rate": round(sum(1 for r in results if r.passed) / len(results), 3),
            "avg_composite": avg([r.composite_score for r in results]),
            "avg_faithfulness": avg([r.faithfulness for r in results]),
            "avg_answer_relevancy": avg([r.answer_relevancy for r in results]),
            "avg_context_precision": avg([r.context_precision for r in results]),
            "avg_context_recall": avg([r.context_recall for r in results]),
            "judge": self._judge_name,
            "per_sample": [r.to_dict() for r in results],
        }

    def _log_to_langsmith(self, run_id: str, result: EvalResult) -> None:
        try:
            from backend.app.monitoring.langsmith_logger import tracer

            tracer.log_evaluation(
                run_id=run_id,
                eval_scores={
                    k: v
                    for k, v in {
                        "faithfulness": result.faithfulness,
                        "answer_relevancy": result.answer_relevancy,
                        "context_precision": result.context_precision,
                        "context_recall": result.context_recall,
                    }.items()
                    if v is not None
                },
                composite_score=result.composite_score,
            )
        except Exception as e:
            logger.debug("Failed to log eval to LangSmith: %s", e)
