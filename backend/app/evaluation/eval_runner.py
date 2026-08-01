"""
EvalRunner — batch evaluation runner for AEGIS v3.0.

Orchestrates end-to-end evaluation:
  1. Load test dataset
  2. Run each query through the live pipeline
  3. Score with RAGAS via AegisEvaluator
  4. Log all results to LangSmith
  5. Generate JSON + summary report

Usage:
    runner = EvalRunner(pipeline)
    report = runner.run_from_dataset(dataset)
    print(report["summary"])
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.app.evaluation.ragas_evaluator import AegisEvaluator, EvalResult
from backend.app.evaluation.eval_dataset import EvalDataset

logger = logging.getLogger(__name__)


class EvalRunner:
    """
    End-to-end evaluation runner.

    Can run evaluation against:
    - A pre-built EvalDataset (with answers already generated)
    - A list of questions (generates answers live via pipeline)
    """

    def __init__(self, pipeline=None, evaluator: Optional[AegisEvaluator] = None):
        """
        Args:
            pipeline: QueryPipeline instance (optional).
                      Required if running live evaluation (generate answers).
            evaluator: Optional pre-built evaluator (lazy-created on first use).
        """
        self.pipeline = pipeline
        self._evaluator = evaluator

    @property
    def evaluator(self) -> AegisEvaluator:
        if self._evaluator is None:
            self._evaluator = AegisEvaluator()
        return self._evaluator

    @evaluator.setter
    def evaluator(self, value: AegisEvaluator) -> None:
        self._evaluator = value

    # ── Public: Run Evaluation ────────────────────────────────────────────────

    def run_from_dataset(
        self,
        dataset: EvalDataset,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a pre-built dataset (answers already provided).

        Returns a report dict with aggregate scores and per-sample results.
        """
        logger.info(f"Starting evaluation: {len(dataset)} samples")
        start = time.perf_counter()

        samples = dataset.to_ragas_samples()
        aggregate = self.evaluator.evaluate_batch_ragas(samples)

        elapsed_s = round(time.perf_counter() - start, 2)
        report = self._build_report(aggregate, elapsed_s, dataset.name)

        if output_dir:
            self._save_report(report, output_dir)

        return report

    def run_live(
        self,
        questions: List[str],
        ground_truths: Optional[List[str]] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run questions through the live pipeline, then evaluate with RAGAS.

        Requires self.pipeline to be set.
        """
        if not self.pipeline:
            raise ValueError("pipeline must be set to run live evaluation")

        logger.info(f"Running live evaluation on {len(questions)} questions")
        start = time.perf_counter()

        samples = []
        for i, question in enumerate(questions):
            logger.info(f"  [{i+1}/{len(questions)}] Running: {question[:60]}")
            try:
                result = self.pipeline.answer(question)
                answer   = result.get("answer", "")
                contexts = result.get("context_used", [])
                # Normalise contexts to list of strings
                if contexts and isinstance(contexts[0], dict):
                    contexts = [c.get("text", "") for c in contexts]
                samples.append({
                    "question":     question,
                    "answer":       answer,
                    "contexts":     contexts,
                    "ground_truth": ground_truths[i] if ground_truths else None,
                })
            except Exception as e:
                logger.error(f"Pipeline failed for question '{question}': {e}")
                samples.append({
                    "question": question,
                    "answer":   "",
                    "contexts": [],
                    "error":    str(e),
                })

        aggregate = self.evaluator.evaluate_batch_ragas(samples)
        elapsed_s = round(time.perf_counter() - start, 2)
        report = self._build_report(aggregate, elapsed_s, "live_evaluation")

        if output_dir:
            self._save_report(report, output_dir)

        return report

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _build_report(
        self,
        aggregate: Dict[str, Any],
        elapsed_s: float,
        dataset_name: str,
    ) -> Dict[str, Any]:
        """Build a structured evaluation report."""
        summary = {
            "total_samples":       aggregate.get("total", 0),
            "passed":              aggregate.get("passed", 0),
            "pass_rate":           aggregate.get("pass_rate", 0.0),
            "avg_composite_score": aggregate.get("avg_composite", 0.0),
            "avg_faithfulness":    aggregate.get("avg_faithfulness"),
            "avg_answer_relevancy": aggregate.get("avg_answer_relevancy"),
            "avg_context_precision": aggregate.get("avg_context_precision"),
            "avg_context_recall":  aggregate.get("avg_context_recall"),
            "judge":               aggregate.get("judge"),
        }

        return {
            "report_id":   f"aegis_eval_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "dataset":     dataset_name,
            "timestamp":   datetime.utcnow().isoformat(),
            "elapsed_s":   elapsed_s,
            "summary":     summary,
            "per_sample":  aggregate.get("per_sample", []),
        }

    def _save_report(self, report: Dict[str, Any], output_dir: str) -> None:
        """Save evaluation report to disk as JSON."""
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        filename = p / f"{report['report_id']}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"Evaluation report saved: {filename}")
