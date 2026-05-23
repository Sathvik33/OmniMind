"""
EvalDataset — Test dataset builder for RAGAS evaluation.

Provides a clean interface to build Q&A evaluation datasets
from uploaded documents or manual inputs.

Usage:
    ds = EvalDataset()
    ds.add("What is AEGIS?", "AEGIS is a RAG system", ["AEGIS is..."], "AEGIS is a RAG system")
    batch = ds.to_ragas_samples()
    evaluator.evaluate_batch(batch)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class QAPair:
    """A single question-answer pair for evaluation."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvalDataset:
    """
    Builds and manages Q&A evaluation datasets.

    Supports:
    - Manual addition of Q&A pairs
    - Loading from JSON files
    - Exporting to RAGAS-compatible format
    - Saving/loading datasets to disk
    """

    def __init__(self, name: str = "aegis_eval"):
        self.name = name
        self._samples: List[QAPair] = []

    # ── Add Samples ──────────────────────────────────────────────────────────

    def add(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single Q&A pair to the dataset."""
        self._samples.append(QAPair(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
            metadata=metadata or {},
        ))

    def add_from_dict(self, data: Dict[str, Any]) -> None:
        """Add a Q&A pair from a dict with keys: question, answer, contexts, ground_truth."""
        self.add(
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            contexts=data.get("contexts", []),
            ground_truth=data.get("ground_truth"),
            metadata=data.get("metadata", {}),
        )

    def add_batch(self, samples: List[Dict[str, Any]]) -> None:
        """Add multiple Q&A pairs from a list of dicts."""
        for s in samples:
            self.add_from_dict(s)

    # ── Export ────────────────────────────────────────────────────────────────

    def to_ragas_samples(self) -> List[Dict[str, Any]]:
        """Export to format expected by AegisEvaluator.evaluate_batch()."""
        return [
            {
                "question":    s.question,
                "answer":      s.answer,
                "contexts":    s.contexts,
                "ground_truth": s.ground_truth,
            }
            for s in self._samples
        ]

    def to_list(self) -> List[Dict[str, Any]]:
        """Export all samples as list of dicts."""
        return [s.to_dict() for s in self._samples]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save dataset to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(
                {"name": self.name, "samples": self.to_list()},
                f,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str) -> "EvalDataset":
        """Load dataset from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        ds = cls(name=data.get("name", "aegis_eval"))
        for sample in data.get("samples", []):
            ds.add_from_dict(sample)
        return ds

    # ── Properties ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._samples)

    def __repr__(self) -> str:
        return f"EvalDataset(name={self.name!r}, samples={len(self._samples)})"

    @property
    def samples(self) -> List[QAPair]:
        return self._samples
