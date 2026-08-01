"""
RAGAS benchmark with local Qwen judge (no Groq).

Modes:
  offline   — score the baked Q/A/context dataset (25 samples)
  retrieve  — BM25/hybrid retrieve real chunks, keep GT answers, then RAGAS
  live      — retrieve + generate with local Ollama, then RAGAS
  score     — score an existing samples_*.json (no retrieval)

Two-phase by default for retrieve/live so BGE (retrieval) and Ollama Qwen
(judge) are not loaded in the same process (avoids OOM on ~16–24GB machines).

Usage (repo root, Ollama running with qwen2.5:7b):
  $env:RAGAS_LLM_PROVIDER="ollama"
  $env:USE_LOCAL_LLM="true"
  python -m backend.scripts.run_ragas_benchmark --mode offline
  python -m backend.scripts.run_ragas_benchmark --mode retrieve
  python -m backend.scripts.run_ragas_benchmark --mode live --limit 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("RAGAS_LLM_PROVIDER", "ollama")
os.environ.setdefault("USE_LOCAL_LLM", "true")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("ragas_benchmark")

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "backend" / "evaluation_data" / "ragas_benchmark_25.json"
OUT_DIR = ROOT / "docs" / "eval"


def _normalize_contexts(context_used) -> list[str]:
    if not context_used:
        return []
    out = []
    for c in context_used:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, dict):
            out.append(c.get("text") or c.get("content") or str(c))
        else:
            out.append(str(c))
    return [x for x in out if x and str(x).strip()]


def _retrieve_contexts(question: str, top_k: int = 5) -> list[str]:
    """Prefer BM25-only (light); fall back to hybrid if BM25 empty."""
    try:
        from backend.app.retrieval.bm25_store import BM25Store

        bm25 = BM25Store()
        hits = bm25.search(question, top_k=top_k) if hasattr(bm25, "search") else []
        if not hits and hasattr(bm25, "retrieve"):
            hits = bm25.retrieve(question, top_k=top_k)
        texts = []
        for h in hits or []:
            if isinstance(h, str):
                texts.append(h)
            elif isinstance(h, dict):
                texts.append(h.get("text") or h.get("content") or "")
            elif isinstance(h, (list, tuple)) and h:
                texts.append(h[0] if isinstance(h[0], str) else str(h[0]))
        texts = [t for t in texts if t.strip()]
        if texts:
            return texts
    except Exception as e:
        logger.warning("BM25 retrieve failed: %s", e)

    from backend.app.pipelines.query_pipeline import QueryPipeline

    pipeline = QueryPipeline()
    retrieved = pipeline.hybrid_retriever.retrieve_with_confidence(question)
    return [r.get("text", "") for r in (retrieved or [])[:top_k] if r.get("text")]


def build_samples(mode: str, limit: int | None) -> list[dict]:
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    if limit:
        raw = raw[:limit]

    if mode == "offline":
        return raw

    samples = []
    pipeline = None
    if mode == "live":
        from backend.app.pipelines.query_pipeline import QueryPipeline

        pipeline = QueryPipeline()

    for i, row in enumerate(raw, 1):
        q = row["question"]
        gt = row.get("ground_truth")
        logger.info("[%s/%s] %s — %s", i, len(raw), mode, q[:70])
        try:
            if mode == "retrieve":
                contexts = _retrieve_contexts(q)
                if not contexts:
                    logger.warning("  no retrieved chunks — using dataset contexts")
                    contexts = row.get("contexts") or []
                samples.append(
                    {
                        "question": q,
                        "answer": row.get("answer") or gt or "",
                        "contexts": contexts,
                        "ground_truth": gt,
                        "metadata": {"mode": mode, "n_contexts": len(contexts)},
                    }
                )
            else:  # live
                assert pipeline is not None
                result = pipeline.answer(q, top_k=5)
                answer = result.get("answer") or ""
                contexts = _normalize_contexts(result.get("context_used"))
                if not contexts:
                    contexts = _retrieve_contexts(q)
                samples.append(
                    {
                        "question": q,
                        "answer": answer,
                        "contexts": contexts,
                        "ground_truth": gt,
                        "metadata": {
                            "mode": mode,
                            "n_contexts": len(contexts),
                            "confidence": result.get("confidence"),
                        },
                    }
                )
        except Exception as e:
            logger.error("  failed: %s", e)
            samples.append(
                {
                    "question": q,
                    "answer": row.get("answer") or "",
                    "contexts": row.get("contexts") or [],
                    "ground_truth": gt,
                    "error": str(e),
                }
            )
    return samples


def score_samples(samples: list[dict], mode: str) -> dict:
    from backend.app.evaluation.eval_dataset import EvalDataset
    from backend.app.evaluation.eval_runner import EvalRunner
    from backend.app.evaluation.ragas_evaluator import AegisEvaluator

    ds = EvalDataset(name=f"aegis_ragas_{mode}_{len(samples)}")
    ds.add_batch(samples)

    runner = EvalRunner()
    runner.evaluator = AegisEvaluator(provider="ollama")
    if runner.evaluator._llm is None:
        raise RuntimeError(
            "Ollama Qwen judge failed to start. Free RAM (stop heavy Python workers), "
            "ensure `ollama serve` is up, then retry."
        )

    report = runner.run_from_dataset(ds, output_dir=str(OUT_DIR))
    summary = report["summary"]
    summary["mode"] = mode
    summary["judge"] = runner.evaluator.judge_name
    summary["samples"] = len(samples)
    return summary


def _spawn_collect(mode: str, limit: int | None) -> Path:
    """Run retrieval/generation in a child process, then exit to free BGE RAM."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"samples_{mode}.json"
    cmd = [
        sys.executable,
        "-m",
        "backend.scripts.run_ragas_benchmark",
        "--mode",
        mode,
        "--collect-only",
        "--samples-out",
        str(out_path),
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    logger.info("Collecting samples in subprocess: %s", " ".join(cmd))
    subprocess.check_call(cmd, cwd=str(ROOT))
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("offline", "retrieve", "live", "score"),
        default="retrieve",
        help="offline|retrieve|live|score",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap number of samples")
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only build samples JSON (no RAGAS). Used by parent two-phase run.",
    )
    parser.add_argument(
        "--samples-out",
        type=str,
        default=None,
        help="Write/read samples JSON path",
    )
    parser.add_argument(
        "--no-subprocess",
        action="store_true",
        help="Run retrieve/live + RAGAS in one process (needs more RAM)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    samples_path = Path(args.samples_out) if args.samples_out else OUT_DIR / f"samples_{args.mode}.json"

    if args.mode == "score":
        if not samples_path.exists():
            logger.error("Samples file missing: %s", samples_path)
            return 1
        samples = json.loads(samples_path.read_text(encoding="utf-8"))
        summary = score_samples(samples, mode="score")
    elif args.collect_only or args.mode == "offline" or args.no_subprocess:
        if not DATASET.exists():
            logger.error("Dataset missing: %s", DATASET)
            return 1
        samples = build_samples(args.mode if args.mode != "score" else "offline", args.limit)
        samples_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Wrote %s samples → %s", len(samples), samples_path)
        if args.collect_only:
            return 0
        summary = score_samples(samples, mode=args.mode)
    else:
        # Two-phase: collect (may load BGE) → score (Ollama + MiniLM only)
        out = _spawn_collect(args.mode, args.limit)
        samples = json.loads(out.read_text(encoding="utf-8"))
        summary = score_samples(samples, mode=args.mode)

    latest = OUT_DIR / "latest_summary.json"
    latest.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Judge: %s", summary.get("judge"))
    logger.info("Summary:\n%s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
