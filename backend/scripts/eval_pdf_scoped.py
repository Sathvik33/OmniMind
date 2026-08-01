"""
Scoped 5-question RAG eval: signup → chat → upload → query → RAGAS (local Qwen).

Usage:
  python -m backend.scripts.eval_pdf_scoped \\
    --file path/to/notes.md \\
    --base-url http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("RAGAS_LLM_PROVIDER", "ollama")
os.environ.setdefault("USE_LOCAL_LLM", "true")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("eval_pdf_scoped")

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "eval"

# 5 questions grounded in the NLP endsem notes
NLP_QA = [
    {
        "question": "What is the Turing Test according to the notes?",
        "ground_truth": (
            "A machine is considered intelligent if humans cannot distinguish "
            "its responses from a human."
        ),
    },
    {
        "question": "What is the TF-IDF formula and what do TF and IDF measure?",
        "ground_truth": (
            "TF-IDF = TF × IDF. TF is term frequency in a document; "
            "IDF is log(N/df(t)) measuring term rarity across documents."
        ),
    },
    {
        "question": "What is BLEU score used for?",
        "ground_truth": (
            "BLEU evaluates machine translation by comparing a generated "
            "translation with a reference; higher BLEU means better quality."
        ),
    },
    {
        "question": "What three vectors does self-attention use and what is each for?",
        "ground_truth": (
            "Query (Q) is what the word searches for, Key (K) represents word "
            "identity, and Value (V) carries the actual information."
        ),
    },
    {
        "question": "Name two advantages and two disadvantages of rule-based NLP from the 1960s–1980s.",
        "ground_truth": (
            "Advantages: interpretable and structured. Disadvantages: difficult "
            "to scale and required manual rule creation."
        ),
    },
]


def _request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    last = None
    for attempt in range(1, 8):
        try:
            r = requests.request(method, url, **kwargs)
            return r
        except (requests.ConnectionError, requests.Timeout) as e:
            last = e
            wait = min(2 ** attempt, 20)
            logger.warning("HTTP %s %s failed (%s); retry in %ss", method, url, e, wait)
            time.sleep(wait)
    raise last  # type: ignore[misc]


def _auth(base: str, email: str, password: str) -> str:
    r = _request_with_retry(
        "POST",
        f"{base}/auth/signup",
        json={"email": email, "password": password},
        timeout=60,
    )
    if r.status_code == 400 and "already" in r.text.lower():
        r = _request_with_retry(
            "POST",
            f"{base}/auth/login",
            json={"email": email, "password": password},
            timeout=60,
        )
    r.raise_for_status()
    token = r.json()["access_token"]
    logger.info("Authenticated as %s", email)
    return token


def _headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _create_chat(base: str, token: str, title: str) -> int:
    r = _request_with_retry(
        "POST",
        f"{base}/chats",
        headers=_headers(token),
        json={"title": title},
        timeout=60,
    )
    r.raise_for_status()
    chat_id = r.json()["id"]
    logger.info("Created chat session_id=%s", chat_id)
    return chat_id


def _upload(base: str, token: str, session_id: int, path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        r = _request_with_retry(
            "POST",
            f"{base}/upload",
            headers=_headers(token),
            data={"session_id": str(session_id)},
            files={"file": (path.name, f, "text/markdown")},
            timeout=180,
        )
    if r.status_code >= 400:
        logger.error("Upload failed: %s %s", r.status_code, r.text)
    r.raise_for_status()
    body = r.json()
    artifact_id = body.get("artifact_id") or body.get("id")
    job_id = body.get("job_id")
    logger.info("Uploaded artifact_id=%s job_id=%s", artifact_id, job_id)
    return int(artifact_id), int(job_id) if job_id is not None else int(artifact_id)


def _wait_job(base: str, token: str, job_id: int, timeout_s: int = 900) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = _request_with_retry(
            "GET",
            f"{base}/jobs/{job_id}",
            headers=_headers(token),
            timeout=60,
        )
        r.raise_for_status()
        body = r.json()
        status = (body.get("status") or "").lower()
        logger.info("Job %s status=%s", job_id, status)
        if status in ("completed", "failed", "dead_letter"):
            return body
        time.sleep(3)
    raise TimeoutError(f"Job {job_id} not finished in {timeout_s}s")


def _normalize_contexts(context_used) -> list[str]:
    out = []
    for c in context_used or []:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, dict):
            out.append(c.get("text") or c.get("content") or str(c))
        else:
            out.append(str(c))
    return [x for x in out if x and str(x).strip()]


def _query(base: str, token: str, session_id: int, question: str, top_k: int = 5) -> dict:
    r = _request_with_retry(
        "POST",
        f"{base}/query",
        headers=_headers(token),
        json={
            "query": question,
            "session_id": session_id,
            "top_k": top_k,
            "evaluate": False,
        },
        timeout=300,
    )
    if r.status_code >= 400:
        logger.error("Query failed: %s %s", r.status_code, r.text)
    r.raise_for_status()
    return r.json()


def _run_direct(path: Path, top_k: int) -> tuple[int, int, list[dict]]:
    """Bypass HTTP: create user/session/artifact in DB and ingest + query in-process."""
    from backend.app.core.security import hash_password
    from backend.app.db.database import SessionLocal
    from backend.app.db.models import (
        Artifact,
        IngestionJob,
        ProcessingStatus,
        Session as ChatSession,
        User,
    )
    from backend.app.pipelines.ingestion_pipeline import MultimodalIngestionPipeline
    from backend.app.pipelines.query_pipeline import QueryPipeline
    from backend.app.retrieval.bm25_store import BM25Store
    from backend.app.storage.minio_client import minio_client

    db = SessionLocal()
    try:
        email = "nlp.eval.direct@aegis.local"
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(
                email=email,
                username="nlpevaldirect",
                password_hash=hash_password("nlp-eval-pass-123"),
            )
            db.add(user)
            db.commit()
            db.refresh(user)

        chat = ChatSession(user_id=user.id, title="NLP endsem notes eval (direct)")
        db.add(chat)
        db.commit()
        db.refresh(chat)

        minio_path = minio_client.upload_file(
            object_name=path.name,
            file_path=str(path),
            content_type="text/markdown",
        )

        art = Artifact(
            filename=path.name,
            file_path=minio_path,
            modality="document",
            user_id=user.id,
            session_id=chat.id,
            upload_status=ProcessingStatus.COMPLETED,
            processing_status=ProcessingStatus.QUEUED,
        )
        db.add(art)
        db.flush()
        job = IngestionJob(artifact_id=art.id, status=ProcessingStatus.QUEUED)
        db.add(job)
        db.commit()
        db.refresh(art)
        db.refresh(job)
        artifact_id, session_id, job_id = art.id, chat.id, job.id
        logger.info(
            "Direct setup user=%s session=%s artifact=%s job=%s minio=%s",
            user.id,
            session_id,
            artifact_id,
            job_id,
            minio_path,
        )
    finally:
        db.close()

    pipeline = MultimodalIngestionPipeline()
    pipeline.process_artifact(artifact_id)
    db = SessionLocal()
    try:
        job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
        if job:
            job.status = ProcessingStatus.COMPLETED
            if job.artifact:
                job.artifact.processing_status = ProcessingStatus.COMPLETED
            db.commit()
        BM25Store().rebuild_from_postgres(db)
    finally:
        db.close()
    logger.info("Direct ingestion completed for artifact %s", artifact_id)

    qp = QueryPipeline()
    samples = []
    for i, qa in enumerate(NLP_QA, 1):
        q = qa["question"]
        logger.info("[%s/%s] %s", i, len(NLP_QA), q)
        result = qp.answer(q, top_k=top_k, evaluate=False, artifact_ids=[artifact_id])
        contexts = _normalize_contexts(result.get("context_used"))
        samples.append(
            {
                "question": q,
                "answer": result.get("answer") or "",
                "contexts": contexts,
                "ground_truth": qa["ground_truth"],
                "metadata": {
                    "session_id": session_id,
                    "artifact_id": artifact_id,
                    "confidence": result.get("confidence"),
                    "grounded": result.get("grounded"),
                    "n_contexts": len(contexts),
                    "run_id": result.get("run_id"),
                    "mode": "direct",
                },
            }
        )
        logger.info(
            "  contexts=%s confidence=%s answer[:120]=%r",
            len(contexts),
            result.get("confidence"),
            (result.get("answer") or "")[:120],
        )
    return session_id, artifact_id, samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to the document to ingest and evaluate (md/pdf/…)",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--email", default="nlp.eval@aegis.local")
    parser.add_argument("--password", default="nlp-eval-pass-123")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--direct",
        action="store_true",
        help="Skip HTTP/Celery — create user/chat in DB and ingest+query in-process",
    )
    args = parser.parse_args()

    if not args.file.exists():
        logger.error("File missing: %s", args.file)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.direct:
        chat_id, artifact_id, samples = _run_direct(args.file, args.top_k)
    else:
        base = args.base_url.rstrip("/")
        token = _auth(base, args.email, args.password)
        chat_id = _create_chat(base, token, "NLP endsem notes eval")
        artifact_id, job_id = _upload(base, token, chat_id, args.file)
        job = _wait_job(base, token, job_id)
        if (job.get("status") or "").lower() != "completed":
            logger.error("Ingestion failed: %s", job)
            return 1

        samples = []
        for i, qa in enumerate(NLP_QA, 1):
            q = qa["question"]
            logger.info("[%s/%s] %s", i, len(NLP_QA), q)
            result = _query(base, token, chat_id, q, top_k=args.top_k)
            contexts = _normalize_contexts(result.get("context_used"))
            samples.append(
                {
                    "question": q,
                    "answer": result.get("answer") or "",
                    "contexts": contexts,
                    "ground_truth": qa["ground_truth"],
                    "metadata": {
                        "session_id": chat_id,
                        "artifact_id": artifact_id,
                        "confidence": result.get("confidence"),
                        "grounded": result.get("grounded"),
                        "n_contexts": len(contexts),
                        "run_id": result.get("run_id"),
                        "mode": "http",
                    },
                }
            )
            logger.info(
                "  contexts=%s confidence=%s answer[:120]=%r",
                len(contexts),
                result.get("confidence"),
                (result.get("answer") or "")[:120],
            )

    samples_path = OUT_DIR / "samples_nlp5_scoped.json"
    samples_path.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Wrote samples → %s", samples_path)

    from backend.app.evaluation.eval_dataset import EvalDataset
    from backend.app.evaluation.eval_runner import EvalRunner
    from backend.app.evaluation.ragas_evaluator import AegisEvaluator

    ds = EvalDataset(name="nlp_endsem_scoped_5")
    ds.add_batch(samples)
    runner = EvalRunner()
    runner.evaluator = AegisEvaluator(provider="ollama")
    if runner.evaluator._llm is None:
        logger.error("Ollama judge unavailable — samples saved, skipping RAGAS")
        return 2

    report = runner.run_from_dataset(ds, output_dir=str(OUT_DIR))
    summary = report["summary"]
    summary.update(
        {
            "mode": "scoped_live",
            "judge": runner.evaluator.judge_name,
            "session_id": chat_id,
            "artifact_id": artifact_id,
            "source_file": str(args.file),
            "samples": len(samples),
        }
    )
    (OUT_DIR / "summary_nlp5_scoped.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "latest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Summary:\n%s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
