import traceback
import logging
from datetime import datetime, timezone, timedelta
from backend.app.worker import celery_app
from backend.app.db.database import SessionLocal
from backend.app.db.models import IngestionJob, ProcessingStatus
from backend.app.core.exceptions import NonRetryableIngestionError, RetryableIngestionError
from backend.app.pipelines.ingestion_pipeline import MultimodalIngestionPipeline

logger = logging.getLogger(__name__)


def _update_job_status(job_id: int, status: ProcessingStatus, stage: str = None, error: str = None, tb: str = None, retryable: bool = True):
    """Update job status in the database with one retry on failure."""
    for attempt in range(2):
        db = SessionLocal()
        try:
            job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
            if job:
                job.status = status
                job.last_heartbeat = datetime.now(timezone.utc)
                if stage:
                    job.failed_stage = stage
                if error:
                    job.error_message = str(error)
                if tb:
                    job.traceback = tb
                job.is_retryable = 1 if retryable else 0
                if job.artifact:
                    job.artifact.processing_status = status
                db.commit()
            return  # Success — exit retry loop
        except Exception as e:
            db.rollback()
            if attempt == 0:
                logger.warning(f"Failed to update job {job_id} status to {status} (retrying): {e}")
            else:
                logger.error(f"Failed to update job {job_id} status to {status} after retry: {e}")
        finally:
            db.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_ingestion_task(self, job_id: int):
    """
    Main Celery ingestion orchestrator with fine-grained stage tracking,
    exception classification, and dead-letter routing.
    """
    _update_job_status(job_id, ProcessingStatus.PARSING, stage="PARSING")

    db = SessionLocal()
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
    if not job:
        db.close()
        return

    artifact_id = job.artifact_id
    db.close()

    pipeline = MultimodalIngestionPipeline()

    try:
        # Pipeline execution with stage callbacks
        pipeline.process_artifact(
            artifact_id=artifact_id,
            on_stage_change=lambda stage: _update_job_status(job_id, getattr(ProcessingStatus, stage, ProcessingStatus.RUNNING), stage=stage)
        )
        _update_job_status(job_id, ProcessingStatus.COMPLETED)

        # Rebuild BM25 index on disk so the query pipeline picks up newly ingested document chunks
        try:
            from backend.app.retrieval.bm25_store import BM25Store
            db_bm25 = SessionLocal()
            try:
                BM25Store().rebuild_from_postgres(db_bm25)
            finally:
                db_bm25.close()
        except Exception as bm25_err:
            logger.warning(f"Could not auto-rebuild BM25 index: {bm25_err}")


    except NonRetryableIngestionError as exc:
        # Non-retryable error (e.g. malformed input, unsupported format) -> fail immediately
        tb_str = traceback.format_exc()
        logger.error(f"Non-retryable ingestion error at stage {exc.stage}: {exc}")
        _update_job_status(job_id, ProcessingStatus.FAILED, stage=exc.stage, error=str(exc), tb=tb_str, retryable=False)

    except RetryableIngestionError as exc:
        tb_str = traceback.format_exc()
        current_retries = self.request.retries
        stage_name = exc.stage

        if current_retries >= self.max_retries:
            logger.error(f"Job {job_id} exhausted max retries at stage {stage_name}. Dead-lettering: {exc}")
            _update_job_status(job_id, ProcessingStatus.DEAD_LETTER, stage=stage_name, error=f"Max retries reached: {str(exc)}", tb=tb_str, retryable=True)
        else:
            logger.warning(f"Retryable error at stage {stage_name} (retry {current_retries + 1}/{self.max_retries}): {exc}")
            _update_job_status(job_id, ProcessingStatus.RUNNING, stage=stage_name, error=str(exc), tb=tb_str, retryable=True)
            # Exponential backoff retry
            raise self.retry(exc=exc, countdown=2 ** current_retries * 5)

    except Exception as exc:
        # Unexpected error — only retry if it looks transient (connection/timeout)
        tb_str = traceback.format_exc()
        current_retries = self.request.retries
        exc_str = str(exc).lower()
        is_transient = any(kw in exc_str for kw in ["connection", "timeout", "unavailable", "503", "reset"])

        if is_transient and current_retries < self.max_retries:
            logger.warning(f"Transient error (retry {current_retries + 1}/{self.max_retries}): {exc}")
            _update_job_status(job_id, ProcessingStatus.RUNNING, stage="RUNNING", error=str(exc), tb=tb_str, retryable=True)
            raise self.retry(exc=exc, countdown=2 ** current_retries * 5)
        else:
            logger.error(f"Non-retryable unexpected error for job {job_id}: {exc}")
            _update_job_status(job_id, ProcessingStatus.FAILED, stage="RUNNING", error=str(exc), tb=tb_str, retryable=False)


@celery_app.task
def cleanup_stuck_jobs(timeout_minutes: int = 15):
    """
    Heartbeat monitor: detects jobs stuck in RUNNING/PARSING/VISION/CHUNKING/STORING
    without heartbeat updates for > timeout_minutes and transitions them to DEAD_LETTER.
    """
    db = SessionLocal()
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)
        active_statuses = [
            ProcessingStatus.RUNNING,
            ProcessingStatus.PARSING,
            ProcessingStatus.VISION_CAPTIONING,
            ProcessingStatus.CHUNKING_EMBEDDING,
            ProcessingStatus.STORING,
        ]
        stuck_jobs = db.query(IngestionJob).filter(
            IngestionJob.status.in_(active_statuses),
            IngestionJob.last_heartbeat < cutoff
        ).all()

        for job in stuck_jobs:
            logger.error(f"Stuck job detected: ID {job.id} last heartbeat at {job.last_heartbeat}. Dead-lettering.")
            job.status = ProcessingStatus.DEAD_LETTER
            job.error_message = f"Stuck job execution timeout (no heartbeat for > {timeout_minutes} mins)"
            if job.artifact:
                job.artifact.processing_status = ProcessingStatus.DEAD_LETTER
        db.commit()
    except Exception as e:
        logger.error(f"Error during stuck job cleanup: {e}")
        db.rollback()
    finally:
        db.close()
