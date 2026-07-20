from backend.app.worker import celery_app
from backend.app.db.database import SessionLocal
from backend.app.db.models import IngestionJob, ProcessingStatus
from backend.app.pipelines.ingestion_pipeline import MultimodalIngestionPipeline

@celery_app.task(bind=True, max_retries=3)
def process_ingestion_task(self, job_id: int):
    """
    Celery task that picks up an ingestion job from the Event Bus.
    """
    db = SessionLocal()
    job = db.query(IngestionJob).filter(IngestionJob.id == job_id).first()
    if not job:
        db.close()
        return

    try:
        job.status = ProcessingStatus.RUNNING
        db.commit()

        pipeline = MultimodalIngestionPipeline()
        pipeline.process_artifact(job.artifact_id)
        
        job.status = ProcessingStatus.COMPLETED
        job.artifact.processing_status = ProcessingStatus.COMPLETED
        db.commit()
    except Exception as exc:
        job.status = ProcessingStatus.FAILED
        job.error_message = str(exc)
        job.retry_count += 1
        db.commit()
        # Retry with exponential backoff
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
    finally:
        db.close()
