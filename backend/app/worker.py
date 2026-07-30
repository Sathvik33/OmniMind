from celery import Celery
from dotenv import load_dotenv

load_dotenv()

from kombu import Queue
from backend.app.core.config import REDIS_BROKER_URL

if not REDIS_BROKER_URL:
    raise ValueError("CELERY_BROKER_URL environment variable is not set")

celery_app = Celery(
    "worker",
    broker=REDIS_BROKER_URL,
    backend=REDIS_BROKER_URL,
    include=["backend.app.tasks.ingestion_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_default_queue="ingestion_queue",
    task_queues=(
        Queue("ingestion_queue", routing_key="ingestion.#"),
    ),
    task_routes={
        "backend.app.tasks.ingestion_tasks.process_ingestion_task": {"queue": "ingestion_queue", "routing_key": "ingestion.task"},
        "backend.app.tasks.ingestion_tasks.cleanup_stuck_jobs": {"queue": "ingestion_queue", "routing_key": "ingestion.cleanup"},
    },
)

