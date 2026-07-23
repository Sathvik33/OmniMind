import os
from celery import Celery
from dotenv import load_dotenv

load_dotenv()

# Bypass SSL proxy for Hugging Face downloads
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

REDIS_URL = os.getenv("CELERY_BROKER_URL")
if not REDIS_URL:
    raise ValueError("CELERY_BROKER_URL environment variable is not set")

celery_app = Celery(
    "worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["backend.app.tasks.ingestion_tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)
