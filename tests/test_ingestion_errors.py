"""Ingestion exception taxonomy (pure helper — no MinIO/Celery required)."""

from backend.app.core.exceptions import NonRetryableIngestionError, RetryableIngestionError
from backend.app.db.models import ProcessingStatus
from backend.app.tasks.ingestion_tasks import resolve_ingestion_failure


class TestIngestionErrorTaxonomy:
    def test_non_retryable_fails_immediately(self):
        d = resolve_ingestion_failure(
            NonRetryableIngestionError("bad file", stage="PARSING"),
            retries=0,
            max_retries=3,
        )
        assert d["status"] == ProcessingStatus.FAILED
        assert d["retryable"] is False
        assert d["should_retry"] is False
        assert d["stage"] == "PARSING"

    def test_retryable_schedules_backoff(self):
        d = resolve_ingestion_failure(
            RetryableIngestionError("429 rate limit", stage="VISION_CAPTIONING"),
            retries=0,
            max_retries=3,
        )
        assert d["status"] == ProcessingStatus.RUNNING
        assert d["should_retry"] is True
        assert d["countdown"] == 5

    def test_retryable_exhausted_dead_letters(self):
        d = resolve_ingestion_failure(
            RetryableIngestionError("429 again", stage="VISION_CAPTIONING"),
            retries=3,
            max_retries=3,
        )
        assert d["status"] == ProcessingStatus.DEAD_LETTER
        assert d["should_retry"] is False

    def test_transient_unexpected_retries(self):
        d = resolve_ingestion_failure(
            ConnectionError("connection reset"),
            retries=1,
            max_retries=3,
        )
        assert d["should_retry"] is True
        assert d["countdown"] == 10
