"""
Custom Ingestion Exceptions for AEGIS Multimodal Pipeline.

Distinguishes between:
  - NonRetryableIngestionError: Corrupt file, malformed format, unsupported MIME type.
    Fails immediately without retrying.
  - RetryableIngestionError: Rate limit (429), API timeout, 503 service unavailable.
    Triggers Celery exponential backoff retries.
"""

class IngestionError(Exception):
    """Base exception for all ingestion pipeline errors."""
    def __init__(self, message: str, stage: str = "UNKNOWN"):
        super().__init__(message)
        self.message = message
        self.stage = stage


class NonRetryableIngestionError(IngestionError):
    """Errors that should NOT be retried (e.g. malformed inputs)."""
    pass


class RetryableIngestionError(IngestionError):
    """Transient errors that SHOULD be retried (e.g. Groq 429/5xx)."""
    pass
