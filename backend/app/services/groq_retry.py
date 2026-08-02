"""Shared Groq 429 / transient-error retry with Retry-After support."""

from __future__ import annotations

import logging
import random
import time
from typing import Callable, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

_RATE_MARKERS = (
    "429",
    "rate_limit",
    "rate limit",
    "too many requests",
    "tokens per minute",
)


def is_rate_limit_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    if any(m in msg for m in _RATE_MARKERS):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status == 429:
        return True
    body = getattr(exc, "body", None) or getattr(exc, "response", None)
    if body is not None and "429" in str(body):
        return True
    return False


def is_transient_error(exc: BaseException) -> bool:
    if is_rate_limit_error(exc):
        return True
    msg = str(exc).lower()
    return any(
        kw in msg
        for kw in (
            "timeout",
            "503",
            "502",
            "unavailable",
            "connection",
            "reset",
            "refused",
            "ollama",
            "out of memory",
            "oom",
            "memory",
            "no such file",
            "model not found",
            "pull model",
        )
    )


def should_use_fallback(exc: BaseException) -> bool:
    """
    Errors that should switch from Groq → OpenRouter (after retries or immediately).
    Includes rate limits, 404/model-not-found, auth/quota, and other provider faults.
    """
    if is_rate_limit_error(exc) or is_transient_error(exc):
        return True
    msg = str(exc).lower()
    markers = (
        "404",
        "not found",
        "model_not_found",
        "does not exist",
        "decommissioned",
        "invalid_api_key",
        "authentication",
        "unauthorized",
        "forbidden",
        "401",
        "403",
        "402",
        "payment",
        "quota",
        "insufficient",
        "over capacity",
        "capacity",
        "no tokens",
        "empty model response",
        "returned no tokens",
    )
    if any(m in msg for m in markers):
        return True
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if status in {401, 402, 403, 404, 408, 429, 500, 502, 503, 529}:
        return True
    return False


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    headers = getattr(exc, "headers", None)
    if headers is None:
        resp = getattr(exc, "response", None)
        headers = getattr(resp, "headers", None) if resp is not None else None
    if not headers:
        return None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
        if raw is None:
            return None
        return float(raw)
    except (TypeError, ValueError):
        return None


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 4,
    base_delay: float = 1.5,
    max_delay: float = 45.0,
    label: str = "groq",
) -> T:
    """
    Call ``fn`` with exponential backoff on 429 / transient errors.
    Honors Retry-After when present. Re-raises the last exception if exhausted.
    """
    last: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if not is_transient_error(exc) or attempt >= max_attempts - 1:
                raise
            retry_after = _retry_after_seconds(exc)
            delay = retry_after if retry_after is not None else min(
                max_delay,
                base_delay * (2 ** attempt) + random.uniform(0, 0.5),
            )
            logger.warning(
                "%s transient error (attempt %s/%s), sleeping %.1fs: %s",
                label,
                attempt + 1,
                max_attempts,
                delay,
                exc,
            )
            time.sleep(delay)
    assert last is not None
    raise last
