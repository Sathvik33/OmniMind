from unittest.mock import patch

import pytest

from backend.app.services.groq_retry import (
    call_with_retry,
    is_rate_limit_error,
    should_use_fallback,
)


class TestGroqRetry:
    def test_detects_429(self):
        assert is_rate_limit_error(Exception("Error 429 rate_limit_exceeded"))

    def test_fallback_on_404_and_rate_limit(self):
        assert should_use_fallback(Exception("404 model_not_found"))
        assert should_use_fallback(Exception("429 Too Many Requests"))
        assert should_use_fallback(RuntimeError("Groq stream returned no tokens"))
        assert not should_use_fallback(ValueError("prompt too long for schema"))

    def test_retries_then_succeeds(self):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise Exception("429 Too Many Requests")
            return "ok"

        with patch("backend.app.services.groq_retry.time.sleep"):
            assert call_with_retry(flaky, max_attempts=4, label="test") == "ok"
        assert calls["n"] == 3

    def test_non_transient_raises_immediately(self):
        def boom():
            raise ValueError("invalid image")

        with pytest.raises(ValueError):
            call_with_retry(boom, max_attempts=4)
