"""LangGraph classify / route helpers (offline, no LLM)."""

from backend.app.workflow.nodes import (
    classify_query,
    route_after_guard,
    route_query_type,
    _detect_time,
)


class TestDetectTime:
    def test_between_seconds(self):
        tr = _detect_time("what happens between 10 and 20 seconds")
        assert tr == {"start": 10, "end": 20}

    def test_first_minute(self):
        tr = _detect_time("summarize the first minute")
        assert tr == {"start": 0, "end": 60}

    def test_no_time_returns_none(self):
        assert _detect_time("What skills are listed in the CV?") is None


class TestClassifyAndRoute:
    def test_classify_temporal(self):
        out = classify_query({"query": "describe the scene at 15 seconds", "latency_ms": {}})
        assert out["query_type"] == "temporal"
        assert out["time_range"] is not None

    def test_classify_semantic(self):
        out = classify_query({"query": "list the programming languages", "latency_ms": {}})
        assert out["query_type"] == "semantic"
        assert out["time_range"] is None

    def test_route_query_type(self):
        assert route_query_type({"query_type": "temporal"}) == "temporal"
        assert route_query_type({"query_type": "semantic"}) == "semantic"
        assert route_query_type({}) == "semantic"

    def test_route_after_guard_blocked(self):
        assert route_after_guard({"guard_result": {"ok": False}}) == "blocked"
        assert route_after_guard({"guard_result": {"ok": True}}) == "ok"
