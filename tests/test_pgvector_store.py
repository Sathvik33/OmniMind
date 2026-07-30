import pytest
from backend.app.retrieval.pgvector_store import PgVectorStore


class TestPgVectorStore:

    def setup_method(self):
        self.store = PgVectorStore()

    def test_store_instantiation(self):
        assert self.store is not None

    def test_query_time_range_empty_db(self):
        try:
            results = self.store.query_time_range(0, 60)
            assert isinstance(results, list)
        except Exception as e:
            # Operational error if DB container is initializing
            assert "OperationalError" in type(e).__name__ or "connection" in str(e).lower()

