"""Tests for TextCollection and MultimodalCollection (dual vector store)."""

import uuid
import pytest

from backend.app.vectorstore.text_collection import TextCollection
from backend.app.vectorstore.multimodal_collection import MultimodalCollection


# ── TextCollection (all-MiniLM-L6-v2 · 384-dim) ───────────────────────────────

class TestTextCollection:

    def setup_method(self):
        self.col = TextCollection()
        # Use isolated collection name for tests
        self.col.collection_name = "test_text_collection"

    def teardown_method(self):
        self.col.delete()

    def test_add_and_query(self):
        doc_id = str(uuid.uuid4())
        self.col.add_documents(
            documents=["OmniMind is a multi-modal RAG engine."],
            ids=[doc_id],
            metadata=[{"source": "unit_test", "modality": "document"}],
        )
        result = self.col.semantic_query("What is OmniMind?", n_results=1)
        assert result["documents"] is not None
        assert len(result["documents"][0]) > 0

    def test_get_all_documents(self):
        doc_id = str(uuid.uuid4())
        self.col.add_documents(
            documents=["Test document for BM25 rebuild."],
            ids=[doc_id],
            metadata=[{"source": "unit_test", "modality": "document"}],
        )
        all_docs = self.col.get_all_documents()
        assert "documents" in all_docs
        assert len(all_docs["documents"]) > 0

    def test_count(self):
        doc_id = str(uuid.uuid4())
        self.col.add_documents(
            documents=["Count test document."],
            ids=[doc_id],
            metadata=[{"source": "unit_test", "modality": "document"}],
        )
        assert self.col.count() >= 1

    def test_delete(self):
        self.col.delete()
        assert self.col.count() == 0


# ── MultimodalCollection (clip-ViT-B-32 · 512-dim) ────────────────────────────

class TestMultimodalCollection:

    def setup_method(self):
        self.col = MultimodalCollection()
        self.col.collection_name = "test_multimodal_collection"

    def teardown_method(self):
        self.col.delete()

    def test_add_image_and_query(self):
        doc_id = str(uuid.uuid4())
        self.col.add_image(
            description="A bar chart showing quarterly revenue growth.",
            doc_id=doc_id,
            metadata={"source": "chart.png", "modality": "image"},
        )
        result = self.col.query_by_text("revenue chart", n_results=1)
        assert result["documents"] is not None
        assert len(result["documents"][0]) > 0

    def test_add_video_segments(self):
        ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        self.col.add_video_segments(
            documents=["Speaker introduces product at t=0.", "Demo of feature X at t=10."],
            ids=ids,
            metadata=[
                {"source": "demo.mp4", "modality": "video", "start_time": 0, "end_time": 10},
                {"source": "demo.mp4", "modality": "video", "start_time": 10, "end_time": 20},
            ],
        )
        assert self.col.count() == 2

    def test_query_time_range(self):
        ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        self.col.add_video_segments(
            documents=["Opening segment.", "Middle segment."],
            ids=ids,
            metadata=[
                {"source": "video.mp4", "modality": "video", "start_time": 0,  "end_time": 30},
                {"source": "video.mp4", "modality": "video", "start_time": 60, "end_time": 90},
            ],
        )
        results = self.col.query_time_range(0, 30)
        assert len(results) == 1
        assert "Opening segment" in results[0]