"""Critical-path tests for RRF fusion and adaptive BM25/dense weights."""

from unittest.mock import MagicMock, patch

from backend.app.retrieval.hybrid_retriever import (
    HybridRetriever,
    _fuse_results_rrf,
    _rrf_score,
)


class TestRRFFusion:
    def test_rrf_score_decreases_with_rank(self):
        assert _rrf_score(1) > _rrf_score(2) > _rrf_score(10)

    def test_docs_in_both_lists_rank_higher(self):
        bm25 = ["shared", "only_bm25"]
        dense = ["shared", "only_dense"]
        fused = _fuse_results_rrf(bm25, dense, bm25_weight=0.4, dense_weight=0.6)
        assert fused[0][0] == "shared"
        scores = {doc: score for doc, score in fused}
        assert scores["shared"] > scores["only_bm25"]
        assert scores["shared"] > scores["only_dense"]

    def test_single_side_still_scores(self):
        fused = _fuse_results_rrf([], ["a", "b"])
        assert [d for d, _ in fused] == ["a", "b"]
        fused_bm25 = _fuse_results_rrf(["a"], [])
        assert [d for d, _ in fused_bm25] == ["a"]


class TestAdaptiveWeights:
    def setup_method(self):
        self.retriever = HybridRetriever(bm25_store=MagicMock(), reranker=MagicMock())

    def test_quoted_phrase_boosts_bm25(self):
        bm25_w, dense_w = self.retriever._get_adaptive_weights('find "exact phrase"')
        assert bm25_w >= dense_w

    def test_explain_boosts_dense(self):
        bm25_w, dense_w = self.retriever._get_adaptive_weights("why does attention work")
        assert dense_w > bm25_w


class TestHybridRetrieveScoped:
    def test_passes_artifact_ids_to_stores(self):
        bm25 = MagicMock()
        bm25.retriever = object()
        bm25.search.return_value = [
            "[Document: x]\n\n" + ("word " * 40),
        ]
        reranker = MagicMock()
        reranker.rerank.return_value = [(bm25.search.return_value[0], 0.9)]

        retriever = HybridRetriever(bm25_store=bm25, reranker=reranker)
        dense_doc = "[Document: y]\n\n" + ("semantic " * 40)

        with patch.object(retriever, "pg_store") as pg:
            pg.search.side_effect = [
                [{"content": dense_doc, "score": 0.8}],  # text dense
                [],  # image
                [],  # video text
                [],  # video vision
            ]
            out = retriever.retrieve(
                "what is attention",
                top_k=5,
                final_k=3,
                artifact_ids=[42, 43],
                adaptive_weights=False,
            )

        bm25.search.assert_called()
        assert bm25.search.call_args.kwargs.get("artifact_ids") == [42, 43]
        first_dense = pg.search.call_args_list[0]
        assert first_dense.kwargs.get("artifact_ids") == [42, 43]
        assert first_dense.kwargs.get("modality") is None  # scoped → all modalities
        assert isinstance(out, list) and len(out) >= 1
