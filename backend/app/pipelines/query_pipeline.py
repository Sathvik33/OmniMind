"""
QueryPipeline — AEGIS v3.0 RAG orchestrator with full monitoring.

Wraps the LangGraph workflow and integrates:
  - AegisTracer for LangSmith monitoring
  - RAGAS evaluation (async background or on-demand)
  - Structured response with run_id, confidence, latency

Streaming approach:
  For /query-stream, we run guard → retrieve → context synchronously,
  then stream generation tokens directly.
"""

from typing import Dict, Any, Generator as StreamGenerator


from backend.app.retrieval.bm25_store import BM25Store
from backend.app.retrieval.reranker import CrossEncoderReranker
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.rag.context_builder import ContextBuilder
from backend.app.rag.generator import Generator as LLMGenerator
# from backend.app.models.groq_model import GroqModel
from backend.app.models.ollama_model import OllamaModel
from backend.app.guardrails.input_guard import InputGuard
from backend.app.guardrails.output_guard import OutputGuard
from backend.app.workflow.graph import build_rag_graph
from backend.app.workflow.nodes import _detect_time
from backend.app.monitoring.langsmith_logger import tracer as aegis_tracer


class QueryPipeline:
    """
    Main AEGIS RAG query orchestrator.
    Constructed once at app startup; all components are reused per-request.
    """

    def __init__(self, bm25_store: BM25Store | None = None):
        # ── Core components ───────────────────────────────────────────────────
        self.collection_manager = None
        self.bm25 = bm25_store or BM25Store()
        self.reranker = CrossEncoderReranker()
        self.hybrid_retriever = HybridRetriever(
            bm25_store=self.bm25,
            reranker=self.reranker,
        )
        self.context_builder = ContextBuilder()
        # self.generator = LLMGenerator(GroqModel())
        self.generator = LLMGenerator(OllamaModel(model_name="qwen2.5:7b"))

        # ── Compile LangGraph ─────────────────────────────────────────────────
        self._graph = build_rag_graph({
            "collection_manager": self.collection_manager,
            "hybrid_retriever":   self.hybrid_retriever,
            "reranker":           self.reranker,
            "generator":          self.generator,
        })

    # ── Blocking query (POST /query) ──────────────────────────────────────────

    def answer(self, query: str, top_k: int = 3, evaluate: bool = False) -> Dict[str, Any]:
        """
        Run full query pipeline synchronously.

        Args:
            query:    User question
            top_k:    Number of results to retrieve
            evaluate: If True, run inline RAGAS evaluation (adds latency)

        Returns structured response with answer, context, confidence, run_id.
        """
        # 1. Input validation
        input_guard = InputGuard.validate(query)
        if not input_guard.get("ok"):
            return {
                "answer":        input_guard.get("reason", "Query validation failed."),
                "context_used":  [],
                "warnings":      [input_guard.get("reason")],
                "confidence":    0.0,
                "grounded":      False,
                "has_hallucination": False,
                "error":         True,
                "run_id":        None,
            }

        # 2. Run full LangGraph workflow
        try:
            final_state = self._graph.invoke({"query": query})
        except Exception as e:
            return {
                "answer":        f"Error processing query: {str(e)}",
                "context_used":  [],
                "warnings":      [f"Pipeline error: {str(e)}"],
                "confidence":    0.0,
                "grounded":      False,
                "has_hallucination": False,
                "error":         True,
                "run_id":        None,
            }

        # 3. Extract result from state
        final_answer  = final_state.get("final_answer", "")
        context_used  = final_state.get("reranked") or final_state.get("candidates", [])
        context_used  = context_used if isinstance(context_used, list) else []
        run_id        = final_state.get("run_id")
        latency_ms    = final_state.get("latency_ms", {})

        response = {
            "answer":            final_answer,
            "context_used":      context_used,
            "warnings":          final_state.get("warnings", []),
            "confidence":        final_state.get("confidence", 0.5),
            "grounded":          final_state.get("grounded", False),
            "has_hallucination": final_state.get("has_hallucination", False),
            "retrieval_metadata": final_state.get("retrieval_metadata", {}),
            "latency_ms":        latency_ms,
            "run_id":            run_id,
            "error":             False,
        }

        # 4. Optional inline RAGAS evaluation
        if evaluate:
            try:
                from backend.app.evaluation.ragas_evaluator import AegisEvaluator
                evaluator = AegisEvaluator()
                eval_result = evaluator.evaluate_single(
                    query=query,
                    answer=final_answer,
                    contexts=context_used if isinstance(context_used[0] if context_used else "", str) else [],
                    run_id=run_id,
                )
                response["eval_scores"] = eval_result.to_dict()
            except Exception:
                pass   # Evaluation failure must not break the main response

        return response

    # ── Streaming query (POST /query-stream) ──────────────────────────────────

    def stream_answer(self, query: str) -> StreamGenerator[str, None, None]:
        """
        Run RAG pipeline with streaming generation and guardrails.
        """
        # 1. Input guard
        guard = InputGuard.validate(query)
        if not guard.get("ok"):
            yield f"⚠️ {guard.get('reason', 'Query validation failed.')}\n"
            return

        if guard.get("warnings"):
            for warning in guard["warnings"]:
                yield f"ℹ️ {warning}\n"

        # 2. Start LangSmith trace
        run_id = aegis_tracer.start_run(query=query)

        # 3. Time-based vs semantic retrieval
        time_range = _detect_time(query)

        try:
            if time_range:
                segments = self.collection_manager.query_time_range(time_range["start"], time_range["end"])
                if not segments:
                    yield "No video content found for that time range."
                    return
                context = "\n\n".join(segments)
            else:
                retrieved = self.hybrid_retriever.retrieve_with_confidence(query)
                if not retrieved:
                    yield "No relevant context found in your uploaded data."
                    return

                # Log retrieval confidence headers
                for i, result in enumerate(retrieved[:3], 1):
                    confidence = result.get("relevance_score", 0.0)
                    yield f"[Retrieved: {result.get('source', 'unknown')} ({confidence:.1%} confidence)]\n"

                chunks = [r["text"] for r in retrieved]
                context = self.context_builder.build(chunks)

                aegis_tracer.log_retrieval(
                    run_id=run_id,
                    bm25_hits=len([r for r in retrieved if r.get("source") == "text"]),
                    dense_hits=len(retrieved),
                    fused_count=len(chunks),
                )

        except Exception as e:
            yield f"❌ Retrieval error: {str(e)}\n"
            return

        # 4. Stream generation tokens
        full_answer = ""
        try:
            for token in self.generator.stream_generate(query, context):
                full_answer += token
                yield token
        except Exception as e:
            yield f"\n\n❌ Generation error: {str(e)}\n"
            return

        # 5. Output guard
        output_validation = OutputGuard.validate(full_answer, context)
        for warning in output_validation.get("warnings", []):
            yield f"\n\n⚠️ {warning}"

        confidence = output_validation.get("confidence", 0.5)
        yield f"\n\n[Response confidence: {confidence:.1%}]"

        # 6. End LangSmith trace
        aegis_tracer.log_generation(run_id=run_id, answer=full_answer)
        aegis_tracer.log_guardrails(
            run_id=run_id,
            input_ok=True,
            output_ok=output_validation.get("ok", True),
            confidence=confidence,
            grounded=output_validation.get("grounded", False),
            has_hallucination=output_validation.get("has_hallucination", False),
        )
        aegis_tracer.end_run(run_id=run_id, final_answer=full_answer)

    # ── Monitoring helpers ────────────────────────────────────────────────────

    def get_retriever_stats(self) -> Dict[str, Any]:
        """Get current retriever statistics for monitoring."""
        return self.hybrid_retriever.get_stats()

    def get_monitor_stats(self) -> Dict[str, Any]:
        """Get LangSmith monitoring statistics."""
        return aegis_tracer.get_run_stats()