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

from typing import Dict, Any, Generator as StreamGenerator, List, Optional


from backend.app.retrieval.bm25_store import BM25Store
from backend.app.retrieval.reranker import CrossEncoderReranker
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.rag.context_builder import ContextBuilder
from backend.app.rag.generator import Generator as LLMGenerator
from backend.app.models.failover_llm import FailoverLLM
from backend.app.core.config import USE_LOCAL_LLM, OLLAMA_MODEL
from backend.app.guardrails.input_guard import InputGuard
from backend.app.guardrails.output_guard import OutputGuard
from backend.app.workflow.graph import build_rag_graph
from backend.app.workflow.nodes import _detect_time
from backend.app.monitoring.langsmith_logger import tracer as aegis_tracer
from backend.app.db.database import SessionLocal
from backend.app.services.query_scope import ensure_query_ready, resolve_artifact_ids


class QueryPipeline:
    """
    Main AEGIS RAG query orchestrator.
    Constructed once at app startup; all components are reused per-request.
    """

    def __init__(self, bm25_store: BM25Store | None = None):
        # ── Core components ───────────────────────────────────────────────────
        self.bm25 = bm25_store or BM25Store()
        self.reranker = CrossEncoderReranker()
        self.hybrid_retriever = HybridRetriever(
            bm25_store=self.bm25,
            reranker=self.reranker,
        )
        self.collection_manager = self.hybrid_retriever.pg_store
        self.context_builder = ContextBuilder()
        
        # Dev default: local Qwen → Groq → OpenRouter free (FailoverLLM)
        # USE_LOCAL_LLM=false skips Ollama (demos / low RAM): Groq → OpenRouter
        self.generator = LLMGenerator(FailoverLLM(prefer_local=USE_LOCAL_LLM))
        if USE_LOCAL_LLM:
            import logging

            logging.getLogger(__name__).info(
                "LLM priority: local %s → Groq → OpenRouter free", OLLAMA_MODEL
            )

        # ── Evaluation (lazy singleton) ───────────────────────────────────────
        self._evaluator = None



        # ── Compile LangGraph ─────────────────────────────────────────────────
        self._graph = build_rag_graph({
            "collection_manager": self.collection_manager,
            "hybrid_retriever":   self.hybrid_retriever,
            "reranker":           self.reranker,
            "generator":          self.generator,
        })


    # ── Blocking query (POST /query) ──────────────────────────────────────────

    def answer(
        self,
        query: str,
        top_k: int = 3,
        evaluate: bool = False,
        artifact_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Run full query pipeline synchronously.

        Answers only from ready embeddings for the scoped artifact_ids
        (or the latest completed upload when none are provided).
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

        db = SessionLocal()
        try:
            scoped_ids = resolve_artifact_ids(db, artifact_ids)
            blocked = ensure_query_ready(db, scoped_ids)
        finally:
            db.close()

        if blocked:
            return {
                "answer":        blocked,
                "context_used":  [],
                "warnings":      [blocked],
                "confidence":    0.0,
                "grounded":      False,
                "has_hallucination": False,
                "error":         True,
                "run_id":        None,
            }

        # 2. Run full LangGraph workflow (retrieval scoped to ready embeddings)
        try:
            final_state = self._graph.invoke({
                "query": query,
                "artifact_ids": scoped_ids,
            })
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
        structured    = final_state.get("structured_answer")

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
            "structured_answer": structured,
        }

        # 4. Optional inline RAGAS evaluation
        if evaluate:
            try:
                if self._evaluator is None:
                    from backend.app.evaluation.ragas_evaluator import AegisEvaluator
                    self._evaluator = AegisEvaluator()
                raw_contexts = context_used if isinstance(context_used, list) else []
                string_contexts = [
                    item if isinstance(item, str) else item.get("text", str(item))
                    for item in raw_contexts
                ]
                eval_result = self._evaluator.evaluate_single(
                    query=query,
                    answer=final_answer,
                    contexts=string_contexts,
                    run_id=run_id,
                )

                response["eval_scores"] = eval_result.to_dict()
            except Exception:
                pass   # Evaluation failure must not break the main response

        return response

    # ── Streaming query (POST /query-stream) ──────────────────────────────────

    def stream_answer(
        self,
        query: str,
        artifact_ids: Optional[List[int]] = None,
    ) -> StreamGenerator[str, None, None]:
        """
        Run RAG pipeline with streaming generation and guardrails.
        Retrieval runs only after embeddings are ready, and only for scoped uploads.
        """
        # 1. Input guard
        guard = InputGuard.validate(query)
        if not guard.get("ok"):
            yield f"⚠️ {guard.get('reason', 'Query validation failed.')}\n"
            return

        if guard.get("warnings"):
            for warning in guard["warnings"]:
                yield f"ℹ️ {warning}\n"

        db = SessionLocal()
        try:
            scoped_ids = resolve_artifact_ids(db, artifact_ids)
            blocked = ensure_query_ready(db, scoped_ids)
        finally:
            db.close()

        if blocked:
            yield f"⚠️ {blocked}\n"
            return

        # 2. Start LangSmith trace — always closed in finally
        run_id = aegis_tracer.start_run(query=query)
        full_answer = ""
        run_error: Optional[str] = None

        # Keep proxies/ngrok from buffering an empty body during slow retrieval
        yield "Working on your question…\n\n"

        # 3. Time-based vs semantic retrieval (scoped to ready embeddings only)
        time_range = _detect_time(query)
        gen_query = query

        try:
            if time_range:
                focus = time_range.get("focus")
                if focus is not None:
                    gen_query = (
                        f"{query}\n\n"
                        f"(Focus on what is visible or happening around t={focus}s. "
                        f"Context may cover nearby seconds.)"
                    )
                segments = self.collection_manager.query_time_range(
                    time_range["start"],
                    time_range["end"],
                    artifact_ids=scoped_ids,
                )
                if not segments:
                    # Fall back to semantic retrieval if the time window is empty
                    retrieved = self.hybrid_retriever.retrieve_with_confidence(
                        query, artifact_ids=scoped_ids, run_id=run_id
                    )
                    if not retrieved:
                        msg = "No video content found for that time range in your ready uploads."
                        run_error = msg
                        yield msg
                        return
                    chunks = [r["text"] for r in retrieved]
                    context = self.context_builder.build(chunks)
                else:
                    context = "\n\n".join(segments)
            else:
                # LangSmith children: bm25_retrieve, dense_retrieve, hybrid_retrieval, rerank
                retrieved = self.hybrid_retriever.retrieve_with_confidence(
                    query, artifact_ids=scoped_ids, run_id=run_id
                )
                if not retrieved:
                    msg = "No relevant context found in your newly embedded uploads."
                    run_error = msg
                    yield msg
                    return

                # Log retrieval confidence headers
                for i, result in enumerate(retrieved[:3], 1):
                    confidence = result.get("relevance_score", 0.0)
                    yield f"[Retrieved: {result.get('source', 'unknown')} ({confidence:.1%} confidence)]\n"

                chunks = [r["text"] for r in retrieved]
                context = self.context_builder.build(chunks)

            # 4. Stream generation tokens
            try:
                for token in self.generator.stream_generate(gen_query, context):
                    full_answer += token
                    yield token
            except Exception as e:
                run_error = f"Generation error: {e}"
                yield f"\n\n❌ Generation error: {str(e)}\n"
                return

            # 5. Output guard
            output_validation = OutputGuard.validate(full_answer, context)
            for warning in output_validation.get("warnings", []):
                yield f"\n\n⚠️ {warning}"

            confidence = output_validation.get("confidence", 0.5)
            yield f"\n\n[Response confidence: {confidence:.1%}]"

            aegis_tracer.log_generation(run_id=run_id, answer=full_answer)
            aegis_tracer.log_guardrails(
                run_id=run_id,
                input_ok=True,
                output_ok=output_validation.get("ok", True),
                confidence=confidence,
                grounded=output_validation.get("grounded", False),
                has_hallucination=output_validation.get("has_hallucination", False),
            )

        except Exception as e:
            run_error = f"Retrieval error: {e}"
            yield f"❌ Retrieval error: {str(e)}\n"
        finally:
            # Always close LangSmith parent — prevents perpetual "running" traces
            aegis_tracer.end_run(
                run_id=run_id,
                final_answer=full_answer,
                error=run_error,
            )

    # ── Monitoring helpers ────────────────────────────────────────────────────

    def get_retriever_stats(self) -> Dict[str, Any]:
        """Get current retriever statistics for monitoring."""
        return self.hybrid_retriever.get_stats()

    def get_monitor_stats(self) -> Dict[str, Any]:
        """Get LangSmith monitoring statistics."""
        return aegis_tracer.get_run_stats()