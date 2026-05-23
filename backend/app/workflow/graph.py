"""
LangGraph RAG StateGraph for OmniMind.

Graph topology
──────────────

  guard_input
      │
      ├─[blocked]──► reject ──► END
      │
      └─[ok]──► classify_query
                    │
                    ├─[temporal]──► temporal_retrieve ──► build_context
                    │                                         │
                    └─[semantic]──► hybrid_retrieve ──► rerank ──► build_context
                                                                        │
                                                                    generate
                                                                        │
                                                                  guard_output
                                                                        │
                                                                       END

The graph is compiled once at module import and reused across requests.
All heavy objects (retriever, reranker, generator) are injected via
`build_rag_graph(services)` so they are constructed once at startup.
"""

from langgraph.graph import StateGraph, END

from backend.app.workflow.state import OmniMindState
from backend.app.workflow.nodes import (
    guard_input,
    classify_query,
    build_context,
    guard_output,
    reject,
    route_after_guard,
    route_query_type,
    make_temporal_retrieve,
    make_hybrid_retrieve,
    make_rerank,
    make_generate,
)


def build_rag_graph(services: dict):
    """
    Build and compile the LangGraph RAG workflow.

    Parameters
    ----------
    services : dict with keys:
        "collection_manager" : CollectionManager
        "hybrid_retriever"   : HybridRetriever
        "reranker"           : CrossEncoderReranker
        "generator"          : Generator

    Returns
    -------
    A compiled LangGraph runnable (supports .invoke() and .stream()).
    """
    # Bind dependencies into node closures
    temporal_retrieve = make_temporal_retrieve(services["collection_manager"])
    hybrid_retrieve   = make_hybrid_retrieve(services["hybrid_retriever"])
    rerank            = make_rerank(services["reranker"])
    generate          = make_generate(services["generator"])

    # ── Build graph ────────────────────────────────────────────────────────────
    graph = StateGraph(OmniMindState)

    graph.add_node("guard_input",       guard_input)
    graph.add_node("classify_query",    classify_query)
    graph.add_node("temporal_retrieve", temporal_retrieve)
    graph.add_node("hybrid_retrieve",   hybrid_retrieve)
    graph.add_node("rerank",            rerank)
    graph.add_node("build_context",     build_context)
    graph.add_node("generate",          generate)
    graph.add_node("guard_output",      guard_output)
    graph.add_node("reject",            reject)

    # ── Entry point ────────────────────────────────────────────────────────────
    graph.set_entry_point("guard_input")

    # ── Edges ──────────────────────────────────────────────────────────────────
    graph.add_conditional_edges(
        "guard_input",
        route_after_guard,
        {"ok": "classify_query", "blocked": "reject"},
    )
    graph.add_conditional_edges(
        "classify_query",
        route_query_type,
        {"temporal": "temporal_retrieve", "semantic": "hybrid_retrieve"},
    )

    graph.add_edge("temporal_retrieve", "build_context")
    graph.add_edge("hybrid_retrieve",   "rerank")
    graph.add_edge("rerank",            "build_context")
    graph.add_edge("build_context",     "generate")
    graph.add_edge("generate",          "guard_output")
    graph.add_edge("guard_output",      END)
    graph.add_edge("reject",            END)

    return graph.compile()
