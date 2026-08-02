"""
Render docs/assets/aegis-system-architecture.png
Style matched to the Aegis reference system-design layout (yellow layers,
blue nodes, parallel modality lanes) — content reflects the real stack.

  python scripts/render_architecture_png.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "aegis-system-architecture.png"

# Match reference aesthetic: soft blue nodes, warm yellow groups, dark edges
BLUE = "#D6EAF8"
BLUE_E = "#2E86AB"
YELLOW = "#FCF3CF"
YELLOW_E = "#B7950B"
INK = "#1C2833"
MUTED = "#566573"
BG = "#FBFCFA"
WHITE = "#FFFFFF"
ACCENT = "#1A5276"
GREEN = "#D5F5E3"
GREEN_E = "#1E8449"
PINK = "#FADBD8"
PINK_E = "#922B21"
ORANGE = "#FDEBD0"
ORANGE_E = "#AF601A"


def zone(ax, x, y, w, h, title):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=1.8,
            edgecolor=YELLOW_E,
            facecolor=YELLOW,
            alpha=0.95,
            zorder=1,
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.014,
        title,
        ha="center",
        va="top",
        fontsize=8.5,
        fontweight="bold",
        color=ORANGE_E,
        zorder=2,
        family="DejaVu Sans",
    )


def node(ax, x, y, w, h, text, fc=BLUE, ec=BLUE_E, fs=7.2, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.01",
            linewidth=1.15,
            edgecolor=ec,
            facecolor=fc,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        fontweight="bold" if bold else "normal",
        color=INK,
        zorder=4,
        family="DejaVu Sans",
        linespacing=1.25,
        wrap=True,
    )
    return (x + w / 2, y + h / 2, x, y, w, h)


def oval(ax, x, y, w, h, text, fc=WHITE, ec=ACCENT):
    # Approximate oval with round box
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.4,
            edgecolor=ec,
            facecolor=fc,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color=ec,
        zorder=4,
        family="DejaVu Sans",
    )
    return (x + w / 2, y + h / 2)


def arrow(ax, a, b, color=MUTED, rad=0.0, lw=1.1):
    """a,b are (cx,cy) or full node tuples."""
    p1 = (a[0], a[1]) if len(a) == 2 else (a[0], a[1])
    p2 = (b[0], b[1]) if len(b) == 2 else (b[0], b[1])
    ax.add_patch(
        FancyArrowPatch(
            p1,
            p2,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
            shrinkA=6,
            shrinkB=6,
        )
    )


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 14), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.985,
        "AEGIS — Multimodal Hybrid RAG System Architecture",
        ha="center",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=INK,
        family="DejaVu Sans",
    )
    ax.text(
        0.5,
        0.962,
        "Actual deployed stack  ·  Chat-scoped retrieval  ·  Local-first LLM with cloud fallback  ·  LangSmith observability",
        ha="center",
        va="top",
        fontsize=8,
        color=MUTED,
        family="DejaVu Sans",
    )

    # ── User (top) ──────────────────────────────────────────────────────────
    user_top = oval(ax, 0.42, 0.915, 0.16, 0.035, "User / Client")

    # ── API Gateway zone ────────────────────────────────────────────────────
    zone(ax, 0.02, 0.78, 0.22, 0.12, "API Edge")
    n_vite = node(ax, 0.04, 0.855, 0.08, 0.03, "Vite / ngrok\nproxy", fs=6.5)
    n_api = node(ax, 0.13, 0.855, 0.09, 0.03, "FastAPI\n:8000", fs=6.5, bold=True)
    n_jwt = node(ax, 0.04, 0.805, 0.08, 0.03, "JWT Auth\n+ chat scope", fs=6.5)
    n_guard_in = node(ax, 0.13, 0.805, 0.09, 0.03, "InputGuard\nvalidation", fs=6.5)

    arrow(ax, user_top, (n_vite[0], n_vite[1] + 0.02), ACCENT)
    arrow(ax, n_vite, n_api, ACCENT)
    arrow(ax, n_api, n_jwt, ACCENT)
    arrow(ax, n_jwt, n_guard_in, ACCENT)

    # ── Semantic router / LangGraph ─────────────────────────────────────────
    n_router = node(
        ax,
        0.27,
        0.84,
        0.12,
        0.04,
        "Semantic Query\nRouter (LangGraph)",
        fc=ORANGE,
        ec=ORANGE_E,
        fs=7,
        bold=True,
    )
    n_exec = node(
        ax,
        0.41,
        0.84,
        0.10,
        0.04,
        "classify_query\ntemporal | hybrid",
        fc=ORANGE,
        ec=ORANGE_E,
        fs=6.8,
        bold=True,
    )
    arrow(ax, n_guard_in, n_router, ACCENT)
    arrow(ax, n_router, n_exec, ACCENT)

    # Agentic-ish branch (temporal) vs simple hybrid
    zone(ax, 0.02, 0.62, 0.18, 0.14, "Temporal Branch")
    n_temp = node(
        ax,
        0.04,
        0.68,
        0.14,
        0.05,
        "temporal_retrieve\nSQL overlap on\nstart/end metadata",
        fs=6.6,
    )
    arrow(ax, (n_exec[0] - 0.02, n_exec[1]), (n_temp[0], n_temp[1] + 0.03), ORANGE_E, rad=0.2)

    # ── Multi-Modal Retrieval Layer ─────────────────────────────────────────
    zone(ax, 0.22, 0.48, 0.40, 0.28, "Multi-Modal Retrieval Layer")
    n_meta = node(ax, 0.24, 0.68, 0.12, 0.035, "artifact_ids filter\n+ metadata pre-filter", fs=6.5)
    arrow(ax, n_exec, n_meta, BLUE_E)

    # Four modality retrievals
    n_rt = node(ax, 0.24, 0.60, 0.085, 0.045, "Text\nRetrieval\nBM25+BGE", fs=6.2)
    n_ri = node(ax, 0.335, 0.60, 0.085, 0.045, "Image\nRetrieval\nSigLIP", fs=6.2)
    n_rv = node(ax, 0.43, 0.60, 0.085, 0.045, "Video\nRetrieval\ntext+vision", fs=6.2)
    n_ra = node(ax, 0.525, 0.60, 0.085, 0.045, "Audio/ASR\nwindows\n(from video)", fs=6.2)

    for n in (n_rt, n_ri, n_rv, n_ra):
        arrow(ax, n_meta, n, BLUE_E)

    n_hyb = node(
        ax,
        0.28,
        0.515,
        0.13,
        0.045,
        "Hybrid Search\nRRF fusion",
        fc=GREEN,
        ec=GREEN_E,
        fs=7,
        bold=True,
    )
    n_rr = node(
        ax,
        0.44,
        0.515,
        0.13,
        0.045,
        "Cross-Encoder\nReranker",
        fc=GREEN,
        ec=GREEN_E,
        fs=7,
        bold=True,
    )
    arrow(ax, n_rt, n_hyb, GREEN_E, rad=0.1)
    arrow(ax, n_ri, n_hyb, GREEN_E)
    arrow(ax, n_rv, n_hyb, GREEN_E, rad=-0.05)
    arrow(ax, n_ra, n_hyb, GREEN_E, rad=-0.12)
    arrow(ax, n_hyb, n_rr, GREEN_E)
    arrow(ax, (n_temp[0] + 0.07, n_temp[1]), (n_hyb[0] - 0.02, n_hyb[1] + 0.02), ORANGE_E, rad=-0.15)

    # ── Event bus / Celery ──────────────────────────────────────────────────
    n_bus = node(
        ax,
        0.64,
        0.84,
        0.12,
        0.045,
        "Task Queue\nRedis + Celery",
        fc=ORANGE,
        ec=ORANGE_E,
        fs=7,
        bold=True,
    )

    # ── Ingestion Layer ─────────────────────────────────────────────────────
    zone(ax, 0.78, 0.62, 0.20, 0.26, "Ingestion Layer")
    n_doc = node(ax, 0.795, 0.80, 0.17, 0.035, "Document · LiteParse → chunks", fs=6.3)
    n_img = node(ax, 0.795, 0.75, 0.17, 0.035, "Image · Groq/OR vision caption", fs=6.3)
    n_vid = node(ax, 0.795, 0.70, 0.17, 0.035, "Video · Whisper ASR + keyframes", fs=6.3)
    n_aud = node(ax, 0.795, 0.65, 0.17, 0.035, "Audio track · ASR segments", fs=6.3)
    arrow(ax, n_bus, n_doc, ORANGE_E, rad=-0.1)
    for a, b in ((n_doc, n_img), (n_img, n_vid), (n_vid, n_aud)):
        arrow(ax, a, b, ORANGE_E)

    # Upload path from API
    arrow(ax, n_api, n_bus, ORANGE_E, rad=-0.25)

    # ── Embedding Service ───────────────────────────────────────────────────
    n_emb = node(
        ax,
        0.64,
        0.68,
        0.12,
        0.08,
        "Embedding\nService\nBGE-M3 1024\nSigLIP 768\n(+ Redis cache)",
        fc=GREEN,
        ec=GREEN_E,
        fs=6.8,
        bold=True,
    )
    arrow(ax, n_aud, n_emb, GREEN_E, rad=0.15)
    arrow(ax, n_emb, n_rt, GREEN_E, rad=0.2)

    # ── Storage Layer ───────────────────────────────────────────────────────
    zone(ax, 0.52, 0.30, 0.46, 0.28, "Storage Layer")
    n_pg = node(
        ax,
        0.54,
        0.42,
        0.14,
        0.08,
        "PostgreSQL\n+ pgvector\nusers · chats\nartifacts · jobs\nembedding_text\nembedding_vision",
        fs=6.4,
        bold=True,
    )
    n_minio = node(
        ax,
        0.70,
        0.45,
        0.12,
        0.055,
        "MinIO\nObject Store\nraw uploads",
        fs=6.5,
        bold=True,
    )
    n_redis = node(
        ax,
        0.84,
        0.45,
        0.12,
        0.055,
        "Redis\nCelery broker\nemb cache",
        fs=6.5,
        bold=True,
    )
    n_bm25 = node(
        ax,
        0.70,
        0.34,
        0.12,
        0.055,
        "BM25 Index\nJSON on disk\nsparse lexical",
        fs=6.5,
        bold=True,
    )
    n_meta_db = node(
        ax,
        0.84,
        0.34,
        0.12,
        0.055,
        "artifact_metadata\ntemporal JSON\nstart/end",
        fs=6.4,
        bold=True,
    )
    arrow(ax, n_emb, n_pg, GREEN_E, rad=-0.1)
    arrow(ax, n_emb, n_bm25, GREEN_E, rad=0.05)
    arrow(ax, n_bus, n_redis, ORANGE_E)
    arrow(ax, n_doc, n_minio, ORANGE_E, rad=-0.2)
    arrow(ax, n_pg, n_meta, BLUE_E, rad=0.35)
    arrow(ax, n_bm25, n_rt, BLUE_E, rad=0.25)
    arrow(ax, n_meta_db, n_temp, BLUE_E, rad=0.3)

    # ── Context / Memory ────────────────────────────────────────────────────
    n_ctx = node(
        ax,
        0.24,
        0.36,
        0.14,
        0.06,
        "Context Fusion\n+ chat history\n(scoped session)",
        fc=ORANGE,
        ec=ORANGE_E,
        fs=6.8,
        bold=True,
    )
    arrow(ax, n_rr, n_ctx, GREEN_E)
    arrow(ax, n_temp, n_ctx, ORANGE_E, rad=0.15)

    # ── Reasoning & Generation ──────────────────────────────────────────────
    zone(ax, 0.22, 0.08, 0.28, 0.26, "Reasoning & Generation")
    n_reg = node(ax, 0.24, 0.26, 0.11, 0.04, "AI Model\nRegistry", fs=6.5)
    n_llm_r = node(
        ax,
        0.37,
        0.26,
        0.11,
        0.04,
        "LLM Router\nFailoverLLM",
        fc=PINK,
        ec=PINK_E,
        fs=6.6,
        bold=True,
    )
    n_sched = node(ax, 0.24, 0.18, 0.11, 0.04, "Local infer\nOllama GPU/CPU", fs=6.4)
    n_serve = node(
        ax,
        0.37,
        0.14,
        0.11,
        0.08,
        "LLM Serving\n1) Qwen 2.5 7B\n2) Groq LLaMA\n3) OR :free",
        fc=PINK,
        ec=PINK_E,
        fs=6.5,
        bold=True,
    )
    arrow(ax, n_ctx, n_llm_r, PINK_E)
    arrow(ax, n_reg, n_llm_r, MUTED)
    arrow(ax, n_llm_r, n_sched, PINK_E)
    arrow(ax, n_sched, n_serve, PINK_E)
    arrow(ax, n_llm_r, n_serve, PINK_E, rad=-0.1)

    # ── Output guardrails ───────────────────────────────────────────────────
    n_out = node(
        ax,
        0.54,
        0.14,
        0.14,
        0.07,
        "Output Guardrails\nPII mask · grounding\nhallucination check\nconfidence score",
        fc=PINK,
        ec=PINK_E,
        fs=6.5,
        bold=True,
    )
    arrow(ax, n_serve, n_out, PINK_E)

    user_bot = oval(ax, 0.72, 0.14, 0.14, 0.04, "User / Client")
    n_sse = node(ax, 0.72, 0.21, 0.14, 0.035, "SSE /query-stream\n(+ /query JSON)", fs=6.5)
    arrow(ax, n_out, n_sse, ACCENT)
    arrow(ax, n_sse, user_bot, ACCENT)

    # ── Feedback + Observability (right of generation, clear of retrieval) ──
    zone(ax, 0.88, 0.08, 0.10, 0.18, "Obs")
    n_ls = node(
        ax,
        0.89,
        0.14,
        0.08,
        0.09,
        "LangSmith\nspans\nchunk·BM25\ndense·RRF\nrerank·gen",
        fc=GREEN,
        ec=GREEN_E,
        fs=6.2,
        bold=True,
    )
    n_fb = node(
        ax,
        0.89,
        0.09,
        0.08,
        0.04,
        "UserFeedback",
        fs=6.3,
    )
    arrow(ax, n_rr, n_ls, GREEN_E, rad=-0.35)
    arrow(ax, n_serve, n_ls, GREEN_E, rad=-0.15)
    arrow(ax, n_emb, n_ls, GREEN_E, rad=0.25)
    arrow(ax, user_bot, n_fb, MUTED, rad=-0.1)

    # RAGAS opt-in
    n_ragas = node(
        ax,
        0.54,
        0.08,
        0.14,
        0.035,
        "RAGAS eval (opt-in · local Qwen judge)",
        fs=6.2,
        fc=GREEN,
        ec=GREEN_E,
    )
    arrow(ax, n_out, n_ragas, GREEN_E, rad=0.1)

    # Legend footer
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.008),
            0.96,
            0.055,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=1,
            edgecolor="#D5D8DC",
            facecolor=WHITE,
            zorder=3,
        )
    )
    ax.text(
        0.03,
        0.042,
        "Yellow = logical layer   ·   Blue = runtime component   ·   Green = retrieval / embeddings / observability   ·   Pink = generation & guards",
        fontsize=7,
        color=INK,
        va="center",
        family="DejaVu Sans",
        zorder=4,
    )
    ax.text(
        0.03,
        0.022,
        "LLM priority: Ollama qwen2.5:7b → Groq → OpenRouter :free   |   Vectors: pgvector (text 1024 + vision 768)   |   Queue: Redis/Celery   |   Isolation: JWT + session artifact_ids",
        fontsize=6.8,
        color=MUTED,
        va="center",
        family="DejaVu Sans",
        zorder=4,
    )

    fig.savefig(OUT, bbox_inches="tight", facecolor=BG, pad_inches=0.2)
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
