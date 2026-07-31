# Guardrails

**Path:** `backend/app/guardrails/`

Safety and quality gates at three boundaries: **upload**, **query in**, **answer out**.

| File | Boundary | Checks |
|------|----------|--------|
| `ingestion_guard.py` | Upload | Extension whitelist, **100 MB** max, safe filename |
| `input_guard.py` | Query | Length 2–2000; injection / SQL / cmd / harmful patterns |
| `output_guard.py` | Answer | PII mask, grounding heuristic (≥60% token overlap), confidence |
| `generation_guard.py` | Legacy agent path | Simple stubs — not main RAG |

---

## Why guard at three layers?

```text
Bad file  → wasted Celery + API cost     → ingestion_guard
Bad query → wasted retrieval + LLM $     → input_guard
Bad answer → user harm / PII leak        → output_guard
```

Defense in depth: retrieval grounding reduces hallucination, but output guard still scores confidence for UI trailers / monitoring.

---

## Design rules

1. Guards return structured `{ok, reason, warnings}` — nodes decide soft vs hard fail.  
2. Do not call LLMs inside input guard (latency + cost).  
3. Keep extension lists in sync with UI `ACCEPT` and docs.
