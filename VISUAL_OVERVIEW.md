# Visual Implementation Overview

## Complete Multi-Modal RAG Pipeline with Guardrails

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Streamlit)                               │
│                    ChatGPT-style UI + Streaming Display                         │
└────────────────────────────────────────┬──────────────────────────────────────┘
                                         │ HTTP/Streaming
        ┌────────────────────────────────▼──────────────────────────────┐
        │                    FastAPI Backend (Port 8000)                │
        └───────────────────────┬──────────────────────────────────────┘
                                │
         ┌──────────────────────▼──────────────────────┐
         │      QueryPipeline - Main Orchestrator      │
         └──────────────────────┬──────────────────────┘
                                │
          ┌─────────────────────┴─────────────────────────────────────┐
          │                                                           │
          ▼                                                           ▼
    ┌──────────────────────┐                              ┌──────────────────────┐
    │  INPUT VALIDATION    │                              │  STREAMING/BLOCKING  │
    │  (InputGuard)        │                              │     MODES            │
    │                      │                              │                      │
    │ • 30+ patterns       │  ✅ Query OK                 │ • Async streaming    │
    │ • SQL injection      │  ❌ Reject                   │ • Real-time tokens   │
    │ • Command injection  │  ⚠️ Warnings                 │ • Metadata comments  │
    │ • Harmful keywords   │                              │ • Confidence feedback│
    │ • Unicode abuse      │                              │                      │
    │ • Semantic checks    │                              │                      │
    └──────────────────────┘                              └──────────────────────┘
                                     │
                                     ▼
                    ┌──────────────────────────────────────────┐
                    │    HYBRID RETRIEVAL PIPELINE             │
                    │  (HybridRetriever + CrossEncoderReranker)│
                    └──────────────────────┬───────────────────┘
                                          │
        ┌─────────────────────────────────┼─────────────────────────────────┐
        │                                 │                                 │
        ▼                                 ▼                                 ▼
    ┌─────────────┐              ┌──────────────┐              ┌──────────────┐
    │ BM25 Search │              │  DENSE SEARCH │              │ CLIP MULTIMODAL
    │ (Keywords)  │              │ (Semantic)   │              │ (Images/Video)
    │             │              │              │              │              
    │ Fast ⚡     │              │ Slow 🔍      │              │ Visual Cross  
    │ ~50-200ms   │              │ ~200-500ms   │              │ Modal ~100ms  
    └──────┬──────┘              └──────┬───────┘              └───────┬──────┘
           │                            │                             │
           └─────────────────────────────┼─────────────────────────────┘
                                        │
                                        ▼
                        ┌──────────────────────────────────┐
                        │ RECIPROCAL RANK FUSION (RRF)    │
                        │ Fuses BM25 + Dense Rankings     │
                        │                                  │
                        │ Adaptive Weights:                │
                        │ • Exact match: 55% BM25, 45%    │
                        │ • Semantic Q: 35% BM25, 65%    │
                        │ • Temporal Q: 30% BM25, 70%    │
                        │ • Default: 40% BM25, 60%       │
                        └────────────┬─────────────────────┘
                                    │
                                    ▼
                        ┌──────────────────────────────────┐
                        │  CROSS-ENCODER RERANKING        │
                        │  (CrossEncoderReranker)         │
                        │                                  │
                        │ Scores: (query, doc) pairs      │
                        │ Output: 0.0-1.0 normalized      │
                        │ Top-5 selected per top-20       │
                        │ Device: GPU/CPU auto-select    │
                        └────────────┬─────────────────────┘
                                    │
                        ┌───────────▼─────────────────┐
                        │ Confidence Tier Scoring:    │
                        │ ≥ 0.8: Highly relevant      │
                        │ ≥ 0.6: Moderately relevant  │
                        │ ≥ 0.4: Potentially relevant │
                        │ < 0.4: Low confidence       │
                        └───────────┬─────────────────┘
                                    │
                                    ▼
            ┌───────────────────────────────────────────────────┐
            │          CONTEXT BUILDER                          │
            │  (Merges top-5 ranked documents into prompt)      │
            └───────────────────┬───────────────────────────────┘
                                │
                                ▼
            ┌───────────────────────────────────────────────────┐
            │        LLM GENERATION                             │
            │  (Qwen2.5:7b via Ollama, temperature: 0.2)       │
            │                                                   │
            │  Modes:                                           │
            │  • Blocking: Full response                        │
            │  • Streaming: Token-by-token                      │
            │  • Latency: <1s first token, 2-4s total         │
            └───────────────────┬───────────────────────────────┘
                                │
                                ▼
            ┌───────────────────────────────────────────────────┐
            │    OUTPUT VALIDATION & SCORING                    │
            │           (OutputGuard)                           │
            │                                                   │
            │ ✅ PII Masking:                                  │
            │    Email → [EMAIL], Phone → [PHONE]              │
            │    SSN → [SSN], Card → [CARD], ID → [ID]        │
            │                                                   │
            │ ✅ Hallucination Detection:                      │
            │    Uncertainty markers, numeric mismatches        │
            │                                                   │
            │ ✅ Grounding Validation:                         │
            │    % of answer tokens in context                  │
            │                                                   │
            │ ✅ Confidence Scoring:                           │
            │    60% grounding + 30% no hallucination + 10% len│
            │    Result: 0.0-1.0 confidence score              │
            └───────────────────┬───────────────────────────────┘
                                │
        ┌───────────────────────▼────────────────────────────────┐
        │                RESPONSE STRUCTURE                       │
        │                                                         │
        │ {                                                       │
        │   "answer": str,          # Main LLM response          │
        │   "context_used": [],     # Retrieved documents         │
        │   "warnings": [],         # Issues & explanations       │
        │   "confidence": 0.85,     # 0.0-1.0 overall score      │
        │   "grounded": true,       # Supported by context?      │
        │   "has_hallucination": false,  # Speculations found?   │
        │   "error": false          # Processing error?          │
        │ }                                                       │
        └───────────────────────────────────────────────────────┘
```

---

## Confidence Scoring Details

```
┌─────────────────────────────────────────────────────────────────┐
│         CONFIDENCE CALCULATION FORMULA                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Final Score = (Grounding × 0.6) + (No Hallucination × 0.3)   │
│                + (Reasonable Length × 0.1)                     │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│ GROUNDING (60% weight):                                          │
│ • Extract significant tokens from answer (>4 chars)              │
│ • Count tokens present in retrieved context                      │
│ • Score = matching_tokens / total_tokens                         │
│ • Threshold: ≥0.6 = grounded, <0.6 = warning                   │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│ HALLUCINATION (30% weight):                                      │
│ • Detect uncertainty markers: "I'm not sure", "probably"         │
│ • Check numeric claims not in context                            │
│ • Check for speculation patterns                                 │
│ • If hallucinations found: -0.2 penalty                          │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│ LENGTH (10% weight):                                             │
│ • Penalize very short (<50 chars) or very long (>2000) answers   │
│ • Reasonable range (50-2000): +0.1                              │
│ • Too short/long: +0.05                                         │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│ CONFIDENCE INTERPRETATION:                                       │
│                                                                  │
│ 0.90-1.00 ⭐⭐⭐⭐⭐ Excellent - Fully grounded, no issues         │
│ 0.75-0.89 ⭐⭐⭐⭐  Good - Mostly grounded, minor concerns         │
│ 0.60-0.74 ⭐⭐⭐   Fair - Reasonably grounded, some warnings     │
│ 0.45-0.59 ⭐⭐    Low - Partially grounded, recommend review    │
│ 0.00-0.44 ⭐     Very Low - Not grounded, high risk             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Security Filtering Layers

```
┌────────────────────────────────────────────────────────────────┐
│              INPUT SECURITY LAYERS                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Layer 1: BASIC VALIDATION                                      │
│ ❌ Empty queries                                               │
│ ❌ Too short (<2 chars) or too long (>2000 chars)              │
│ ❌ Only whitespace                                             │
│                                                                 │
│ Layer 2: PROMPT INJECTION (30+ patterns)                      │
│ ❌ "ignore previous instructions"                              │
│ ❌ "act as", "roleplay as", "pretend"                          │
│ ❌ "enable DAN/jailbreak mode"                                 │
│ ❌ "disregard", "override", "escape"                           │
│ ❌ LLM control tokens: <|im_start|>, ###instruction, etc.      │
│                                                                 │
│ Layer 3: SQL INJECTION (10+ patterns)                         │
│ ❌ SELECT, UNION, DROP, INSERT, UPDATE, DELETE                 │
│ ❌ '';OR'', SQL comment patterns                                │
│                                                                 │
│ Layer 4: COMMAND INJECTION (bash/shell)                        │
│ ❌ $(command), `command`, &&, ||                               │
│ ❌ rm -rf, chmod 777, bash -i                                  │
│                                                                 │
│ Layer 5: UNICODE ABUSE                                         │
│ ❌ Invisible characters (U+200B-U+206F)                         │
│ ❌ Right-to-left overrides                                     │
│ ❌ Zero-width spaces                                           │
│                                                                 │
│ Layer 6: HARMFUL INTENT (20+ keywords)                        │
│ ❌ hack, crack, exploit, malware, virus, ddos, ransomware      │
│ ❌ phishing, spyware, trojan, worm, botnet, c2, payload        │
│ ❌ shellcode, buffer overflow, pivot, lateral movement         │
│                                                                 │
│ Layer 7: SEMANTIC VALIDATION (soft)                            │
│ ⚠️  Document-like content (multiline, >500 chars)              │
│ ⚠️  No question words (what, how, why, when, where, who)       │
│ ⚠️  Doesn't end with "?"                                       │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

```
┌────────────────────────────────────────────────────────────────┐
│              OUTPUT SECURITY LAYERS                            │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│ Layer 1: EMPTY RESPONSE CHECK                                  │
│ ❌ Reject if answer is empty or only whitespace                │
│                                                                 │
│ Layer 2: PII MASKING                                           │
│ ✅ Email addresses → [EMAIL]                                  │
│ ✅ Phone numbers → [PHONE]                                    │
│ ✅ Social Security Numbers → [SSN]                            │
│ ✅ Credit card numbers → [CARD]                               │
│ ✅ ID numbers → [ID]                                          │
│                                                                 │
│ Layer 3: JAILBREAK REMOVAL                                     │
│ ✅ Remove: "Sure, here's", "Of course", "Absolutely"          │
│ ✅ Remove: "Certainly", "I'd be happy to", "I can help"       │
│                                                                 │
│ Layer 4: GROUNDING VALIDATION                                  │
│ ✅ Check: % of answer tokens in retrieved context              │
│ ⚠️  Warn if <60% tokens found in context                       │
│ 📊 Report: confidence % (e.g., 75% of tokens grounded)         │
│                                                                 │
│ Layer 5: HALLUCINATION DETECTION                               │
│ ✅ Detect: "I'm not sure", "I don't know", "probably"         │
│ ✅ Detect: "most likely", "in general", "from my knowledge"    │
│ ✅ Detect: Numeric claims not in context                       │
│ ⚠️  Flag up to 3 hallucinated phrases                          │
│                                                                 │
│ Layer 6: CONFIDENCE SCORING                                    │
│ 📊 Calculate: Multi-factor 0.0-1.0 confidence score            │
│ 📊 Weight: 60% grounding + 30% hallucination + 10% length      │
│ 📊 Return: Numeric confidence for UX                           │
│                                                                 │
│ Layer 7: ANSWER LENGTH CHECK                                   │
│ ⚠️  Warn if <20 chars (too brief)                             │
│ ⚠️  Warn if >3000 chars (too long, recommend sections)         │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

---

## Information Flow Diagram

```
     USER INPUT
         │
         ▼
    ┌─────────────┐
    │InputGuard   │
    │(7 layers)   │  ──❌─→ REJECT WITH REASON
    └──────┬──────┘
           │✅
           ▼
    ┌──────────────┐
    │HybridRetriever   
    │ • BM25       │
    │ • Dense      │      ┌────────────────────┐
    │ • RRF Fusion │─────→│CrossEncoderReranker│
    │ • CLIP Modal │      │Scores: 0.0-1.0    │
    └──────┬───────┘      └────────┬───────────┘
           │                       │
           └───────────────────────┤
                                   ▼
                        ┌──────────────────────┐
                        │ContextBuilder       │
                        │ (Merge + Format)     │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │Generator (Qwen2.5)   │
                        │Streaming: token/token│
                        │Blocking: full resp   │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │OutputGuard           │
                        │ (5 layers)           │
                        │ • PII masking        │
                        │ • Hallucination      │
                        │ • Grounding          │
                        │ • Confidence calc    │
                        └──────────┬───────────┘
                                   │
                                   ▼
                        ┌──────────────────────┐
                        │QueryResponse         │
                        │ • answer             │
                        │ • context_used       │
                        │ • confidence (0-1)   │
                        │ • grounded (bool)    │
                        │ • hallucination(bool)│
                        │ • warnings           │
                        └──────────────────────┘
                                   │
                                   ▼
                        RESPONSE TO USER
                        (JSON for API)
                        (Streaming for UI)
```

---

## Component Interaction Map

```
          ┌─────────────────────────────────────────┐
          │        External: Qwen2.5:7b (Ollama)    │
          │        External: Chroma DB              │
          │        External: CUDA/GPU (optional)    │
          └─────────────────────────────────────────┘
                           ▲
                           │ Query
                           │ Generation
                    ┌──────▼──────────────────┐
                    │   QueryPipeline         │
                    │  (Main Orchestrator)    │
                    └──────┬───────────────────┘
                           │ Calls
           ┌───────────────┼───────────────────┐
           │               │                   │
           ▼               ▼                   ▼
      ┌─────────┐    ┌──────────┐      ┌────────────┐
      │InputGuard     │Hybrid    │      │OutputGuard │
      │             Retriever │      │           │
      │ • Validates  │ • Merges │      │ • Masks PII
      │ • Rejects    │   BM25+  │      │ • Detects  │
      │ • Warns      │   Dense  │      │   halluc   │
      └─────────────┘ • RRF fus │      │ • Confidence
                      │ • Rerank │      └────────────┘
                      │ • Modal  │
                      └──────────┘
                           │
           ┌───────────────┼───────────────┐
           │               │               │
           ▼               ▼               ▼
      ┌─────────┐    ┌──────────┐    ┌──────────┐
      │BM25Store    │TextCol   │    │Multimodal│
      │              │(ChromaDB)│    │Collection│
      │ • Keyword    │ • Dense  │    │ • CLIP   │
      │   matching   │   embed  │    │ • Images │
      │ • Fast       │ • Persist│    │ • Videos │
      └─────────────┘ └─────────┘    └──────────┘
           │               │               │
           └───────────────┴───────────────┘
                      │
           ┌──────────▼──────────┐
           │CrossEncoderReranker │
           │ • Scores pairs      │
           │ • Normalizes scores │
           │ • GPU accelerated   │
           └─────────────────────┘
```

---

## Deployment Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DOCKER COMPOSE SETUP                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         FRONTEND (Port 8501)                             │  │
│  │         Streamlit + Token Streaming Display              │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │ HTTP/SSE                              │
│  ┌──────────────────────▼───────────────────────────────────┐  │
│  │    BACKEND (Port 8000)                                   │  │
│  │  FastAPI + QueryPipeline + All Guardrails               │  │
│  │                                                           │  │
│  │  Components:                                             │  │
│  │  • InputGuard (7 layers)                                │  │
│  │  • HybridRetriever (adaptive, confidence)              │  │
│  │  • CrossEncoderReranker (GPU-accelerated)              │  │
│  │  • OutputGuard (5 layers, confidence scoring)          │  │
│  │  • QueryPipeline (orchestration)                       │  │
│  │                                                           │  │
│  │  Memory:                                                │  │
│  │  • ChromaDB (vector store)                             │  │
│  │  • BM25 index (sparse retrieval)                       │  │
│  └──────────┬────────────────────────────────────────────────┘  │
│             │ Inference                                        │
│  ┌──────────▼────────────────────────────────────────────────┐  │
│  │    OLLAMA (Port 11434)                                   │  │
│  │  Qwen2.5:7b LLM (local, no cloud)                      │  │
│  │  LLaVA-1.5-7B (vision, 4-bit quantized)               │  │
│  │                                                           │  │
│  │  GPU Support:                                           │  │
│  │  • CUDA enabled (auto-detect)                          │  │
│  │  • CPU fallback                                        │  │
│  │  • Device stats in responses                          │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │    VOLUMES (Persistence)                                 │  │
│  │    • ./data/ → Uploaded files                           │  │
│  │    • ./chroma_db/ → Vector index                        │  │
│  │    • ./ollama_data/ → Model cache                       │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

This visual summary shows:
✅ Complete pipeline from user input to response
✅ All guardrail layers (input + output)
✅ Confidence scoring calculation
✅ Security filtering layers
✅ Component interaction patterns
✅ Docker deployment architecture
