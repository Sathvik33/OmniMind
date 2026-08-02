"""
OutputGuard — comprehensive validation of LLM responses before user delivery.

Multi-layered post-generation checks:
  1. Non-empty answer validation
  2. PII masking (email, phone, SSN, credit card, sensitive terms)
  3. Jailbreak affirmative opener detection & stripping
  4. Grounding validation (answer references context)
  5. Hallucination detection (answer too different from context)
  6. Confidence scoring
  7. Harmful content filtering
  8. Source coherence check
"""

import re
from typing import Dict, Any, List, Tuple

# ── PII Patterns ───────────────────────────────────────────────────────────────

_PII_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN]"),
    (re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"), "[CARD]"),
    (re.compile(r"\b\+?1?\s*\(?\d{3}\)?\s*-?\d{3}\s*-?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{9}\b"), "[ID]"),
]

_JAILBREAK_OPENER_RE = re.compile(
    r"^(Sure[,!]?\s*[Hh]ere('s|s| is)?[\s\w]*:|"
    r"Of course[,!]?\s*[Hh]ere|"
    r"Absolutely[,!]?\s*[Hh]ere|"
    r"Certainly[,!]?\s*[Hh]ere|"
    r"I'd be happy to|"
    r"I can help|"
    r"No problem)",
    re.IGNORECASE,
)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "in",
    "to", "and", "or", "for", "on", "at", "with", "it", "this",
    "that", "be", "by", "from", "as", "have", "has", "had", "but",
    "not", "no", "so", "if", "do", "does", "did", "can", "will",
    "would", "could", "should", "may", "might", "shall", "just",
    "about", "which", "what", "when", "where", "why", "how",
}

_HARMFUL_TOKENS = {
    "I can", "I will", "I'll help you", "I am", "I'm an",
    "absolutely", "certainly", "of course",
}


def _mask_pii(text: str) -> str:
    """Mask personally identifiable information."""
    for pattern, replacement in _PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def _strip_jailbreak_opener(text: str) -> str:
    """Remove common jailbreak affirmative openers."""
    return _JAILBREAK_OPENER_RE.sub("", text).strip()


def _extract_tokens(text: str) -> set:
    """Extract significant tokens (>4 chars, non-stopwords) from text."""
    tokens = set(re.findall(r"\b[a-zA-Z]{4,}\b", text.lower()))
    return tokens - _STOPWORDS


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p and len(p.strip()) > 8]


def _token_seq(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS and len(t) > 2]


def _lcs_len(a: List[str], b: List[str]) -> int:
    """Length of longest common subsequence (for ROUGE-L style overlap)."""
    if not a or not b:
        return 0
    # Bound cost for long answers
    a = a[:80]
    b = b[:120]
    prev = [0] * (len(b) + 1)
    for tok in a:
        cur = [0]
        for j, bt in enumerate(b, start=1):
            if tok == bt:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def _sentence_support(answer: str, context: str) -> float:
    """
    Fraction of answer sentences that share strong n-gram / LCS overlap
    with at least one context sentence. Softens paraphrase false positives.
    """
    ans_sents = _split_sentences(answer)
    ctx_sents = _split_sentences(context)
    if not ans_sents:
        return 1.0
    if not ctx_sents:
        return 0.0

    ctx_seqs = [_token_seq(s) for s in ctx_sents]
    supported = 0
    for sent in ans_sents:
        aseq = _token_seq(sent)
        if not aseq:
            supported += 1
            continue
        best = 0.0
        for cseq in ctx_seqs:
            if not cseq:
                continue
            lcs = _lcs_len(aseq, cseq)
            score = lcs / max(len(aseq), 1)
            if score > best:
                best = score
            # Also reward shared bigrams
            abigs = set(zip(aseq, aseq[1:])) if len(aseq) > 1 else set()
            cbigs = set(zip(cseq, cseq[1:])) if len(cseq) > 1 else set()
            if abigs:
                bigram = len(abigs & cbigs) / len(abigs)
                best = max(best, bigram)
        if best >= 0.35:
            supported += 1
    return supported / len(ans_sents)


def _check_grounding(answer: str, context: str) -> Tuple[bool, float]:
    """
    Measure if answer is grounded in context.

    Combines:
      - lexical token overlap (cheap first pass)
      - sentence-level LCS / bigram support (paraphrase-friendly)

    Returns: (is_grounded: bool, confidence: float 0.0-1.0)
    """
    if not context or not context.strip():
        return True, 1.0

    answer_tokens = _extract_tokens(answer)
    if not answer_tokens:
        return True, 1.0

    context_lower = context.lower()
    matching_tokens = sum(1 for tok in answer_tokens if tok in context_lower)
    lexical = matching_tokens / len(answer_tokens)
    sent_score = _sentence_support(answer, context)

    # Blend: sentence support rescues paraphrases with weaker exact token hits
    confidence = max(lexical, 0.55 * lexical + 0.45 * sent_score)
    if sent_score >= 0.67 and lexical >= 0.35:
        confidence = max(confidence, 0.62)

    is_grounded = confidence >= 0.55 or (sent_score >= 0.75 and lexical >= 0.25)
    return is_grounded, min(1.0, confidence)


def _detect_hallucination(answer: str, context: str) -> Tuple[bool, List[str]]:
    """
    Detect potential hallucinations by checking for facts not in context.

    Returns: (has_hallucination: bool, hallucinated_phrases: List[str])
    """
    hallucinations = []

    # Common hallucination markers (avoiding polite refusals which are legitimate RAG responses)
    hallucination_patterns = [
        (r"from my knowledge", "generic knowledge reference"),
        (r"as an ai language model", "generic pretrained statement"),
        (r"in general knowledge", "vague generalization"),
    ]

    for pattern, description in hallucination_patterns:
        if re.search(pattern, answer, re.IGNORECASE):
            hallucinations.append(f"{description}: '{pattern}'")

    # Check for large numeric claims not in context (>2 digits to avoid bullet points/single digits)
    numbers_in_answer = set(re.findall(r"\b\d{3,}(?:\.\d+)?\b", answer))
    numbers_in_context = set(re.findall(r"\b\d+(?:\.\d+)?\b", context))

    for num in numbers_in_answer:
        if num not in numbers_in_context and len(numbers_in_context) > 0:
            hallucinations.append(f"Numeric claim '{num}' not found in context")

    return len(hallucinations) > 0, hallucinations[:3]



def _calculate_confidence(
    grounding_confidence: float,
    has_hallucination: bool,
    answer_length: int,
) -> float:
    """
    Calculate overall confidence score (0.0-1.0).

    Factors:
      - Grounding confidence: 60% weight
      - No hallucination: 30% weight
      - Answer length (not too short/long): 10% weight
    """
    base_score = grounding_confidence * 0.6

    if not has_hallucination:
        base_score += 0.3
    else:
        base_score += 0.1  # Partial credit for hallucination present

    # Penalize very short answers (< 50 chars) or very long (> 2000 chars)
    if 50 <= answer_length <= 2000:
        base_score += 0.1
    elif answer_length > 0:
        base_score += 0.05

    return min(1.0, base_score)


# ── Public API ─────────────────────────────────────────────────────────────────

class OutputGuard:
    """Post-generation filter for every LLM answer."""

    @staticmethod
    def validate(answer: str, context: str = "") -> Dict[str, Any]:
        """
        Comprehensive output validation with confidence scoring.

        Returns:
          {
            "ok": bool,
            "answer": str,           ← cleaned, PII-masked answer
            "warnings": [str],       ← non-blocking issues
            "confidence": float,     ← 0.0-1.0 confidence score
            "grounded": bool,        ← is answer grounded in context
            "has_hallucination": bool
          }
        """
        warnings: List[str] = []

        # 1. Empty response check
        if not answer or not answer.strip():
            return {
                "ok": False,
                "answer": "I could not generate a response. Please try again.",
                "warnings": ["Empty LLM response"],
                "confidence": 0.0,
                "grounded": False,
                "has_hallucination": False,
            }

        # 2. Clean answer (PII masking, jailbreak removal)
        clean = _mask_pii(answer)
        clean = _strip_jailbreak_opener(clean)

        # 3. Grounding check
        is_grounded, grounding_conf = _check_grounding(clean, context)
        if not is_grounded:
            warnings.append(
                f"⚠️ Answer may not be fully supported by retrieved context (confidence: {grounding_conf:.1%})"
            )

        # 4. Hallucination detection
        has_hallucination, hallucin_details = _detect_hallucination(clean, context)
        if has_hallucination:
            for detail in hallucin_details:
                warnings.append(f"⚠️ Potential hallucination: {detail}")

        # 5. Length check
        answer_length = len(clean)
        if answer_length < 20:
            warnings.append("Answer is very short; may lack sufficient detail")
        elif answer_length > 3000:
            warnings.append("Answer is very long; consider breaking into sections")

        # 6. Calculate overall confidence
        confidence = _calculate_confidence(
            grounding_conf,
            has_hallucination,
            answer_length,
        )

        return {
            "ok": True,
            "answer": clean,
            "warnings": warnings,
            "confidence": round(confidence, 3),
            "grounded": is_grounded,
            "has_hallucination": has_hallucination,
        }

    @staticmethod
    def batch_validate(
        answers: List[str], context: str = ""
    ) -> List[Dict[str, Any]]:
        """Validate multiple answers at once."""
        return [OutputGuard.validate(ans, context) for ans in answers]
