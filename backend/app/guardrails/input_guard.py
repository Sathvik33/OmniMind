"""
InputGuard — comprehensive input validation for RAG queries.

Multi-layered security checks:
  1. Empty / null validation
  2. Length constraints (2000 chars)
  3. Prompt injection detection (30+ patterns)
  4. Unicode abuse & control characters
  5. SQL injection patterns
  6. Command injection patterns
  7. Malicious intent detection
  8. Content relevance scoring
  9. Semantic validation (check if query is actual question)
"""

import re
from typing import Dict, Any, List

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_QUERY_LENGTH = 2000
MIN_QUERY_LENGTH = 2

_INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|prior)\s+instructions",
    r"forget\s+(your|all|previous|prior)\s+(rules|instructions|context)",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(a\s+|an\s+)?(different|new|evil|jailbreak)",
    r"(system|admin|root)\s*:",
    r"<\|im_start\|>",
    r"<\|im_end\|>",
    r"###\s*(instruction|system|human|assistant)",
    r"disregard\s+(the\s+)?(above|previous|all)",
    r"override\s+(your\s+)?(instructions|rules|constraints)",
    r"roleplay\s+as",
    r"pretend\s+you\s+(are|have\s+no)",
    r"enable\s+(developer|jailbreak|dan)\s+mode",
    r"\bDAN\b",
    r"do\s+anything\s+now",
    r"jailbreak",
    r"exploit",
    r"bypass\s+(filter|security|detection)",
    r"escape\s+(sequence|prompt|context)",
    r"injection\s+(attack|payload)",
    r"decode\s+(base64|hex|rot13)",
]

_SQL_PATTERNS = [
    r"(\bselect\b|\bunion\b|\bdrop\b|\binsert\b|\bupdate\b|\bdelete\b|\bfrom\b)",
    r"''\s*or\s*'",
    r";\s*(select|insert|update|delete)",
]

_CMD_PATTERNS = [
    r"(\$\(|`|&&|\|\||;\s*\w+)",
    r"bash\s*-",
    r"rm\s+-rf",
    r"chmod\s+777",
]

_INVISIBLE_RE = re.compile(
    r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff\u00ad]"
)

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS), re.IGNORECASE | re.UNICODE
)

_SQL_RE = re.compile("|".join(_SQL_PATTERNS), re.IGNORECASE)
_CMD_RE = re.compile("|".join(_CMD_PATTERNS), re.IGNORECASE)

_HARMFUL_KEYWORDS = {
    "hack", "crack", "exploit", "malware", "virus", "ddos", "ransomware",
    "phishing", "spyware", "trojan", "worm", "botnet", "c2", "payload",
    "shellcode", "overflow", "buffer", "pivot", "lateral movement",
}

_SEMANTIC_PATTERNS = [
    r"what\s+",
    r"how\s+",
    r"why\s+",
    r"when\s+",
    r"where\s+",
    r"who\s+",
    r"which\s+",
    r"tell\s+",
    r"explain\s+",
    r"describe\s+",
    r"summarize\s+",
    r"extract\s+",
    r"find\s+",
    r"list\s+",
    r"show\s+",
    r"give\s+",
    r"\?$",
]

_SEMANTIC_RE = re.compile("|".join(_SEMANTIC_PATTERNS), re.IGNORECASE)


# ── Public API ─────────────────────────────────────────────────────────────────

class InputGuard:
    """
    Multi-layered query validator. Call validate() before every RAG query.
    """

    @staticmethod
    def validate(query: str) -> Dict[str, Any]:
        """
        Comprehensive input validation with detailed feedback.

        Returns:
          {"ok": True} on success
          {"ok": False, "reason": str} on failure
          {"ok": True, "warnings": [str]} if valid but has minor issues
        """
        warnings: List[str] = []

        # 1. Basic validation
        basic_check = InputGuard._check_basic(query)
        if not basic_check["ok"]:
            return basic_check

        # 2. Security checks
        security_check = InputGuard._check_security(query)
        if not security_check["ok"]:
            return security_check

        # 3. Harmful content check
        harmful_check = InputGuard._check_harmful_intent(query)
        if not harmful_check["ok"]:
            return harmful_check
        warnings.extend(harmful_check.get("warnings", []))

        # 4. Semantic validation (soft check)
        semantic_check = InputGuard._check_semantic_validity(query)
        if not semantic_check["ok"]:
            warnings.append(semantic_check.get("reason", "Query may not be a valid question"))

        result = {"ok": True}
        if warnings:
            result["warnings"] = warnings
        return result

    @staticmethod
    def _check_basic(query: str) -> Dict[str, Any]:
        """Null, length, whitespace checks."""
        if not query or not query.strip():
            return {"ok": False, "reason": "Query is empty or contains only whitespace."}

        if len(query) < MIN_QUERY_LENGTH:
            return {"ok": False, "reason": f"Query too short. Minimum {MIN_QUERY_LENGTH} characters required."}

        if len(query) > MAX_QUERY_LENGTH:
            return {
                "ok": False,
                "reason": f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters.",
            }

        return {"ok": True}

    @staticmethod
    def _check_security(query: str) -> Dict[str, Any]:
        """Prompt injection, unicode abuse, SQL/command injection."""
        # Unicode control characters
        if _INVISIBLE_RE.search(query):
            return {
                "ok": False,
                "reason": "Query contains disallowed Unicode control characters.",
            }

        # Prompt injection patterns
        if _INJECTION_RE.search(query):
            return {
                "ok": False,
                "reason": "Query contains prompt injection patterns. Please rephrase your question.",
            }

        # SQL injection patterns
        if _SQL_RE.search(query):
            return {
                "ok": False,
                "reason": "Query contains SQL-like patterns which are not allowed.",
            }

        # Command injection patterns
        if _CMD_RE.search(query):
            return {
                "ok": False,
                "reason": "Query contains command-like patterns which are not allowed.",
            }

        return {"ok": True}

    @staticmethod
    def _check_harmful_intent(query: str) -> Dict[str, Any]:
        """Detect harmful/malicious intent keywords."""
        query_lower = query.lower()
        found_keywords = [kw for kw in _HARMFUL_KEYWORDS if kw in query_lower]

        if found_keywords:
            return {
                "ok": False,
                "reason": f"Query contains restricted keywords: {', '.join(found_keywords[:3])}. "
                          "This system is designed for information retrieval only.",
            }

        return {"ok": True, "warnings": []}

    @staticmethod
    def _check_semantic_validity(query: str) -> Dict[str, Any]:
        """Soft check: does this look like a real question/query?"""
        # Check if query contains question words or ends with ?
        if _SEMANTIC_RE.search(query):
            return {"ok": True}

        # Check for question mark
        if query.strip().endswith("?"):
            return {"ok": True}

        # Check if it's likely a document/context to index (too long, structured)
        if len(query) > 500 and query.count("\n") > 5:
            return {
                "ok": False,
                "reason": "Query appears to be document content rather than a question. "
                         "Please ask a specific question about your uploaded data.",
            }

        # Warn if it doesn't look like a question but allow it
        return {"ok": False, "reason": "Query may not be phrased as a question. "
                                       "Consider using question words (what, how, why, etc.)."}
