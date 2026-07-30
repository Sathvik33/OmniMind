"""Convert LLM markdown (or mixed markup) into readable plain text."""

from __future__ import annotations

import html
import re


_CODE_FENCE = re.compile(r"```[\w+-]*\n?(.*?)```", re.DOTALL)
_INLINE_CODE = re.compile(r"`([^`]+)`")
_BOLD = re.compile(r"(\*\*|__)(.*?)\1")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|(?<!_)_(?!_)(.+?)(?<!_)_(?!_)")
_HEADING = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_HR = re.compile(r"^(-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
_BLOCKQUOTE = re.compile(r"^>\s?", re.MULTILINE)
_UL = re.compile(r"^\s*[-*+]\s+", re.MULTILINE)
_OL = re.compile(r"^\s*\d+\.\s+")
_HTML_TAG = re.compile(r"<[^>]+>")
_MULTI_NL = re.compile(r"\n{3,}")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def markdown_to_plain_text(text: str) -> str:
    """
    Strip common markdown / HTML into plain text suitable for UI display.
    Safe no-op for already-plain text.
    """
    if not text:
        return ""

    out = text.replace("\r\n", "\n").replace("\r", "\n")
    out = _THINK.sub("", out)
    out = _CODE_FENCE.sub(lambda m: m.group(1).strip(), out)
    out = _IMAGE.sub(lambda m: m.group(1) or "", out)
    out = _LINK.sub(lambda m: m.group(1), out)
    out = _HEADING.sub("", out)
    out = _HR.sub("", out)
    out = _BLOCKQUOTE.sub("", out)
    out = _BOLD.sub(lambda m: m.group(2), out)
    out = _ITALIC.sub(lambda m: m.group(1) or m.group(2) or "", out)
    out = _INLINE_CODE.sub(lambda m: m.group(1), out)
    out = _UL.sub("• ", out)
    # Keep numbered lists as "1. " but normalize spacing
    out = re.sub(r"^\s*(\d+)\.\s+", r"\1. ", out, flags=re.MULTILINE)
    out = _HTML_TAG.sub("", out)
    out = html.unescape(out)
    out = _MULTI_NL.sub("\n\n", out)
    return out.strip()
