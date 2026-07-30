from __future__ import annotations

import json
import logging
import re
from typing import Any, Generator, Optional

from dotenv import load_dotenv
from langsmith import traceable

from backend.app.rag.schemas import StructuredAnswer
from backend.app.rag.text_format import markdown_to_plain_text

load_dotenv()
logger = logging.getLogger(__name__)

_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


class Generator:
    """
    RAG answer generator with:
      - reframed grounded prompt
      - Pydantic StructuredAnswer (when the LLM supports structured output)
      - markdown → plain text normalization
    """

    def __init__(self, llm):
        self.llm = llm

    @traceable(name="llm_generation")
    def generate(self, query: str, context: str) -> str:
        structured = self.generate_structured(query, context)
        return structured.to_plain_text()

    def generate_structured(self, query: str, context: str) -> StructuredAnswer:
        prompt = self._build_prompt(query, context)
        raw: Any = None

        # Prefer native structured output (ChatOllama / ChatGroq)
        try:
            if hasattr(self.llm, "llm") and hasattr(self.llm.llm, "with_structured_output"):
                structured_llm = self.llm.llm.with_structured_output(StructuredAnswer)
                raw = structured_llm.invoke(prompt)
                if isinstance(raw, StructuredAnswer):
                    return self._normalize_structured(raw)
                if isinstance(raw, dict):
                    return self._normalize_structured(StructuredAnswer.model_validate(raw))
        except Exception as e:
            logger.warning("Structured LLM bind failed (%s); falling back to JSON parse.", e)

        # Fallback: free-form generation + JSON / plain parse
        try:
            text = self.llm.generate(prompt)
        except Exception as e:
            logger.error("LLM generate failed: %s", e)
            return StructuredAnswer(
                answer="I could not generate an answer right now.",
                no_context=False,
            )

        parsed = self._parse_structured_text(text)
        return self._normalize_structured(parsed)

    def stream_generate(self, query: str, context: str) -> Generator[str, None, None]:
        """
        Produce a plain-text answer. Uses structured generation then streams
        the flattened plain text in small chunks for UI compatibility.
        """
        plain = self.generate(query, context)
        # Stream by paragraph so the UI still feels progressive
        parts = plain.split("\n\n")
        for i, part in enumerate(parts):
            chunk = part if i == len(parts) - 1 else part + "\n\n"
            if chunk:
                yield chunk

    def _normalize_structured(self, structured: StructuredAnswer) -> StructuredAnswer:
        answer = markdown_to_plain_text(structured.answer)
        sections = []
        for sec in structured.sections:
            sections.append(
                sec.model_copy(
                    update={
                        "title": markdown_to_plain_text(sec.title),
                        "content": markdown_to_plain_text(sec.content),
                    }
                )
            )
        note = (
            markdown_to_plain_text(structured.confidence_note)
            if structured.confidence_note
            else None
        )
        return structured.model_copy(
            update={"answer": answer, "sections": sections, "confidence_note": note}
        )

    def _parse_structured_text(self, text: str) -> StructuredAnswer:
        cleaned = (text or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        candidate = cleaned
        match = _JSON_BLOCK.search(cleaned)
        if match:
            candidate = match.group(0)

        try:
            data = json.loads(candidate)
            return StructuredAnswer.model_validate(data)
        except Exception:
            plain = markdown_to_plain_text(cleaned)
            no_ctx = "does not contain information" in plain.lower()
            return StructuredAnswer(answer=plain, no_context=no_ctx)

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return f"""You are AEGIS, a grounded retrieval assistant.

TASK
Answer the user question using ONLY the Context below.

OUTPUT FORMAT (STRICT)
Return a single JSON object matching this schema:
{{
  "answer": "plain text primary answer",
  "sections": [
    {{"title": "optional section title", "content": "plain text details"}}
  ],
  "no_context": false,
  "confidence_note": null
}}

PLAIN TEXT RULES
- Do NOT use markdown: no # headings, no **, no *, no `, no ---, no bullet "-" lists.
- For lists use "1. " "2. " "3. " or the "• " character.
- Write complete sentences. Prefer concrete facts from context (names, dates, skills, projects).
- If context is partial, still answer what is available and set confidence_note briefly.
- Set no_context=true ONLY when context is empty or clearly unrelated; then answer may be empty.

Context:
{context}

Question:
{query}
"""
