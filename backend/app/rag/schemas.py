from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class AnswerSection(BaseModel):
    """One titled block inside a structured answer. Content must be plain text."""

    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1, description="Plain text only — no markdown")

    @field_validator("title", "content", mode="before")
    @classmethod
    def _strip(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v


class StructuredAnswer(BaseModel):
    """
    Canonical LLM answer schema for AEGIS.

    The model must fill these fields with plain text (no markdown markers).
    """

    answer: str = Field(
        ...,
        description=(
            "Primary answer in plain text. Use short paragraphs and numbered lists "
            "with '1.' '2.' if needed. Never use markdown (#, **, `, bullets with -)."
        ),
    )
    sections: List[AnswerSection] = Field(
        default_factory=list,
        description="Optional detail sections (e.g. Project 1, Education). Plain text only.",
    )
    no_context: bool = Field(
        default=False,
        description="True only when the provided context has nothing relevant to the question.",
    )
    confidence_note: Optional[str] = Field(
        default=None,
        description="Optional short plain-text caveat when context is partial.",
    )

    @field_validator("answer", mode="before")
    @classmethod
    def _strip_answer(cls, v: object) -> object:
        return v.strip() if isinstance(v, str) else v

    def to_plain_text(self) -> str:
        """Flatten structured fields into a single plain-text response."""
        if self.no_context and not (self.answer or "").strip():
            return "The uploaded data does not contain information about this topic."

        parts: List[str] = []
        if self.answer.strip():
            parts.append(self.answer.strip())

        for section in self.sections:
            title = section.title.strip().rstrip(":")
            body = section.content.strip()
            parts.append(f"{title}\n{body}")

        if self.confidence_note and self.confidence_note.strip():
            parts.append(self.confidence_note.strip())

        text = "\n\n".join(parts).strip()
        return text or "The uploaded data does not contain information about this topic."
