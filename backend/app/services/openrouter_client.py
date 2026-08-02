"""OpenRouter OpenAI-compatible client — fallback when Groq fails (free-tier models)."""

from __future__ import annotations

import base64
import json
import logging
from typing import Any, Dict, Generator, List, Optional

import httpx

from backend.app.core.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_VISION_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_APP_NAME,
    OPENROUTER_SITE_URL,
)

logger = logging.getLogger(__name__)


def openrouter_configured() -> bool:
    return bool(OPENROUTER_API_KEY and OPENROUTER_API_KEY.strip())


def openrouter_vision_configured() -> bool:
    key = (OPENROUTER_VISION_API_KEY or OPENROUTER_API_KEY or "").strip()
    return bool(key)


def _headers(api_key: Optional[str] = None) -> Dict[str, str]:
    key = (api_key or OPENROUTER_API_KEY or "").strip()
    h = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if OPENROUTER_SITE_URL:
        h["HTTP-Referer"] = OPENROUTER_SITE_URL
    if OPENROUTER_APP_NAME:
        h["X-Title"] = OPENROUTER_APP_NAME
    return h


def chat_completion(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 1500,
    timeout: float = 120.0,
    api_key: Optional[str] = None,
) -> str:
    key = (api_key or OPENROUTER_API_KEY or "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=_headers(key), json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter chat {resp.status_code}: {resp.text[:400]}"
            )
        data = resp.json()
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"Unexpected OpenRouter response: {data!r}") from e


def chat_completion_stream(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 1500,
    timeout: float = 180.0,
    api_key: Optional[str] = None,
) -> Generator[str, None, None]:
    key = (api_key or OPENROUTER_API_KEY or "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/chat/completions"
    with httpx.Client(timeout=timeout) as client:
        with client.stream("POST", url, headers=_headers(key), json=payload) as resp:
            if resp.status_code >= 400:
                body = resp.read().decode("utf-8", errors="ignore")[:400]
                raise RuntimeError(f"OpenRouter stream {resp.status_code}: {body}")
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    data = line[5:].strip()
                else:
                    data = line.strip()
                if not data or data == "[DONE]":
                    if data == "[DONE]":
                        break
                    continue
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                try:
                    delta = chunk["choices"][0].get("delta") or {}
                    content = delta.get("content")
                    if content:
                        yield content
                except (KeyError, IndexError, TypeError):
                    continue


def vision_chat_completion(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: float = 0.1,
    max_tokens: int = 220,
    timeout: float = 120.0,
) -> str:
    """Vision fallback uses OPENROUTER_VISION_API_KEY when set."""
    key = (OPENROUTER_VISION_API_KEY or OPENROUTER_API_KEY or "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_VISION_API_KEY / OPENROUTER_API_KEY is not set")
    return chat_completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        api_key=key,
    )


def transcribe_audio(
    *,
    model: str,
    filename: str,
    audio_bytes: bytes,
    timeout: float = 300.0,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    OpenRouter STT via JSON body (base64 input_audio).
    Only used when OPENROUTER_WHISPER_MODEL is set to a free STT model.
    """
    key = (api_key or OPENROUTER_API_KEY or "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set")
    if not model or not model.strip():
        raise RuntimeError("OPENROUTER_WHISPER_MODEL is empty — ASR fallback disabled")

    fmt = "mp3"
    if "." in filename:
        fmt = filename.rsplit(".", 1)[-1].lower() or "mp3"

    url = f"{OPENROUTER_BASE_URL.rstrip('/')}/audio/transcriptions"
    payload = {
        "model": model,
        "input_audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": fmt,
        },
    }
    with httpx.Client(timeout=timeout) as client:
        resp = client.post(url, headers=_headers(key), json=payload)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"OpenRouter ASR {resp.status_code}: {resp.text[:400]}"
            )
        return resp.json()
