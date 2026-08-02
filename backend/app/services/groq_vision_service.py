"""
Vision captioning for images / video keyframes.

Primary: Groq vision.
On Groq rate-limit / hard failure: OpenRouter free VL (OPENROUTER_VISION_API_KEY).
Once a 429 is seen, subsequent frames in this process prefer OpenRouter so
video keyframe batches do not keep hammering Groq TPM.
"""

import base64
import io
import logging
import os
import threading

from dotenv import load_dotenv
load_dotenv()

from PIL import Image
from groq import Groq

from backend.app.core.config import (
    GROQ_VISION_API_KEY,
    GROQ_VISION_MODEL,
    OPENROUTER_VISION_MODEL,
)
from backend.app.services.groq_retry import is_rate_limit_error, should_use_fallback
from backend.app.services.openrouter_client import (
    openrouter_vision_configured,
    vision_chat_completion,
)

logger = logging.getLogger(__name__)

_DESCRIBE_PROMPT = (
    "Describe this video frame in 2-3 factual sentences. "
    "Name key objects: vehicles (type, color, count if clear), people, buildings, "
    "signs or on-screen text, and the setting/lighting/action. "
    "Be specific and concrete. Do not speculate."
)

# Shared across VideoService thread-pool workers in one Celery process
_force_openrouter = False
_force_lock = threading.Lock()


def _mark_openrouter_preferred(reason: str) -> None:
    global _force_openrouter
    with _force_lock:
        if not _force_openrouter:
            _force_openrouter = True
            logger.warning(
                "Groq vision limited (%s) — remaining frames use OpenRouter VL",
                reason,
            )


class GroqVisionService:
    """
    Image/frame describe() for VideoService + image ingest.

    Groq first; OpenRouter free VL on rate-limit / provider failure.
    """

    def __init__(self, model: str | None = None):
        api_key = GROQ_VISION_API_KEY or os.getenv("GROQ_VISION_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_VISION_API_KEY is not set. Add it to your .env file."
            )
        self.client = Groq(api_key=api_key)
        self.model = model or GROQ_VISION_MODEL or "qwen/qwen3.6-27b"
        logger.info(
            "GroqVisionService initialized — model: %s (OpenRouter VL fallback: %s)",
            self.model,
            openrouter_vision_configured(),
        )

    def describe(self, image_path: str) -> str:
        """Describe an image / video keyframe. Falls back to OpenRouter on Groq 429."""
        try:
            b64 = self._encode_image(image_path)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _DESCRIBE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                }
            ]

            use_or = _force_openrouter and openrouter_vision_configured()
            if use_or:
                return self._openrouter_describe(messages, image_path=image_path)

            try:
                return self._groq_describe(b64, image_path=image_path)
            except Exception as groq_err:
                if openrouter_vision_configured() and (
                    is_rate_limit_error(groq_err) or should_use_fallback(groq_err)
                ):
                    _mark_openrouter_preferred(str(groq_err)[:120])
                    return self._openrouter_describe(messages, image_path=image_path)
                raise

        except Exception as e:
            logger.error("GroqVisionService.describe failed for %s: %s", image_path, e)
            if is_rate_limit_error(e):
                return f"[Vision processing failed: 429 rate limit: {e}]"
            return f"[Vision processing failed: {e}]"

    def _groq_describe(self, b64: str, *, image_path: str = "") -> str:
        from backend.app.monitoring.langsmith_logger import tracer

        with tracer.model_call(
            name=f"vision-llm:groq:{self.model}",
            tags=["vision-llm", "groq-vision", "groq", "caption"],
            provider="groq",
            model=self.model,
            inputs={
                "image_path": image_path,
                "prompt": _DESCRIBE_PROMPT,
                "b64_chars": len(b64),
            },
            run_type="llm",
        ) as span:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            },
                            {"type": "text", "text": _DESCRIBE_PROMPT},
                        ],
                    }
                ],
                max_tokens=220,
                temperature=0.1,
            )
            text = response.choices[0].message.content.strip()
            span["output"] = text
            return text

    def _openrouter_describe(self, messages, *, image_path: str = "") -> str:
        from backend.app.monitoring.langsmith_logger import tracer

        logger.info("Vision caption via OpenRouter (%s)", OPENROUTER_VISION_MODEL)
        with tracer.model_call(
            name=f"vision-llm:openrouter:{OPENROUTER_VISION_MODEL}",
            tags=["vision-llm", "openrouter-vision", "openrouter", "caption", "free-tier"],
            provider="openrouter",
            model=OPENROUTER_VISION_MODEL,
            inputs={"image_path": image_path, "prompt": _DESCRIBE_PROMPT},
            run_type="llm",
        ) as span:
            text = vision_chat_completion(
                model=OPENROUTER_VISION_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=220,
            )
            span["output"] = text
            return text

    def _encode_image(self, image_path: str) -> str:
        with Image.open(image_path).convert("RGB") as img:
            img.thumbnail((512, 512), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
