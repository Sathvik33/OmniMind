"""
GroqVisionService — Cloud-powered image/frame captioning for AEGIS v3.0.

Replaces local LLaVA (slow, GPU-heavy) with Groq's vision model:
  - meta-llama/llama-4-scout-17b-16e-instruct  (vision capable, fast)

How it works:
  1. Load image from disk
  2. Resize + compress to JPEG (reduces API payload)
  3. Base64-encode → send to Groq vision API
  4. Returns a concise 1-2 sentence description

Production pattern:
  Google uses Gemini Vision, OpenAI uses GPT-4o Vision, Meta uses
  Llama-4 Vision. All follow the same encode→API→caption pattern.
  Groq gives us near-OpenAI quality at very high speed (speculative decoding).
"""

import base64
import io
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from PIL import Image
from groq import Groq

logger = logging.getLogger(__name__)

_GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
_DESCRIBE_PROMPT   = (
    "Describe this image concisely in 1-2 sentences. "
    "Include: what is shown, any visible text, key objects or actions. "
    "Be factual and precise."
)


class GroqVisionService:
    """
    Cloud vision service using Groq's Llama-4 Scout model.

    Same .describe(image_path) interface as the local LLaVA VisionService,
    so VideoService and image API require zero changes.
    """

    def __init__(self, model: str = _GROQ_VISION_MODEL):
        api_key = (
            os.getenv("GROQ_GENERATION_API_KEY")
            or os.getenv("GROQ_API_KEY", "")
        )
        if not api_key:
            raise ValueError(
                "GROQ_GENERATION_API_KEY is not set. Add it to your .env file."
            )
        self.client = Groq(api_key=api_key)
        self.model  = model
        logger.info(f"✅ GroqVisionService initialized — model: {model}")

    # ── Public ─────────────────────────────────────────────────────────────────

    def describe(self, image_path: str) -> str:
        """
        Describe an image using Groq's vision model.

        Args:
            image_path: Absolute path to the image file (JPEG, PNG, etc.)

        Returns:
            1-2 sentence description string.
        """
        try:
            b64 = self._encode_image(image_path)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type":      "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": _DESCRIBE_PROMPT,
                            },
                        ],
                    }
                ],
                max_tokens=120,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"GroqVisionService.describe failed for {image_path}: {e}")
            return f"[Vision processing failed: {e}]"

    # ── Private ────────────────────────────────────────────────────────────────

    def _encode_image(self, image_path: str) -> str:
        """
        Resize image to max 512px wide (reduces payload), encode as JPEG base64.
        Groq vision API accepts base64-encoded data URIs.
        """
        with Image.open(image_path).convert("RGB") as img:
            # Resize: keep aspect ratio, max 512px on longest side
            img.thumbnail((512, 512), Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
