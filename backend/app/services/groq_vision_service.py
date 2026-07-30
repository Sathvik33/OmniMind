"""
GroqVisionService — Cloud-powered image/frame captioning for AEGIS v3.0.

Uses Groq's current vision-capable model:
  - qwen/qwen3.6-27b  (multimodal: text + image, OCR, captioning)

How it works:
  1. Load image from disk
  2. Resize + compress to JPEG (reduces API payload)
  3. Base64-encode → send to Groq vision API
  4. Returns a concise 1-2 sentence description

Requires GROQ_VISION_API_KEY (falls back to GROQ_API_KEY via config).
"""

import base64
import io
import logging
import os

from dotenv import load_dotenv
load_dotenv()

from PIL import Image
from groq import Groq

from backend.app.core.config import GROQ_VISION_API_KEY, GROQ_VISION_MODEL

logger = logging.getLogger(__name__)

_DESCRIBE_PROMPT = (
    "Describe this image concisely in 1-2 sentences. "
    "Include: what is shown, any visible text, key objects or actions. "
    "Be factual and precise."
)


class GroqVisionService:
    """
    Cloud vision service using Groq's Qwen 3.6 27B multimodal model.

    Same .describe(image_path) interface as other vision services,
    so VideoService and ingestion require zero changes.
    """

    def __init__(self, model: str | None = None):
        api_key = GROQ_VISION_API_KEY or os.getenv("GROQ_VISION_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_VISION_API_KEY is not set. Add it to your .env file."
            )
        self.client = Groq(api_key=api_key)
        self.model = model or GROQ_VISION_MODEL or "qwen/qwen3.6-27b"
        logger.info(f"GroqVisionService initialized — model: {self.model}")

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
                                "type": "image_url",
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

    def _encode_image(self, image_path: str) -> str:
        """
        Resize image to max 512px wide (reduces payload), encode as JPEG base64.
        Groq vision API accepts base64-encoded data URIs.
        """
        with Image.open(image_path).convert("RGB") as img:
            img.thumbnail((512, 512), Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            buf.seek(0)
            return base64.b64encode(buf.read()).decode("utf-8")
