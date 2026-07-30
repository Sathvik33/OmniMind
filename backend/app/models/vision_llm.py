import logging
from backend.app.core.config import GROQ_VISION_MODEL
from backend.app.services.groq_vision_service import GroqVisionService

logger = logging.getLogger(__name__)


class VisionLLM:
    """
    Cloud-powered vision analysis using Groq Qwen 3.6 via GROQ_VISION_API_KEY.
    """

    def __init__(self, model_name: str | None = None):
        self.vision_service = GroqVisionService(model=model_name or GROQ_VISION_MODEL)

    def describe_image(self, image_path: str) -> dict:
        """
        Describes image using Groq vision model and formats response as dict.
        """
        try:
            description = self.vision_service.describe(image_path)
            return {
                "image_type": "Figure/Diagram",
                "topic": "Extracted Image Content",
                "ocr_text": [],
                "description": description,
                "relationships": [],
                "keywords": [],
            }
        except Exception as e:
            logger.error(f"VisionLLM Error processing {image_path}: {e}")
            return {
                "image_type": "Unknown",
                "topic": "Error parsing image",
                "ocr_text": [],
                "description": f"Failed to generate description: {str(e)}",
                "relationships": [],
                "keywords": [],
            }
