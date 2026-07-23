import base64
import json
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from backend.app.core.config import OLLAMA_TEMP

class VisionLLM:
    """
    Wrapper around Qwen2.5-VL via Ollama for semantic image understanding.
    """
    def __init__(self, model_name: str = "qwen2.5-vl"):
        # We enforce format="json" if supported by the Ollama version, 
        # or rely on prompt engineering to get pure JSON.
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.0, # Low temperature for factual image description
            num_predict=1024,
            format="json"
        )
        
        self.system_prompt = """You are an image analysis model used in a Multimodal RAG ingestion pipeline.
Analyze the provided extracted PDF image and return a JSON object.

Tasks:
1. Identify the image type: Diagram, Flowchart, Architecture Diagram, Table, Chart, Graph, Screenshot, Photograph, Equation, UI Screenshot, Other.
2. Extract all visible text (OCR).
3. Describe the image in detail.
4. Explain the relationships between components.
5. If it is a chart, summarize the trend.
6. If it is a table, summarize the data.
7. If it is a workflow, explain each step.
8. Generate 10-20 retrieval keywords.

Return ONLY valid JSON in this exact format, with no markdown wrappers or extra text:
{
  "image_type": "",
  "topic": "",
  "ocr_text": [],
  "description": "",
  "relationships": [],
  "keywords": []
}"""

    def describe_image(self, image_path: str) -> dict:
        """
        Takes an image path, sends it to Qwen2.5-VL, and returns the parsed JSON dict.
        """
        with open(image_path, "rb") as img_file:
            img_b64 = base64.b64encode(img_file.read()).decode("utf-8")
            
        message = HumanMessage(content=[
            {"type": "text", "text": self.system_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ])
        
        try:
            response = self.llm.invoke([message])
            content = response.content.strip()
            
            # Clean up potential markdown formatting if the model ignored `format="json"`
            if content.startswith("```json"):
                content = content.replace("```json", "", 1)
            if content.endswith("```"):
                content = content.rsplit("```", 1)[0]
                
            return json.loads(content.strip())
        except Exception as e:
            print(f"VisionLLM Error processing {image_path}: {e}")
            return {
                "image_type": "Unknown",
                "topic": "Error parsing image",
                "ocr_text": [],
                "description": f"Failed to generate description: {str(e)}",
                "relationships": [],
                "keywords": []
            }
