from typing import Dict, Any
import os
from backend.app.db.database import SessionLocal
from backend.app.db.models import AIModelRegistry, ModelCategory

class DocumentParsingService:
    def __init__(self):
        self._load_active_parser()

    def _load_active_parser(self):
        """Fetch active parsing model from the Model Registry."""
        db = SessionLocal()
        try:
            self.parser_model = db.query(AIModelRegistry).filter_by(
                category=ModelCategory.PARSER, 
                is_active=1
            ).first()
        finally:
            db.close()

    def parse_document(self, file_path: str, original_filename: str = "") -> Dict[str, Any]:
        """
        Dynamically route to the best parser based on registry configuration and file complexity.
        """
        parser_name = self.parser_model.name if self.parser_model else "liteparse"
        
        # Simple routing logic
        if parser_name.lower() == "liteparse":
            return self._run_liteparse(file_path, original_filename)
        elif parser_name.lower() == "docling":
            return self._run_docling(file_path)
        elif parser_name.lower() == "llamaparse":
            return self._run_llamaparse(file_path)
        else:
            return self._run_liteparse(file_path, original_filename) # fallback

    def _run_liteparse(self, file_path: str, original_filename: str) -> Dict[str, Any]:
        text_content = ""
        try:
            if original_filename.lower().endswith('.pdf'):
                import fitz
                with fitz.open(file_path) as doc:
                    text_content = chr(10).join([page.get_text() for page in doc])
            else:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text_content = f.read()
        except Exception as e:
            text_content = f"Error parsing document {original_filename}: {e}"
            
        return {"text": text_content, "metadata": {}}

    def _run_docling(self, file_path: str) -> Dict[str, Any]:
        # TODO: Implement advanced structure-aware parsing
        return {"text": f"Parsed content of {file_path} using Docling", "metadata": {}}
        
    def _run_llamaparse(self, file_path: str) -> Dict[str, Any]:
        # TODO: Implement cloud API parsing
        return {"text": f"Parsed content of {file_path} using LlamaParse", "metadata": {}}

parsing_service = DocumentParsingService()
