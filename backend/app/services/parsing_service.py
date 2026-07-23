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
        import subprocess
        import tempfile
        import shutil
        
        # Create a unique output directory in the project root
        base_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(base_dir, exist_ok=True)
        
        # Use a temporary directory inside output to avoid collisions
        out_dir = tempfile.mkdtemp(dir=base_dir, prefix="liteparse_")
        
        md_path = os.path.join(out_dir, "document.md")
        json_path = os.path.join(out_dir, "document.json")
        images_dir = os.path.join(out_dir, "images")
        
        if original_filename.lower().endswith('.pdf'):
            try:
                # 1. Generate JSON
                subprocess.run(
                    ["lit", "parse", file_path, "--format", "json", "-o", json_path],
                    check=True, capture_output=True
                )
                
                # 2. Generate Markdown & Images
                subprocess.run(
                    ["lit", "parse", file_path, "--format", "markdown", "--image-mode", "embed", "--image-output-dir", images_dir, "-o", md_path],
                    check=True, capture_output=True
                )
                
                return {
                    "markdown_path": md_path,
                    "json_path": json_path,
                    "images_dir": images_dir
                }
            except subprocess.CalledProcessError as e:
                error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
                raise RuntimeError(f"LiteParse CLI failed: {error_msg}")
        else:
            # Fallback for non-PDFs
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text_content = f.read()
            
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(text_content)
                
            return {
                "markdown_path": md_path,
                "json_path": None,
                "images_dir": None
            }

    def _run_docling(self, file_path: str) -> Dict[str, Any]:
        # TODO: Implement advanced structure-aware parsing
        return {"text": f"Parsed content of {file_path} using Docling", "metadata": {}}
        
    def _run_llamaparse(self, file_path: str) -> Dict[str, Any]:
        # TODO: Implement cloud API parsing
        return {"text": f"Parsed content of {file_path} using LlamaParse", "metadata": {}}

parsing_service = DocumentParsingService()
