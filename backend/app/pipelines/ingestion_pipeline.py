import tempfile
import os
from backend.app.db.database import SessionLocal
from backend.app.db.models import Artifact, VectorEmbedding, Metadata
from backend.app.storage.minio_client import minio_client
from backend.app.services.parsing_service import parsing_service
from backend.app.services.embedding_service import embedding_service
from backend.app.ingestion.chunking.semantic_chunker import SemanticChunker

class MultimodalIngestionPipeline:
    def __init__(self):
        self.chunker = SemanticChunker()

    def process_artifact(self, artifact_id: int):
        db = SessionLocal()
        try:
            artifact = db.query(Artifact).filter_by(id=artifact_id).first()
            if not artifact:
                return
            
            # Download from MinIO
            bucket, obj_name = artifact.file_path.split("/", 1)
            
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                
            minio_client.client.fget_object(bucket, obj_name, temp_path)

            if artifact.modality == "document":
                self._process_document(db, artifact, temp_path)
            elif artifact.modality == "image":
                self._process_image(db, artifact, temp_path)
            elif artifact.modality == "video":
                self._process_video(db, artifact, temp_path)

            os.remove(temp_path)
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def _process_document(self, db, artifact, file_path):
        import re
        from backend.app.models.vision_llm import VisionLLM
        from backend.app.ingestion.chunking.markdown_chunker import MarkdownHierarchicalChunker
        import shutil
        
        # 1. Parse PDF using LiteParse
        parsed = parsing_service.parse_document(file_path, original_filename=artifact.filename)
        md_path = parsed.get("markdown_path")
        images_dir = parsed.get("images_dir")
        
        if not md_path or not os.path.exists(md_path):
            raise ValueError("Markdown generation failed.")
            
        with open(md_path, "r", encoding="utf-8") as f:
            markdown_content = f.read()
            
        # 3 & 4. Image Understanding & Merge
        if images_dir and os.path.exists(images_dir):
            vision_llm = VisionLLM()
            
            # Find all markdown images: ![](path/to/image.png) or ![alt](path/to/image.png)
            # We use a regex replacement function to swap them out
            def replace_image(match):
                alt_text = match.group(1)
                img_rel_path = match.group(2)
                
                # Resolve actual image path
                img_filename = os.path.basename(img_rel_path)
                full_img_path = os.path.join(images_dir, img_filename)
                
                if os.path.exists(full_img_path):
                    try:
                        desc_json = vision_llm.describe_image(full_img_path)
                        # Format into a nice markdown block
                        desc_md = f"\n\n**Figure Description ({img_filename}):**\n"
                        desc_md += f"- **Type**: {desc_json.get('image_type', 'Unknown')}\n"
                        desc_md += f"- **Topic**: {desc_json.get('topic', '')}\n"
                        desc_md += f"- **Description**: {desc_json.get('description', '')}\n"
                        if desc_json.get('ocr_text'):
                            desc_md += f"- **Visible Text**: {', '.join(desc_json.get('ocr_text', []))}\n"
                        desc_md += "\n"
                        return desc_md
                    except Exception as e:
                        print(f"Failed to describe image {img_filename}: {e}")
                        return match.group(0) # fallback to original
                else:
                    return match.group(0)

            markdown_content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image, markdown_content)
            
        # 5. Sentence-aware Chunking
        chunker = MarkdownHierarchicalChunker()
        chunks = chunker.chunk(markdown_content, artifact_name=artifact.filename)
        
        # 6 & 7. Metadata and Embedding
        for chunk_data in chunks:
            text = chunk_data["text"]
            if not text.strip():
                continue
                
            # Prepend metadata to the content to enrich the embedding
            meta = chunk_data["metadata"]
            enriched_text = (
                f"[Document: {meta.get('document_name')}]\n"
                f"[Hierarchy: {meta.get('hierarchy')}]\n"
                f"[Contains Table: {meta.get('contains_table')}]\n"
                f"[Contains Image: {meta.get('contains_image')}]\n"
                f"\n{text}"
            )
            
            embedding = embedding_service.embed_text(enriched_text)
            vec = VectorEmbedding(
                artifact_id=artifact.id,
                embedding_type="text",
                content=enriched_text,
                embedding=embedding
            )
            db.add(vec)
            
        db.commit()
        
        # Cleanup output dir
        output_dir = os.path.dirname(md_path)
        shutil.rmtree(output_dir, ignore_errors=True)

    def _process_image(self, db, artifact, file_path):
        # Placeholder for Object Detection (e.g., Florence-2)
        # bounding_boxes = florence2.detect_objects(file_path)
        bounding_boxes = [{"label": "placeholder_object", "box": [0,0,10,10]}]
        
        meta = Metadata(artifact_id=artifact.id, key="detected_objects", value=bounding_boxes)
        db.add(meta)
        
        # Visual Embedding using SigLIP2
        embedding = embedding_service.embed_image(file_path)
        vec = VectorEmbedding(
            artifact_id=artifact.id,
            embedding_type="vision",
            content=f"[Image File: {artifact.filename}]",
            embedding=embedding
        )
        db.add(vec)
        db.commit()
        
    def _process_video(self, db, artifact, file_path):
        # Placeholder for Keyframe Extraction & Scene Summarization
        # Here we would extract frames, and run them through _process_image logic
        # Or summarize scenes using a Vision LLM and embed the text
        meta = Metadata(artifact_id=artifact.id, key="video_scenes", value=["Scene 1 placeholder"])
        db.add(meta)
        db.commit()