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
        parsed = parsing_service.parse_document(file_path)
        chunks = self.chunker.chunk(parsed.get("text", ""))
        
        for chunk in chunks:
            if not chunk.strip():
                continue
            embedding = embedding_service.embed_text(chunk)
            vec = VectorEmbedding(
                artifact_id=artifact.id,
                embedding_type="text",
                content=chunk,
                embedding=embedding
            )
            db.add(vec)
        db.commit()

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