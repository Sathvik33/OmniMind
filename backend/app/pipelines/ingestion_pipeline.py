import tempfile
import os
import shutil
import re
import logging
from typing import Callable, Optional
from backend.app.db.database import SessionLocal
from backend.app.db.models import Artifact, VectorEmbedding, Metadata
from backend.app.storage.minio_client import minio_client
from backend.app.services.parsing_service import parsing_service
from backend.app.services.embedding_service import embedding_service
from backend.app.core.exceptions import NonRetryableIngestionError, RetryableIngestionError

logger = logging.getLogger(__name__)



class MultimodalIngestionPipeline:
    _vision_llm = None

    def __init__(self):
        pass

    @classmethod
    def _get_vision_llm(cls):
        """Lazy-initialize VisionLLM once and reuse across all calls."""
        if cls._vision_llm is None:
            from backend.app.models.vision_llm import VisionLLM
            cls._vision_llm = VisionLLM()
        return cls._vision_llm

    def process_artifact(self, artifact_id: int, on_stage_change: Optional[Callable[[str], None]] = None):
        def notify(stage: str):
            if on_stage_change:
                on_stage_change(stage)

        db = SessionLocal()
        temp_path = None
        try:
            artifact = db.query(Artifact).filter_by(id=artifact_id).first()
            if not artifact:
                raise NonRetryableIngestionError(f"Artifact {artifact_id} not found in database.", stage="PARSING")
            
            # Download from MinIO
            bucket, obj_name = artifact.file_path.split("/", 1)
            ext = os.path.splitext(artifact.filename)[1] or ""
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                temp_path = temp_file.name

            try:
                minio_client.client.fget_object(bucket, obj_name, temp_path)
            except Exception as e:
                raise RetryableIngestionError(f"Failed to fetch object from MinIO: {e}", stage="PARSING")

            if artifact.modality == "document":
                self._process_document(db, artifact, temp_path, notify)
            elif artifact.modality == "image":
                self._process_image(db, artifact, temp_path, notify)
            elif artifact.modality == "video":
                self._process_video(db, artifact, temp_path, notify)

        except (NonRetryableIngestionError, RetryableIngestionError):
            db.rollback()
            raise
        except Exception as e:
            db.rollback()
            raise NonRetryableIngestionError(f"Unexpected pipeline error: {e}", stage="RUNNING")
        finally:
            # Always clean up temp file, even on exception
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
            db.close()

    def _process_document(self, db, artifact, file_path, notify: Callable[[str], None]):
        import fitz
        from backend.app.ingestion.chunking.markdown_chunker import MarkdownHierarchicalChunker
        
        # 1. PARSING Stage
        notify("PARSING")
        md_path = None
        images_dir = None
        try:
            parsed = parsing_service.parse_document(file_path, original_filename=artifact.filename)
            md_path = parsed.get("markdown_path")
            images_dir = parsed.get("images_dir")
            
            markdown_content = ""
            if md_path and os.path.exists(md_path):
                with open(md_path, "r", encoding="utf-8") as f:
                    markdown_content = f.read()
        except Exception as e:
            logger.warning(f"LiteParse document parsing warning: {e}")
            markdown_content = ""
            
        # 2. VISION CAPTIONING & SCANNED PDF PAGE RENDERING Stage
        notify("VISION_CAPTIONING")
        vision_llm = self._get_vision_llm()
        processed_images = set()

        # If document is a PDF: check if pages contain images or sparse text (< 40 words per page or < 300 chars overall)
        if artifact.filename.lower().endswith('.pdf'):
            try:
                pdf_doc = fitz.open(file_path)
                should_render_pages = False
                if len(markdown_content.strip()) < 300:
                    should_render_pages = True
                else:
                    for page in pdf_doc:
                        if len(page.get_images()) > 0 or len(page.get_text().split()) < 40:
                            should_render_pages = True
                            break

                if should_render_pages:
                    logger.info(f"📷 Certificate/Scanned PDF page detected ({artifact.filename}). Rendering pages to images with PyMuPDF...")
                    render_dir = tempfile.mkdtemp(prefix="rendered_pdf_")
                    for page_num in range(len(pdf_doc)):
                        page = pdf_doc[page_num]
                        pix = page.get_pixmap(dpi=200)
                        img_file = os.path.join(render_dir, f"page_{page_num + 1}.png")
                        pix.save(img_file)
                    images_dir = render_dir
            except Exception as render_err:
                logger.error(f"Failed to render PDF pages with PyMuPDF: {render_err}")


        vision_descriptions = []
        vision_vectors_to_add = []

        if images_dir and os.path.exists(images_dir):
            def replace_image(match):
                img_rel_path = match.group(2)
                img_filename = os.path.basename(img_rel_path)
                full_img_path = os.path.join(images_dir, img_filename)
                
                if os.path.exists(full_img_path):
                    processed_images.add(full_img_path)
                    try:
                        desc_json = vision_llm.describe_image(full_img_path)
                        desc_md = f"\n\n<!-- image: true -->\n**Figure/Page Description ({img_filename}):**\n"
                        desc_md += f"- **Type**: {desc_json.get('image_type', 'Unknown')}\n"
                        desc_md += f"- **Topic**: {desc_json.get('topic', '')}\n"
                        desc_md += f"- **Description**: {desc_json.get('description', '')}\n"
                        if desc_json.get('ocr_text'):
                            desc_md += f"- **Visible Text / OCR**: {', '.join(desc_json.get('ocr_text', []))}\n"
                        desc_md += "\n"
                        return desc_md
                    except Exception as e:
                        if "429" in str(e) or "500" in str(e) or "timeout" in str(e).lower():
                            raise RetryableIngestionError(f"Groq Vision API rate limit or timeout: {e}", stage="VISION_CAPTIONING")
                        return match.group(0)
                return match.group(0)

            markdown_content = re.sub(r'!\[(.*?)\]\((.*?)\)', replace_image, markdown_content)

            # Process any remaining un-captioned images in images_dir (e.g. rendered PDF pages)
            for img_name in os.listdir(images_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    full_img_path = os.path.join(images_dir, img_name)
                    if full_img_path not in processed_images:
                        processed_images.add(full_img_path)
                        try:
                            desc_json = vision_llm.describe_image(full_img_path)
                            desc_md = f"\n\n<!-- image: true -->\n**[Document Page/Certificate Description ({img_name})]:**\n"
                            desc_md += f"- **Type**: {desc_json.get('image_type', 'Certificate/Document')}\n"
                            desc_md += f"- **Topic**: {desc_json.get('topic', '')}\n"
                            desc_md += f"- **Description**: {desc_json.get('description', '')}\n"
                            if desc_json.get('ocr_text'):
                                desc_md += f"- **Visible Text / Certificate OCR**: {', '.join(desc_json.get('ocr_text', []))}\n"
                            desc_md += "\n"
                            vision_descriptions.append(desc_md)

                            # Visual embedding via SigLIP
                            img_emb = embedding_service.embed_image(full_img_path)
                            vec_vis = VectorEmbedding.make(
                                artifact_id=artifact.id,
                                embedding_type="vision",
                                content=f"[Image File: {artifact.filename} - {img_name}]\n{desc_json.get('description', '')}",
                                embedding=img_emb
                            )
                            vision_vectors_to_add.append(vec_vis)
                        except Exception as e:
                            if "429" in str(e) or "500" in str(e) or "timeout" in str(e).lower():
                                raise RetryableIngestionError(f"Groq Vision API rate limit or timeout: {e}", stage="VISION_CAPTIONING")

        if vision_descriptions:
            markdown_content += "\n\n" + "\n".join(vision_descriptions)

        if not markdown_content.strip():
            markdown_content = f"# Document: {artifact.filename}\n\n[Scanned document content - no readable text layer found]"

        # Guard: binary / corrupted "text" produces endless micro-chunks and looks like an infinite Celery loop
        sample = markdown_content[:4000]
        non_printable = sum(1 for ch in sample if ord(ch) < 9 or (13 < ord(ch) < 32))
        if sample and (non_printable / max(len(sample), 1)) > 0.08:
            raise NonRetryableIngestionError(
                f"Extracted content for {artifact.filename} looks binary/corrupt. "
                "Use LiteParse with LibreOffice for Office files, or re-export the document.",
                stage="PARSING",
            )

        # 3. CHUNKING & EMBEDDING Stage
        notify("CHUNKING_EMBEDDING")
        ingest_run_id = None
        try:
            from backend.app.monitoring.langsmith_logger import tracer as aegis_tracer

            if aegis_tracer.is_active():
                ingest_run_id = aegis_tracer.start_ingest_run(
                    filename=artifact.filename,
                    artifact_id=artifact.id,
                    metadata={"modality": artifact.modality},
                )
        except Exception:
            ingest_run_id = None

        try:
            import time as _time

            t_chunk = _time.perf_counter()
            chunker = MarkdownHierarchicalChunker()
            chunks = chunker.chunk(markdown_content, artifact_name=artifact.filename)
            max_chunks = int(os.getenv("MAX_INGEST_CHUNKS", "200"))
            truncated_to = None
            if len(chunks) > max_chunks:
                logger.warning(
                    "Truncating chunks for %s from %s to %s",
                    artifact.filename,
                    len(chunks),
                    max_chunks,
                )
                truncated_to = max_chunks
                chunks = chunks[:max_chunks]

            if ingest_run_id:
                try:
                    from backend.app.monitoring.langsmith_logger import tracer as aegis_tracer

                    aegis_tracer.log_chunking(
                        run_id=ingest_run_id,
                        filename=artifact.filename,
                        chunk_count=len(chunks),
                        truncated_to=truncated_to,
                        markdown_chars=len(markdown_content or ""),
                        sample_hierarchies=[
                            (c.get("metadata") or {}).get("hierarchy", "")
                            for c in chunks[:8]
                        ],
                        sample_previews=[(c.get("text") or "")[:200] for c in chunks[:5]],
                        latency_ms=round((_time.perf_counter() - t_chunk) * 1000, 2),
                    )
                except Exception as ls_err:
                    logger.debug("LangSmith chunking log skipped: %s", ls_err)

            logger.info(
                "Embedding %s text chunks for artifact %s (%s)",
                len(chunks),
                artifact.id,
                artifact.filename,
            )

            vectors_to_add = list(vision_vectors_to_add)
            for idx, chunk_data in enumerate(chunks, start=1):
                text = chunk_data["text"]
                if not text.strip():
                    continue

                meta = chunk_data["metadata"]
                enriched_text = (
                    f"[Document: {meta.get('document_name')}]\n"
                    f"[Hierarchy: {meta.get('hierarchy')}]\n"
                    f"[Contains Table: {meta.get('contains_table')}]\n"
                    f"[Contains Image: {meta.get('contains_image')}]\n"
                    f"\n{text}"
                )

                embedding = embedding_service.embed_text(enriched_text)
                if not embedding:
                    raise NonRetryableIngestionError(
                        f"Empty embedding returned for chunk {idx}/{len(chunks)}",
                        stage="CHUNKING_EMBEDDING",
                    )

                vec = VectorEmbedding.make(
                    artifact_id=artifact.id,
                    embedding_type="text",
                    content=enriched_text,
                    embedding=embedding,
                )
                vectors_to_add.append(vec)

                if idx % 25 == 0:
                    notify("CHUNKING_EMBEDDING")  # heartbeat while embedding large docs
                    logger.info("Embedded %s/%s chunks for artifact %s", idx, len(chunks), artifact.id)

            if not vectors_to_add:
                raise NonRetryableIngestionError(
                    f"No embeddable content extracted from {artifact.filename}",
                    stage="CHUNKING_EMBEDDING",
                )
        except NonRetryableIngestionError as e:
            if ingest_run_id:
                try:
                    from backend.app.monitoring.langsmith_logger import tracer as aegis_tracer
                    aegis_tracer.end_ingest_run(ingest_run_id, error=str(e))
                except Exception:
                    pass
            raise
        except Exception as e:
            if ingest_run_id:
                try:
                    from backend.app.monitoring.langsmith_logger import tracer as aegis_tracer
                    aegis_tracer.end_ingest_run(ingest_run_id, error=str(e))
                except Exception:
                    pass
            raise NonRetryableIngestionError(f"Chunking or embedding failed: {e}", stage="CHUNKING_EMBEDDING")

        # 4. STORING Stage

        notify("STORING")
        try:
            for vec in vectors_to_add:
                db.add(vec)
            db.commit()
            if ingest_run_id:
                try:
                    from backend.app.monitoring.langsmith_logger import tracer as aegis_tracer
                    aegis_tracer.end_ingest_run(
                        ingest_run_id, chunk_count=len(vectors_to_add)
                    )
                except Exception:
                    pass
        except Exception as e:
            db.rollback()
            if ingest_run_id:
                try:
                    from backend.app.monitoring.langsmith_logger import tracer as aegis_tracer
                    aegis_tracer.end_ingest_run(ingest_run_id, error=str(e))
                except Exception:
                    pass
            raise RetryableIngestionError(f"Database vector insertion failed: {e}", stage="STORING")
        
        if md_path and os.path.exists(md_path):
            output_dir = os.path.dirname(md_path)
            shutil.rmtree(output_dir, ignore_errors=True)
        if images_dir and os.path.exists(images_dir) and "rendered_pdf_" in images_dir:
            shutil.rmtree(images_dir, ignore_errors=True)

    def _process_image(self, db, artifact, file_path, notify: Callable[[str], None]):
        notify("VISION_CAPTIONING")
        
        vision_llm = self._get_vision_llm()
        desc_text = f"[Image File: {artifact.filename}]"
        try:
            desc_json = vision_llm.describe_image(file_path)
            desc_text = (
                f"[Image Document: {artifact.filename}]\n"
                f"- Type: {desc_json.get('image_type', 'Image')}\n"
                f"- Topic: {desc_json.get('topic', '')}\n"
                f"- Description: {desc_json.get('description', '')}\n"
            )
            if desc_json.get("ocr_text"):
                ocr = desc_json.get("ocr_text")
                ocr_str = ", ".join(ocr) if isinstance(ocr, list) else str(ocr)
                desc_text += f"- Visible Text / OCR: {ocr_str}\n"
        except Exception as e:
            if "429" in str(e) or "500" in str(e) or "timeout" in str(e).lower():
                raise RetryableIngestionError(f"Groq Vision API rate limit or timeout: {e}", stage="VISION_CAPTIONING")
            logger.warning(f"Vision captioning warning for image artifact {artifact.id}: {e}")

        notify("CHUNKING_EMBEDDING")
        try:
            # 1. Visual embedding via SigLIP
            img_emb = embedding_service.embed_image(file_path)
            vec_vis = VectorEmbedding.make(
                artifact_id=artifact.id,
                embedding_type="vision",
                content=desc_text,
                embedding=img_emb
            )
            db.add(vec_vis)

            # 2. Text embedding via BGE-M3
            txt_emb = embedding_service.embed_text(desc_text)
            vec_txt = VectorEmbedding.make(
                artifact_id=artifact.id,
                embedding_type="text",
                content=desc_text,
                embedding=txt_emb
            )
            db.add(vec_txt)
        except Exception as e:
            raise NonRetryableIngestionError(f"Image embedding failed: {e}", stage="CHUNKING_EMBEDDING")

        notify("STORING")
        try:
            meta = Metadata(artifact_id=artifact.id, key="vision_caption", value=desc_text)
            db.add(meta)
            db.commit()
        except Exception as e:
            db.rollback()
            raise RetryableIngestionError(f"Database insertion failed for image artifact: {e}", stage="STORING")

    def _process_video(self, db, artifact, file_path, notify: Callable[[str], None]):
        from backend.app.services.video_service import VideoService
        from backend.app.services.groq_vision_service import GroqVisionService
        from backend.app.core.config import VIDEO_VISION_ENABLED, VIDEO_ASR_ENABLED

        video_svc = VideoService()
        vision = None
        if VIDEO_VISION_ENABLED:
            try:
                vision = GroqVisionService()
            except Exception as e:
                logger.warning("Groq vision unavailable for video; ASR-only if enabled: %s", e)

        try:
            segments = video_svc.process(
                file_path=file_path,
                source_name=artifact.filename,
                vision_service=vision,
                notify=notify,
            )
        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate limit" in err or "rate_limit" in err:
                raise RetryableIngestionError(
                    f"Groq ASR/vision rate limit: {e}", stage="VISION_CAPTIONING"
                )
            if "ffmpeg" in err:
                raise NonRetryableIngestionError(
                    f"ffmpeg required for video ASR: {e}", stage="PARSING"
                )
            raise NonRetryableIngestionError(
                f"Video processing failed: {e}", stage="VISION_CAPTIONING"
            )

        if not segments:
            raise NonRetryableIngestionError(
                f"No searchable video segments from {artifact.filename} "
                f"(ASR={VIDEO_ASR_ENABLED}, vision={VIDEO_VISION_ENABLED})",
                stage="VISION_CAPTIONING",
            )

        max_chunks = int(os.getenv("MAX_INGEST_CHUNKS", "200"))
        if len(segments) > max_chunks:
            logger.warning(
                "Capping video segments from %s to %s for artifact %s",
                len(segments),
                max_chunks,
                artifact.id,
            )
            segments = segments[:max_chunks]

        notify("CHUNKING_EMBEDDING")
        vectors_to_add = []
        temporal_metas = []
        try:
            for idx, seg in enumerate(segments, start=1):
                text = seg["text"]
                txt_emb = embedding_service.embed_text(text)
                vectors_to_add.append(
                    VectorEmbedding.make(
                        artifact_id=artifact.id,
                        embedding_type="text",
                        content=text,
                        embedding=txt_emb,
                    )
                )

                frame_path = seg.get("frame_path")
                if frame_path and os.path.exists(frame_path):
                    try:
                        img_emb = embedding_service.embed_image(frame_path)
                        vectors_to_add.append(
                            VectorEmbedding.make(
                                artifact_id=artifact.id,
                                embedding_type="vision",
                                content=text,
                                embedding=img_emb,
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            "SigLIP video frame embed failed at t=%s: %s",
                            seg.get("start"),
                            e,
                        )

                temporal_metas.append(
                    Metadata(
                        artifact_id=artifact.id,
                        key="temporal",
                        value={
                            "start_time": int(seg["start"]),
                            "end_time": int(seg["end"]),
                            "source": artifact.filename,
                            "has_asr": bool(seg.get("has_asr")),
                            "has_visual": bool(seg.get("has_visual")),
                            "content": text,
                        },
                    )
                )

                if idx % 5 == 0:
                    notify("CHUNKING_EMBEDDING")
        except Exception as e:
            raise NonRetryableIngestionError(
                f"Video embedding failed: {e}", stage="CHUNKING_EMBEDDING"
            )
        finally:
            video_svc.cleanup_segment_files(segments)

        notify("STORING")
        try:
            for vec in vectors_to_add:
                db.add(vec)
            for meta in temporal_metas:
                db.add(meta)
            db.commit()
            logger.info(
                "Stored %s video vectors + %s temporal metas for artifact %s",
                len(vectors_to_add),
                len(temporal_metas),
                artifact.id,
            )
        except Exception as e:
            db.rollback()
            raise RetryableIngestionError(
                f"Database insertion failed for video artifact: {e}", stage="STORING"
            )
