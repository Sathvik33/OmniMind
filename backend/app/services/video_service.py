"""
VideoService — Optimized video ingestion for AEGIS v3.0.

Production strategy:
  1. Sample 1 keyframe every N seconds (default 10s, was 2s)
  2. Scene-change detection via pixel diff (skip near-duplicate frames)
  3. Describe each keyframe using GroqVisionService (cloud, fast)
  4. Merge consecutive identical-scene segments to reduce redundancy
  5. Store in PostgreSQL vector_embeddings table (text + vision embeddings)


Why 10s intervals?
  - 2s = 30 API calls/minute for a 1-min video → rate limits + slow
  - 10s = 6 API calls/minute → fast, still fine-grained enough for RAG

Production systems (YouTube, Netflix, Google Photos) use:
  - Scene boundary detection (histogram diff, optical flow)
  - Shot-level captioning (1 caption per scene, not per frame)
  - Parallel processing across scenes (concurrent API calls)
  - We implement all three here.
"""

import cv2
import os
import uuid
import logging
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Optional

from backend.app.core.config import (
    DATA_DIR,
    VIDEO_FRAME_INTERVAL_SEC,
    VIDEO_SCENE_DIFF_THRESHOLD,
)

logger = logging.getLogger(__name__)

# Max concurrent vision API calls (avoids Groq rate limits)
_MAX_WORKERS = 3


class VideoService:
    """
    Optimized video ingestion pipeline.

    Flow:
      video → keyframe extraction (every 10s) → scene-change filter
            → parallel Groq vision captioning → segment merging
            → ChromaDB (text + multimodal collections)
    """

    def __init__(self, vision_service, collection_manager):
        self.vision_service     = vision_service
        self.collection_manager = collection_manager

    # ── Public ─────────────────────────────────────────────────────────────────

    def process(self, video_path: str, source_name: str, job_id: str, app) -> None:
        """
        Main entry point — called as a FastAPI background task.
        Processes the video and stores results in ChromaDB.
        """
        app.state.video_jobs[job_id] = "processing"
        logger.info(f"[VideoService] Starting: {source_name} (job={job_id})")

        try:
            # 1. Extract keyframes
            keyframes = self._extract_keyframes(video_path)
            if not keyframes:
                logger.warning(f"[VideoService] No keyframes extracted from {source_name}")
                app.state.video_jobs[job_id] = "completed"
                return

            logger.info(f"[VideoService] Extracted {len(keyframes)} keyframes")

            # 2. Caption keyframes in parallel (Groq cloud — fast)
            captions = self._caption_frames_parallel(keyframes)

            # 3. Merge into segments
            segments = self._merge_segments(captions, video_path)

            # 4. Store in ChromaDB
            if segments:
                documents = [s["text"] for s in segments]
                ids       = [str(uuid.uuid4()) for _ in segments]
                metadatas = [
                    {
                        "source":     source_name,
                        "modality":   "video",
                        "start_time": s["start"],
                        "end_time":   s["end"],
                        "frame_count": s.get("frame_count", 1),
                    }
                    for s in segments
                ]
                self.collection_manager.add_documents(documents, ids, metadatas)
                logger.info(f"[VideoService] Stored {len(segments)} segments for {source_name}")

            app.state.video_jobs[job_id] = "completed"

        except Exception as e:
            logger.error(f"[VideoService] Failed: {e}")
            app.state.video_jobs[job_id] = f"failed: {e}"

        finally:
            # Clean up temp files
            self._cleanup_temp_frames(job_id)

    # ── Private: Keyframe Extraction ───────────────────────────────────────────

    def _extract_keyframes(self, video_path: str) -> List[Tuple[int, str]]:
        """
        Extract one keyframe every VIDEO_FRAME_INTERVAL_SEC seconds.
        Applies scene-change detection to skip near-duplicate frames.

        Returns: list of (timestamp_sec, temp_image_path) tuples.
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps

        frame_interval = max(1, int(fps * VIDEO_FRAME_INTERVAL_SEC))
        logger.info(
            f"[VideoService] fps={fps:.1f}, duration={duration_sec:.0f}s, "
            f"sampling every {VIDEO_FRAME_INTERVAL_SEC}s ({frame_interval} frames)"
        )

        keyframes: List[Tuple[int, str]] = []
        prev_frame = None
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp_sec = int(frame_count / fps)

                # Scene-change filter: skip if too similar to previous frame
                if prev_frame is not None:
                    diff = cv2.absdiff(
                        cv2.resize(prev_frame, (64, 64)),
                        cv2.resize(frame, (64, 64)),
                    ).mean()
                    if diff < VIDEO_SCENE_DIFF_THRESHOLD:
                        frame_count += 1
                        continue  # near-duplicate, skip

                # Save keyframe to temp file (tagged with job info)
                temp_path = str(DATA_DIR / f"_frame_{frame_count}_{timestamp_sec}.jpg")
                cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                keyframes.append((timestamp_sec, temp_path))
                prev_frame = frame

            frame_count += 1

        cap.release()
        return keyframes

    # ── Private: Parallel Captioning ──────────────────────────────────────────

    def _caption_frames_parallel(
        self, keyframes: List[Tuple[int, str]]
    ) -> List[Tuple[int, str]]:
        """
        Caption all keyframes concurrently using a thread pool.
        Returns: list of (timestamp_sec, caption) sorted by timestamp.
        """
        results: List[Tuple[int, str]] = []

        def describe_one(item: Tuple[int, str]) -> Tuple[int, str]:
            ts, path = item
            try:
                caption = self.vision_service.describe(path)
                return (ts, caption)
            except Exception as e:
                logger.warning(f"Caption failed at t={ts}s: {e}")
                return (ts, "")
            finally:
                # Remove temp file immediately after captioning
                try:
                    os.remove(path)
                except Exception:
                    pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
            futures = {executor.submit(describe_one, kf): kf for kf in keyframes}
            for future in concurrent.futures.as_completed(futures):
                ts, caption = future.result()
                if caption:
                    results.append((ts, caption))

        # Sort by timestamp
        results.sort(key=lambda x: x[0])
        return results

    # ── Private: Segment Merging ───────────────────────────────────────────────

    def _merge_segments(
        self,
        captions: List[Tuple[int, str]],
        video_path: str,
    ) -> List[dict]:
        """
        Merge consecutive frames with identical/very-similar captions
        into single segments with start/end timestamps.

        Returns list of {text, start, end, frame_count} dicts.
        """
        if not captions:
            return []

        segments = []
        seg_text   = captions[0][1]
        seg_start  = captions[0][0]
        seg_end    = captions[0][0] + VIDEO_FRAME_INTERVAL_SEC
        seg_frames = 1

        for ts, caption in captions[1:]:
            # Simple similarity: if captions share >60% tokens → same scene
            if self._similar(caption, seg_text):
                seg_end    = ts + VIDEO_FRAME_INTERVAL_SEC
                seg_frames += 1
            else:
                segments.append({
                    "text":        seg_text,
                    "start":       seg_start,
                    "end":         seg_end,
                    "frame_count": seg_frames,
                })
                seg_text   = caption
                seg_start  = ts
                seg_end    = ts + VIDEO_FRAME_INTERVAL_SEC
                seg_frames = 1

        # Flush last segment
        segments.append({
            "text":        seg_text,
            "start":       seg_start,
            "end":         seg_end,
            "frame_count": seg_frames,
        })

        return segments

    @staticmethod
    def _similar(a: str, b: str, threshold: float = 0.5) -> bool:
        """Token-overlap similarity check (no heavy models needed)."""
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a or not tokens_b:
            return False
        overlap = len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))
        return overlap >= threshold

    def _cleanup_temp_frames(self, job_id: str) -> None:
        """Remove any leftover temp frame files."""
        try:
            for f in Path(DATA_DIR).glob("_frame_*.jpg"):
                f.unlink(missing_ok=True)
        except Exception:
            pass