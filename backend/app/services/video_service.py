"""
VideoService — cloud-first Video RAG for AEGIS.

Local: OpenCV seek + histogram scene sampling, ffmpeg (via GroqASRService).
Cloud: Groq Whisper ASR + capped Groq vision captions (free-tier safe).

Returns segment dicts for the Celery ingestion pipeline to embed into pgvector.
Does not write Chroma or touch FastAPI app.state.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2

from backend.app.core.config import (
    DATA_DIR,
    VIDEO_ASR_ENABLED,
    VIDEO_FRAME_INTERVAL_SEC,
    VIDEO_MAX_KEYFRAMES,
    VIDEO_SCENE_DIFF_THRESHOLD,
    VIDEO_VISION_ENABLED,
)

logger = logging.getLogger(__name__)

_MAX_VISION_WORKERS = 1  # serialize captions — avoids Groq TPM bursts on video keyframes
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
# Merge short Whisper segments into ~30s windows for RAG chunk quality
_ASR_WINDOW_SEC = 30


class VideoService:
    """
    Build timed video segments for pgvector ingest.

    Flow:
      probe → Groq ASR (optional) → seek keyframes → capped vision captions
            → merge ASR + nearest caption → return segments
    """

    def __init__(self, vision_service=None):
        self.vision_service = vision_service
        self._temp_files: List[str] = []

    def process(
        self,
        file_path: str,
        source_name: str,
        vision_service=None,
        notify: Optional[Callable[[str], None]] = None,
    ) -> List[Dict]:
        """
        Process a local video file into segment dicts:

          {
            text, start, end, frame_path?, has_asr, has_visual
          }

        Caller embeds/stores and must not rely on Chroma.
        """
        if vision_service is not None:
            self.vision_service = vision_service
        self._temp_files = []

        def _notify(stage: str):
            if notify:
                notify(stage)

        _notify("PARSING")
        duration = self.probe_duration(file_path)
        logger.info(
            "[VideoService] %s duration=%.1fs asr=%s vision=%s",
            source_name,
            duration,
            VIDEO_ASR_ENABLED,
            VIDEO_VISION_ENABLED,
        )

        _notify("VISION_CAPTIONING")
        asr_segments: List[Dict] = []
        if VIDEO_ASR_ENABLED:
            asr_segments = self._run_asr(file_path)

        captions: List[Tuple[int, str, Optional[str]]] = []  # ts, caption, frame_path
        vision_degraded = False
        if VIDEO_VISION_ENABLED and self.vision_service is not None:
            keyframes = self._extract_keyframes(file_path, duration)
            logger.info("[VideoService] Extracted %s keyframes", len(keyframes))
            try:
                captions = self._caption_frames(keyframes, keep_paths=True)
            except _VisionRateLimitError as e:
                self._cleanup_paths([p for _, p in keyframes])
                if asr_segments:
                    logger.warning(
                        "[VideoService] Vision rate-limited; degrading to ASR-only: %s", e
                    )
                    vision_degraded = True
                else:
                    raise
            except Exception as e:
                if self._is_rate_limit(e):
                    self._cleanup_paths([p for _, p in keyframes])
                    if asr_segments:
                        logger.warning(
                            "[VideoService] Vision 429; degrading to ASR-only: %s", e
                        )
                        vision_degraded = True
                    else:
                        raise
                else:
                    logger.warning("[VideoService] Vision captioning failed: %s", e)
                    self._cleanup_paths([p for _, p in keyframes])
        elif VIDEO_VISION_ENABLED and self.vision_service is None:
            logger.warning("[VideoService] VIDEO_VISION_ENABLED but no vision_service")

        segments = self._merge_asr_and_captions(
            asr_segments=asr_segments,
            captions=captions,
            source_name=source_name,
            duration=duration,
        )

        if not segments:
            logger.warning(
                "[VideoService] No segments produced for %s (vision_degraded=%s)",
                source_name,
                vision_degraded,
            )

        return segments

    # ── Probe ──────────────────────────────────────────────────────────────────

    @staticmethod
    def probe_duration(video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps <= 0:
                fps = 25.0
            if total > 0:
                return total / fps
            # Fallback: some containers report 0 frame count
            return 0.0
        finally:
            cap.release()

    # ── ASR ────────────────────────────────────────────────────────────────────

    def _run_asr(self, video_path: str) -> List[Dict]:
        try:
            from backend.app.services.groq_asr_service import GroqASRService

            asr = GroqASRService()
            segments = asr.transcribe(video_path)
            logger.info("[VideoService] ASR returned %s segments", len(segments))
            return segments
        except Exception as e:
            if self._is_rate_limit(e):
                # Retries already exhausted in GroqASRService — degrade to vision-only
                logger.warning(
                    "[VideoService] ASR rate-limited after retries; continuing without speech: %s",
                    e,
                )
                return []
            logger.warning("[VideoService] ASR failed (continuing without speech): %s", e)
            return []

    # ── Keyframes (seek-based) ─────────────────────────────────────────────────

    def _extract_keyframes(
        self, video_path: str, duration: float
    ) -> List[Tuple[int, str]]:
        """
        Seek to candidate timestamps; keep frames with histogram/mean change.
        Caps at VIDEO_MAX_KEYFRAMES. Falls back to fixed interval if few cuts.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            interval = max(1, VIDEO_FRAME_INTERVAL_SEC)
            if duration <= 0:
                # Probe failed — sample first few minutes by interval guesses
                candidates = list(range(0, interval * VIDEO_MAX_KEYFRAMES * 2, interval))
            else:
                candidates = list(range(0, int(duration) + 1, interval))
                # Also sample midpoints if very short
                if len(candidates) < 2 and duration > 1:
                    candidates = [0, int(duration / 2)]

            keyframes: List[Tuple[int, str]] = []
            prev_hist = None
            prev_small = None
            temp_dir = tempfile.mkdtemp(prefix="video_frames_")
            self._temp_files.append(temp_dir)

            for ts in candidates:
                if len(keyframes) >= VIDEO_MAX_KEYFRAMES:
                    break
                cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                small = cv2.resize(frame, (64, 64))
                hist = cv2.calcHist(
                    [small], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
                )
                cv2.normalize(hist, hist)

                if prev_hist is not None and prev_small is not None:
                    # Higher correlation → more similar; skip near-duplicates
                    corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    mean_diff = float(cv2.absdiff(small, prev_small).mean())
                    if corr > 0.92 and mean_diff < VIDEO_SCENE_DIFF_THRESHOLD:
                        continue

                out_path = os.path.join(temp_dir, f"frame_{ts:06d}.jpg")
                cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                keyframes.append((int(ts), out_path))
                prev_hist = hist
                prev_small = small

            # If scene filter was too aggressive, force interval samples up to cap
            if len(keyframes) < min(3, VIDEO_MAX_KEYFRAMES) and duration > 0:
                self._cleanup_paths([p for _, p in keyframes])
                keyframes = []
                step = max(interval, int(duration / max(1, VIDEO_MAX_KEYFRAMES)))
                for ts in range(0, int(duration) + 1, step):
                    if len(keyframes) >= VIDEO_MAX_KEYFRAMES:
                        break
                    cap.set(cv2.CAP_PROP_POS_MSEC, float(ts) * 1000.0)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        continue
                    out_path = os.path.join(temp_dir, f"frame_{ts:06d}.jpg")
                    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    keyframes.append((int(ts), out_path))

            return keyframes[:VIDEO_MAX_KEYFRAMES]
        finally:
            cap.release()

    # ── Captions ───────────────────────────────────────────────────────────────

    def _caption_frames(
        self,
        keyframes: List[Tuple[int, str]],
        keep_paths: bool = True,
    ) -> List[Tuple[int, str, Optional[str]]]:
        """Return (timestamp, caption, frame_path|None) sorted by time."""
        if not keyframes or self.vision_service is None:
            return []

        results: List[Tuple[int, str, Optional[str]]] = []
        rate_limited = False

        # ContextVars do not always propagate into ThreadPool workers — capture parent.
        from backend.app.monitoring.langsmith_logger import tracer

        parent_run_id = tracer.current_parent()

        def describe_one(item: Tuple[int, str]) -> Tuple[int, str, Optional[str]]:
            nonlocal rate_limited
            if parent_run_id:
                tracer.bind_parent(parent_run_id)
            ts, path = item
            try:
                caption = self.vision_service.describe(path)
                caption = _THINK_RE.sub("", caption or "").strip()
                if self._is_rate_limit_message(caption):
                    rate_limited = True
                    return (ts, "", path if keep_paths else None)
                return (ts, caption, path if keep_paths else None)
            except Exception as e:
                if self._is_rate_limit(e):
                    rate_limited = True
                logger.warning("Caption failed at t=%ss: %s", ts, e)
                return (ts, "", path if keep_paths else None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=_MAX_VISION_WORKERS) as pool:
            futures = [pool.submit(describe_one, kf) for kf in keyframes]
            for fut in concurrent.futures.as_completed(futures):
                results.append(fut.result())

        if rate_limited and not any(c for _, c, _ in results):
            raise _VisionRateLimitError("Groq vision rate limited (429)")

        # Drop empty captions; cleanup unused frame files
        kept: List[Tuple[int, str, Optional[str]]] = []
        for ts, caption, path in results:
            if caption:
                kept.append((ts, caption, path))
            elif path and not keep_paths:
                self._cleanup_paths([path])
            elif path and not caption:
                # Keep path only if we had a caption; else cleanup
                self._cleanup_paths([path])

        kept.sort(key=lambda x: x[0])
        return kept

    # ── Merge ──────────────────────────────────────────────────────────────────

    def _merge_asr_and_captions(
        self,
        asr_segments: List[Dict],
        captions: List[Tuple[int, str, Optional[str]]],
        source_name: str,
        duration: float,
    ) -> List[Dict]:
        """
        Build timed RAG segments from ASR windows + every vision caption.

        Previously, unused keyframe captions were discarded whenever ASR existed,
        so object questions (cars, people, signs) often had nothing to retrieve.
        Now every caption becomes its own visual segment, and ASR windows still
        attach in-range (or nearest) visuals.
        """
        windows = self._window_asr(asr_segments, duration)
        segments: List[Dict] = []
        used_caption_ts: set[int] = set()

        for win in windows:
            in_range = [c for c in captions if win["start"] <= c[0] <= win["end"]]
            if not in_range:
                nearest = self._nearest_caption(win["start"], win["end"], captions)
                mid = (win["start"] + win["end"]) / 2.0
                # Only attach if reasonably close (avoid wrong-scene captions)
                if nearest and abs(nearest[0] - mid) <= max(_ASR_WINDOW_SEC, 20):
                    in_range = [nearest]

            spoken = win["text"].strip()
            visual_lines: List[str] = []
            frame_path = None
            for ts, cap, path in in_range:
                used_caption_ts.add(ts)
                visual_lines.append(f"At {ts}s: {cap}")
                if frame_path is None:
                    frame_path = path

            parts = [f"[Video: {source_name}] t={win['start']}–{win['end']}"]
            if spoken:
                parts.append(f"Spoken: {spoken}")
            if visual_lines:
                parts.append("Visual:\n" + "\n".join(visual_lines))
            if len(parts) == 1:
                continue
            segments.append(
                {
                    "text": "\n".join(parts),
                    "start": int(win["start"]),
                    "end": int(win["end"]),
                    "frame_path": frame_path,
                    "has_asr": bool(spoken),
                    "has_visual": bool(visual_lines),
                }
            )

        # Always keep caption-only segments for keyframes not folded into ASR
        for i, (ts, caption, frame_path) in enumerate(captions):
            if ts in used_caption_ts:
                continue
            if i + 1 < len(captions):
                end = captions[i + 1][0]
            else:
                end = int(ts + VIDEO_FRAME_INTERVAL_SEC)
            if duration > 0:
                end = min(end, int(duration))
            end = max(end, ts + 1)
            # Small pad so point temporal queries near the keyframe still hit
            start = max(0, int(ts) - 2)
            segments.append(
                {
                    "text": (
                        f"[Video: {source_name}] t={start}–{end}\n"
                        f"Visual: At {ts}s: {caption}"
                    ),
                    "start": start,
                    "end": int(end),
                    "frame_path": frame_path,
                    "has_asr": False,
                    "has_visual": True,
                }
            )

        segments.sort(key=lambda s: (s["start"], s["end"]))
        return segments

    def _window_asr(self, asr_segments: List[Dict], duration: float) -> List[Dict]:
        if not asr_segments:
            return []
        windows: List[Dict] = []
        buf_text: List[str] = []
        buf_start = int(asr_segments[0]["start"])
        buf_end = int(asr_segments[0]["end"])

        for seg in asr_segments:
            start = int(seg.get("start", 0))
            end = int(seg.get("end", start + 1))
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            if not buf_text:
                buf_start, buf_end = start, end
                buf_text = [text]
                continue
            if end - buf_start <= _ASR_WINDOW_SEC:
                buf_text.append(text)
                buf_end = max(buf_end, end)
            else:
                windows.append(
                    {"start": buf_start, "end": max(buf_end, buf_start + 1), "text": " ".join(buf_text)}
                )
                buf_start, buf_end = start, end
                buf_text = [text]

        if buf_text:
            windows.append(
                {"start": buf_start, "end": max(buf_end, buf_start + 1), "text": " ".join(buf_text)}
            )

        if duration > 0 and windows:
            windows[-1]["end"] = max(windows[-1]["end"], min(int(duration), windows[-1]["end"]))
        return windows

    @staticmethod
    def _nearest_caption(
        start: int,
        end: int,
        captions: List[Tuple[int, str, Optional[str]]],
    ) -> Optional[Tuple[int, str, Optional[str]]]:
        if not captions:
            return None
        mid = (start + end) / 2.0
        # Prefer captions inside the window; else nearest by midpoint
        inside = [c for c in captions if start <= c[0] <= end]
        pool = inside or captions
        return min(pool, key=lambda c: abs(c[0] - mid))

    # ── Cleanup helpers ────────────────────────────────────────────────────────

    def cleanup_segment_files(self, segments: List[Dict]) -> None:
        paths = [s["frame_path"] for s in segments if s.get("frame_path")]
        self._cleanup_paths(paths)
        for p in list(self._temp_files):
            if os.path.isdir(p):
                try:
                    import shutil

                    shutil.rmtree(p, ignore_errors=True)
                except Exception:
                    pass
            self._temp_files.clear()
        # Legacy DATA_DIR leftovers
        try:
            for f in Path(DATA_DIR).glob("_frame_*.jpg"):
                f.unlink(missing_ok=True)
        except Exception:
            pass

    @staticmethod
    def _cleanup_paths(paths: List[str]) -> None:
        for p in paths:
            if not p:
                continue
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass

    @staticmethod
    def _is_rate_limit(exc: Exception) -> bool:
        msg = str(exc).lower()
        return "429" in msg or "rate limit" in msg or "rate_limit" in msg

    @staticmethod
    def _is_rate_limit_message(text: str) -> bool:
        t = (text or "").lower()
        return "429" in t or "rate limit" in t


class _VisionRateLimitError(Exception):
    pass
