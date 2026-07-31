"""
GroqASRService — cloud Whisper transcription for AEGIS video RAG.

Free-tier friendly: uses Groq whisper-large-v3-turbo via the existing groq SDK.
Local work is only ffmpeg audio extraction (must be on PATH).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

from groq import Groq

from backend.app.core.config import GROQ_ASR_API_KEY, GROQ_WHISPER_MODEL

logger = logging.getLogger(__name__)

# Groq free tier max upload ~25 MB — keep extracted audio small
_MAX_AUDIO_MB = 24


class GroqASRService:
    def __init__(self, model: Optional[str] = None):
        api_key = GROQ_ASR_API_KEY or os.getenv("GROQ_ASR_API_KEY", "")
        if not api_key:
            raise ValueError(
                "GROQ_ASR_API_KEY / GROQ_API_KEY is not set. "
                "Add a Groq key to .env for Whisper ASR."
            )
        if not shutil.which("ffmpeg"):
            raise RuntimeError(
                "ffmpeg not found on PATH. Install ffmpeg to enable video ASR "
                "(e.g. winget install ffmpeg / choco install ffmpeg)."
            )
        self.client = Groq(api_key=api_key)
        self.model = model or GROQ_WHISPER_MODEL
        logger.info("GroqASRService initialized — model: %s", self.model)

    def transcribe(self, video_path: str) -> List[Dict]:
        """
        Extract audio from video and return timed segments:
          [{start: int, end: int, text: str}, ...]
        """
        audio_path = None
        try:
            audio_path = self._extract_audio(video_path)
            if not audio_path or not os.path.exists(audio_path):
                logger.warning("No audio track extracted from %s", video_path)
                return []

            size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            if size_mb > _MAX_AUDIO_MB:
                logger.info(
                    "Audio %.1fMB exceeds free-tier upload cap; chunking before Whisper",
                    size_mb,
                )
                return self._transcribe_chunked(audio_path)

            return self._transcribe_file(audio_path)
        finally:
            if audio_path and os.path.exists(audio_path):
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

    def _extract_audio(self, video_path: str) -> Optional[str]:
        """ffmpeg → mono 16kHz mp3 (small, Whisper-friendly)."""
        fd, out_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "64k",
            out_path,
        ]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=600,
            )
        except subprocess.CalledProcessError as e:
            err = (e.stderr or b"").decode("utf-8", errors="ignore")[:400]
            # Videos with no audio track
            if "does not contain any stream" in err.lower() or "output file is empty" in err.lower():
                logger.warning("Video has no usable audio: %s", video_path)
                try:
                    os.remove(out_path)
                except OSError:
                    pass
                return None
            raise RuntimeError(f"ffmpeg audio extract failed: {err}") from e
        except FileNotFoundError as e:
            raise RuntimeError("ffmpeg not found on PATH") from e

        if not os.path.exists(out_path) or os.path.getsize(out_path) < 256:
            try:
                os.remove(out_path)
            except OSError:
                pass
            return None
        return out_path

    def _transcribe_file(self, audio_path: str, time_offset: float = 0.0) -> List[Dict]:
        with open(audio_path, "rb") as f:
            result = self.client.audio.transcriptions.create(
                file=(Path(audio_path).name, f.read()),
                model=self.model,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )

        segments: List[Dict] = []
        raw_segments = getattr(result, "segments", None) or []
        if raw_segments:
            for seg in raw_segments:
                if isinstance(seg, dict):
                    start = float(seg.get("start", 0))
                    end = float(seg.get("end", start))
                    text = (seg.get("text") or "").strip()
                else:
                    start = float(getattr(seg, "start", 0) or 0)
                    end = float(getattr(seg, "end", start) or start)
                    text = (getattr(seg, "text", "") or "").strip()
                if not text:
                    continue
                segments.append(
                    {
                        "start": int(max(0, start + time_offset)),
                        "end": int(max(start + time_offset, end + time_offset)),
                        "text": text,
                    }
                )
            return segments

        # Fallback: whole transcript as one segment
        text = (getattr(result, "text", None) or str(result) or "").strip()
        if not text:
            return []
        duration = getattr(result, "duration", None)
        end = int(duration + time_offset) if duration else int(time_offset) + 1
        return [{"start": int(time_offset), "end": max(int(time_offset) + 1, end), "text": text}]

    def _transcribe_chunked(self, audio_path: str) -> List[Dict]:
        """Split long audio into ~10-minute chunks under the upload size cap."""
        chunk_dir = tempfile.mkdtemp(prefix="asr_chunks_")
        pattern = os.path.join(chunk_dir, "chunk_%03d.mp3")
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    audio_path,
                    "-f",
                    "segment",
                    "-segment_time",
                    "600",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-b:a",
                    "64k",
                    pattern,
                ],
                check=True,
                capture_output=True,
                timeout=900,
            )
            chunks = sorted(Path(chunk_dir).glob("chunk_*.mp3"))
            all_segments: List[Dict] = []
            offset = 0.0
            for chunk in chunks:
                segs = self._transcribe_file(str(chunk), time_offset=offset)
                all_segments.extend(segs)
                # Advance offset by segment duration (approx 600s)
                if segs:
                    offset = float(max(s["end"] for s in segs))
                else:
                    offset += 600.0
            return all_segments
        finally:
            shutil.rmtree(chunk_dir, ignore_errors=True)
