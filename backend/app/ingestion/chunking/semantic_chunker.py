import re
import torch
from typing import List
from sentence_transformers import SentenceTransformer, util

from backend.app.core.config import (
    SEMANTIC_THRESHOLD,
    MIN_CHUNK_SENTENCES,
    MAX_CHUNK_SENTENCES,
)


class SemanticChunker:
    """
    Production semantic chunker.

    Groups sentences into topically coherent chunks using cosine similarity
    between adjacent sentence embeddings. A similarity drop below
    `similarity_threshold` signals a topic boundary → start a new chunk.

    NOTE: This runs at INGESTION time (background task) and does NOT affect
    query latency. The extra seconds are invisible to the user.

    Strategy:
        1. Split text → sentences (regex, no NLTK dependency)
        2. GPU-batch encode all sentences via BAAI/bge-m3
        3. Walk pairs; accumulate until similarity < threshold OR max reached
        4. Enforce min size to avoid micro-chunks
        5. Fall back to raw text for very short inputs
    """

    def __init__(
        self,
        similarity_threshold: float = SEMANTIC_THRESHOLD,
        min_sentences: int = MIN_CHUNK_SENTENCES,
        max_sentences: int = MAX_CHUNK_SENTENCES,
        model_name: str = "BAAI/bge-m3",
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.threshold = similarity_threshold
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

    # ── Sentence splitting ─────────────────────────────────────────────────────

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split on sentence boundaries (.!?) followed by whitespace + capital letter,
        plus paragraph breaks. Handles lists and newline-heavy docs.
        """
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Split on: sentence-ending punctuation → space → capital letter
        sentence_end = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\'])")
        parts = sentence_end.split(text)

        sentences: List[str] = []
        for part in parts:
            # Further split on blank lines (paragraph boundaries)
            for sub in re.split(r"\n{2,}", part):
                sub = sub.strip()
                if sub:
                    sentences.append(sub)

        return sentences

    # ── Main chunking logic ────────────────────────────────────────────────────

    def chunk(self, text: str) -> List[str]:
        sentences = self._split_sentences(text)

        # Too few sentences → return as single chunk
        if len(sentences) < 3:
            return [text.strip()] if text.strip() else []

        # GPU batch-encode all sentences at once (fast)
        embeddings = self.model.encode(
            sentences,
            convert_to_tensor=True,
            batch_size=64,
            show_progress_bar=False,
        )

        chunks: List[str] = []
        current: List[str] = [sentences[0]]

        for i in range(1, len(sentences)):
            similarity = util.cos_sim(embeddings[i - 1], embeddings[i]).item()

            # Hard cap: flush current group
            if len(current) >= self.max_sentences:
                chunks.append(" ".join(current))
                current = [sentences[i]]
                continue

            if similarity >= self.threshold:
                # Same topic → keep accumulating
                current.append(sentences[i])
            else:
                # Topic boundary detected
                if len(current) >= self.min_sentences:
                    chunks.append(" ".join(current))
                    current = [sentences[i]]
                else:
                    # Micro-group: absorb into current to avoid tiny chunks
                    current.append(sentences[i])

        # Flush remaining sentences
        if current:
            chunks.append(" ".join(current))

        return [c for c in chunks if c.strip()]
