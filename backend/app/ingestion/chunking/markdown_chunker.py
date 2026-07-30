import re
from typing import List, Dict, Any
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core import Document

# Drop / merge fragments that are just dates, bare headers, or separators.
_MIN_CHUNK_CHARS = 120
_HEADER_ONLY = re.compile(r"^#{1,6}\s+\S.*$", re.MULTILINE)
_DATEISH = re.compile(
    r"^[#*\s\-]*"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*['’]?\d{0,2}"
    r"(?:\s|$|[,\-–—])",
    re.IGNORECASE,
)


class MarkdownHierarchicalChunker:
    """
    Structure-aware Markdown chunker for Multimodal RAG.
    Utilizes LlamaIndex's MarkdownNodeParser for hierarchy detection
    and SentenceSplitter for sentence-aware chunking.
    Tiny header/date-only fragments are merged so retrieval stays useful.
    """

    def __init__(
        self,
        target_tokens: int = 600,
        overlap_tokens: int = 60,
        min_chunk_chars: int = _MIN_CHUNK_CHARS,
    ):
        self.chunk_size = target_tokens
        self.chunk_overlap = overlap_tokens
        self.min_chunk_chars = min_chunk_chars

        self.md_parser = MarkdownNodeParser()
        self.sent_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk(self, markdown: str, artifact_name: str) -> List[Dict[str, Any]]:
        doc = Document(text=markdown)
        md_nodes = self.md_parser.get_nodes_from_documents([doc])
        final_nodes = self.sent_splitter(md_nodes)

        raw_chunks: List[Dict[str, Any]] = []
        for node in final_nodes:
            text = node.get_content().strip()
            if not text:
                continue

            headers = []
            for i in range(1, 7):
                h_key = f"Header_{i}"
                if h_key in node.metadata:
                    headers.append(node.metadata[h_key].strip())

            hierarchy_str = " > ".join(headers) if headers else "Document Root"
            contains_table = bool(
                re.search(r"\|.*\|", text) and re.search(r"[-:]+[-| :]*", text)
            ) or "<table" in text.lower()
            contains_image = "<!-- image: true -->" in text

            raw_chunks.append(
                {
                    "text": text,
                    "metadata": {
                        "document_name": artifact_name,
                        "hierarchy": hierarchy_str,
                        "contains_table": contains_table,
                        "contains_image": contains_image,
                    },
                }
            )

        return self._merge_small_chunks(raw_chunks)

    def _substantive_len(self, text: str) -> int:
        """Length after stripping markdown chrome that isn't useful alone."""
        cleaned = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
        cleaned = re.sub(r"^---+$", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return len(cleaned)

    def _is_weak_fragment(self, text: str) -> bool:
        body = text.strip()
        if self._substantive_len(body) < self.min_chunk_chars:
            return True
        lines = [ln.strip() for ln in body.splitlines() if ln.strip() and ln.strip() != "---"]
        if not lines:
            return True
        if len(lines) <= 2 and all(
            _HEADER_ONLY.match(ln) or _DATEISH.match(ln) or len(ln) < 40 for ln in lines
        ):
            return True
        return False

    def _merge_small_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        merged: List[Dict[str, Any]] = []
        for chunk in chunks:
            if not merged:
                merged.append(chunk)
                continue

            if self._is_weak_fragment(chunk["text"]):
                prev = merged[-1]
                prev["text"] = f"{prev['text'].rstrip()}\n\n{chunk['text'].lstrip()}"
                prev["metadata"]["contains_table"] = (
                    prev["metadata"].get("contains_table")
                    or chunk["metadata"].get("contains_table")
                )
                prev["metadata"]["contains_image"] = (
                    prev["metadata"].get("contains_image")
                    or chunk["metadata"].get("contains_image")
                )
                if prev["metadata"].get("hierarchy") == "Document Root":
                    prev["metadata"]["hierarchy"] = chunk["metadata"].get(
                        "hierarchy", "Document Root"
                    )
            else:
                merged.append(chunk)

        # If the first chunk itself is weak, fold it into the next when possible.
        if len(merged) >= 2 and self._is_weak_fragment(merged[0]["text"]):
            first = merged.pop(0)
            merged[0]["text"] = f"{first['text'].rstrip()}\n\n{merged[0]['text'].lstrip()}"

        # Drop any leftover micro-fragments (e.g. single-page junk).
        kept = [c for c in merged if not self._is_weak_fragment(c["text"])]
        return kept if kept else merged[:1]
