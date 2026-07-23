import re
from typing import List, Dict, Any
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core import Document

class MarkdownHierarchicalChunker:
    """
    Structure-aware Markdown chunker for Multimodal RAG.
    Utilizes LlamaIndex's MarkdownNodeParser for hierarchy detection
    and SentenceSplitter for sentence-aware chunking.
    """
    
    def __init__(self, target_tokens: int = 600, overlap_tokens: int = 60):
        self.chunk_size = target_tokens
        self.chunk_overlap = overlap_tokens
        
        self.md_parser = MarkdownNodeParser()
        self.sent_splitter = SentenceSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )

    def chunk(self, markdown: str, artifact_name: str) -> List[Dict[str, Any]]:
        # 1. Create a Document
        doc = Document(text=markdown)
        
        # 2. Parse Markdown nodes (extracts headers)
        md_nodes = self.md_parser.get_nodes_from_documents([doc])
        
        # 3. Split by sentences within those nodes
        final_nodes = self.sent_splitter.get_nodes_from_nodes(md_nodes)
        
        chunks = []
        for node in final_nodes:
            text = node.get_content().strip()
            if not text:
                continue
                
            # Extract header path from LlamaIndex metadata
            # LlamaIndex stores headers as 'Header_1', 'Header_2', etc.
            headers = []
            for i in range(1, 7):
                h_key = f"Header_{i}"
                if h_key in node.metadata:
                    headers.append(node.metadata[h_key].strip())
            
            hierarchy_str = " > ".join(headers) if headers else "Document Root"
            
            # Robust table detection: Markdown tables or HTML tables
            contains_table = bool(
                re.search(r'\|.*\|', text) and re.search(r'[-:]+[-| :]*', text)
            ) or "<table" in text.lower()
            contains_image = "<!-- image: true -->" in text
            
            chunks.append({
                "text": text,
                "metadata": {
                    "document_name": artifact_name,
                    "hierarchy": hierarchy_str,
                    "contains_table": contains_table,
                    "contains_image": contains_image
                }
            })
            
        return chunks
