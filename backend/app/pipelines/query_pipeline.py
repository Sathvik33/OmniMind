import re
from langsmith import traceable
from backend.app.vectorstore.collection_manager import CollectionManager
from backend.app.rag.retriever import Retriever
from backend.app.rag.context_builder import ContextBuilder
from backend.app.rag.generator import Generator
from backend.app.models.ollama_model import OllamaModel
from backend.app.utils.time_utils import timestamp_to_seconds


class QueryPipeline:

    def __init__(self):
        self.collection_manager = CollectionManager()
        self.retriever = Retriever(self.collection_manager)
        self.context_builder = ContextBuilder()
        self.generator = Generator(OllamaModel())

    def detect_time_query(self, query: str):
        query = query.lower()

        hhmmss = re.findall(r"\d{2}:\d{2}:\d{2}", query)
        seconds = re.findall(r"\b\d+\s*seconds?\b", query)

        return hhmmss, seconds

    def handle_time_query(self, query, hhmmss, seconds):

        if hhmmss:
            times = [timestamp_to_seconds(t) for t in hhmmss]
        else:
            times = [int(re.findall(r"\d+", s)[0]) for s in seconds]

        if len(times) == 1:
            start_time = times[0] 
            end_time = times[0]
        else:
            start_time = min(times)
            end_time = max(times)

        segments = self.collection_manager.query_time_range(start_time, end_time)

        if not segments:
            return {
                "answer": "No video content found for that time range.",
                "context_used": []
            }

        context = "\n\n".join(segments)

        prompt = f"""
You are analyzing a video segment.

The following descriptions correspond to video content between {start_time} and {end_time} seconds.

Summarize clearly what is happening during that time window.

Segments:
{context}

Answer:
"""

        answer = self.generator.llm.generate(prompt)

        return {
            "answer": answer,
            "context_used": segments
        }

    @traceable(name="rag_pipeline")
    def answer(self, query: str, top_k: int = 3):

        hhmmss, seconds = self.detect_time_query(query)

        if hhmmss or seconds:
            return self.handle_time_query(query, hhmmss, seconds)

        chunks, metadata = self.retriever.retrieve(query, top_k)

        context = self.context_builder.build(chunks)

        answer = self.generator.generate(query, context)

        return {
            "answer": answer,
            "context_used": chunks
        }