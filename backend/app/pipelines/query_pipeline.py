from dotenv import load_dotenv
load_dotenv()
from backend.app.vectorstore.collection_manager import CollectionManager
from backend.app.models.ollama_model import OllamaModel
from langsmith import traceable
from backend.app.rag.retriever import Retriever
from backend.app.rag.context_builder import ContextBuilder
from backend.app.rag.generator import Generator
import re
from backend.app.utils.time_utils import timestamp_to_seconds

class QueryPipeline:

    def __init__(self):
        self.collection_manager = CollectionManager()
        self.retriever = Retriever(self.collection_manager)
        self.context_builder = ContextBuilder()
        self.generator = Generator(OllamaModel())

    @traceable(name="rag_pipeline")
    def answer(self, query: str, top_k: int = 3):
        time_matches = re.findall(r"\d{2}:\d{2}:\d{2}", query)
        seconds_match = re.findall(r"\b\d+\s*seconds?\b", query.lower())

        if time_matches or seconds_match:
            return self.handle_time_query(query, time_matches, seconds_match)
        chunks, metadata = self.retriever.retrieve(query, top_k)
        context = self.context_builder.build(chunks)
        answer = self.generator.generate(query, context)

        return {
            "answer": answer,
            "context_used": chunks
        }

    
    def handle_time_query(self, query, time_matches, seconds_match):

        if time_matches:
            times = [timestamp_to_seconds(t) for t in time_matches]
        else:
            times = [int(re.findall(r"\d+", s)[0]) for s in seconds_match]

        if len(times) == 1:
            start_time = times[0]
            end_time = times[0]
        else:
            start_time = min(times)
            end_time = max(times)

        segments = self.collection_manager.query_time_range(start_time, end_time)

        if not segments:
            return {"answer": "No video content found for that time.", "context_used": []}

        context = "\n\n".join(segments)

        answer = self.generator.generate(query, context)

        return {
            "answer": answer,
            "context_used": segments
        }