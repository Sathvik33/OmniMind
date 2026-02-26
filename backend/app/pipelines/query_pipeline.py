from dotenv import load_dotenv
load_dotenv()
from backend.app.vectorstore.collection_manager import CollectionManager
from backend.app.models.ollama_model import OllamaModel
from langsmith import traceable
from backend.app.rag.retriever import Retriever
from backend.app.rag.context_builder import ContextBuilder
from backend.app.rag.generator import Generator


class QueryPipeline:

    def __init__(self):
        self.collection_manager = CollectionManager()
        self.retriever = Retriever(self.collection_manager)
        self.context_builder = ContextBuilder()
        self.generator = Generator(OllamaModel())

    @traceable(name="rag_pipeline")
    def answer(self, query: str, top_k: int = 3):
        chunks, metadata = self.retriever.retrieve(query, top_k)
        context = self.context_builder.build(chunks)
        answer = self.generator.generate(query, context)

        return {
            "answer": answer,
            "context_used": chunks
        }