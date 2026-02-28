from dotenv import load_dotenv
load_dotenv()

from langsmith import traceable


class Generator:

    def __init__(self, llm):
        self.llm = llm

    @traceable(name="llm_generation")
    def generate(self, query: str, context: str):
        prompt = f"""
You are a technical knowledge assistant.

Provide a detailed and well-structured explanation.
Use only the information from the context.

Context:
{context}

Question:
{query}

Answer:
"""
        return self.llm.generate(prompt)

    def stream_generate(self, query: str, context: str):
        prompt = f"""
You are a technical knowledge assistant.

Provide a detailed and well-structured explanation.
Use only the information from the context.

Context:
{context}

Question:
{query}

Answer:
"""
        for token in self.llm.stream(prompt):
            yield token