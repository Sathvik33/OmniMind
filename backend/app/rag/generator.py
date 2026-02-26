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
If the answer is not present, say you don't know.

Context:
{context}

Question:
{query}

Answer:
"""
        return self.llm.generate(prompt)