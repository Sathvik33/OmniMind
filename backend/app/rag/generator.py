from dotenv import load_dotenv
load_dotenv()

from langsmith import traceable


class Generator:

    def __init__(self, llm):
        self.llm = llm

    @traceable(name="llm_generation")
    def generate(self, query: str, context: str) -> str:
        prompt = self._build_prompt(query, context)
        return self.llm.generate(prompt)

    def stream_generate(self, query: str, context: str):
        prompt = self._build_prompt(query, context)
        for token in self.llm.stream(prompt):
            yield token

    @staticmethod
    def _build_prompt(query: str, context: str) -> str:
        return f"""You are a precise knowledge assistant for OmniMind.

RULES:
- Answer ONLY using information from the provided context.
- If the context does not contain the answer, say: "The uploaded data does not contain information about this topic."
- Be concise, factual, and well-structured.
- Do NOT fabricate facts or use knowledge outside the context.

Context:
{context}

Question:
{query}

Answer:"""