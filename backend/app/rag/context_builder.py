from dotenv import load_dotenv
import os

load_dotenv()

class ContextBuilder:

    def build(self, chunks):
        return "\n\n".join(chunks)