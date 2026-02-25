from langchain.text_splitter import RecursiveCharacterTextSplitter

class LangchainChunker:
    def __init__(self, chunk_size=600, overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap
        )

    def chunk(self, text: str):
        return self.splitter.split_text(text)