from ingestion.load_documents import load_documents
from ingestion.chunking import process_documents
from embeddings.generate_embeddings import embed_texts
from embeddings.embedding_model import EmbeddingModel

DATA_DIR = "data/documents"

class DocumentState:
    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.reload()

    def reload(self):
        self.docs = load_documents(DATA_DIR)
        self.chunks = process_documents(self.docs)
        self.chunk_embeddings = embed_texts(
            self.chunks, self.embedding_model
        )

state = DocumentState()

