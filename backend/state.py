from ingestion.load_documents import load_documents
from ingestion.chunking import process_documents
from embeddings.generate_embeddings import embed_texts
#from embeddings.embedding_model import EmbeddingModel
from retrieval.bm25_retriever import BM25Retriever
import os

DATA_DIR = "data/documents"

class DocumentState:

    def __init__(self):
        self.embedding_model = None
        

    def get_index(self, business_id, client_id):

        folder = os.path.join(DATA_DIR, business_id, client_id)

        if not os.path.exists(folder):
            return None

        docs = load_documents(folder)

        if not docs:
            return None
    

        #self.docs = load_documents(DATA_DIR)
        self.docs = docs
        for doc in self.docs:
            print("Loaded file:", doc["source"])

        # if not docs:
        #     return None

        fine_chunks, coarse_chunks, list_chunks = process_documents(docs, business_id)

        # 🔥 DEBUG
        for chunk in fine_chunks[:5]:
            print("FINE CHUNK:", chunk["text"][:100])

        for chunk in coarse_chunks[:3]:
            print("COARSE CHUNK:", chunk["text"][:100])


        # -------------------------
        # EMBEDDINGS
        # -------------------------
        # fine_embeddings = embed_texts(
        #     [c["text"] for c in fine_chunks],
        #     self.embedding_model
        # )

        # coarse_embeddings = embed_texts(
        #     [c["text"] for c in coarse_chunks],
        #     self.embedding_model
        # )

        # -------------------------
        # EMBEDDINGS (DISABLED)
        # -------------------------
        if self.embedding_model:
            fine_embeddings = embed_texts(
                [c["text"] for c in fine_chunks],
                self.embedding_model
            )

            coarse_embeddings = embed_texts(
                [c["text"] for c in coarse_chunks],
                self.embedding_model
            )
        else:
            fine_embeddings = None
            coarse_embeddings = None

        # -------------------------
        # BM25
        # -------------------------
        fine_bm25 = BM25Retriever(fine_chunks)
        coarse_bm25 = BM25Retriever(coarse_chunks)


        
        index = {
            "fine_chunks": fine_chunks,
            "coarse_chunks": coarse_chunks,
            "list_chunks": list_chunks,   # ✅ NEW
            "fine_embeddings": fine_embeddings,
            "coarse_embeddings": coarse_embeddings,
            "fine_bm25": fine_bm25,
            "coarse_bm25": coarse_bm25
        }

        print("\n📦 INDEX DEBUG")
        print("Fine chunks:", len(index["fine_chunks"]))
        print("Coarse chunks:", len(index["coarse_chunks"]))
        print("List chunks:", len(index["list_chunks"]))  # ✅ THIS ONE

        # -------------------------
        # RETURN
        # -------------------------
        return index
        
    
state = DocumentState()