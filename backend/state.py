from ingestion.load_documents import load_documents
from ingestion.chunking import process_documents
from embeddings.generate_embeddings import embed_texts
from embeddings.embedding_model import EmbeddingModel
from retrieval.bm25_retriever import BM25Retriever
import os

DATA_DIR = "data/documents"

# class DocumentState:

#     def __init__(self):
#         self.business_id = "hotel"
#         self.embedding_model = EmbeddingModel()
#         self.reload()

#     def reload(self):

#         self.docs = load_documents(DATA_DIR)

#         print("Documents loaded:", len(self.docs))   #DEBUG TOOL

#         indices = {}

#         for doc in self.docs:

#             source = doc["source"].replace("\\", "/")

#             # extract business + client from path
#             parts = source.split("/")

#             # expected path:
#             # data/documents/hotel/taj/file.pdf

#             if len(parts) >= 3:
#                 business_id = parts[-3]
#                 client_id = parts[-2]
#             else:
#                 # fallback (old docs)
#                 business_id = "hotel"
#                 client_id = "default"

#             chunks = process_documents([doc], business_id)

#             # initialize business
#             if business_id not in indices:
#                 indices[business_id] = {}

#             # initialize client
#             if client_id not in indices[business_id]:
#                 indices[business_id][client_id] = {
#                     "chunks": []
#                 }

#             indices[business_id][client_id]["chunks"].extend(chunks)

#         # Build embeddings and BM25 per client
#         for business_id, clients in indices.items():

#             for client_id, data in clients.items():
 
#                 chunks = data["chunks"]

#                 embeddings = embed_texts(
#                     chunks,
#                     self.embedding_model
#                 )

#                 bm25 = BM25Retriever(chunks)

#                 indices[business_id][client_id]["embeddings"] = embeddings
#                 indices[business_id][client_id]["bm25"] = bm25

#         self.indices = indices

#         print("Indexed businesses:", list(self.indices.keys()))
#         for business in indices:
#             print(business, list(indices[business].keys()))


# state = DocumentState()
class DocumentState:

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        

    def get_index(self, business_id, client_id):

        folder = os.path.join(DATA_DIR, business_id, client_id)

        if not os.path.exists(folder):
            return None

        docs = load_documents(folder)

        self.docs = load_documents(DATA_DIR)
        for doc in self.docs:
            print("Loaded file:", doc["source"])

        if not docs:
            return None

        fine_chunks, coarse_chunks, list_chunks = process_documents(docs, business_id)

        # 🔥 DEBUG
        for chunk in fine_chunks[:5]:
            print("FINE CHUNK:", chunk["text"][:100])

        for chunk in coarse_chunks[:3]:
            print("COARSE CHUNK:", chunk["text"][:100])


        # -------------------------
        # EMBEDDINGS
        # -------------------------
        fine_embeddings = embed_texts(
            [c["text"] for c in fine_chunks],
            self.embedding_model
        )

        coarse_embeddings = embed_texts(
            [c["text"] for c in coarse_chunks],
            self.embedding_model
        )

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