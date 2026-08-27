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

        fine_chunks, coarse_chunks, list_chunks, entity_section_map = process_documents(docs, business_id)

        print("\n" + "=" * 80)
        print("🔎 SEARCHING FOR SWIMMING POOL IN FINE CHUNKS")
        print("=" * 80)

        swimming_chunks = [
            c for c in fine_chunks
            if "swimming" in c.get("text", "").lower()
            or "pool" in c.get("text", "").lower()
        ]

        print("Swimming pool chunks:", len(swimming_chunks))

        for c in swimming_chunks:
            print(c)

        print("=" * 80)

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
        # # -------------------------
        # from collections import Counter

        # texts = [c["text"].strip().lower() for c in fine_chunks]

        # counts = Counter(texts)

        # print("\n========== DUPLICATE CHUNK DEBUG ==========")

        # for text, count in counts.items():

        #     if count > 1:

        #         print(
        #             f"DUPLICATE x{count}:",
        #             text
        #         )

        # print("TOTAL FINE CHUNKS:", len(fine_chunks))
        # print("UNIQUE FINE CHUNKS:", len(counts))
        # print("==========================================")

        print("\n" + "=" * 80)
        print("🔎 CHECKING BM25 INPUT")
        print("=" * 80)

        for i, c in enumerate(fine_chunks):
            text = c.get("text", "").lower()

        if "swimming" in text or "pool" in text:
            print("BM25 INPUT INDEX:", i)
            print("TEXT:", c["text"])
            print("SOURCE:", c)


        fine_bm25 = BM25Retriever(fine_chunks)
        coarse_bm25 = BM25Retriever(coarse_chunks)


        

        print("\n========== BM25 METADATA ==========")

        for i, chunk in enumerate(fine_bm25.chunks):

            if (
                "laundry" in chunk["text"].lower()
                or "swimming pool" in chunk["text"].lower()
                or "room service" in chunk["text"].lower()
            ):
                print(
                        i,
                        repr(chunk["text"]),
                        "→",
                        chunk.get("section")
                    )


        
        index = {
            "fine_chunks": fine_chunks,
            "coarse_chunks": coarse_chunks,
            "list_chunks": list_chunks,   # ✅ NEW
            "entity_section_map": entity_section_map,
            "fine_embeddings": fine_embeddings,
            "coarse_embeddings": coarse_embeddings,
            "fine_bm25": fine_bm25,
            "coarse_bm25": coarse_bm25
        }

        print("\n========== INDEX ENTITY SECTION MAP ==========")

        for entity, section in index["entity_section_map"].items():
            print(
                entity,
                "→",
                section
            )

        print("\n📦 INDEX DEBUG")
        print("Fine chunks:", len(index["fine_chunks"]))
        print("Coarse chunks:", len(index["coarse_chunks"]))
        print("List chunks:", len(index["list_chunks"]))  # ✅ THIS ONE

        print("\n" + "=" * 80)
        print("📦 FINAL INDEX ASSEMBLY") 
        print("=" * 80)
 
        print("fine_chunks :", len(fine_chunks))
        print("coarse_chunks:", len(coarse_chunks))

        print("fine_bm25   :", len(fine_bm25.chunks))
        print("coarse_bm25 :", len(coarse_bm25.chunks))

        print("\nFINE BM25 TARGET CHECK:")

        for i, c in enumerate(fine_bm25.chunks):
            if "laundry" in c["text"].lower():
                print("FOUND IN FINE BM25:", i, repr(c["text"]))

        print("\nCOARSE BM25 TARGET CHECK:")

        for i, c in enumerate(coarse_bm25.chunks):
            if "laundry" in c["text"].lower():
                print("FOUND IN COARSE BM25:", i, repr(c["text"]))

        # -------------------------
        # RETURN
        # -------------------------
        return index
        
    
state = DocumentState()