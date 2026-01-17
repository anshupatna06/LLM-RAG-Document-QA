import numpy as np
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



# Retrieve Top-K Chunks
def retrieve_top_k(query_embedding, chunk_embeddings, chunks, k=3):
    scored = []

    for emb, chunk in zip(chunk_embeddings, chunks):
        score = cosine_similarity(query_embedding, emb)
        scored.append((score, chunk["text"], chunk["source"]))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]



