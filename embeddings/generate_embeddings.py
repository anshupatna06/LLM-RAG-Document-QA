import numpy as np

def embed_texts(chunks, embedding_model):
    embeddings = []

    for chunk in chunks:
        vector = embedding_model(chunk["text"])  # embedding_model is already an instance
        embeddings.append(vector)

    return embeddings
