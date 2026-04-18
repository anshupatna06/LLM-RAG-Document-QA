def embed_texts(texts, embedding_model):
    embeddings = []

    for text in texts:
        vector = embedding_model(text)
        embeddings.append(vector)

    return embeddings