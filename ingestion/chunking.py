def process_documents(documents, chunk_size=300, overlap=50):
    chunks = []

    for doc in documents:
        text = doc["text"]
        source = doc["source"]

        start = 0
        while start < len(text):
            chunk_text = text[start:start + chunk_size]

            chunks.append({
                "text": chunk_text,
                "source": source
            })

            start += chunk_size - overlap

    return chunks
