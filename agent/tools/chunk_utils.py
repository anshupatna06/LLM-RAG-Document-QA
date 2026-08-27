def normalize_chunks(
    chunk,
    score_type
):
    # -----------------------------
    # Extract common fields
    # -----------------------------
    if isinstance(chunk, tuple):

        score, text, source = chunk

    elif isinstance(chunk, dict):

        score = chunk.get("score", 0.0)
        text = chunk.get("text", "")
        source = chunk.get("source", {})

    else:
        raise ValueError(
            f"Unsupported chunk type: {type(chunk)}"
        )

    # -----------------------------
    # Chunk ID
    # -----------------------------
    chunk_id = None

    if isinstance(source, dict):
        chunk_id = source.get("chunk_id")

    if not chunk_id:
        chunk_id = hash(text.strip().lower())

    # -----------------------------
    # Metadata
    # -----------------------------
    section = ""
    business_id = ""

    if isinstance(source, dict):

        section = source.get("section", "")

        business_id = source.get(
            "business_id",
            ""
        )

    # -----------------------------
    # Scores
    # -----------------------------
    vector_score = 0.0
    bm25_score = 0.0

    if score_type == "vector":
        vector_score = score

    elif score_type == "bm25":
        bm25_score = score

    # -----------------------------
    # Canonical Chunk
    # -----------------------------
    return {

        "chunk_id": chunk_id,

        "text": text,

        "source": source,

        "section": section,

        "business_id": business_id,

        "vector_score": vector_score,

        "bm25_score": bm25_score,

        "final_score": 0.0

    }
