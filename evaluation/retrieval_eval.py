def evaluate_retrieval(
    eval_queries,
    embedding_model,
    chunk_embeddings,
    chunks,
    retrieve_fn,
    k=3
):
    hits = []
    reciprocal_ranks = []

    for item in eval_queries:
        question = item["question"]
        relevant_sources = item["relevant_sources"]

        query_embedding = embedding_model(question)
        retrieved = retrieve_fn(query_embedding, chunk_embeddings, chunks, k=k)

        hit = 0
        rr = 0

        for rank, (_, _, source) in enumerate(retrieved, start=1):
            if any(rel in source for rel in relevant_sources):

                hit = 1
                rr = 1 / rank
                break

        hits.append(hit)
        reciprocal_ranks.append(rr)

    hit_at_k = sum(hits) / len(hits)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)

    return {
        "Hit@K": hit_at_k,
        "MRR": mrr
    }
