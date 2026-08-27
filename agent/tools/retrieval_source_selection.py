def select_retrieval_index(
    query_type,
    index
):
    if query_type in [
        "binary",
        "feature",
        "contact",
        "time",
        "action"
    ]:

        return (
            index["fine_chunks"],
            index["fine_embeddings"],
            index["fine_bm25"]
        )

    elif query_type == "list":

        return (
            index["list_chunks"],
            None,
            None
        )

    else:

        return (
            index["coarse_chunks"],
            index["coarse_embeddings"],
            index["coarse_bm25"]
        )
