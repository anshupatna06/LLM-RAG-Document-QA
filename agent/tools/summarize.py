# agent/tools/summarize.py

def summarize_chunks(memory, retrieval, pipeline):
    """
    Compress retrieved chunks into shorter context.
    Used when too many chunks are retrieved.
    """

    used_chunks = []

    for c in retrieval["chunks"]:

        score = c.get("score", 1.0)   # default score
        text = c.get("text", "")
        source = c.get("source", "")

        # 🔥 if "used" missing → assume True
        if c.get("used", True):
            used_chunks.append(c)


    if not used_chunks:
        return retrieval

    # Simple summarization strategy (safe start)
    summary_prompt = (
        "Summarize the following context while keeping key facts:\n\n"
        + "\n\n".join(used_chunks)
    )

    summary = pipeline.llm.generate(summary_prompt)

    memory.log("summarize", "Context summarized")

    retrieval["summary"] = summary
    return retrieval
