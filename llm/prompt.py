def build_prompt(context_chunks, question):
    context = "\n\n".join(context_chunks)

    prompt = f"""
You are an assistant that answers questions strictly using the provided context.
If the answer is not present in the context, say:
"I cannot find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt
