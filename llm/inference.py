# def generate_answer(prompt, llm):
#     return llm(prompt)
def generate_answer(context_chunks, question):
    # 🔥 simple deterministic answer
    if not context_chunks:
        return "I cannot find this information in the provided documents."

    # You can return best chunk or join top chunks
    return context_chunks[0]