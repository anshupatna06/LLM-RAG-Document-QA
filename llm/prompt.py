def build_prompt(context_chunks, question, system_prompt):

    context = "\n\n".join(context_chunks[:3])

    prompt = f"""
{system_prompt}

Answer the user's question using the provided context.

Rules:
1. Use ONLY the provided context.
2. If the answer is not present, say:
   "I cannot find this information in the provided documents."
3. Be concise and natural.
4. Avoid repeating information.
5. Do not copy raw document formatting.

Context:
{context}

Question:
{question}

Answer:
"""

    return prompt