import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def grounding_score(answer, context_chunks, embedding_model):
    """
    Measures how well the answer is supported by retrieved context.
    """
    answer_emb = embedding_model(answer)

    context_text = " ".join(context_chunks)
    context_emb = embedding_model(context_text)

    score = cosine_similarity(answer_emb, context_emb)
    return score
