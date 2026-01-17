def recall_at_k(retrieved, threshold=0.3):
    """
    retrieved: list of (score, text, source)
    """
    relevant = [score for score, _, _ in retrieved if score >= threshold]
    return 1.0 if len(relevant) > 0 else 0.0
