def recall_at_k(retrieved, threshold=0.3):

    relevant = [
        c.get("score", 0)
        for c in retrieved
        if c.get("score", 0) >= threshold
    ]

    return 1.0 if len(relevant) > 0 else 0.0