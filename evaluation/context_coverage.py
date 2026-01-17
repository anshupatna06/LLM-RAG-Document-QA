import re

def tokenize(text):
    return set(re.findall(r"\b\w+\b", text.lower()))

def context_coverage(answer, context_chunks):
    answer_tokens = tokenize(answer)
    context_tokens = set()

    for chunk in context_chunks:
        context_tokens |= tokenize(chunk)

    if not context_tokens:
        return 0.0

    overlap = answer_tokens & context_tokens
    return len(overlap) / len(answer_tokens) if answer_tokens else 0.0
