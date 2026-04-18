# agent/policies/decision_rules.py

def should_refuse(memory):
    """
    Refuse ONLY when query is clearly out-of-domain
    """
    return (
        memory.execution["max_score"] < 0.15  # very low semantic overlap
    )


def should_clarify(memory):

    # clarify ONLY when nothing retrieved
    if memory.execution["used_chunks"] == 0:


        return True

    return False

