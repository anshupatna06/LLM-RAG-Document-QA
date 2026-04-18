# agent/planner.py

def plan(memory):

    used_chunks = memory.execution.get("used_chunks", 0)
    max_score = memory.execution.get("max_score", 0)

    # FIRST STEP → always retrieve
    if used_chunks == 0 and max_score == 0:
        return "RETRIEVE"

    # -----------------------------
    # CONFIDENCE BANDS
    # -----------------------------

    if max_score < 0.15:
        return "REFUSE"

    return "ANSWER"