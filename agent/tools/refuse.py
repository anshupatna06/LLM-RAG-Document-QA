# agent/tools/refuse.py
import time

def refuse(memory, retrieval, start_time):
    """
    Return a structured refusal response when no grounded answer is possible.
    """

    end = time.time()

    return {
        "decision": "REFUSE",
        "answer": None,
        "failure": {
            "type": "NO_GROUNDED_CONTEXT",
            "reason": "No retrieved chunks passed the similarity threshold",
            "threshold": memory.execution["threshold"],
            "max_score": memory.execution["max_score"]
        },
        "retrieval": retrieval,
        "performance": {
            "latency": {
                "total_sec": round(end - start_time, 3),
                "retrieval_sec": 0.0,
                "llm_sec": 0.0
            },
            "cost": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated_cost_usd": 0.0
            }
        },
        "agent_trace": memory.trace
    }
