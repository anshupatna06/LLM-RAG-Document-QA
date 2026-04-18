import time

def clarify(memory, retrieval, start):

    business_name = memory.config.get("business_name", "our service")

    question = (
        f"I’m not fully sure about your request. "
        f"Could you please clarify what you would like to know about {business_name}?"
    )

    # execution memory update
    memory.execution["clarification_question"] = question

    return {
        "decision": "CLARIFY",
        "clarification": question,
        "retrieval": retrieval,
        "agent_trace": memory.trace,
        "performance": {
            "latency": {"total_sec": round(time.time() - start, 3)},
            "cost": {"total_tokens": 0, "estimated_cost_usd": 0.0}
        }
    }
