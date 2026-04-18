#agent/memory.py
class AgentMemory:
    def __init__(self, query):

        # ----------------------
        # EXECUTION MEMORY
        # ----------------------
        self.execution = {
            "original_query": query,
            "rewritten_query": None,
            "max_score": 0.0,
            "used_chunks": 0,
            "threshold": 0.0,
            "clarification_question": None,
            "clarification_answer": None,
            "was_clarified": False,
        }

        # ----------------------
        # CONVERSATION MEMORY
        # ----------------------
        self.conversation = {
            "last_topic": None,
            "last_answer": None
        }

        # ----------------------
        # CONFIG MEMORY
        # ----------------------
        self.config = {
            "business_id": None,
            "business_name": None,
            "tone_prompt": None,
            "system_prompt": None
        }

        self.trace = []

    def log(self, step, detail=None):
        self.trace.append({"step": step, "detail": detail})
