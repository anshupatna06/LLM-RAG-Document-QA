# agent/schemas.py
from typing import List, Dict, Optional
from pydantic import BaseModel


class AgentTraceStep(BaseModel):
    step: str
    detail: Optional[str] = None


class AgentResponse(BaseModel):
    decision: str                # ANSWER | REFUSE | CLARIFY
    answer: str
    clarification_question: Optional[str] = None
    sources: List[str]
    retrieval: Dict
    metrics: Dict
    performance: Dict
    agent_trace: List[AgentTraceStep]
