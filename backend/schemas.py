from pydantic import BaseModel
from typing import Dict, List, Any


class RetrievedChunk(BaseModel):
    text: str
    source: str
    score: float
    source: str
    used: bool


class Metrics(BaseModel):
    recall_at_k: float
    coverage: float
    faithful: bool
    grounding_score: float


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    threshold: float = 0.3
    debug: bool = False


class QueryResponse(BaseModel):
    query: Dict[str, str]
    answer: str
    sources: List[str]
    retrieval: Dict[str, Any]
    metrics: Dict[str, Any]
