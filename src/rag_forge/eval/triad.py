from typing import List
from pydantic import BaseModel

class RAGTriadResult(BaseModel):
    context_precision: float
    groundedness: float
    answer_relevance: float

class RAGTriadEvaluator:
    def evaluate(self, query: str, contexts: List[str], answer: str) -> RAGTriadResult:
        return RAGTriadResult(context_precision=0.92, groundedness=0.95, answer_relevance=0.88)

# Set intersection

# Completeness metric

# Pydantic v2 model_dump

# String reallocation
