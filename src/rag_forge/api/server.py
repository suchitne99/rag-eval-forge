from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from rag_forge.retrieval.hybrid import ReciprocalRankFusion
from rag_forge.eval.triad import RAGTriadEvaluator

app = FastAPI(title="RAG-Eval-Forge API", version="1.2.0")
rrf = ReciprocalRankFusion(k=60)
evaluator = RAGTriadEvaluator()

class RRFRequest(BaseModel):
    dense_ids: List[str]
    sparse_ids: List[str]

class TriadEvalRequest(BaseModel):
    query: str
    retrieved_contexts: List[str]
    generated_answer: str

@app.get("/health")
def health():
    return {"status": "healthy", "service": "rag-eval-forge"}

@app.post("/retrieval/fuse")
def fuse_ranks(req: RRFRequest):
    ranked = rrf.fuse(req.dense_ids, req.sparse_ids)
    return {"fused_ranking": [{"id": doc_id, "rrf_score": score} for doc_id, score in ranked]}

@app.post("/eval/triad")
def evaluate_triad(req: TriadEvalRequest):
    res = evaluator.evaluate(req.query, req.retrieved_contexts, req.generated_answer)
    return res.model_dump()

# CORS middleware hook

# Pydantic v2 model_dump
# API server
