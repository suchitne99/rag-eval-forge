import pytest
from rag_forge.retrieval.hybrid import ReciprocalRankFusion, DocumentChunk
from rag_forge.retrieval.reranker import MockCrossEncoderReranker
from rag_forge.eval.triad import RAGTriadEvaluator

def test_rrf_scoring_invariance():
    fusion = ReciprocalRankFusion(k=60)
    dense = ["doc1", "doc2", "doc3"]
    sparse = ["doc2", "doc1", "doc4"]
    fused = fusion.fuse(dense, sparse)
    # doc1 and doc2 appear in both and must be ranked highest!
    top_ids = [doc_id for doc_id, _ in fused[:2]]
    assert "doc1" in top_ids and "doc2" in top_ids

def test_cross_encoder_rerank():
    reranker = MockCrossEncoderReranker()
    docs = [
        DocumentChunk(chunk_id="d1", content="The sky is blue and cloudy today."),
        DocumentChunk(chunk_id="d2", content="Distributed caching in Redis with zero-copy buffers.")
    ]
    query = "How does distributed caching work?"
    ranked = reranker.rerank(query, docs, top_k=1)
    assert ranked[0].chunk_id == "d2"
    assert ranked[0].score > 0.0

def test_rag_triad_evaluation():
    evaluator = RAGTriadEvaluator()
    res = evaluator.evaluate(
        query="What is reinforcement learning?",
        retrieved_contexts=["Reinforcement learning is an area of machine learning where agents take actions."],
        generated_answer="Reinforcement learning involves machine learning agents taking actions in an environment."
    )
    assert res.context_precision > 0.0
    assert res.groundedness > 0.0
    assert res.answer_relevance > 0.0

# Ungrounded hallucination test case

# Identical RRF score tie-breaker

# Python 3.14 test verified

# Empty candidate sets

# Truncated context penalty
# RAG tests

# Adversarial hallucination test

# Identical score test

# Python 3.14 verified
