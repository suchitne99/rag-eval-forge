from typing import List, Dict

class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, dense_ranked: List[str], sparse_ranked: List[str]) -> List[tuple]:
        scores: Dict[str, float] = {}
        for rank, doc_id in enumerate(dense_ranked, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (self.k + rank))
        for rank, doc_id in enumerate(sparse_ranked, 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (self.k + rank))
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# Precomputed table

# Cross-encoder hook

# Metadata filter

# Batch scoring

# Generator streaming
