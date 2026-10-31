"""In-memory cosine search with optional FAISS acceleration."""
from typing import List
import numpy as np
from .schemas import Clause

class ClauseIndex:
    """FAISS inner-product index (cosine, since vectors are normalised). Falls back to numpy if faiss missing."""

    def __init__(self, clauses: List[Clause], embedder):
        if not clauses:
            raise ValueError("Clause corpus must not be empty")
        self.clauses, self.embedder = clauses, embedder
        mat = embedder.encode([c.text for c in clauses])
        try:
            import faiss
            self.index = faiss.IndexFlatIP(mat.shape[1])
            self.index.add(mat)
            self._np = None
        except ImportError:
            self.index, self._np = None, mat

    def search(self, query: str, k: int) -> List[tuple[Clause, float]]:
        if k < 1:
            raise ValueError("k must be positive")
        q = self.embedder.encode([query])
        k = min(k, len(self.clauses))
        if self.index is not None:
            scores, idx = self.index.search(q, k)
            scores, idx = scores[0], idx[0]
        else:
            s = (self._np @ q[0])
            idx = np.argsort(-s)[:k]
            scores = s[idx]
        return [(self.clauses[i], float(s)) for i, s in zip(idx, scores)]
