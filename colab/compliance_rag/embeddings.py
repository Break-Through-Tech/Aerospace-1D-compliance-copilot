"""Embedding providers; optional SDKs load only when selected."""
import os
from typing import List, Optional
import numpy as np

class TfidfEmbedder:
    """Offline fallback (no model download). Fit on the clause corpus; dense L2-normalised vectors."""
    name = "tfidf"

    def __init__(self, corpus: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit(corpus)

    def encode(self, texts: List[str]) -> np.ndarray:
        x = self.vec.transform(texts).toarray().astype("float32")
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, 1e-9)


class SbertEmbedder:
    name = "sbert"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")


class OllamaEmbedder:
    """Local embeddings through Ollama (default nomic-embed-text). No API key."""
    name = "ollama"

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        from ollama import Client
        self.client = Client(host=host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    def encode(self, texts: List[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):                       # batch to keep requests small
            r = self.client.embed(model=self.model, input=texts[i:i + 64], keep_alive="30m")
            out.extend(r.embeddings)
        x = np.asarray(out, dtype="float32")
        return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)
