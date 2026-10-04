"""Text encoders for the semantic-similarity term.

``SentenceEncoder`` uses sentence-transformers (default all-MiniLM-L6-v2) with a
persistent cache. ``TfidfEncoder`` is a dependency-light fallback (TF-IDF ->
LSA) used when sentence-transformers is unavailable or in fast unit tests.
Both return L2-normalized float32 matrices so cosine similarity is a dot product.
"""
from __future__ import annotations

import logging
from typing import Protocol, Sequence

import numpy as np

from .cache import EmbeddingCache, text_key

log = logging.getLogger(__name__)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder(Protocol):
    name: str

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _normalize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


class TfidfEncoder:
    """Corpus-fitted TF-IDF + truncated SVD. Deterministic, no downloads."""

    name = "tfidf-lsa"

    def __init__(self, dim: int = 128) -> None:
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        texts = [t or "" for t in texts]
        if not any(t.strip() for t in texts):
            return np.zeros((len(texts), 1), dtype=np.float32)
        vec = TfidfVectorizer(stop_words="english", max_features=20000, sublinear_tf=True, min_df=1)
        X = vec.fit_transform(texts)
        k = min(self.dim, X.shape[1] - 1, X.shape[0] - 1)
        if k >= 2:
            X = TruncatedSVD(n_components=k, random_state=0).fit_transform(X)
        else:
            X = X.toarray()
        return _normalize(X)


class SentenceEncoder:
    """sentence-transformers encoder with a persistent per-text cache."""

    def __init__(self, model_name: str = DEFAULT_MODEL, batch_size: int = 128, max_chars: int = 1000) -> None:
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.batch_size = batch_size
        self.max_chars = max_chars
        try:  # avoid a network round-trip per load when the model is already cached
            self._model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(model_name)
        self._cache = EmbeddingCache(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        texts = [(t or "")[: self.max_chars] for t in texts]
        keys = [text_key(t) for t in texts]
        missing = sorted({k: t for k, t in zip(keys, texts) if self._cache.get(k) is None}.items())
        if missing:
            log.info("encoding %d new texts (%d cached)", len(missing), len(texts) - len(missing))
            vecs = self._model.encode(
                [t for _, t in missing], batch_size=self.batch_size,
                show_progress_bar=len(missing) > 500, normalize_embeddings=True,
            )
            for (k, _), v in zip(missing, vecs):
                self._cache.put(k, v)
            self._cache.save()
        X = np.stack([self._cache.get(k).astype(np.float32) for k in keys]) if keys else np.zeros((0, 384), np.float32)
        return _normalize(X)


def get_encoder(kind: str = "auto", model_name: str = DEFAULT_MODEL) -> Encoder:
    """kind: 'auto' (MiniLM if installed, else TF-IDF), 'minilm', or 'tfidf'."""
    if kind == "tfidf":
        return TfidfEncoder()
    try:
        return SentenceEncoder(model_name)
    except Exception as e:  # ImportError, offline model download failure, ...
        if kind == "minilm":
            raise
        log.warning("sentence-transformers unavailable (%s); falling back to TF-IDF", type(e).__name__)
        return TfidfEncoder()
