"""On-disk embedding cache keyed by (model, sha1(text)).

Stored outside the repo by default (``~/.cache/swarmguard``) because the repo
may live in an iCloud-synced folder where large files get evicted. Override with
``SWARMGUARD_CACHE``.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np


def cache_root() -> Path:
    root = Path(os.environ.get("SWARMGUARD_CACHE", Path.home() / ".cache" / "swarmguard"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


class EmbeddingCache:
    """A dict-like float16 vector store persisted as one pickle per model."""

    def __init__(self, model_name: str, root: Path | None = None) -> None:
        safe = model_name.replace("/", "__")
        self.path = (root or cache_root()) / "embeddings" / f"{safe}.pkl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, np.ndarray] = {}
        self._dirty = False
        if self.path.exists():
            try:
                with open(self.path, "rb") as f:
                    self._data = pickle.load(f)
            except Exception:
                self._data = {}

    def get(self, key: str) -> np.ndarray | None:
        return self._data.get(key)

    def put(self, key: str, vec: np.ndarray) -> None:
        self._data[key] = vec.astype(np.float16)
        self._dirty = True

    def __len__(self) -> int:
        return len(self._data)

    def save(self) -> None:
        if not self._dirty:
            return
        tmp = self.path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self._data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.path)
        self._dirty = False
