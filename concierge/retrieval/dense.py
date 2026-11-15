"""Dense retrieval: nearest catalog documents to a query vector by cosine similarity.

Purpose: the first stage of `search_catalog` (E3, brief section 4) and the semantic half of the
re-implemented baseline. Vectors are unit length, so cosine is a dot product; 25k x 384 floats
fit in memory and an exact scan is faster than any index at this size.
Input: a query vector, k, optionally a boolean mask of allowed rows (filters applied BEFORE
ranking, unlike the baseline's post-filter, brief W2). Output: (row, score) pairs, best first.
`document_vectors` embeds documents once per (model, texts) and caches them under .cache/vectors.
"""

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from concierge.config import CACHE_DIR
from concierge.embed import Embedder

VECTOR_CACHE_DIR = CACHE_DIR / "vectors"


class DenseIndex:
    """Exact cosine search over unit row vectors."""

    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.vectors)

    def search(
        self, query: np.ndarray, k: int, allowed: np.ndarray | None = None
    ) -> list[tuple[int, float]]:
        """Top-k rows by cosine to `query`; rows where `allowed` is False are never returned."""
        scores = self.vectors @ query.astype(np.float32)
        candidates = np.arange(len(scores)) if allowed is None else np.flatnonzero(allowed)
        k = min(k, len(candidates))
        if k == 0:
            return []
        top = candidates[np.argpartition(-scores[candidates], k - 1)[:k]]
        top = top[np.lexsort((top, -scores[top]))]  # best first, ties by row for determinism
        return [(int(i), float(scores[i])) for i in top]


def document_vectors(
    embedder: Embedder, texts: Sequence[str], cache_dir: Path = VECTOR_CACHE_DIR
) -> np.ndarray:
    """Embed `texts` with `embedder`, reusing the cached array if the same model and texts
    were embedded before."""
    digest = hashlib.sha256(json.dumps([embedder.name, list(texts)]).encode()).hexdigest()
    path = cache_dir / f"{digest[:24]}.npy"
    if path.exists():
        return np.load(path)
    vectors = embedder.encode(texts, progress=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, vectors)
    tmp.replace(path)
    return vectors
