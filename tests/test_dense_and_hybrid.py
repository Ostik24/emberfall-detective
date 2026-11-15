"""Dense top-k with a filter mask, the vector cache, and the baseline's RRF fusion."""

import numpy as np
import pytest

from concierge.retrieval.dense import DenseIndex, document_vectors
from evals.baseline.hybrid import EmberfallHybrid, HybridConfig


def unit(*xs: float) -> np.ndarray:
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


VECTORS = np.stack([unit(1, 0), unit(0.8, 0.6), unit(0, 1), unit(-1, 0)])


def test_dense_returns_best_first_and_respects_the_mask():
    index = DenseIndex(VECTORS)
    assert [row for row, _ in index.search(unit(1, 0), k=3)] == [0, 1, 2]
    allowed = np.array([False, True, True, True])
    assert [row for row, _ in index.search(unit(1, 0), k=2, allowed=allowed)] == [1, 2]
    assert index.search(unit(1, 0), k=5, allowed=np.zeros(4, bool)) == []


class FakeEmbedder:
    name = "fake"

    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector
        self.calls = 0

    def encode(self, texts, **_):
        self.calls += 1
        return np.stack([self.vector for _ in texts])


def test_document_vectors_are_cached_per_model_and_texts(tmp_path):
    embedder = FakeEmbedder(unit(1, 0))
    first = document_vectors(embedder, ["a", "b"], tmp_path)
    again = document_vectors(embedder, ["a", "b"], tmp_path)
    document_vectors(embedder, ["a", "c"], tmp_path)
    assert embedder.calls == 2 and np.array_equal(first, again)


def test_hybrid_fuses_like_the_sql():
    # Semantic order for query (1, 0): rows 0, 1, 2, 3. Keyword order: row 2 ("tornado" x2),
    # then row 3. Popularity nudges row 1.
    docs = ["alpha", "beta", "tornado tornado", "tornado"]
    popularity = np.array([0.0, 100.0, 0.0, 0.0])
    config = HybridConfig(match_count=3)
    hybrid = EmberfallHybrid(docs, VECTORS, popularity, FakeEmbedder(unit(1, 0)), config)
    expected = {
        0: 1 / 51,
        1: 1 / 52 + 0.02 * 100 / 200,
        2: 1 / 53 + 0.35 / 51,
        3: 1 / 54 + 0.35 / 52,
    }
    best = sorted(expected, key=lambda r: -expected[r])[:3]
    assert hybrid.search("tornado") == best
    assert expected[2] > expected[0]  # the keyword hit overtakes the top semantic hit


def test_hybrid_type_filter_applies_to_both_lists():
    hybrid = EmberfallHybrid(
        ["a", "b", "tornado", "tornado"], VECTORS, np.zeros(4), FakeEmbedder(unit(1, 0))
    )
    allowed = np.array([True, False, False, True])
    assert set(hybrid.search("tornado", allowed)) == {0, 3}


@pytest.mark.parametrize("depth", [(40, 80), (10, 20), (60, 80)])
def test_each_list_is_cut_at_twice_the_capped_match_count(depth):
    match_count, expected = depth
    assert HybridConfig(match_count=match_count).depth == expected
