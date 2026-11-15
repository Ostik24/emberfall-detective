"""Uncertainty for per-query metrics.

Purpose: say whether a difference between two systems on the same queries is more than noise
(paired bootstrap, as in the Colab retriever experiments, brief 7.1).
Input: per-query scores. Output: mean and a 95% percentile confidence interval.
"""

from collections.abc import Sequence

import numpy as np

SEED = 2026


def bootstrap_ci(
    values: Sequence[float], samples: int = 10_000, seed: int = SEED
) -> tuple[float, float, float]:
    """(mean, low, high) of the mean of `values`, resampling queries with replacement."""
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(samples, len(x)))].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(x.mean()), float(low), float(high)


def paired_difference(
    a: Sequence[float], b: Sequence[float], samples: int = 10_000, seed: int = SEED
) -> tuple[float, float, float]:
    """(mean, low, high) of b - a over the same queries; the CI excluding 0 means significant."""
    if len(a) != len(b):
        raise ValueError("paired scores need the same queries")
    return bootstrap_ci(np.asarray(b, float) - np.asarray(a, float), samples, seed)
