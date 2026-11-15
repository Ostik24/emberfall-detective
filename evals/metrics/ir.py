"""Ranking metrics for retrieval evals (Lecture 3): Precision, Recall, P@k, Recall@k, Success@k,
AP/MAP, RR/MRR, nDCG@k.

Input: `ranked`, the item ids a system returned in rank order, and `relevant`, the ground-truth
relevant ids (a mapping id -> gain for graded nDCG). Output: a float in [0, 1].
For known-item search there is one relevant id, so MRR = MAP and Recall@k = Success@k.
"""

import math
from collections.abc import Collection, Hashable, Mapping, Sequence

Ranking = Sequence[Hashable]
Run = tuple[Ranking, Collection[Hashable]]


def precision(ranked: Ranking, relevant: Collection[Hashable]) -> float:
    """Share of returned items that are relevant; 0 for an empty ranking."""
    if not ranked:
        return 0.0
    return sum(item in relevant for item in ranked) / len(ranked)


def recall(ranked: Ranking, relevant: Collection[Hashable]) -> float:
    """Share of relevant items that were returned; 0 when nothing is relevant."""
    if not relevant:
        return 0.0
    return len(set(ranked) & set(relevant)) / len(relevant)


def precision_at_k(ranked: Ranking, relevant: Collection[Hashable], k: int) -> float:
    """Relevant items in the top k, divided by k (also when fewer than k items were returned)."""
    return sum(item in relevant for item in ranked[:k]) / k


def recall_at_k(ranked: Ranking, relevant: Collection[Hashable], k: int) -> float:
    """Recall of the top k."""
    return recall(ranked[:k], relevant)


def success_at_k(ranked: Ranking, relevant: Collection[Hashable], k: int) -> float:
    """1 if any relevant item is in the top k, else 0 (Hit@k, Lecture 4)."""
    return float(any(item in relevant for item in ranked[:k]))


def average_precision(ranked: Ranking, relevant: Collection[Hashable]) -> float:
    """Mean of P@i over the ranks i that hold a relevant item.

    The denominator is the number of relevant items in the ground truth (TREC convention), so a
    relevant item that was never returned counts as precision 0. The lecture divides by the
    relevant items found; both agree when every relevant item was returned.
    """
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant)


def reciprocal_rank(ranked: Ranking, relevant: Collection[Hashable]) -> float:
    """1 / rank of the first relevant item; 0 if none was returned."""
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def mean_average_precision(runs: Sequence[Run]) -> float:
    """MAP: average_precision averaged over queries."""
    return sum(average_precision(r, rel) for r, rel in runs) / len(runs)


def mean_reciprocal_rank(runs: Sequence[Run]) -> float:
    """MRR: reciprocal_rank averaged over queries."""
    return sum(reciprocal_rank(r, rel) for r, rel in runs) / len(runs)


def ndcg_at_k(
    ranked: Ranking, relevant: Collection[Hashable] | Mapping[Hashable, float], k: int
) -> float:
    """DCG@k / ideal DCG@k with linear gain and discount 1/log2(rank + 1).

    `relevant` is a set (gain 1 each) or a mapping id -> graded gain. The ideal ranking is built
    from all ground-truth gains, not only from the returned items. 0 when nothing is relevant.
    """
    gains = relevant if isinstance(relevant, Mapping) else dict.fromkeys(relevant, 1.0)
    dcg = _dcg([gains.get(item, 0.0) for item in ranked[:k]])
    ideal = _dcg(sorted(gains.values(), reverse=True)[:k])
    return dcg / ideal if ideal > 0 else 0.0


def _dcg(gains: Sequence[float]) -> float:
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(gains, start=1))
