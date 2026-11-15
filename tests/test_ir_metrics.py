"""IR metrics against hand-computed values.

The R/N lists are the exercises on Lecture 3 slides 24-25 (the slides give no answers; the
expected values below are computed by hand in the comments). The lecture has no nDCG exercise,
so the nDCG cases are our own, also computed by hand.
"""

import pytest

from evals.metrics.ir import (
    average_precision,
    mean_average_precision,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision,
    precision_at_k,
    recall,
    recall_at_k,
    reciprocal_rank,
    success_at_k,
)


def judged(labels: str) -> tuple[list[int], set[int]]:
    """'RNNR' -> ranking [0, 1, 2, 3] with relevant ids {0, 3}."""
    return list(range(len(labels))), {i for i, label in enumerate(labels) if label == "R"}


SLIDE_24 = "RNNRRNNNRN"  # relevant at ranks 1, 4, 5, 9


def test_precision_slide_24():
    ranked, relevant = judged(SLIDE_24)
    assert precision(ranked, relevant) == pytest.approx(4 / 10)


def test_recall_slide_24_five_relevant_in_ground_truth():
    ranked, found = judged("NNNRNNNN")
    relevant = found | {100, 101, 102, 103}  # 5 relevant in total, 1 retrieved
    assert recall(ranked, relevant) == pytest.approx(1 / 5)


def test_precision_at_5_slide_24():
    ranked, relevant = judged(SLIDE_24)
    assert precision_at_k(ranked, relevant, 5) == pytest.approx(3 / 5)


def test_average_precision_slide_24():
    ranked, relevant = judged(SLIDE_24)
    # (1/1 + 2/4 + 3/5 + 4/9) / 4 = 2.5444 / 4
    assert average_precision(ranked, relevant) == pytest.approx(0.636111, abs=1e-6)


def test_map_slide_25():
    q1 = judged("RNRNR")  # (1 + 2/3 + 3/5) / 3 = 0.755556
    q2 = judged("NRN")  # 1/2
    assert average_precision(*q1) == pytest.approx(0.755556, abs=1e-6)
    assert average_precision(*q2) == pytest.approx(0.5)
    assert mean_average_precision([q1, q2]) == pytest.approx(0.627778, abs=1e-6)


def test_mrr_slide_25_first_relevant_at_1_3_none():
    runs = [judged("RNN"), judged("NNR"), judged("NNN")]
    assert [reciprocal_rank(*r) for r in runs] == pytest.approx([1.0, 1 / 3, 0.0])
    assert mean_reciprocal_rank(runs) == pytest.approx(4 / 9)


def test_average_precision_counts_unretrieved_relevant_as_zero():
    ranked, found = judged("RN")
    assert average_precision(ranked, found | {99}) == pytest.approx(0.5)


def test_precision_at_k_divides_by_k_for_short_rankings():
    ranked, relevant = judged("R")
    assert precision_at_k(ranked, relevant, 5) == pytest.approx(1 / 5)


def test_recall_and_success_at_k_known_item():
    ranked, relevant = judged("NNNR")
    assert recall_at_k(ranked, relevant, 3) == 0.0
    assert recall_at_k(ranked, relevant, 4) == 1.0
    assert success_at_k(ranked, relevant, 3) == 0.0
    assert success_at_k(ranked, relevant, 4) == 1.0


def test_ndcg_binary():
    ranked, relevant = judged("NRNR")
    # DCG = 1/log2(3) + 1/log2(5) = 1.061606; ideal = 1 + 1/log2(3) = 1.630930
    assert ndcg_at_k(ranked, relevant, 4) == pytest.approx(0.650920, abs=1e-6)


def test_ndcg_graded():
    gains = {"d1": 3, "d2": 2, "d3": 3, "d4": 0, "d5": 1, "d6": 2}
    ranked = ["d1", "d2", "d3", "d4", "d5", "d6"]
    # DCG = 3 + 2/log2(3) + 3/2 + 0 + 1/log2(6) + 2/log2(7) = 6.861127
    # ideal [3,3,2,2,1,0] = 3 + 3/log2(3) + 1 + 2/log2(5) + 1/log2(6) = 7.140995
    assert ndcg_at_k(ranked, gains, 6) == pytest.approx(0.960808, abs=1e-6)


def test_ndcg_ideal_uses_unretrieved_relevant_items():
    # The only relevant item was not returned, so the score is 0, not undefined.
    assert ndcg_at_k(["a", "b"], {"z"}, 2) == 0.0
    # One of two relevant items returned, at rank 2: (1/log2(3)) / (1 + 1/log2(3)) = 0.386853.
    # An ideal built from the returned items only would give 0.630930.
    assert ndcg_at_k(["a", "x"], {"x", "z"}, 2) == pytest.approx(0.386853, abs=1e-6)


def test_empty_inputs_score_zero():
    assert precision([], {1}) == 0.0
    assert recall([1], set()) == 0.0
    assert average_precision([1], set()) == 0.0
    assert ndcg_at_k([1], set(), 10) == 0.0
