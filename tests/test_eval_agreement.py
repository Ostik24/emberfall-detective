"""Agreement and uncertainty helpers used by the evals."""

from evals.c1_agent import mcc
from evals.c12_explain import cohen_kappa
from evals.metrics.stats import paired_difference


def test_cohen_kappa_hand_computed():
    # 10 items: agree on 6 True + 2 False; judge True/human False once, judge False/human True once
    pairs = [(True, True)] * 6 + [(False, False)] * 2 + [(True, False), (False, True)]
    # p_o = 0.8; p_judge = p_human = 0.7; p_e = 0.49 + 0.09 = 0.58; kappa = 0.22 / 0.42
    assert cohen_kappa(pairs) == round(0.22 / 0.42, 4)
    assert cohen_kappa([]) is None


def test_mcc_perfect_inverse_and_undefined():
    assert mcc([(True, True), (False, False)]) == 1.0
    assert mcc([(True, False), (False, True)]) == -1.0
    assert mcc([(True, True), (True, True)]) == 0.0  # no negatives: undefined, reported as 0


def test_paired_difference_of_identical_runs_is_zero():
    assert paired_difference([1, 0, 1], [1, 0, 1]) == (0.0, 0.0, 0.0)
