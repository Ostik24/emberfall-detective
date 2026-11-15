"""Listwise rerank with a mocked LLM."""

import pytest

from concierge.retrieval.rerank import Ranking, candidate_block, merge_order, rerank


def test_merge_keeps_valid_unique_numbers_then_fills_missing_then_tail():
    merged = merge_order([2, 2, 7, -1, 0], head=3, total=5)
    assert merged.order == [2, 0, 1, 3, 4]
    assert (merged.invalid, merged.missing) == (3, 1)


def test_candidates_are_delimited_and_truncated():
    block = candidate_block(["Twister. A storm film.", "x" * 50], max_chars=10)
    assert block.startswith('<candidates>\n<candidate n="0">\nTwister. A')
    assert '<candidate n="1">\nxxxxxxxxxx\n</candidate>' in block


class FakeLLM:
    def __init__(self, order: list[int]) -> None:
        self.order = order
        self.requests: list[dict] = []

    def call_tool(self, **kwargs) -> Ranking:
        self.requests.append(kwargs)
        return kwargs["output_model"](order=self.order)


def test_rerank_reorders_the_head_only_and_sends_the_recollection_as_data():
    llm = FakeLLM([1, 0])
    result = rerank(llm, "storm chasers", ["a", "b", "c"], depth=2)
    assert result.order == [1, 0, 2]
    user = llm.requests[0]["messages"][0]["content"]
    assert "<recollection>\nstorm chasers\n</recollection>" in user
    assert "numbers 0 to 1" in user


@pytest.mark.parametrize("texts", [[], ["only one"]])
def test_nothing_to_rerank_makes_no_call(texts):
    llm = FakeLLM([])
    assert rerank(llm, "q", texts).order == list(range(len(texts)))
    assert llm.requests == []
