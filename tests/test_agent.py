"""Agent rules that are enforced in code, with a scripted LLM (no network)."""

import copy
import hashlib
from typing import Any

import pandas as pd
import pytest

from concierge.agent import MAX_TOOL_CALLS, DetectiveAgent, is_confirmation
from concierge.catalog import Catalog
from concierge.llm import LLMResult, Spend, ToolCall, Usage
from concierge.tools.registry import TOOLS_BY_NAME
from concierge.tools.search_catalog import Candidate, SearchCatalogResult
from concierge.tools.watchlist import Watchlist


def reply(*calls: tuple[str, dict[str, Any]]) -> LLMResult:
    """An assistant message that calls the given tools."""
    blocks = [{"type": "tool_use", "id": f"t{i}", "name": n, "input": a}
              for i, (n, a) in enumerate(calls)]  # fmt: skip
    return LLMResult(
        model="m", prompt_version="v", text="", content=blocks, stop_reason="tool_use",
        tool_calls=[ToolCall(id=b["id"], name=b["name"], input=b["input"]) for b in blocks],
        usage=Usage(input_tokens=1, output_tokens=1), cost_usd=0.0,
    )  # fmt: skip


class ScriptedLLM:
    def __init__(self, *script: LLMResult) -> None:
        self.script = list(script)
        self.requests: list[dict] = []
        self.spend = Spend()

    def complete(self, **request) -> LLMResult:
        self.requests.append(request)
        self.spend.api_calls += 1
        return self.script.pop(0)


class FakeToolbox:
    """Real watchlist on a two-title catalog; search returns The Matrix."""

    def __init__(self, llm: ScriptedLLM, tmp_path) -> None:
        frame = pd.DataFrame({"item_type": ["movie", "movie"], "item_id": [603, 604],
                              "title": ["The Matrix", "The Matrix Reloaded"]})  # fmt: skip
        self.llm = llm
        self.watchlist = Watchlist(Catalog(frame), tmp_path / "w.db")

    def run(self, name: str, arguments: dict) -> Any:
        args = TOOLS_BY_NAME[name].args.model_validate(arguments)
        if name == "watchlist":
            return self.watchlist.run(args)
        candidate = Candidate(id="movie/603", title="The Matrix", year=1999, genres=[], snippet="")
        return SearchCatalogResult(candidates=[candidate], clues=None, searched_text="",
                                   rows_after_filters=2, reranked=True)  # fmt: skip


def agent(tmp_path, *script: LLMResult) -> tuple[DetectiveAgent, FakeToolbox]:
    box = FakeToolbox(ScriptedLLM(*script), tmp_path)
    return DetectiveAgent(box), box


RESPOND = ("respond", {"message": "ok"})


def token(action: str, item_id: str) -> str:
    """The watchlist's deterministic token for a proposed change."""
    return hashlib.sha256(f"{action}:{item_id}".encode()).hexdigest()[:8]


def test_confirm_in_the_same_turn_as_the_proposal_is_refused(tmp_path):
    bot, box = agent(tmp_path,
        reply(("watchlist", {"action": "add", "id": "movie/603"})),
        reply(("watchlist", {"action": "confirm", "token": token("add", "movie/603")})),
        reply(RESPOND))  # fmt: skip
    turn = bot.turn("yes, add The Matrix")
    assert box.watchlist.items() == []
    assert [c.ok for c in turn.tool_calls] == [True, False]


def test_confirm_after_a_yes_in_the_next_turn_writes(tmp_path):
    bot, box = agent(tmp_path, reply(("watchlist", {"action": "add", "id": "movie/603"})),
                     reply(RESPOND))  # fmt: skip
    bot.turn("add The Matrix to my watchlist")
    token = box.watchlist.pending.token
    box.llm.script += [reply(("watchlist", {"action": "confirm", "token": token})), reply(RESPOND)]
    bot.turn("так, додай")
    assert [i.id for i in box.watchlist.items()] == ["movie/603"]


def test_without_a_yes_the_next_turn_cannot_confirm(tmp_path):
    bot, box = agent(tmp_path, reply(("watchlist", {"action": "add", "id": "movie/603"})),
                     reply(RESPOND))  # fmt: skip
    bot.turn("add The Matrix to my watchlist")
    token = box.watchlist.pending.token
    box.llm.script += [reply(("watchlist", {"action": "confirm", "token": token})), reply(RESPOND)]
    turn = bot.turn("actually, what else is like it?")
    assert box.watchlist.items() == []
    assert turn.tool_calls[0].error.startswith("not executed")


def test_a_new_proposal_cannot_be_confirmed_with_an_old_yes(tmp_path):
    bot, box = agent(tmp_path, reply(("watchlist", {"action": "add", "id": "movie/603"})),
                     reply(RESPOND))  # fmt: skip
    bot.turn("add The Matrix")
    box.llm.script += [  # the model swaps in another title and confirms it with this "yes"
        reply(("watchlist", {"action": "add", "id": "movie/604"})),
        reply(("watchlist", {"action": "confirm", "token": token("add", "movie/604")})),
        reply(RESPOND),
    ]
    bot.turn("yes")
    assert box.watchlist.items() == []


def test_picks_must_come_from_tool_results(tmp_path):
    picks = {"message": "m", "picks": ["movie/603", "movie/999"]}
    bot, _ = agent(tmp_path, reply(("search_catalog", {"query": "red pill"})),
                   reply(("respond", picks)))  # fmt: skip
    turn = bot.turn("a hacker learns reality is a simulation")
    assert [p.id for p in turn.picks] == ["movie/603"]
    assert turn.dropped_picks == ["movie/999"]


def test_tool_budget_forces_respond(tmp_path):
    calls = [reply(("search_catalog", {"query": f"q{i}"})) for i in range(MAX_TOOL_CALLS)]
    bot, box = agent(tmp_path, *calls, reply(RESPOND))
    turn = bot.turn("find it")
    assert len(turn.tool_calls) == MAX_TOOL_CALLS and turn.forced_respond
    assert box.llm.requests[-1]["tool_choice"] == {"type": "tool", "name": "respond"}


@pytest.mark.parametrize(
    ("text", "yes"),
    [("yes", True), ("Yes please!", True), ("так, додай", True), ("ок", True),
     ("no", False), ("ні, не треба", False), ("don't add it", False),
     ("yes, and also a long new description of another film I remember", False)],
)  # fmt: skip
def test_is_confirmation(text, yes):
    assert is_confirmation(text) is yes


class BrokenToolbox(FakeToolbox):
    def run(self, name: str, arguments: dict) -> Any:
        if name == "search_catalog":
            raise RuntimeError("GET api.themoviedb.org/3/search/multi -> HTTP 503")
        return super().run(name, arguments)


def test_a_failing_tool_becomes_a_tool_error_the_model_sees(tmp_path):
    box = BrokenToolbox(ScriptedLLM(reply(("search_catalog", {"query": "q"})), reply(RESPOND)),
                        tmp_path)  # fmt: skip
    turn = DetectiveAgent(box).turn("find it")
    assert turn.tool_calls[0].ok is False and "HTTP 503" in turn.tool_calls[0].error


class FailingLLM(ScriptedLLM):
    def complete(self, **request) -> LLMResult:
        if not self.script:
            raise ConnectionError("network gone mid-turn")
        return super().complete(**request)


def test_a_failed_turn_leaves_a_valid_conversation(tmp_path):
    llm = FailingLLM(reply(("search_catalog", {"query": "q"})))  # then the API fails
    box = FakeToolbox(llm, tmp_path)
    bot = DetectiveAgent(box)
    with pytest.raises(ConnectionError):
        bot.turn("first message")
    assert bot.messages == [] and bot.user_texts == [] and bot.turn_index == 0
    llm.script += [reply(RESPOND)]
    bot.turn("second message")
    roles = [m["role"] for m in bot.messages]
    assert roles == ["user", "assistant", "user"]  # no tool_use left without its result


def test_rollback_also_undoes_text_joined_into_the_last_tool_result_message(tmp_path):
    llm = FailingLLM(reply(("search_catalog", {"query": "q"})), reply(RESPOND))
    bot = DetectiveAgent(FakeToolbox(llm, tmp_path))
    bot.turn("first message")
    before = copy.deepcopy(bot.messages)
    with pytest.raises(ConnectionError):
        bot.turn("second message")  # joins the last tool-result message, then the API fails
    assert bot.messages == before
