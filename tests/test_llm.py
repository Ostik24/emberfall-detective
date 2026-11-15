"""LLM wrapper against a fake Anthropic client: block parsing, caching, tool choice, failures."""

from typing import Any

import pytest
from anthropic._models import add_response_ids
from anthropic.types import Message
from pydantic import BaseModel

from concierge import config
from concierge.cache import DiskCache
from concierge.llm import LLM, LLMError, Usage

TOOL = {
    "name": "submit_title",
    "description": "Return the identified title.",
    "input_schema": {
        "type": "object",
        "properties": {"title": {"type": "string"}, "year": {"type": "integer"}},
        "required": ["title", "year"],
    },
}


class Title(BaseModel):
    title: str
    year: int


def message(*blocks: dict[str, Any], stop_reason: str = "end_turn") -> Message:
    """A Message as the SDK returns it, including the request id it attaches to responses."""
    msg = Message.model_validate(
        {
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": config.HAIKU,
            "content": list(blocks),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 200,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 2000,
            },
        }
    )
    add_response_ids(msg, request_id="req_test", workspace_id=None)
    return msg


THINKING = {"type": "thinking", "thinking": "", "signature": "sig"}
TEXT = {"type": "text", "text": "Twister (1996)"}
TOOL_USE = {
    "type": "tool_use",
    "id": "toolu_1",
    "name": "submit_title",
    "input": {"title": "Twister", "year": 1996},
}


class FakeClient:
    """Stands in for anthropic.Anthropic: records requests, returns queued messages."""

    def __init__(self, *responses: Message) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, Any]] = []
        self.messages = self

    def create(self, **request: Any) -> Message:
        self.requests.append(request)
        return self.responses.pop(0)


@pytest.fixture
def make_llm(tmp_path):
    def make(*responses: Message) -> tuple[LLM, FakeClient]:
        client = FakeClient(*responses)
        llm = LLM(
            client=client, cache=DiskCache(tmp_path / "cache"), usage_log=tmp_path / "u.jsonl"
        )
        return llm, client

    return make


def ask(llm: LLM, model: str = config.HAIKU):
    return llm.complete(model, "v1", "system prompt", [{"role": "user", "content": "tornado film"}])


def call(llm: LLM, model: str = config.HAIKU) -> Title:
    messages = [{"role": "user", "content": "tornado film"}]
    return llm.call_tool(model, "v1", "system prompt", messages, TOOL, Title)


def test_reads_blocks_by_type_when_thinking_comes_first(make_llm):
    llm, _ = make_llm(message(THINKING, TEXT, TOOL_USE))
    result = ask(llm)
    assert result.text == "Twister (1996)"
    assert [c.name for c in result.tool_calls] == ["submit_title"]
    assert result.content[0]["type"] == "thinking"  # kept for replay in a tool loop


def test_second_identical_call_is_served_from_cache(make_llm, tmp_path):
    llm, client = make_llm(message(TEXT))
    first = ask(llm)
    second = ask(llm)
    assert len(client.requests) == 1
    assert (first.from_cache, second.from_cache) == (False, True)
    assert second.text == first.text
    log = (tmp_path / "u.jsonl").read_text().splitlines()
    assert len(log) == 2 and '"cost_usd": 0.0' in log[1]
    assert (llm.spend.api_calls, llm.spend.cached_calls) == (1, 1)
    assert llm.spend.cost_usd == pytest.approx(first.cost_usd)  # the replay is free


def test_prompt_version_is_part_of_the_cache_key(make_llm):
    llm, client = make_llm(message(TEXT), message(TEXT))
    llm.complete(config.HAIKU, "v1", "s", [{"role": "user", "content": "q"}])
    llm.complete(config.HAIKU, "v2", "s", [{"role": "user", "content": "q"}])
    assert len(client.requests) == 2


def test_request_caches_system_prompt_and_sends_no_temperature(make_llm):
    llm, client = make_llm(message(TEXT))
    ask(llm)
    request = client.requests[0]
    assert request["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "temperature" not in request


def test_cost_from_usage():
    usage = Usage(input_tokens=1000, output_tokens=200, cache_read_input_tokens=2000)
    # Haiku: (1000 + 0.1 * 2000) * 0.10 / 1e6 + 200 * 0.50 / 1e6
    assert usage.cost_usd(config.HAIKU) == pytest.approx(0.00022)


def test_call_tool_forces_tool_choice_on_haiku(make_llm):
    llm, client = make_llm(message(TOOL_USE, stop_reason="tool_use"))
    assert call(llm) == Title(title="Twister", year=1996)
    assert client.requests[0]["tool_choice"] == {"type": "tool", "name": "submit_title"}


def test_call_tool_uses_auto_and_instruction_on_sonnet(make_llm):
    llm, client = make_llm(message(THINKING, TOOL_USE, stop_reason="tool_use"))
    call(llm, config.SONNET)
    request = client.requests[0]
    assert request["tool_choice"] == {"type": "auto"}
    assert "submit_title" in request["system"][0]["text"]


def test_missing_tool_call_raises_and_is_not_cached(make_llm):
    llm, client = make_llm(message(TEXT), message(TOOL_USE, stop_reason="tool_use"))
    with pytest.raises(LLMError, match="did not call"):
        call(llm, config.SONNET)
    assert call(llm, config.SONNET).title == "Twister"  # retried against the API, not the cache
    assert len(client.requests) == 2


def test_invalid_tool_input_raises_and_is_not_cached(make_llm):
    bad = {**TOOL_USE, "input": {"title": "Twister"}}
    llm, client = make_llm(message(bad, stop_reason="tool_use"), message(TOOL_USE))
    with pytest.raises(LLMError, match="does not match"):
        call(llm)
    call(llm)
    assert len(client.requests) == 2


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_refusal_and_truncation_raise_and_are_not_cached(make_llm, stop_reason):
    llm, client = make_llm(message(TEXT, stop_reason=stop_reason), message(TEXT))
    with pytest.raises(LLMError):
        ask(llm)
    assert ask(llm).from_cache is False
    assert len(client.requests) == 2


def test_structured_calls_use_strict_tools_with_closed_objects(make_llm):
    llm, client = make_llm(message(TOOL_USE))
    call(llm)
    sent = client.requests[0]["tools"][0]
    assert sent["strict"] is True
    assert sent["input_schema"]["additionalProperties"] is False
