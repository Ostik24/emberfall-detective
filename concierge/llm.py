"""Single entry point for every Claude call in the project.

Purpose: make LLM steps cheap and reproducible. Every call goes through `LLM`, which
- replays a response from the disk cache when (model, prompt version, full request) was seen before;
- marks the system prompt for prompt caching;
- appends token usage and estimated cost of each call to a JSONL log;
- reads response blocks by type (thinking blocks may come first);
- refuses to cache failures: a refusal, a truncated response, or a missing/invalid tool call raise.

Input: model id, prompt version, system prompt, messages (and a tool for structured output).
Output: `LLMResult` (text, tool calls, raw content blocks, usage, cost), or a validated pydantic
model for structured output.
"""

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from anthropic import Anthropic
from anthropic.types import Message
from pydantic import BaseModel, ValidationError

from concierge import config
from concierge.cache import DiskCache

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """A response that must not be used or cached."""


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    web_search_requests: int = 0

    def cost_usd(self, model: str) -> float:
        """Estimated cost from config.PRICES (cache write 1.25x input, cache read 0.1x) plus
        web searches at config.WEB_SEARCH_USD each."""
        price = config.PRICES[model]
        input_equivalent = (
            self.input_tokens
            + 1.25 * self.cache_creation_input_tokens
            + 0.1 * self.cache_read_input_tokens
        )
        tokens = (input_equivalent * price.input + self.output_tokens * price.output) / 1_000_000
        return tokens + self.web_search_requests * config.WEB_SEARCH_USD


class ToolCall(BaseModel):
    id: str
    name: str
    input: dict[str, Any]


class LLMResult(BaseModel):
    model: str
    prompt_version: str
    text: str
    tool_calls: list[ToolCall]
    content: list[dict[str, Any]]  # raw blocks, thinking included, to send back in a tool loop
    stop_reason: str | None
    usage: Usage
    cost_usd: float
    from_cache: bool = False


class Spend(BaseModel):
    """Calls and cost since this client was created (for eval result files)."""

    api_calls: int = 0
    cached_calls: int = 0
    web_searches: int = 0
    cost_usd: float = 0.0

    def add(self, result: LLMResult) -> None:
        if result.from_cache:
            self.cached_calls += 1
        else:
            self.api_calls += 1
            self.web_searches += result.usage.web_search_requests
            self.cost_usd += result.cost_usd


class LLM:
    """Cached, logged Claude client. The API client is created on the first uncached call,
    so a fully cached run (graders, USE_CACHE=True) needs no API key."""

    def __init__(
        self,
        client: Anthropic | None = None,
        cache: DiskCache | None = None,
        usage_log: Path = config.LLM_USAGE_LOG,
    ) -> None:
        self._client = client
        self.cache = cache or DiskCache(config.LLM_CACHE_DIR)
        self.usage_log = usage_log
        self.spend = Spend()

    def complete(
        self,
        model: str,
        prompt_version: str,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
        effort: str | None = None,
        check: Callable[[LLMResult], object] | None = None,
        tool_choice: dict[str, Any] | None = None,
    ) -> LLMResult:
        """One Messages API call (or its cached replay). Tools run with `auto` choice unless
        `tool_choice` says otherwise. `check` may raise to reject a response; a rejected
        response is not cached."""
        request = _request(model, system, messages, tools, tool_choice, max_tokens, effort)
        return self._cached(prompt_version, request, check)

    def call_tool(
        self,
        model: str,
        prompt_version: str,
        system: str,
        messages: list[dict[str, Any]],
        tool: dict[str, Any],
        output_model: type[T],
        max_tokens: int = 4096,
        effort: str | None = None,
    ) -> T:
        """Structured output: make the model call `tool` and return its input as `output_model`.

        Models in config.FORCED_TOOL_CHOICE_MODELS get a forced tool_choice; the others get
        `auto` plus an instruction to call the tool. The tool is sent with `strict: true`, so
        the API constrains its input to the schema. A missing or invalid call raises LLMError.
        """
        tool = strict_tool(tool)
        name = tool["name"]
        if model in config.FORCED_TOOL_CHOICE_MODELS:
            tool_choice = {"type": "tool", "name": name}
        else:
            tool_choice = {"type": "auto"}
            system = f"{system}\n\nAnswer only by calling the `{name}` tool."
        request = _request(model, system, messages, [tool], tool_choice, max_tokens, effort)

        def parse(result: LLMResult) -> T:
            call = next((c for c in result.tool_calls if c.name == name), None)
            if call is None:
                raise LLMError(f"{model} did not call {name} (stop_reason={result.stop_reason})")
            try:
                return output_model.model_validate(call.input)
            except ValidationError as e:
                raise LLMError(f"{name} input does not match {output_model.__name__}: {e}") from e

        return parse(self._cached(prompt_version, request, check=parse))

    def _cached(
        self,
        prompt_version: str,
        request: dict[str, Any],
        check: Callable[[LLMResult], object] | None = None,
    ) -> LLMResult:
        """Replay from cache, or call the API, run `check` (raises on a bad result), then cache."""
        key = {"model": request["model"], "prompt_version": prompt_version, "request": request}
        hit = self.cache.get(key)
        if hit is not None:
            result = LLMResult.model_validate({**hit, "from_cache": True})
            self._log(result, request_id=None)
            return result

        message = self._api().messages.create(**request)
        result = _to_result(message, request["model"], prompt_version)
        self._log(result, request_id=message._request_id)
        if result.stop_reason == "refusal":
            raise LLMError(f"{result.model} refused: {message.stop_details}")
        if result.stop_reason == "max_tokens":
            raise LLMError(f"{result.model} hit max_tokens={request['max_tokens']}")
        if check is not None:
            check(result)
        self.cache.put(key, result.model_dump(mode="json", exclude={"from_cache"}))
        return result

    def _api(self) -> Anthropic:
        if self._client is None:
            self._client = Anthropic(api_key=config.get_secret(config.CLAUDE_KEY_NAME))
        return self._client

    def _log(self, result: LLMResult, request_id: str | None) -> None:
        self.spend.add(result)
        self.usage_log.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "model": result.model,
            "prompt_version": result.prompt_version,
            "from_cache": result.from_cache,
            "stop_reason": result.stop_reason,
            # A replay costs nothing; the original call's cost is in its own log line.
            "cost_usd": 0.0 if result.from_cache else result.cost_usd,
            "usage": result.usage.model_dump(),
            "request_id": request_id,
        }
        with self.usage_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def strict_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """`tool` with `strict: true` and `additionalProperties: false` on every object schema
    (required by strict tool use)."""

    def close(schema: Any) -> Any:
        if isinstance(schema, dict):
            schema = {k: close(v) for k, v in schema.items()}
            if schema.get("type") == "object":
                schema.setdefault("additionalProperties", False)
        elif isinstance(schema, list):
            schema = [close(v) for v in schema]
        return schema

    return {**tool, "strict": True, "input_schema": close(tool["input_schema"])}


def _request(
    model: str,
    system: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: dict[str, Any] | None,
    max_tokens: int,
    effort: str | None,
) -> dict[str, Any]:
    """Messages API kwargs. No `temperature`: current models reject it.

    The cache breakpoint sits on the system block, so tools + system form the cached prefix.
    Prefixes under 512 tokens are not cached by the API (no error, just no cache write).
    """
    request: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        "messages": messages,
    }
    if tools:
        request["tools"] = tools
    if tool_choice:
        request["tool_choice"] = tool_choice
    if effort:
        request["output_config"] = {"effort": effort}
    return request


def _to_result(message: Message, model: str, prompt_version: str) -> LLMResult:
    """Read blocks by type; thinking blocks are kept only in `content`.

    `model` is the requested id: prices and cache keys use it, not the id echoed in the response.
    """
    content = message.to_dict()["content"]
    usage = Usage(
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        cache_creation_input_tokens=message.usage.cache_creation_input_tokens or 0,
        cache_read_input_tokens=message.usage.cache_read_input_tokens or 0,
        web_search_requests=getattr(message.usage.server_tool_use, "web_search_requests", 0) or 0,
    )
    return LLMResult(
        model=model,
        prompt_version=prompt_version,
        text="".join(b["text"] for b in content if b["type"] == "text"),
        tool_calls=[
            ToolCall(id=b["id"], name=b["name"], input=b["input"])
            for b in content
            if b["type"] == "tool_use"
        ],
        content=content,
        stop_reason=message.stop_reason,
        usage=usage,
        cost_usd=usage.cost_usd(model),
    )
