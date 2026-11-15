"""End-to-end: the whole agent on known-item queries, as a user would meet it.

Gold: evals/gold/known_item.jsonl (+ known_item_uk for the Ukrainian set). Each query is one
conversation with the deployed agent (tools, verifier, explanations). When the agent asks a
question, the C3 user simulator (it knows the hidden target) answers, at most twice.
Per-case metrics:
    success@1 / success@5  the answer is the first / among the agent's picks
    rr                     reciprocal rank of the answer in the picks (0 if absent)
    dropped_picks          ids the model proposed that no tool returned (blocked by the guard;
                           the "unverified hallucination" count; shown hallucinations are 0
                           by construction)
    tool_calls, cost_usd, latency_s, questions
Groups: in-catalog vs out-of-catalog (out-of-catalog answers can only come from
external_lookup), source dataset, language.
"""

import hashlib
from dataclasses import dataclass

import anthropic

from concierge.agent import PROMPT_VERSION, AgentTurn, DetectiveAgent
from concierge.finalize import evidence_text, verify_and_explain
from concierge.llm import LLMError
from concierge.tools.registry import Toolbox
from evals.c3_clarify import simulate
from evals.gold import KnownItem, Split, load

METRICS = ("success@1", "success@5", "rr", "dropped_picks", "tool_calls", "cost_usd",
           "latency_s", "questions")  # fmt: skip
GROUPS = ("catalog", "source", "lang")
MAX_TURNS = 3


@dataclass(frozen=True)
class E2EConfig:
    gold: str = "known_item"
    finalize: bool = True
    limit: int | None = None  # first N queries in a fixed hash order (cheap tuning subset)
    parent: str | None = None


CONFIGS = {
    "default": E2EConfig(),
    "sub40": E2EConfig(limit=40),
    "uk": E2EConfig(gold="known_item_uk"),
    "no_finalize": E2EConfig(finalize=False, parent="default"),
}


def converse(box: Toolbox, config: E2EConfig, query: str, answer: str) -> list[AgentTurn]:
    """One conversation; the simulator answers the agent's questions (at most MAX_TURNS)."""
    agent = DetectiveAgent(box, finalize=verify_and_explain if config.finalize else None)
    turns, message = [], query
    for _ in range(MAX_TURNS):
        turn = agent.turn(message)
        turns.append(turn)
        if not turn.options or turn.picks:
            break
        reply = simulate(box.llm, query, evidence_text(box, answer), turn.message, turn.options)
        message = reply.answer
    return turns


def run(config_name: str, split: Split) -> dict:
    config = CONFIGS[config_name]
    box = Toolbox.create(watchlist_path=":memory:")
    cases = []
    examples = load(config.gold, split=split)
    if config.limit:
        examples = sorted(examples, key=lambda e: hashlib.sha256(e.id.encode()).hexdigest())
        examples = examples[: config.limit]
    for example in examples:
        assert isinstance(example, KnownItem)
        answer = f"{example.answer_item_type}/{example.answer_item_id}"
        try:
            turns = converse(box, config, example.query, answer)
            error = None
        except (LLMError, anthropic.APIError) as e:  # a crashed conversation is a failed case
            turns, error = [], f"{type(e).__name__}: {str(e)[:300]}"
        if not turns:
            cases.append({"id": example.id, "catalog": "in" if example.in_catalog else "out",
                          "source": example.source, "lang": example.lang, "error": error,
                          **dict.fromkeys(METRICS, 0.0)})  # fmt: skip
            continue
        picks = [p.id for p in turns[-1].picks]
        rank = picks.index(answer) + 1 if answer in picks else None
        cases.append(
            {
                "id": example.id,
                "catalog": "in" if example.in_catalog else "out",
                "source": example.source,
                "lang": example.lang,
                "answer_title": example.answer_title,
                "picks": [f"{p.id} {p.title}" for p in turns[-1].picks],
                "reply": turns[-1].message,
                "tools": [[c.name for c in t.tool_calls] for t in turns],
                "success@1": float(rank == 1),
                "success@5": float(rank is not None and rank <= 5),
                "rr": 1 / rank if rank else 0.0,
                "dropped_picks": float(sum(len(t.dropped_picks) for t in turns)),
                "tool_calls": float(sum(len(t.tool_calls) for t in turns)),
                "cost_usd": sum(t.cost_usd for t in turns),
                "latency_s": sum(t.latency_s for t in turns),
                "questions": float(sum(bool(t.options) for t in turns)),
                "explanations": [p.notes for p in turns[-1].picks[:3]],
                "error": None,
            }
        )
    return {
        "config": {"name": config_name, **config.__dict__, "agent_prompt": PROMPT_VERSION},
        "cases": cases,
        "metrics": METRICS,
        "groups": GROUPS,
        "failed": [c["id"] for c in cases if c["success@5"] == 0],
        "llm": box.llm.spend.model_dump(),
    }
