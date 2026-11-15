"""C1 agent: is the first move right, and are the search arguments faithful?

Gold: evals/gold/agent_decision.jsonl. One request per example, built by the agent itself
(`DetectiveAgent.request`), so the eval sees exactly the deployed prompt and tools; no tool
is executed.
Predicted action: the first tool called; `respond` with options counts as `ask_user`.
Per-case metrics:
    action_correct       first move equals the expected one
    filters_exact        (search_catalog in gold and prediction) every stated filter set and
                         nothing else; years +-1, genres: non-empty subset of the expected ones,
                         item_type "movie" neither required nor penalised
    hallucinated_filter  (same cases) a filter the user did not state
    query_recall         (same cases) share of the user's content words kept in `query`
                         (the tool asks for the user's own words)
Summary extras: macro-F1 over actions, confusion matrix, clarify-vs-act MCC.
Config `examples` (experiment, dev only, after the test run; not deployed): the same request
with `input_examples` on search_catalog (tool use examples, supported on Haiku 5.5); the
examples are written for this purpose and are not gold items.
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

from concierge.agent import PROMPT_VERSION, DetectiveAgent
from concierge.llm import LLM
from concierge.normalize import fold
from evals.gold import AgentDecision, Split, load

METRICS = ("action_correct", "filters_exact", "hallucinated_filter", "query_recall")
GROUPS = ("expected", "lang")
ACTIONS = ("search_catalog", "find_title", "watchlist", "external_lookup", "ask_user", "respond")
FILTERS = ("item_type", "year_from", "year_to", "genres", "exclude_genres")
STOP = {"a", "an", "the", "and", "or", "of", "to", "in", "on", "it", "i", "is", "was", "where",
        "with", "about", "that", "this", "there", "me", "my", "for", "but", "be"}  # fmt: skip


SEARCH_EXAMPLES = [
    {"query": "a Korean series where contestants play deadly children's games", "item_type": "tv"},
    {"query": "an 80s horror movie about a killer doll, nothing animated", "year_from": 1980,
     "year_to": 1989, "exclude_genres": ["Animation"]},
    {"query": "a documentary about the history of the internet", "genres": ["Documentary"]},
    {"query": "maybe from the 2000s? a man wakes up on a train with no memory"},
]  # fmt: skip


@dataclass(frozen=True)
class AgentConfig:
    examples: bool = False
    parent: str | None = None


CONFIGS = {"default": AgentConfig(), "examples": AgentConfig(examples=True, parent="default")}


class _NoToolsBox:
    """The first move needs only the LLM; no tool runs in this eval."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm


def first_move(llm: LLM, message: str, examples: bool = False) -> tuple[str, dict]:
    """(predicted action, arguments of the first tool call)."""
    agent = DetectiveAgent(_NoToolsBox(llm))  # type: ignore[arg-type]
    agent.add_user_message(message)
    request = agent.request()
    if examples:
        request["tools"] = [
            {**t, "input_examples": SEARCH_EXAMPLES} if t["name"] == "search_catalog" else t
            for t in request["tools"]
        ]
    result = llm.complete(**request)
    if not result.tool_calls:
        return "respond", {}
    call = result.tool_calls[0]
    if call.name == "respond":
        return ("ask_user" if call.input.get("options") else "respond"), call.input
    return call.name, call.input


def filter_scores(expected: dict, args: dict) -> tuple[float, float]:
    """(all stated filters right and nothing extra, any extra filter)."""
    right, extra = True, False
    for name in FILTERS:
        want, got = expected.get(name), args.get(name) or None
        if name == "item_type" and got == "movie" and want is None:
            continue
        if want is None:
            extra |= got is not None
            right &= got is None
        elif name in ("year_from", "year_to"):
            right &= got is not None and abs(int(got) - int(want)) <= 1
        elif name in ("genres", "exclude_genres"):
            got_set = {g.casefold() for g in got or []}
            right &= bool(got_set) and got_set <= {g.casefold() for g in want}
        else:
            right &= got == want
    return float(right), float(extra)


def query_recall(message: str, query: str) -> float:
    words = [w for w in re.findall(r"\w+", fold(message)) if w not in STOP]
    kept = set(re.findall(r"\w+", fold(query)))
    return sum(w in kept for w in words) / len(words) if words else 1.0


def run(config_name: str, split: Split) -> dict:
    llm = LLM()
    config = CONFIGS[config_name]
    cases = []
    for example in load("agent_decision", split=split):
        assert isinstance(example, AgentDecision)
        action, args = first_move(llm, example.message, config.examples)
        both_search = action == example.expected_action == "search_catalog"
        exact, extra = filter_scores(example.expected_filters, args) if both_search else (None,) * 2
        cases.append(
            {
                "id": example.id,
                "lang": example.lang,
                "expected": example.expected_action,
                "predicted": action,
                "message": example.message,
                "args": args,
                "action_correct": float(action == example.expected_action),
                "filters_exact": exact,
                "hallucinated_filter": extra,
                "query_recall": query_recall(example.message, args.get("query", ""))
                if both_search
                else None,
            }
        )
    return {
        "config": {"name": config_name, "prompt": PROMPT_VERSION, **config.__dict__},
        "cases": cases,
        "metrics": METRICS,
        "groups": GROUPS,
        "failed": [c["id"] for c in cases if c["action_correct"] == 0 or c["filters_exact"] == 0],
        "llm": llm.spend.model_dump(),
    }


def summarize(cases: list[dict]) -> dict:
    """Macro-F1 over the actions present in gold, confusion matrix, clarify-vs-act MCC."""
    labels = sorted({c["expected"] for c in cases})
    f1s = {}
    for label in labels:
        tp = sum(c["expected"] == label == c["predicted"] for c in cases)
        fp = sum(c["predicted"] == label != c["expected"] for c in cases)
        fn = sum(c["expected"] == label != c["predicted"] for c in cases)
        f1s[label] = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    confusion = Counter(f"{c['expected']} -> {c['predicted']}" for c in cases)
    ask = [(c["expected"] == "ask_user", c["predicted"] == "ask_user") for c in cases]
    return {
        "macro_f1": round(sum(f1s.values()) / len(f1s), 4),
        "f1_per_action": {k: round(v, 3) for k, v in f1s.items()},
        "confusion": dict(sorted(confusion.items())),
        "clarify_mcc": round(mcc(ask), 4),
    }


def mcc(pairs: list[tuple[bool, bool]]) -> float:
    """Matthews correlation of (expected, predicted) booleans; 0 when undefined."""
    tp = sum(e and p for e, p in pairs)
    tn = sum(not e and not p for e, p in pairs)
    fp = sum(p and not e for e, p in pairs)
    fn = sum(e and not p for e, p in pairs)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denominator if denominator else 0.0
