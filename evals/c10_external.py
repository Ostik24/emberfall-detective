"""C10 external_lookup: does the fallback rescue titles the catalog does not have?

Gold: evals/gold/known_item.jsonl, queries whose answer is NOT in the catalog. The tool runs
on the description alone (no title guesses), so this measures the web-search path; the
agent's own guesses are measured end to end (evals/e2e.py).
Per-case metrics:
    rescue@1 / rescue@3  the answer is the first / among the resolved candidates
    web_searches         searches used (billed $10 per 1,000)
    cost_usd             LLM + search cost of the call
Summary: cost per rescue (total cost / rescues@3). The false-fallback rate (fallback used
although the catalog had the answer) is read from the E2E run.
"""

from dataclasses import dataclass

from concierge.catalog import Catalog
from concierge.llm import LLM, LLMError
from concierge.tmdb import TMDB
from concierge.tools.external_lookup import (
    PROMPT_VERSION,
    ExternalLookupArgs,
    ExternalLookupResult,
    external_lookup,
)
from evals.gold import KnownItem, Split, load

METRICS = ("rescue@1", "rescue@3", "web_searches", "cost_usd")
GROUPS = ("source",)


@dataclass(frozen=True)
class ExternalConfig:
    parent: str | None = None


CONFIGS = {"web": ExternalConfig()}


def run(config_name: str, split: Split) -> dict:
    llm, tmdb, catalog = LLM(), TMDB(), Catalog.load()
    cases = []
    for example in load("known_item", split=split):
        assert isinstance(example, KnownItem)
        if example.in_catalog:
            continue
        answer = f"{example.answer_item_type}/{example.answer_item_id}"
        before = llm.spend.cost_usd
        try:
            args = ExternalLookupArgs(description=example.query)
            result = external_lookup(tmdb, catalog, llm, args)
        except LLMError as e:  # a failed lookup is a failed case, not a crashed run
            result = ExternalLookupResult(candidates=[], web_searches=0, unresolved_titles=[])
            error = str(e)
        else:
            error = None
        ids = [c.id for c in result.candidates]
        cases.append(
            {
                "id": example.id,
                "source": example.source,
                "answer_title": example.answer_title,
                "found": [f"{c.id} {c.title} ({c.year})" for c in result.candidates],
                "unresolved": result.unresolved_titles,
                "error": error,
                "rescue@1": float(ids[:1] == [answer]),
                "rescue@3": float(answer in ids[:3]),
                "web_searches": float(result.web_searches),
                "cost_usd": llm.spend.cost_usd - before,
            }
        )
    return {
        "config": {"name": config_name, "prompt": PROMPT_VERSION},
        "cases": cases,
        "metrics": METRICS,
        "groups": GROUPS,
        "failed": [c["id"] for c in cases if not c["rescue@3"]],
        "llm": llm.spend.model_dump(),
    }


def summarize(cases: list[dict]) -> dict:
    rescues = sum(c["rescue@3"] for c in cases)
    cost = sum(c["cost_usd"] for c in cases)
    return {
        "total_cost_usd": round(cost, 4),
        "cost_per_rescue_usd": round(cost / rescues, 4) if rescues else None,
    }
