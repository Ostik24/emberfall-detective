"""C6 find_title: is the named title resolved to the right item?

Gold: evals/gold/title_lookup.jsonl (clean / typo / Ukrainian variants of catalog titles).
Configs (one factor each):
    wratio         catalog fuzzy match with WRatio, TMDB when the best score < 90 (first try)
    default        `ratio` instead of WRatio                        parent wratio
    catalog_only   never ask TMDB (accept = 0)                      parent default
    tmdb_always    always ask TMDB (accept = 101)                   parent default
    accept80       TMDB only below 80                               parent default
Metrics: Success@1, Success@5; grouped by variant (the brief's clean / typo / UK).
TMDB calls are cached, no LLM is involved.
"""

from dataclasses import dataclass

from concierge.catalog import Catalog
from concierge.tmdb import TMDB
from concierge.tools.find_title import ACCEPT, TitleFinder
from evals.gold import Split, TitleLookup, load

METRICS = ("success@1", "success@5")
GROUPS = ("variant", "lang")


@dataclass(frozen=True)
class FindTitleConfig:
    accept: float
    scorer: str = "ratio"
    parent: str | None = None


CONFIGS = {
    "wratio": FindTitleConfig(ACCEPT, scorer="wratio"),
    "default": FindTitleConfig(ACCEPT, parent="wratio"),
    "catalog_only": FindTitleConfig(0.0, parent="default"),
    "tmdb_always": FindTitleConfig(101.0, parent="default"),
    "accept80": FindTitleConfig(80.0, parent="default"),
}


def run(config_name: str, split: Split) -> dict:
    config = CONFIGS[config_name]
    finder = TitleFinder(Catalog.load(), TMDB(), config.accept, config.scorer)
    cases = []
    for example in load("title_lookup", split=split):
        assert isinstance(example, TitleLookup)
        result = finder.find(example.query)
        keys = [(m.item_type, m.item_id) for m in result.matches]
        answer = (example.answer_item_type, example.answer_item_id)
        top = result.matches[0] if result.matches else None
        cases.append(
            {
                "id": example.id,
                "variant": example.variant,
                "lang": example.lang,
                "query": example.query,
                "answer_title": example.answer_title,
                "got": f"{top.title} ({top.year}, {top.source}, {top.score})" if top else None,
                "success@1": float(keys[:1] == [answer]),
                "success@5": float(answer in keys[:5]),
            }
        )
    return {
        "config": {"name": config_name, **config.__dict__},
        "cases": cases,
        "metrics": METRICS,
        "groups": GROUPS,
        "failed": [c["id"] for c in cases if c["success@1"] == 0],
    }
