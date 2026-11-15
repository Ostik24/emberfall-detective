"""C8 get_title_details: are the facts it returns right?

Gold: evals/gold/details.jsonl, facts from Wikidata (independent of TMDB, where the tool reads).
Per film (a metric is None when Wikidata lacks that fact, and is then left out of its mean):
    director_match     a returned director equals a Wikidata director (folded names)
    director_fuzzy     same, with token-set similarity >= 85 (middle names, transliteration:
                       "Jimmy T. Murakami" / "Jimmy Murakami", "Aleksandr" / "Alexander")
    runtime_within_3   |runtime - Wikidata duration| <= 3 minutes (cuts differ slightly)
    under_120_agrees   both sources agree whether it is under two hours (the constraint the
                       agent checks for "something short")
    year_within_1      release year within one year (festival vs. theatrical dates)
"""

from dataclasses import dataclass

from rapidfuzz import fuzz

from concierge.catalog import Catalog
from concierge.normalize import fold
from concierge.tmdb import TMDB
from concierge.tools.get_title_details import get_title_details
from evals.gold import DetailsFacts, Split, load

METRICS = (
    "director_match",
    "director_fuzzy",
    "runtime_within_3",
    "under_120_agrees",
    "year_within_1",
)
NAME_SIMILARITY = 85
GROUPS = ("source",)


@dataclass(frozen=True)
class DetailsConfig:
    parent: str | None = None


CONFIGS = {"default": DetailsConfig()}


def run(config_name: str, split: Split) -> dict:
    tmdb, catalog = TMDB(), Catalog.load()
    cases = []
    for example in load("details", split=split):
        assert isinstance(example, DetailsFacts)
        got = get_title_details(tmdb, catalog, example.item_type, example.item_id)
        directors = {fold(d) for d in got.directors}
        gold_runtime, runtime = example.runtime_minutes, got.runtime_minutes
        both_runtime = gold_runtime is not None and runtime is not None
        cases.append(
            {
                "id": example.id,
                "source": example.source,
                "title": example.title,
                "got": {"directors": got.directors, "runtime": runtime, "year": got.year},
                "gold": {
                    "directors": example.directors,
                    "runtime": gold_runtime,
                    "year": example.year,
                },  # fmt: skip
                "director_match": float(any(fold(d) in directors for d in example.directors))
                if example.directors
                else None,
                "director_fuzzy": float(
                    any(
                        fuzz.token_set_ratio(fold(d), g) >= NAME_SIMILARITY
                        for d in example.directors
                        for g in directors
                    )
                )
                if example.directors
                else None,
                "runtime_within_3": float(abs(runtime - gold_runtime) <= 3)
                if both_runtime
                else None,
                "under_120_agrees": float((runtime < 120) == (gold_runtime < 120))
                if both_runtime
                else None,
                "year_within_1": float(abs(got.year - example.year) <= 1)
                if got.year and example.year
                else None,
            }
        )
    return {
        "config": {"name": config_name},
        "cases": cases,
        "metrics": METRICS,
        "groups": GROUPS,
        "failed": [c["id"] for c in cases if 0.0 in [c[m] for m in METRICS]],
    }
