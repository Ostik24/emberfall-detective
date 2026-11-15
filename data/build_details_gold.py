"""Build the `details` gold set for C8 get_title_details from Wikidata (no LLM, no TMDB facts).

    uv run python data/build_details_gold.py

Sample: 150 catalog films (75 baseline, 75 extension, fixed seed). Each is linked to Wikidata
through the TMDB external ids already cached by the extension run; directors (P57, English
labels), duration (P2047) and earliest publication year (P577) come from Wikidata, so the C8
check does not grade TMDB against itself. Films with none of the three facts are skipped.
Output: evals/gold/details.jsonl.
"""

from collections import Counter

from concierge.catalog import Catalog
from concierge.tmdb import TMDB
from concierge.wikimedia import Wikimedia
from evals.gold import DetailsFacts, assign_split, write

SEED = 2026
PER_SOURCE = 75


def main() -> None:
    catalog = Catalog.load()
    tmdb, wiki = TMDB(), Wikimedia()
    films = catalog.frame[catalog.frame.item_type == "movie"]
    sample = [
        films[films.source == source].sample(PER_SOURCE, random_state=SEED)
        for source in ("emberfall", "extension")
    ]
    rows = [(int(r.item_id), r.title, r.source) for part in sample for r in part.itertuples()]
    qids = {}
    for item_id, _, _ in rows:
        details = tmdb.details("movie", item_id, append=("keywords", "external_ids")) or {}
        qid = (details.get("external_ids") or {}).get("wikidata_id")
        if qid:
            qids[item_id] = qid
    facts = wiki.film_facts(qids.values())
    examples = []
    for item_id, title, source in rows:
        fact = facts.get(qids.get(item_id, "")) or {}
        if not (fact.get("directors") or fact.get("duration_minutes") or fact.get("year")):
            continue
        examples.append(
            DetailsFacts(
                id=f"movie-{item_id}",
                lang="en",
                source=f"dataset:wikidata-{source}",
                split=assign_split(f"movie-{item_id}"),
                item_type="movie",
                item_id=item_id,
                title=title,
                wikidata_id=qids[item_id],
                directors=fact["directors"],
                runtime_minutes=fact["duration_minutes"],
                year=fact["year"],
            )
        )
    path = write("details", examples)
    print(f"wrote {path}: {len(examples)} of {len(rows)} sampled films")
    print(dict(Counter((e.split, e.source) for e in examples)))
    print("with directors / runtime / year:", sum(bool(e.directors) for e in examples),
          sum(e.runtime_minutes is not None for e in examples),
          sum(e.year is not None for e in examples))  # fmt: skip


if __name__ == "__main__":
    main()
