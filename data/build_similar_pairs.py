"""Build the `similar_pairs` set for C7 similar_to from TMDB recommendations (silver labels).

    uv run python data/build_similar_pairs.py

Sample: 150 catalog titles (fixed seed; movies and TV from both sources). Expected neighbours
= TMDB's first page of recommendations that are in the catalog; anchors with fewer than 3 such
titles are skipped. TMDB recommendations reflect co-watching, not text similarity, so these
labels are silver: they are not checked by hand (the add-gold-example skill asks for a manual
check; the report states this gap).
Output: evals/gold/similar_pairs.jsonl.
"""

from collections import Counter

from concierge.catalog import Catalog
from concierge.tmdb import TMDB
from evals.gold import SimilarPairs, assign_split, write

SEED = 2026
SAMPLE = 150
MIN_EXPECTED = 3


def main() -> None:
    catalog = Catalog.load()
    tmdb = TMDB()
    sample = catalog.frame.sample(SAMPLE, random_state=SEED)
    examples = []
    for row in sample.itertuples():
        anchor = f"{row.item_type}/{int(row.item_id)}"
        recs = tmdb.recommendations(row.item_type, int(row.item_id))
        expected = [
            f"{r.get('media_type', row.item_type)}/{r['id']}"
            for r in recs
            if catalog.row((r.get("media_type", row.item_type), int(r["id"]))) is not None
        ]
        if len(expected) < MIN_EXPECTED:
            continue
        examples.append(
            SimilarPairs(
                id=anchor.replace("/", "-"),
                lang="en",
                source=f"dataset:tmdb-recommendations-{row.source}",
                split=assign_split(anchor.replace("/", "-")),
                notes="silver: TMDB recommendations in the catalog, not hand-checked",
                anchor_id=anchor,
                anchor_title=row.title,
                expected_ids=expected,
            )
        )
    path = write("similar_pairs", examples)
    print(f"wrote {path}: {len(examples)} of {SAMPLE} anchors")
    print(dict(Counter((e.split, e.source) for e in examples)))


if __name__ == "__main__":
    main()
