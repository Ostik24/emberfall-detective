"""Validate a gold set: schema, duplicates, split rule, catalog membership.

    uv run python -m evals.validate_gold known_item

Input: a set name from `evals.gold.SCHEMAS`. Output: counts per split / lang / source and a
list of problems; exit code 1 if there is any problem.
"""

import sys
from collections import Counter

import pyarrow.parquet as pq

from concierge.config import CATALOG_EXTENDED_PATH
from evals.gold import SCHEMAS, GoldExample, KnownItem, TitleLookup, assign_split, load


def catalog_keys() -> set[tuple[str, int]]:
    table = pq.read_table(CATALOG_EXTENDED_PATH, columns=["item_type", "item_id"])
    return set(zip(table["item_type"].to_pylist(), table["item_id"].to_pylist(), strict=True))


def problems(examples: list[GoldExample]) -> list[str]:
    """Every rule violation, one line each."""
    found: list[str] = []
    for key, n in Counter(e.id for e in examples).items():
        if n > 1:
            found.append(f"duplicate id {key}")
    for e in examples:
        if e.split != assign_split(e.split_key()):
            found.append(f"{e.id}: split {e.split}, hash rule says the other one")
    keys = catalog_keys()
    for e in examples:
        if isinstance(e, TitleLookup) and (e.answer_item_type, e.answer_item_id) not in keys:
            found.append(f"{e.id}: answer is not in the catalog")
    known = [e for e in examples if isinstance(e, KnownItem)]
    if known:
        for e in known:
            if ((e.answer_item_type, e.answer_item_id) in keys) != e.in_catalog:
                found.append(f"{e.id}: in_catalog={e.in_catalog} disagrees with the catalog")
        for query, n in Counter(e.query for e in known).items():
            if n > 1:
                found.append(f"duplicate query: {query[:60]}...")
    return found


def main() -> None:
    name = sys.argv[1]
    if name not in SCHEMAS:
        sys.exit(f"unknown set {name}; known: {sorted(SCHEMAS)}")
    examples = load(name)
    counts = Counter((e.split, e.lang, e.source) for e in examples)
    for (split, lang, source), n in sorted(counts.items()):
        print(f"{split:5} {lang} {source:40} {n}")
    if isinstance(examples[0], KnownItem):
        flags = Counter((e.split, e.in_catalog) for e in examples if isinstance(e, KnownItem))
        print("in_catalog per split:", dict(sorted(flags.items())))
    issues = problems(examples)
    for issue in issues:
        print("PROBLEM", issue)
    print(f"{len(examples)} examples, {len(issues)} problems")
    sys.exit(1 if issues else 0)


if __name__ == "__main__":
    main()
