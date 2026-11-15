"""Build the `false_memory` set for C11: real recollections plus one wrong detail.

    uv run python data/build_false_memory.py

Source: known_item queries whose answer is in the catalog, 60 per split (fixed hash order).
One false sentence is appended to each real recollection, alternating two kinds, both wrong
by construction and both checkable from the evidence the verifier sees:
- actor: "I think it starred X." - X is a well-known actor absent from the film's FULL TMDB
  cast (the evidence lists the top-billed cast);
- year: "I'm pretty sure it came out around Y." - Y is the TMDB release year +-15.
An earlier version let Haiku rewrite a detail: 59% of its outputs failed the code checks and
some "wrong" details were true ("probably after 2000" for a 2004 film), so injection is now
programmatic (DEV_LOG). No LLM calls.
Output: evals/gold/false_memory.jsonl.
"""

import hashlib
from collections import Counter

from concierge.catalog import Catalog
from concierge.tmdb import TMDB
from evals.gold import FalseMemory, KnownItem, load, write

PER_SPLIT = 60
YEAR_SHIFT = 15
FAMOUS = ["Tom Hanks", "Meryl Streep", "Denzel Washington", "Julia Roberts", "Robin Williams",
          "Sandra Bullock", "Jackie Chan", "Nicole Kidman", "Morgan Freeman", "Cate Blanchett",
          "Eddie Murphy", "Whoopi Goldberg"]  # fmt: skip


def _hash(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def false_sentence(example: KnownItem, details: dict, kind: str) -> tuple[str, str] | None:
    """(sentence, true detail) or None when the needed fact is missing."""
    if kind == "actor":
        cast = {c["name"] for c in (details.get("credits") or {}).get("cast", [])}
        billed = [c["name"] for c in (details.get("credits") or {}).get("cast", [])[:3]]
        options = [a for a in FAMOUS if a not in cast]
        if not billed or not options:
            return None
        actor = options[_hash(example.id) % len(options)]
        return f"I think it starred {actor}.", f"cast: {', '.join(billed)}"
    date = details.get("release_date") or details.get("first_air_date") or ""
    if not date[:4].isdigit():
        return None
    year = int(date[:4])
    wrong = (
        year - YEAR_SHIFT
        if _hash(example.id) % 2 or year + YEAR_SHIFT > 2025
        else year + YEAR_SHIFT
    )
    return f"I'm pretty sure it came out around {wrong}.", f"year: {year}"


def main() -> None:
    catalog, tmdb = Catalog.load(), TMDB()
    examples: list[FalseMemory] = []
    skipped = Counter()
    for split in ("dev", "test"):
        items = [e for e in load("known_item", split=split) if isinstance(e, KnownItem)]
        items = sorted((e for e in items if e.in_catalog), key=lambda e: _hash(e.id))
        for n, example in enumerate(items[:PER_SPLIT]):
            kind = ("actor", "year")[n % 2]
            media, item_id = example.answer_item_type, example.answer_item_id
            details = tmdb.details(media, item_id, append=("credits", "keywords")) or {}
            made = false_sentence(example, details, kind)
            if made is None:
                skipped[kind] += 1
                continue
            sentence, true_detail = made
            examples.append(
                FalseMemory(
                    id=f"fm-{example.id}",
                    lang="en",
                    source=f"synthetic:{kind}",
                    split=split,
                    query=example.query,
                    altered_query=f"{example.query.rstrip()} {sentence}",
                    wrong_detail=sentence,
                    true_detail=true_detail,
                    answer_id=f"{media}/{item_id}",
                    answer_title=catalog.frame.title[catalog.row((media, item_id))],
                )
            )
    path = write("false_memory", examples)
    print(f"wrote {path}: {len(examples)} examples; skipped {dict(skipped)}")
    print(dict(Counter((e.split, e.source) for e in examples)))


if __name__ == "__main__":
    main()
