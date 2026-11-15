"""Build the title_lookup gold set for C6 find_title (no LLM involved).

    uv run python data/build_title_lookup.py

Sample: 200 catalog titles, fixed seed, half baseline rows and half pre-2000 extension rows.
Titles whose folded form occurs more than once in the catalog are skipped (the query would be
ambiguous without a year). Every title yields up to three queries:
- clean: the catalog title;
- typo: one deterministic edit per word of 5+ letters, at most two (swap, drop or double a
  letter), the kind of slip people make when typing;
- uk: the official Ukrainian title from TMDB translations, when one exists and differs.
Output: evals/gold/title_lookup.jsonl.
"""

import hashlib
import random
from collections import Counter

from concierge.catalog import Catalog
from concierge.normalize import detect_language, fold
from concierge.tmdb import TMDB
from evals.gold import TitleLookup, assign_split, write

SEED = 2026
PER_SOURCE = 100


def typo(title: str) -> str | None:
    """Up to two single-letter edits in the longer words, chosen from a hash of the title."""
    rng = random.Random(hashlib.sha256(title.encode()).hexdigest())
    words = title.split(" ")
    long_words = [i for i, w in enumerate(words) if sum(c.isalpha() for c in w) >= 5]
    if not long_words:
        return None
    for i in rng.sample(long_words, min(2, len(long_words))):
        w = words[i]
        j = rng.randrange(1, len(w) - 1)
        edit = rng.choice(["swap", "drop", "double"])
        if edit == "swap":
            w = w[:j] + w[j + 1] + w[j] + w[j + 2 :]
        elif edit == "drop":
            w = w[:j] + w[j + 1 :]
        else:
            w = w[:j] + w[j] + w[j:]
        words[i] = w
    edited = " ".join(words)
    return edited if fold(edited) != fold(title) else None


def ukrainian_title(tmdb: TMDB, media: str, item_id: int) -> str | None:
    details = tmdb.details(media, item_id, append=("translations",)) or {}
    for t in details.get("translations", {}).get("translations", []):
        if t.get("iso_639_1") == "uk":
            data = t.get("data") or {}
            name = (data.get("title") or data.get("name") or "").strip()
            if name and detect_language(name) == "uk":
                return name
    return None


def main() -> None:
    catalog = Catalog.load()
    frame = catalog.frame
    counts = Counter(fold(t) for t in frame.title)
    unique = frame[[counts[fold(t)] == 1 for t in frame.title]]
    tmdb = TMDB()
    examples: list[TitleLookup] = []
    for source in ("emberfall", "extension"):
        rows = unique[unique.source == source].sample(PER_SOURCE, random_state=SEED)
        for row in rows.itertuples():
            media, item_id, title = row.item_type, int(row.item_id), row.title
            base = f"{media}-{item_id}"
            variants = {
                "clean": title,
                "typo": typo(title),
                "uk": ukrainian_title(tmdb, media, item_id),
            }
            for variant, query in variants.items():
                if query:
                    examples.append(
                        TitleLookup(
                            id=f"{base}-{variant}",
                            lang="uk" if variant == "uk" else "en",
                            source=f"dataset:catalog-{source}",
                            split=assign_split(base),
                            notes="" if variant != "uk" else "TMDB uk translation",
                            query=query,
                            variant=variant,
                            answer_item_id=item_id,
                            answer_item_type=media,
                            answer_title=title,
                        )
                    )
    path = write("title_lookup", examples)
    print(f"wrote {path}: {len(examples)} examples")
    print(dict(Counter((e.split, e.variant) for e in examples)))


if __name__ == "__main__":
    main()
