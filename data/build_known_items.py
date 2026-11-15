"""Build the known-item gold set from TREC ToT queries.

    uv run python data/build_known_items.py

Source: the TREC ToT 2025 release on Zenodo (record 15356599, CC BY 4.0), queries and qrels
only (the 9 GB corpus is not needed). Of its splits, only those the retriever never saw are used:
- dev2 = TREC ToT 2023 test (MS-ToT, human movie queries) -> source dataset:trec-tot-2023-test
- dev3 = TREC ToT 2024 test (movies, landmarks, celebrities) -> source dataset:trec-tot-2024-test
`train` and `dev1` are TREC ToT 2023 train/dev, which selected and evaluated E3 in Colab
(brief 7.1); reusing them would leak.
Each answer (an English Wikipedia page id) is linked to TMDB via Wikidata (P4947 movie, P4983
TV); answers without a TMDB id (landmarks, people) are dropped with a reason.
Output: evals/gold/known_item.jsonl and evals/gold/known_item.dropped.json.
"""

import json
from collections import Counter
from pathlib import Path

import httpx
import pyarrow.parquet as pq

from concierge import config
from concierge.tmdb import TMDB
from concierge.wikimedia import Wikimedia
from evals.gold import GOLD_DIR, KnownItem, assign_split, write

ZENODO = "https://zenodo.org/records/15356599/files"
RAW_DIR = config.DATA_DIR / "raw" / "trec_tot_2025"
SPLITS = {"dev2": "trec-tot-2023-test", "dev3": "trec-tot-2024-test"}


def raw_file(name: str) -> Path:
    """Local copy of a Zenodo file, downloaded once."""
    path = RAW_DIR / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(f"{ZENODO}/{name}?download=1", follow_redirects=True, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
    return path


def read_split(split: str) -> list[tuple[str, str, str]]:
    """(query_id, query text, answer page id) for every query with a qrel."""
    queries = {}
    for line in raw_file(f"{split}-2025-queries.jsonl").read_text().splitlines():
        row = json.loads(line)
        queries[row["query_id"]] = row["query"]
    answers = {}
    for line in raw_file(f"{split}-2025-qrel.txt").read_text().splitlines():
        query_id, _, page_id, relevance = line.split()
        if int(relevance) > 0:
            answers[query_id] = page_id
    return [(q, queries[q], answers[q]) for q in sorted(answers, key=int) if q in queries]


def catalog_titles() -> dict[tuple[str, int], str]:
    table = pq.read_table(config.CATALOG_EXTENDED_PATH, columns=["item_type", "item_id", "title"])
    cols = (table[c].to_pylist() for c in ("item_type", "item_id", "title"))
    return {(t, i): title for t, i, title in zip(*cols, strict=True)}


def build(wiki: Wikimedia, tmdb: TMDB) -> tuple[list[KnownItem], list[dict]]:
    """Gold examples and the dropped queries with their reason."""
    titles = catalog_titles()
    examples: list[KnownItem] = []
    dropped: list[dict] = []
    for split, dataset in SPLITS.items():
        rows = read_split(split)
        qids = wiki.wikidata_ids(page for _, _, page in rows)
        tmdb_ids = wiki.tmdb_ids(q for q in qids.values() if q)
        for query_id, text, page in rows:
            example_id = f"{dataset}-{query_id}"
            qid = qids.get(page)
            ids = tmdb_ids.get(qid) if qid else None
            if not ids:
                reason = "no_wikidata_item" if not qid else "no_tmdb_match"
                dropped.append({"id": example_id, "page_id": page, "reason": reason})
                continue
            media = "movie" if "movie" in ids else "tv"  # TREC ToT targets films
            key = (media, ids[media])
            title = titles.get(key) or tmdb_title(tmdb, media, ids[media])
            if title is None:
                dropped.append({"id": example_id, "page_id": page, "reason": "tmdb_404"})
                continue
            examples.append(
                KnownItem(
                    id=example_id,
                    lang="en",
                    source=f"dataset:{dataset}",
                    split=assign_split(example_id),
                    query=text,
                    answer_item_id=ids[media],
                    answer_item_type=media,
                    answer_title=title,
                    in_catalog=key in titles,
                    wikidata_id=qid,
                )
            )
    return examples, dropped


def tmdb_title(tmdb: TMDB, media: str, item_id: int) -> str | None:
    details = tmdb.details(media, item_id)
    return (details.get("title") or details.get("name")) if details else None


def main() -> None:
    examples, dropped = build(Wikimedia(), TMDB())
    path = write("known_item", examples)
    (GOLD_DIR / "known_item.dropped.json").write_text(json.dumps(dropped, indent=1) + "\n")
    print(f"wrote {path}: {len(examples)} examples")
    print("kept:", dict(Counter((e.source, e.split, e.in_catalog) for e in examples)))
    print("dropped:", dict(Counter((d["id"].rsplit("-", 1)[0], d["reason"]) for d in dropped)))


if __name__ == "__main__":
    main()
