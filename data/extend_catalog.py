"""Extend the catalog: Wikipedia intros for every title, plus movies and TV from before 2000.

Run by the owner:  uv run python data/extend_catalog.py
Smoke run:         uv run python data/extend_catalog.py --sample 10 --out /tmp/ext.parquet
Needs TMDB_API_KEY in `.env`. Reads TMDB, Wikidata and English Wikipedia; writes one local file.

Input: `data/catalog.parquet` (the Supabase export). Output: `data/catalog_extended.parquet`:
- every exported row unchanged, `source = "emberfall"` (the baseline keeps its `doc` and vectors);
- new movie and TV rows released 1950-1999, `source = "extension"`, the most-voted titles of each
  year on TMDB, with no `doc` and no vectors (the baseline never had them);
- for every row `doc_wiki`, the document in the format of the baseline's populate-local builder,
  "Title. Genre, Genre. keyword, keyword. <English Wikipedia intro, else TMDB overview>", and
  `wiki_title`, the article used (null when TMDB -> Wikidata -> enwiki found none).
Every HTTP response is cached on disk, so an interrupted run resumes where it stopped.
"""

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from concierge import config
from concierge.tmdb import TMDB, MediaType
from concierge.wikimedia import Wikimedia

MAX_KEYWORDS = 15  # as in populate-local
VOTE_FLOOR = {"movie": 10, "tv": 5}  # populate-local's per-year passes
DATE_FIELD = {"movie": "primary_release_date", "tv": "first_air_date"}
MEDIA_TYPES: tuple[MediaType, ...] = ("movie", "tv")


def build_doc(
    title: str, genres: list[str], keywords: list[str], intro: str | None, overview: str
) -> str:
    """populate-local's document: non-empty parts of
    [title, genres, keywords, intro or overview] joined by '. '."""
    parts = [title, ", ".join(genres), ", ".join(keywords), intro or overview]
    return ". ".join(p for p in parts if p)


def keyword_names(details: dict[str, Any], media: MediaType) -> list[str]:
    """Keyword names from a details response with `keywords` appended (the key differs by type)."""
    block = details.get("keywords") or {}
    items = block.get("keywords" if media == "movie" else "results") or []
    return [k["name"] for k in items][:MAX_KEYWORDS]


def discover_year(tmdb: TMDB, media: MediaType, year: int, limit: int) -> list[dict]:
    """The `limit` most-voted titles first released in `year`."""
    field = DATE_FIELD[media]
    filters = {
        "sort_by": "vote_count.desc",
        "vote_count.gte": VOTE_FLOOR[media],
        f"{field}.gte": f"{year}-01-01",
        f"{field}.lte": f"{year}-12-31",
    }
    items: list[dict] = []
    for page in range(1, math.ceil(limit / 20) + 1):
        results = tmdb.discover(media, filters, page)
        items += results
        if len(results) < 20:
            break
    return items[:limit]


def extension_row(item: dict[str, Any], media: MediaType) -> dict[str, Any]:
    """A discover result in the catalog's columns (no doc, no vectors)."""
    return {
        "item_id": item["id"],
        "item_type": media,
        "title": item.get("title") or item.get("name"),
        "overview": item.get("overview") or "",
        "poster_path": item.get("poster_path"),
        "release_date": item.get("release_date") or item.get("first_air_date") or None,
        "vote_average": item.get("vote_average"),
        "popularity": item.get("popularity"),
        "genre_ids": item.get("genre_ids") or [],
    }


def extend(
    catalog: pa.Table,
    tmdb: TMDB,
    wiki: Wikimedia,
    years: range,
    per_year: dict[MediaType, int],
    workers: int,
) -> pa.Table:
    """Catalog rows + pre-2000 rows, each with `source`, `doc_wiki` and `wiki_title`."""
    new_rows = new_titles(tmdb, catalog, years, per_year)
    print(f"extension: {len(new_rows)} new titles from {years.start}-{years.stop - 1}")
    extension = pa.Table.from_pylist(new_rows, schema=catalog.schema)
    table = pa.concat_tables(
        [
            catalog.append_column("source", pa.array(["emberfall"] * catalog.num_rows)),
            extension.append_column("source", pa.array(["extension"] * extension.num_rows)),
        ]
    )
    docs, wiki_titles = documents(table, tmdb, wiki, workers)
    return table.append_column("doc_wiki", pa.array(docs)).append_column(
        "wiki_title", pa.array(wiki_titles, type=pa.string())
    )


def new_titles(
    tmdb: TMDB, catalog: pa.Table, years: range, per_year: dict[MediaType, int]
) -> list[dict[str, Any]]:
    """The most-voted titles of each year, minus anything the catalog already has."""
    known = set(zip(catalog["item_type"].to_pylist(), catalog["item_id"].to_pylist(), strict=True))
    rows: list[dict[str, Any]] = []
    for media in MEDIA_TYPES:
        for year in years:
            for item in discover_year(tmdb, media, year, per_year[media]):
                row = extension_row(item, media)
                if row["title"] and (media, row["item_id"]) not in known:
                    known.add((media, row["item_id"]))
                    rows.append(row)
    return rows


def documents(
    table: pa.Table, tmdb: TMDB, wiki: Wikimedia, workers: int
) -> tuple[list[str], list[str | None]]:
    """`doc_wiki` and the Wikipedia article it used (or None), per row of `table`."""
    # Keywords and Wikidata ids: one TMDB details request per title, in parallel.
    keys = list(zip(table["item_type"].to_pylist(), table["item_id"].to_pylist(), strict=True))
    print(f"TMDB details for {len(keys)} titles (cached ones cost nothing)")
    with ThreadPoolExecutor(workers) as pool:
        details = list(
            pool.map(lambda k: tmdb.details(k[0], k[1], ("keywords", "external_ids")), keys)
        )
    gone = sum(d is None for d in details)
    qids = [(d or {}).get("external_ids", {}).get("wikidata_id") or "" for d in details]

    # Wikipedia intros: Wikidata id -> enwiki title -> lead section, batched.
    enwiki = wiki.enwiki_titles(qids)
    titles = [enwiki.get(q) for q in qids]
    intros = wiki.intros(t for t in titles if t)
    print(f"TMDB 404 (title removed): {gone}; with an enwiki article: {sum(map(bool, titles))}")

    genre_names = {media: tmdb.genres(media) for media in MEDIA_TYPES}
    docs: list[str] = []
    wiki_titles: list[str | None] = []
    columns = ["item_type", "title", "overview", "genre_ids"]
    for row, detail, title in zip(table.select(columns).to_pylist(), details, titles, strict=True):
        intro = intros.get(title) if title else None
        names = genre_names[row["item_type"]]
        docs.append(
            build_doc(
                row["title"],
                [names[g] for g in row["genre_ids"] or [] if g in names],
                keyword_names(detail or {}, row["item_type"]),
                intro,
                row["overview"] or "",
            )
        )
        wiki_titles.append(title if intro else None)
    return docs, wiki_titles


def summarize(table: pa.Table) -> None:
    """Rows and Wikipedia coverage per source and media type; document length."""
    df = table.select(["source", "item_type", "doc_wiki", "wiki_title"]).to_pandas()
    df["has_wiki"] = df["wiki_title"].notna()
    print(df.groupby(["source", "item_type"])["has_wiki"].agg(rows="size", wiki_share="mean"))
    chars = df["doc_wiki"].str.len().quantile([0.5, 0.95, 1.0]).round().tolist()
    print(f"doc_wiki chars p50/p95/max: {chars}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=config.CATALOG_PATH)
    parser.add_argument("--out", type=Path, default=config.CATALOG_EXTENDED_PATH)
    parser.add_argument("--from-year", type=int, default=1950)
    parser.add_argument("--to-year", type=int, default=1999)
    parser.add_argument("--movies-per-year", type=int, default=100)
    parser.add_argument("--tv-per-year", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--sample", type=int, default=0, help="smoke run: N catalog rows, N titles of one year"
    )
    args = parser.parse_args()

    catalog = pq.read_table(args.catalog)
    years = range(args.from_year, args.to_year + 1)
    per_year: dict[MediaType, int] = {"movie": args.movies_per_year, "tv": args.tv_per_year}
    if args.sample:
        catalog = catalog.slice(0, args.sample)
        years = range(args.from_year, args.from_year + 1)
        per_year = {"movie": args.sample, "tv": args.sample}

    table = extend(catalog, TMDB(), Wikimedia(), years, per_year, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)
    print(f"wrote {args.out}")
    summarize(table)


if __name__ == "__main__":
    main()
