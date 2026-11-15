"""Export the Emberfall catalog (Supabase `public.title_embeddings`) to `data/catalog.parquet`.

Run once by the owner, locally:  uv run python data/export_catalog.py
Needs SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in `.env`. Read-only: it only issues GETs.

Input: the PostgREST API of the Supabase project. Output: one parquet row per title with the
baseline's columns, `embedding` (gte-small, 384 floats) and `embedding_openai`
(text-embedding-3-small, 1536 floats, null where never computed) as float32 lists.
Pages are fetched per item_type in item_id order (keyset paging), so a server-side row cap
smaller than the page size cannot silently end the export early.
"""

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from concierge import config
from concierge.http import get_json

# `fts` (a tsvector derived from `doc`) is left out: it is rebuilt from `doc` when needed.
COLUMNS = [
    "item_id",
    "item_type",
    "title",
    "overview",
    "doc",
    "poster_path",
    "release_date",
    "vote_average",
    "popularity",
    "genre_ids",
    "embedding",
    "embedding_openai",
    "updated_at",
]
VECTOR_DIMS = {"embedding": 384, "embedding_openai": 1536}
ITEM_TYPES = ("movie", "tv")

SCHEMA = pa.schema(
    [
        ("item_id", pa.int64()),
        ("item_type", pa.string()),
        ("title", pa.string()),
        ("overview", pa.string()),
        ("doc", pa.string()),
        ("poster_path", pa.string()),
        ("release_date", pa.string()),
        ("vote_average", pa.float32()),
        ("popularity", pa.float32()),
        ("genre_ids", pa.list_(pa.int32())),
        ("embedding", pa.list_(pa.float32())),
        ("embedding_openai", pa.list_(pa.float32())),
        ("updated_at", pa.string()),
    ]
)


def parse_vector(raw: str | None, column: str) -> np.ndarray | None:
    """pgvector/halfvec text form '[0.1,0.2,...]' -> float32 array; checks the dimension."""
    if raw is None:
        return None
    vector = np.asarray(json.loads(raw), dtype=np.float32)
    if vector.shape != (VECTOR_DIMS[column],):
        raise ValueError(f"{column} has shape {vector.shape}, expected ({VECTOR_DIMS[column]},)")
    return vector


def parse_row(row: dict[str, Any]) -> dict[str, Any]:
    """One PostgREST row -> one parquet row (vectors parsed, everything else as returned)."""
    return {**row, **{col: parse_vector(row[col], col) for col in VECTOR_DIMS}}


def fetch_pages(client: httpx.Client, item_type: str, page_size: int) -> Iterator[list[dict]]:
    """Yield pages of raw rows of one item_type, ordered by item_id, until a page is empty."""
    last_id = -1
    while True:
        params = {
            "select": ",".join(COLUMNS),
            "item_type": f"eq.{item_type}",
            "item_id": f"gt.{last_id}",
            "order": "item_id.asc",
            "limit": str(page_size),
        }
        rows = get_json(client, "/rest/v1/title_embeddings", params)
        if not rows:
            return
        yield rows
        last_id = rows[-1]["item_id"]


def summarize(table: pa.Table) -> None:
    """Print the counts that answer the open data questions (WORKING_STATE.md)."""
    df = table.select(["item_type", "doc", "release_date", "embedding", "embedding_openai"])
    df = df.to_pandas()
    years = df["release_date"].str[:4].astype("Int64")
    doc_chars = df["doc"].str.len()
    print(f"rows: {len(df)}")
    print(df["item_type"].value_counts().to_string())
    print(f"doc is null (embedded from title+overview only): {df['doc'].isna().sum()}")
    print(f"released before 2000: {(years < 2000).sum()}   no release year: {years.isna().sum()}")
    print(f"embedding is null: {df['embedding'].isna().sum()}")
    print(f"embedding_openai is null: {df['embedding_openai'].isna().sum()}")
    print(f"doc chars p50/p95/max: {doc_chars.quantile([0.5, 0.95, 1.0]).round().tolist()}")


def export(client: httpx.Client, page_size: int) -> pa.Table:
    """All catalog rows as a typed table. Raises if the API returned no rows at all."""
    rows: list[dict[str, Any]] = []
    for item_type in ITEM_TYPES:
        for page in fetch_pages(client, item_type, page_size):
            rows.extend(parse_row(r) for r in page)
            print(f"{item_type}: {len(rows)} rows so far", end="\r")
    print()
    if not rows:
        raise SystemExit("The export returned 0 rows: wrong project URL or key, nothing written.")
    return pa.Table.from_pylist(rows, schema=SCHEMA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=config.CATALOG_PATH)
    parser.add_argument("--page-size", type=int, default=500)
    args = parser.parse_args()

    key = config.get_secret("SUPABASE_SERVICE_ROLE_KEY")
    client = httpx.Client(
        base_url=config.get_secret("SUPABASE_URL"),
        headers={"apikey": key, "Authorization": f"Bearer {key}"},
        timeout=120,
        transport=httpx.HTTPTransport(retries=3),
    )
    with client:
        table = export(client, args.page_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.out)
    print(f"wrote {args.out}")
    summarize(table)


if __name__ == "__main__":
    main()
