"""Catalog export against a fake PostgREST: keyset paging, retries, vector parsing, empty export."""

import importlib.util
import json
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from concierge import http
from concierge.config import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "export_catalog", REPO_ROOT / "data/export_catalog.py"
)
export_catalog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_catalog)


def row(item_id: int, item_type: str, openai: bool = True) -> dict:
    return {
        "item_id": item_id,
        "item_type": item_type,
        "title": f"title {item_id}",
        "overview": "overview",
        "doc": None if item_id == 3 else "doc text",
        "poster_path": None,
        "release_date": "1996-05-10",
        "vote_average": 6.5,
        "popularity": 12.0,
        "genre_ids": [28, 12],
        "embedding": json.dumps([0.5] * 384),
        "embedding_openai": json.dumps([0.25] * 1536) if openai else None,
        "updated_at": "2026-01-01T00:00:00+00:00",
    }


TABLE = {"movie": [row(i, "movie") for i in (1, 2, 3, 7, 9)], "tv": [row(4, "tv", openai=False)]}


def fake_postgrest(server_row_cap: int, fail_first: int = 0):
    """PostgREST stand-in: honours item_type/item_id/limit, caps rows like Supabase max-rows,
    and answers the first `fail_first` requests with 503."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] <= fail_first:
            return httpx.Response(503)
        q = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        item_type = q["item_type"].removeprefix("eq.")
        after = int(q["item_id"].removeprefix("gt."))
        rows = [r for r in TABLE[item_type] if r["item_id"] > after]
        return httpx.Response(200, json=rows[: min(int(q["limit"]), server_row_cap)])

    return httpx.Client(base_url="https://x.supabase.co", transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda _: None)


def test_exports_every_row_even_when_server_caps_pages_below_page_size():
    table = export_catalog.export(fake_postgrest(server_row_cap=2), page_size=500)
    assert sorted(table["item_id"].to_pylist()) == [1, 2, 3, 4, 7, 9]


def test_vectors_are_float32_with_expected_dims_and_nulls_kept():
    table = export_catalog.export(fake_postgrest(server_row_cap=10), page_size=3)
    assert table.schema.field("embedding").type == pa.list_(pa.float32())
    rows = {r["item_id"]: r for r in table.to_pylist()}
    assert len(rows[1]["embedding"]) == 384
    assert len(rows[1]["embedding_openai"]) == 1536
    assert rows[4]["embedding_openai"] is None
    assert rows[3]["doc"] is None


def test_retries_transient_503():
    table = export_catalog.export(fake_postgrest(server_row_cap=10, fail_first=2), page_size=500)
    assert table.num_rows == 6


def test_gives_up_after_repeated_503():
    with pytest.raises(http.HTTPError):
        export_catalog.export(fake_postgrest(server_row_cap=10, fail_first=99), page_size=500)


def test_wrong_vector_dimension_raises():
    with pytest.raises(ValueError, match="expected"):
        export_catalog.parse_vector(json.dumps([0.1] * 383), "embedding")


def test_parse_vector_returns_float32_and_passes_null_through():
    vector = export_catalog.parse_vector(json.dumps([0.5] * 384), "embedding")
    assert vector.dtype == np.float32
    assert export_catalog.parse_vector(None, "embedding_openai") is None


def test_empty_export_is_an_error(monkeypatch):
    monkeypatch.setitem(TABLE, "movie", [])
    monkeypatch.setitem(TABLE, "tv", [])
    with pytest.raises(SystemExit):
        export_catalog.export(fake_postgrest(server_row_cap=10), page_size=500)


def test_writes_readable_parquet(tmp_path: Path):
    table = export_catalog.export(fake_postgrest(server_row_cap=10), page_size=500)
    pq.write_table(table, tmp_path / "c.parquet")
    assert pq.read_table(tmp_path / "c.parquet").num_rows == 6
