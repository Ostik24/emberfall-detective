"""Catalog extension against fake TMDB, Wikidata and Wikipedia APIs."""

import importlib.util
from urllib.parse import parse_qs

import httpx
import pyarrow as pa
import pytest

from concierge import http
from concierge.cache import DiskCache
from concierge.config import REPO_ROOT
from concierge.tmdb import BASE_URL, TMDB
from concierge.wikimedia import USER_AGENT, Wikimedia

spec = importlib.util.spec_from_file_location("extend", REPO_ROOT / "data/extend_catalog.py")
ext = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ext)

KEY = "secret-tmdb-key"
LONG_INTRO = "A storm-chasing team pursues tornadoes  across\nOklahoma."


# ---- pure parts -----------------------------------------------------------------------------


def test_build_doc_matches_populate_local_format():
    doc = ext.build_doc("Twister", ["Action"], ["tornado", "storm"], "Intro.", "Overview.")
    assert doc == "Twister. Action. tornado, storm. Intro."


def test_build_doc_falls_back_to_overview_and_skips_empty_parts():
    assert ext.build_doc("Twister", [], [], None, "Overview.") == "Twister. Overview."
    assert ext.build_doc("Twister", [], [], None, "") == "Twister"


def test_keyword_names_by_media_type_capped_at_15():
    movie = {"keywords": {"keywords": [{"name": f"k{i}"} for i in range(20)]}}
    tv = {"keywords": {"results": [{"name": "detective"}]}}
    assert ext.keyword_names(movie, "movie") == [f"k{i}" for i in range(15)]
    assert ext.keyword_names(tv, "tv") == ["detective"]
    assert ext.keyword_names({}, "movie") == []


# ---- fake world -----------------------------------------------------------------------------

MOVIES_1950 = [{"id": 500 + i, "title": f"Old Film {i}", "genre_ids": [28]} for i in range(25)]
MOVIES_1950[3] = {"id": 503, "title": None}  # untitled: skipped, as in populate-local
SHOWS_1950 = [
    {"id": 900, "name": "Old Show", "first_air_date": "1950-01-01", "genre_ids": [18]},
    {"id": 7, "name": "Show", "genre_ids": [18]},  # already in the catalog: not added twice
]


def tmdb_response(path: str, q: dict[str, str]) -> httpx.Response:
    assert q.pop("api_key") == KEY
    if path.endswith("/genre/movie/list"):
        return httpx.Response(200, json={"genres": [{"id": 28, "name": "Action"}]})
    if path.endswith("/genre/tv/list"):
        return httpx.Response(200, json={"genres": [{"id": 18, "name": "Drama"}]})
    if "/discover/" in path:
        items = MOVIES_1950 if path.endswith("movie") else SHOWS_1950
        page = int(q["page"])
        return httpx.Response(200, json={"results": items[(page - 1) * 20 : page * 20]})
    media, item_id = path.split("/")[-2], int(path.split("/")[-1])
    if item_id == 404:
        return httpx.Response(404, json={})
    keywords = [{"name": "tornado"}, {"name": "storm"}]
    block = {"keywords": keywords} if media == "movie" else {"results": keywords}
    qid = "" if item_id == 501 else f"Q{item_id}"  # 501 has no Wikidata link
    return httpx.Response(200, json={"keywords": block, "external_ids": {"wikidata_id": qid}})


def wikidata_response(q: dict[str, str]) -> httpx.Response:
    entities = {}
    for qid in q["ids"].split("|"):
        n = int(qid[1:])
        sitelinks = {} if n == 502 else {"enwiki": {"title": f"film {n}"}}  # 502: no enwiki
        entities[qid] = {"sitelinks": sitelinks}
    return httpx.Response(200, json={"entities": entities})


def wikipedia_response(q: dict[str, str]) -> httpx.Response:
    """Normalises 'film N' -> 'Film N', redirects Film 1 -> Twister, pages 3 extracts a call."""
    requested = q["titles"].split("|")
    normalized = [{"from": t, "to": t.capitalize()} for t in requested]
    canonical = [t.capitalize() for t in requested]
    redirects = [{"from": "Film 1", "to": "Twister"}] if "Film 1" in canonical else []
    finals = ["Twister" if t == "Film 1" else t for t in canonical]
    offset = int(q.get("excontinue", 0))
    pages = [
        {"title": t, **({"extract": LONG_INTRO} if offset <= i < offset + 3 else {})}
        for i, t in enumerate(finals)
    ]
    body = {"query": {"normalized": normalized, "redirects": redirects, "pages": pages}}
    if offset + 3 < len(finals):
        body["continue"] = {"excontinue": offset + 3, "continue": "||"}
    return httpx.Response(200, json=body)


class FakeWorld:
    """One transport for all three APIs; counts the requests it served."""

    def __init__(self) -> None:
        self.requests = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests += 1
        q = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
        host = request.url.host
        if host == "api.themoviedb.org":
            return tmdb_response(request.url.path, q)
        if host == "www.wikidata.org":
            return wikidata_response(q)
        assert request.headers["User-Agent"] == USER_AGENT
        return wikipedia_response(q)


CATALOG_SCHEMA = pa.schema(
    [
        ("item_id", pa.int64()),
        ("item_type", pa.string()),
        ("title", pa.string()),
        ("overview", pa.string()),
        ("doc", pa.string()),
        ("release_date", pa.string()),
        ("genre_ids", pa.list_(pa.int32())),
    ]
)


def catalog() -> pa.Table:
    rows = [
        {"item_id": 1, "item_type": "movie", "title": "Twister", "overview": "Storm chasers.",
         "doc": "baseline doc", "release_date": "2000-01-01", "genre_ids": [28]},
        {"item_id": 404, "item_type": "movie", "title": "Gone", "overview": "Removed title.",
         "doc": "baseline doc", "release_date": "2001-01-01", "genre_ids": []},
        {"item_id": 7, "item_type": "tv", "title": "Show", "overview": "A series.",
         "doc": None, "release_date": "2005-01-01", "genre_ids": [18]},
    ]  # fmt: skip
    return pa.Table.from_pylist(rows, schema=CATALOG_SCHEMA)


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(http.time, "sleep", lambda _: None)
    monkeypatch.setattr("concierge.config.get_secret", lambda name: KEY)
    world = FakeWorld()

    def run_once() -> pa.Table:
        transport = httpx.MockTransport(world)
        tmdb = TMDB(httpx.Client(base_url=BASE_URL, transport=transport), DiskCache(tmp_path / "t"))
        client = httpx.Client(headers={"User-Agent": USER_AGENT}, transport=transport)
        wiki = Wikimedia(client, DiskCache(tmp_path / "w"))
        per_year = {"movie": 22, "tv": 5}
        return ext.extend(catalog(), tmdb, wiki, range(1950, 1951), per_year, workers=2)

    return run_once, world


def by_key(table: pa.Table) -> dict[tuple[str, int], dict]:
    return {(r["item_type"], r["item_id"]): r for r in table.to_pylist()}


def test_existing_rows_keep_baseline_columns_and_gain_wiki_docs(run):
    run_once, _ = run
    rows = by_key(run_once())
    twister = rows[("movie", 1)]
    assert twister["source"] == "emberfall"
    assert twister["doc"] == "baseline doc"
    # 'film 1' -> normalised 'Film 1' -> redirect 'Twister'
    intro = "A storm-chasing team pursues tornadoes across Oklahoma."
    assert twister["doc_wiki"] == f"Twister. Action. tornado, storm. {intro}"
    assert twister["wiki_title"] == "film 1"
    show = rows[("tv", 7)]
    assert show["doc_wiki"].startswith("Show. Drama. tornado, storm.")


def test_removed_title_keeps_overview_doc_without_keywords(run):
    run_once, _ = run
    gone = by_key(run_once())[("movie", 404)]
    assert gone["doc_wiki"] == "Gone. Removed title."
    assert gone["wiki_title"] is None


def test_extension_adds_most_voted_titles_skipping_untitled(run):
    run_once, _ = run
    table = run_once()
    extension = [r for r in table.to_pylist() if r["source"] == "extension"]
    movies = sorted(r["item_id"] for r in extension if r["item_type"] == "movie")
    assert movies == [500, 501, 502] + list(range(504, 522))  # 22 asked, 503 untitled
    assert [r["title"] for r in extension if r["item_type"] == "tv"] == ["Old Show"]
    assert all(r["doc"] is None for r in extension)


def test_titles_without_wikidata_or_enwiki_fall_back_to_overview(run):
    run_once, _ = run
    rows = by_key(run_once())
    assert rows[("movie", 501)]["wiki_title"] is None  # no Wikidata id
    assert rows[("movie", 502)]["wiki_title"] is None  # no enwiki sitelink


def test_paged_wikipedia_extracts_are_all_collected(run):
    # The fake returns 3 extracts per call, so each batch of 20 needs several pages.
    run_once, _ = run
    without = {r["item_id"] for r in run_once().to_pylist() if r["wiki_title"] is None}
    assert without == {404, 501, 502}  # exactly the titles that have no article


def test_second_run_is_served_from_cache_except_the_uncached_404(run):
    run_once, world = run
    first = run_once()
    served = world.requests
    second = run_once()
    assert world.requests == served + 1  # only the removed title (404) is asked again
    assert first.equals(second)


def test_tmdb_key_never_reaches_errors_or_cache(tmp_path, monkeypatch):
    monkeypatch.setattr("concierge.config.get_secret", lambda name: KEY)
    transport = httpx.MockTransport(lambda r: httpx.Response(401, json={}))
    tmdb = TMDB(httpx.Client(base_url=BASE_URL, transport=transport), DiskCache(tmp_path))
    with pytest.raises(http.HTTPError) as error:
        tmdb.details("movie", 1)
    assert KEY not in str(error.value)

    ok = httpx.MockTransport(lambda r: httpx.Response(200, json={"genres": []}))
    TMDB(httpx.Client(base_url=BASE_URL, transport=ok), DiskCache(tmp_path)).genres("movie")
    cached = "".join(p.read_text() for p in tmp_path.rglob("*.json"))
    assert cached and KEY not in cached
