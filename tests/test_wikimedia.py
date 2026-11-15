"""Wikimedia id lookups against a fake Wikipedia / Wikidata API."""

from urllib.parse import parse_qs

import httpx
import pytest

from concierge.cache import DiskCache
from concierge.wikimedia import Wikimedia, clean_intro


def test_clean_intro_collapses_whitespace_and_caps_length():
    assert clean_intro(" a\n\n b  c ") == "a b c"
    assert len(clean_intro("x" * 3000)) == 2500
    assert clean_intro("   ") is None


def statement(value: str, rank: str = "normal") -> dict:
    return {"rank": rank, "mainsnak": {"datavalue": {"value": value}}}


CLAIMS = {
    "Q1": {"P4947": [statement("603")]},  # a film
    "Q2": {"P4983": [statement("1396")]},  # a TV series
    "Q3": {"P4947": [statement("1", "deprecated"), statement("11"), statement("12", "preferred")]},
    "Q4": {"P31": [statement("Q5")]},  # no TMDB link
    "Q10": {  # Twister: director, duration in minutes, two publication dates
        "P57": [{"rank": "normal", "mainsnak": {"datavalue": {"value": {"id": "Q20"}}}}],
        "P2047": [{"rank": "normal", "mainsnak": {"datavalue": {"value": {
            "amount": "+113", "unit": "http://www.wikidata.org/entity/Q7727"}}}}],
        "P577": [{"rank": "normal", "mainsnak": {"datavalue": {"value": {"time": t}}}}
                 for t in ("+1996-05-10T00:00:00Z", "+1995-12-01T00:00:00Z")],
    },
    "Q11": {"P2047": [{"rank": "normal", "mainsnak": {"datavalue": {"value": {
        "amount": "+1.5", "unit": "http://www.wikidata.org/entity/Q25235"}}}}]},
}  # fmt: skip
LABELS = {"Q20": "Jan de Bont"}
PAGES = {"10": "Q1", "20": "Q2", "30": None}  # 30: page without a Wikidata item; 40: deleted


def fake(request: httpx.Request) -> httpx.Response:
    q = {k: v[0] for k, v in parse_qs(request.url.query.decode()).items()}
    if request.url.host == "www.wikidata.org":
        ids = q["ids"].split("|")
        if q["props"] == "labels":
            labels = {i: {"labels": {"en": {"value": LABELS[i]}}} for i in ids if i in LABELS}
            return httpx.Response(200, json={"entities": labels})
        return httpx.Response(
            200, json={"entities": {i: {"claims": CLAIMS.get(i, {})} for i in ids}}
        )
    pages = []
    for pid in q["pageids"].split("|"):
        if pid not in PAGES:
            pages.append({"pageid": int(pid), "missing": True})
        elif PAGES[pid]:
            pages.append({"pageid": int(pid), "pageprops": {"wikibase_item": PAGES[pid]}})
        else:
            pages.append({"pageid": int(pid)})
    return httpx.Response(200, json={"query": {"pages": pages}})


@pytest.fixture
def wiki(tmp_path) -> Wikimedia:
    return Wikimedia(httpx.Client(transport=httpx.MockTransport(fake)), DiskCache(tmp_path))


def test_page_ids_map_to_wikidata_ids(wiki):
    assert wiki.wikidata_ids(["10", "20", "30", "40", "not-a-number"]) == {
        "10": "Q1",
        "20": "Q2",
        "30": None,
        "40": None,
    }


def test_tmdb_ids_by_media_type_preferred_rank_first_deprecated_ignored(wiki):
    assert wiki.tmdb_ids(["Q1", "Q2", "Q3", "Q4"]) == {
        "Q1": {"movie": 603},
        "Q2": {"tv": 1396},
        "Q3": {"movie": 12},
        "Q4": None,
    }


def test_film_facts_resolve_director_names_minutes_and_earliest_year(wiki):
    facts = wiki.film_facts(["Q10", "Q11"])
    assert facts["Q10"] == {"directors": ["Jan de Bont"], "duration_minutes": 113, "year": 1995}
    assert facts["Q11"] == {"directors": [], "duration_minutes": 90, "year": None}
