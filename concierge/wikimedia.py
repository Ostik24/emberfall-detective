"""Cached, batched Wikidata / English Wikipedia lookups (data preparation only, not the agent).

Purpose: link the three id spaces the project meets - Wikipedia page ids (TREC ToT answers),
Wikidata ids, TMDB ids (the catalog) - and fetch Wikipedia intros for catalog documents.
Input: lists of ids or titles. Output: dicts keyed by the input; None where nothing was found.
Every answer, including "nothing found", is cached on disk per key, so reruns cost no requests.
"""

import re
from collections.abc import Callable, Iterable
from typing import Any

import httpx

from concierge import config
from concierge.cache import DiskCache
from concierge.http import get_json

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
# Wikimedia asks API clients for a contact in the User-Agent; the owner chose this one.
USER_AGENT = "EmberfallDetective/0.1 (https://github.com/ostik24; university NLP project)"
WIKIDATA_BATCH = 50  # wbgetentities limit
PAGEIDS_BATCH = 50  # action=query limit for anonymous clients
INTRO_BATCH = 20  # extracts limit when exintro is set
INTRO_MAX_CHARS = 2500  # as in the baseline's populate-local
QID = re.compile(r"^Q\d+$")
TMDB_PROPERTIES = {"P4947": "movie", "P4983": "tv"}  # Wikidata "TMDB movie / TV series ID"
DIRECTOR, DURATION, PUBLICATION_DATE = "P57", "P2047", "P577"
MINUTE, HOUR = "http://www.wikidata.org/entity/Q7727", "http://www.wikidata.org/entity/Q25235"


def clean_intro(extract: str) -> str | None:
    """Collapse whitespace and cap the length, as populate-local does; None for an empty text."""
    return " ".join(extract.split())[:INTRO_MAX_CHARS] or None


class Wikimedia:
    """Wikidata and English Wikipedia API client with a per-key disk cache."""

    def __init__(self, client: httpx.Client | None = None, cache: DiskCache | None = None) -> None:
        self.client = client or httpx.Client(
            headers={"User-Agent": USER_AGENT},
            timeout=60,
            transport=httpx.HTTPTransport(retries=3),
        )
        self.cache = cache or DiskCache(config.WIKIMEDIA_CACHE_DIR)

    def enwiki_titles(self, qids: Iterable[str]) -> dict[str, str | None]:
        """English Wikipedia article title per Wikidata id."""
        valid = sorted({q for q in qids if QID.match(q)})
        return self._batched("enwiki", valid, WIKIDATA_BATCH, self._fetch_enwiki)

    def intros(self, titles: Iterable[str]) -> dict[str, str | None]:
        """Lead section per article title (redirects followed), cleaned by `clean_intro`."""
        return self._batched("intro", sorted(set(titles)), INTRO_BATCH, self._fetch_intros)

    def wikidata_ids(self, page_ids: Iterable[str]) -> dict[str, str | None]:
        """Wikidata id per English Wikipedia page id (None for deleted or unlinked pages)."""
        valid = sorted({p for p in page_ids if p.isdigit()}, key=int)
        return self._batched("pageid", valid, PAGEIDS_BATCH, self._fetch_wikidata_ids)

    def tmdb_ids(self, qids: Iterable[str]) -> dict[str, dict[str, int] | None]:
        """TMDB ids per Wikidata id, e.g. {"movie": 603}; None if the item has neither."""
        valid = sorted({q for q in qids if QID.match(q)})
        return self._batched("tmdb", valid, WIKIDATA_BATCH, self._fetch_tmdb_ids)

    def film_facts(self, qids: Iterable[str]) -> dict[str, dict[str, Any] | None]:
        """Directors (English labels), duration in minutes and earliest publication year per
        Wikidata id, from statements independent of TMDB (gold for C8)."""
        valid = sorted({q for q in qids if QID.match(q)})
        return self._batched("facts", valid, WIKIDATA_BATCH, self._fetch_facts)

    def labels(self, qids: Iterable[str]) -> dict[str, str | None]:
        """English label per Wikidata id."""
        valid = sorted({q for q in qids if QID.match(q)})
        return self._batched("label", valid, WIKIDATA_BATCH, self._fetch_labels)

    def _batched(
        self, kind: str, keys: list[str], size: int, fetch: Callable[[list[str]], dict[str, Any]]
    ) -> dict[str, Any]:
        found: dict[str, Any] = {}
        missing: list[str] = []
        for key in keys:
            hit = self.cache.get({kind: key})
            if hit is None:
                missing.append(key)
            else:
                found[key] = hit["value"]
        for start in range(0, len(missing), size):
            batch = missing[start : start + size]
            fetched = fetch(batch)
            for key in batch:
                self.cache.put({kind: key}, {"value": fetched.get(key)})
                found[key] = fetched.get(key)
        return found

    def _entities(self, qids: list[str], **params: str) -> dict[str, Any]:
        query = {"action": "wbgetentities", "ids": "|".join(qids), "format": "json", **params}
        data = get_json(self.client, WIKIDATA_API, query)
        if "error" in data:
            raise RuntimeError(f"Wikidata rejected batch {qids[0]}..: {data['error'].get('code')}")
        return data.get("entities", {})

    def _fetch_enwiki(self, qids: list[str]) -> dict[str, str | None]:
        entities = self._entities(qids, props="sitelinks", sitefilter="enwiki")
        return {
            q: entities.get(q, {}).get("sitelinks", {}).get("enwiki", {}).get("title") for q in qids
        }

    def _fetch_tmdb_ids(self, qids: list[str]) -> dict[str, dict[str, int] | None]:
        entities = self._entities(qids, props="claims")
        out: dict[str, dict[str, int] | None] = {}
        for q in qids:
            claims = entities.get(q, {}).get("claims", {})
            ids = {
                media: int(value)
                for prop, media in TMDB_PROPERTIES.items()
                if (value := _first_value(claims.get(prop, []))) and value.isdigit()
            }
            out[q] = ids or None
        return out

    def _fetch_labels(self, qids: list[str]) -> dict[str, str | None]:
        entities = self._entities(qids, props="labels", languages="en")
        return {q: entities.get(q, {}).get("labels", {}).get("en", {}).get("value") for q in qids}

    def _fetch_facts(self, qids: list[str]) -> dict[str, dict[str, Any] | None]:
        entities = self._entities(qids, props="claims")
        raw = {q: entities.get(q, {}).get("claims", {}) for q in qids}
        director_ids = {
            v["id"]
            for claims in raw.values()
            for st in _usable(claims.get(DIRECTOR, []))
            if isinstance(v := _value(st), dict) and "id" in v
        }
        names = self.labels(director_ids)
        out: dict[str, dict[str, Any] | None] = {}
        for q, claims in raw.items():
            directors = [
                names.get(v["id"])
                for st in _usable(claims.get(DIRECTOR, []))
                if isinstance(v := _value(st), dict) and v.get("id") in names
            ]
            years = [
                int(v["time"][1:5])
                for st in _usable(claims.get(PUBLICATION_DATE, []))
                if isinstance(v := _value(st), dict) and str(v.get("time", ""))[1:5].isdigit()
            ]
            out[q] = {
                "directors": [d for d in directors if d],
                "duration_minutes": _minutes(claims.get(DURATION, [])),
                "year": min(years) if years else None,
            }
        return out

    def _fetch_wikidata_ids(self, page_ids: list[str]) -> dict[str, str | None]:
        params = {
            "action": "query",
            "prop": "pageprops",
            "ppprop": "wikibase_item",
            "pageids": "|".join(page_ids),
            "format": "json",
            "formatversion": 2,
        }
        pages = get_json(self.client, WIKIPEDIA_API, params).get("query", {}).get("pages", [])
        return {
            str(p["pageid"]): p.get("pageprops", {}).get("wikibase_item")
            for p in pages
            if "pageid" in p
        }

    def _fetch_intros(self, titles: list[str]) -> dict[str, str | None]:
        params: dict[str, Any] = {
            "action": "query",
            "prop": "extracts",
            "exintro": 1,
            "explaintext": 1,
            "redirects": 1,
            "exlimit": "max",
            "titles": "|".join(titles),
            "format": "json",
            "formatversion": 2,
        }
        aliases: dict[str, str] = {}
        extracts: dict[str, str] = {}
        while True:
            data = get_json(self.client, WIKIPEDIA_API, params)
            query = data.get("query", {})
            for hop in query.get("normalized", []) + query.get("redirects", []):
                aliases[hop["from"]] = hop["to"]
            for page in query.get("pages", []):
                if page.get("extract"):
                    extracts[page["title"]] = page["extract"]
            if "continue" not in data:  # extracts are paged when a batch is long
                break
            params = {**params, **data["continue"]}
        return {t: clean_intro(extracts.get(_follow(t, aliases), "")) for t in titles}


def _usable(statements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-deprecated statements, preferred rank first."""
    usable = [s for s in statements if s.get("rank") != "deprecated"]
    return sorted(usable, key=lambda s: s.get("rank") != "preferred")


def _value(statement: dict[str, Any]) -> Any:
    return statement.get("mainsnak", {}).get("datavalue", {}).get("value")


def _minutes(statements: list[dict[str, Any]]) -> int | None:
    """First duration given in minutes or hours, as whole minutes."""
    for st in _usable(statements):
        v = _value(st)
        if isinstance(v, dict) and v.get("unit") in (MINUTE, HOUR):
            amount = float(v["amount"])
            return round(amount * (60 if v["unit"] == HOUR else 1))
    return None


def _first_value(statements: list[dict[str, Any]]) -> str | None:
    """The first string value among non-deprecated statements, preferred rank first."""
    for statement in _usable(statements):
        value = statement.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(value, str):
            return value
    return None


def _follow(title: str, aliases: dict[str, str]) -> str:
    """Apply normalisation and redirect hops (bounded, in case of a redirect loop)."""
    for _ in range(5):
        if title not in aliases:
            break
        title = aliases[title]
    return title
