"""C6 find_title: resolve a title the user names (possibly misspelled or in Ukrainian) to ids.

Purpose: when the user already knows (part of) the title - "is Twister in the catalog?",
"something like Інтерстеллар" - the agent needs the exact item, not a plot search.
Steps:
1. English/Latin input: fuzzy match against catalog titles (rapidfuzz `ratio`, normalised
   edit distance, on folded titles), a year that matches the release year (+-1) breaks ties;
2. Ukrainian input, or no catalog match scoring at least `accept`: TMDB /search/multi, which
   also knows translated titles; results are flagged `in_catalog` or not.
Input: title, optional year and type. Output: `FindTitleResult` with up to `limit` matches,
best first, each with its source ("catalog" | "tmdb") and score (0-100).
"""

from typing import Literal

from pydantic import BaseModel
from rapidfuzz import fuzz, process

from concierge.catalog import Catalog
from concierge.normalize import detect_language, fold
from concierge.tmdb import TMDB

ACCEPT = 90.0  # catalog score from which TMDB is not asked (tuned on dev, C6)
SCORERS = {"ratio": fuzz.ratio, "wratio": fuzz.WRatio}
YEAR_BONUS = 3.0


class TitleMatch(BaseModel):
    item_type: Literal["movie", "tv"]
    item_id: int
    title: str
    year: int | None
    score: float
    source: Literal["catalog", "tmdb"]
    in_catalog: bool


class FindTitleResult(BaseModel):
    query: str
    matches: list[TitleMatch]


def _year(date: str | None) -> int | None:
    return int(date[:4]) if date and date[:4].isdigit() else None


class TitleFinder:
    """Title lookup over one catalog, with TMDB as the fallback."""

    def __init__(
        self, catalog: Catalog, tmdb: TMDB, accept: float = ACCEPT, scorer: str = "ratio"
    ) -> None:
        self.catalog = catalog
        self.tmdb = tmdb
        self.accept = accept
        self.scorer = SCORERS[scorer]
        self.folded = [fold(t) for t in catalog.frame.title]
        self.years = [_year(d) for d in catalog.frame.release_date]

    def find(
        self,
        title: str,
        year: int | None = None,
        item_type: Literal["movie", "tv"] | None = None,
        limit: int = 5,
    ) -> FindTitleResult:
        """Best matches for `title`, catalog first when it is confident."""
        matches: list[TitleMatch] = []
        if detect_language(title) == "en":
            matches = self._catalog(title, year, item_type, limit)
        if not matches or matches[0].score < self.accept:
            matches = self._tmdb(title, year, item_type, limit) or matches
        return FindTitleResult(query=title, matches=matches[:limit])

    def _catalog(self, title, year, item_type, limit) -> list[TitleMatch]:
        hits = process.extract(fold(title), self.folded, scorer=self.scorer, limit=50)
        scored = []
        for _, score, row in hits:
            key = self.catalog.keys[row]
            if item_type and key[0] != item_type:
                continue
            row_year = self.years[row]
            bonus = YEAR_BONUS if year and row_year and abs(row_year - year) <= 1 else 0.0
            scored.append((min(score + bonus, 100.0), score, row))
        scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
        return [
            TitleMatch(
                item_type=self.catalog.keys[row][0],
                item_id=self.catalog.keys[row][1],
                title=self.catalog.frame.title[row],
                year=self.years[row],
                score=round(total, 1),
                source="catalog",
                in_catalog=True,
            )
            for total, _, row in scored[:limit]
        ]

    def _tmdb(self, title, year, item_type, limit) -> list[TitleMatch]:
        results = self.tmdb.search(title)
        if item_type:
            results = [r for r in results if r["media_type"] == item_type]
        if year:  # stable sort: TMDB order, but a matching year first
            results.sort(key=lambda r: not _close(_year(_date(r)), year))
        matches = []
        for i, r in enumerate(results[:limit]):
            key = (r["media_type"], int(r["id"]))
            matches.append(
                TitleMatch(
                    item_type=key[0],
                    item_id=key[1],
                    title=r.get("title") or r.get("name") or "",
                    year=_year(_date(r)),
                    score=round(100.0 - 10 * i, 1),
                    source="tmdb",
                    in_catalog=self.catalog.row(key) is not None,
                )
            )
        return matches


def _date(result: dict) -> str | None:
    return result.get("release_date") or result.get("first_air_date")


def _close(a: int | None, b: int) -> bool:
    return a is not None and abs(a - b) <= 1
