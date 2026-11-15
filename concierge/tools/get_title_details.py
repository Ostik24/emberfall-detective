"""C8 get_title_details: facts the catalog does not hold (runtime, director, cast).

Purpose: answer "who directed it", "is it under two hours", and give the verifier (C11) and the
explainer (C12) more evidence than the catalog document.
Input: item type and TMDB id. Output: `TitleDetails` (None fields when TMDB has no value);
`found=False` when TMDB no longer knows the id.
"""

from typing import Literal

from pydantic import BaseModel

from concierge.catalog import Catalog
from concierge.tmdb import TMDB

MAX_CAST = 8
MAX_KEYWORDS = 15


class TitleDetails(BaseModel):
    item_type: Literal["movie", "tv"]
    item_id: int
    found: bool
    in_catalog: bool
    title: str | None = None
    original_title: str | None = None
    year: int | None = None
    runtime_minutes: int | None = None  # movie runtime, or typical episode length for TV
    seasons: int | None = None
    genres: list[str] = []
    directors: list[str] = []  # TV: creators
    cast: list[str] = []
    keywords: list[str] = []
    overview: str | None = None
    countries: list[str] = []
    original_language: str | None = None


def get_title_details(
    tmdb: TMDB, catalog: Catalog, item_type: Literal["movie", "tv"], item_id: int
) -> TitleDetails:
    """TMDB details with credits and keywords, flattened."""
    in_catalog = catalog.row((item_type, item_id)) is not None
    data = tmdb.details(item_type, item_id, append=("credits", "keywords"))
    if data is None:
        return TitleDetails(item_type=item_type, item_id=item_id, found=False, in_catalog=False)
    credits = data.get("credits") or {}
    keyword_block = data.get("keywords") or {}
    keywords = keyword_block.get("keywords" if item_type == "movie" else "results") or []
    date = data.get("release_date") or data.get("first_air_date") or ""
    if item_type == "movie":
        runtime = data.get("runtime") or None
        directors = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
    else:
        runtimes = data.get("episode_run_time") or []
        runtime = runtimes[0] if runtimes else None
        directors = [c["name"] for c in data.get("created_by", [])]
    return TitleDetails(
        item_type=item_type,
        item_id=item_id,
        found=True,
        in_catalog=in_catalog,
        title=data.get("title") or data.get("name"),
        original_title=data.get("original_title") or data.get("original_name"),
        year=int(date[:4]) if date[:4].isdigit() else None,
        runtime_minutes=runtime,
        seasons=data.get("number_of_seasons"),
        genres=[g["name"] for g in data.get("genres", [])],
        directors=directors,
        cast=[c["name"] for c in credits.get("cast", [])[:MAX_CAST]],
        keywords=[k["name"] for k in keywords][:MAX_KEYWORDS],
        overview=data.get("overview") or None,
        countries=[c["iso_3166_1"] for c in data.get("production_countries", [])]
        or data.get("origin_country", []),
        original_language=data.get("original_language"),
    )
