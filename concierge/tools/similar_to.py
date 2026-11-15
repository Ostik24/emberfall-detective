"""C7 similar_to: titles close to one the user names ("something like Twister, but older").

Purpose: discovery mode. The anchor's own E3 document vector is the query, so "similar" means
similar plot, setting and tone as the catalog text describes them; the same filters as
`search_catalog` apply before ranking.
Input: `SimilarToArgs` (anchor id + filters). Output: `SimilarToResult` (anchor title,
neighbours best first, the anchor itself excluded). An anchor outside the catalog is reported
with `found=False` (resolve it with find_title first).
"""

from typing import Literal

from pydantic import BaseModel, Field

from concierge.tools.search_catalog import (
    Candidate,
    CatalogSearch,
    SearchCatalogArgs,
    parse_key,
)


class SimilarToArgs(BaseModel):
    id: str = Field(description='The anchor title, as "movie/603" or "tv/1396".')
    item_type: Literal["movie", "tv"] | None = Field(None, description="Restrict the type.")
    year_from: int | None = None
    year_to: int | None = None
    genres: list[str] = Field(default_factory=list)
    exclude_genres: list[str] = Field(default_factory=list)
    exclude_ids: list[str] = Field(default_factory=list)
    limit: int = Field(10, description="How many titles to return (1-20).")


class SimilarToResult(BaseModel):
    found: bool
    anchor: str | None
    similar: list[Candidate]


def similar_to(search: CatalogSearch, args: SimilarToArgs) -> SimilarToResult:
    """Nearest catalog neighbours of the anchor's document vector."""
    key = parse_key(args.id)
    row = search.catalog.row(key) if key else None
    if row is None:
        return SimilarToResult(found=False, anchor=None, similar=[])
    filters = SearchCatalogArgs(
        query="",
        item_type=args.item_type,
        year_from=args.year_from,
        year_to=args.year_to,
        genres=args.genres,
        exclude_genres=args.exclude_genres,
        exclude_ids=[*args.exclude_ids, args.id],
    )
    allowed = search.mask(filters)
    limit = max(1, min(args.limit, 20))
    hits = search.index.search(search.index.vectors[row], limit, allowed)
    return SimilarToResult(
        found=True,
        anchor=search.catalog.frame.title[row],
        similar=[search.candidate(r) for r, _ in hits],
    )
