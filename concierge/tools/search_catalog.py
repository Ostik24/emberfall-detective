"""C4 search_catalog (+ C2 clues, C5 rerank): find titles in the catalog from a description.

Purpose: the main detective tool. Pipeline, configuration chosen on dev (DEV_LOG, C4/C5 loops):
1. the dense query is the whole recollection (`query_mode="whole"`). The C2 ablation
   (`"clues"`: query = film-related clues only) lost on dev (S@1 -0.087, CI excludes 0), so
   clue decomposition serves the verifier (C11), not retrieval;
2. filters (type, years, genres, excluded ids) become a row mask applied BEFORE ranking,
   so a filter can never empty the result list the way the baseline's post-filter can (W2);
3. E3 dense top 100 over `doc_wiki` of the extended catalog;
4. C5: Haiku reranks the top 20 against the whole recollection (1000 characters per
   candidate);
5. the top `limit` are returned with a short snippet each.
Input: `SearchCatalogArgs`. Output: `SearchCatalogResult`.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field

from concierge.catalog import Catalog
from concierge.clues import Clues, decompose
from concierge.embed import Embedder
from concierge.llm import LLM
from concierge.retrieval.dense import DenseIndex, document_vectors
from concierge.retrieval.rerank import rerank

SNIPPET_CHARS = 300


class SearchCatalogArgs(BaseModel):
    """What the agent passes (also the MCP tool input)."""

    query: str = Field(description="The user's description of the title, in their own words.")
    item_type: Literal["movie", "tv"] | None = Field(
        None, description="Only movies or only TV shows, if the user said so."
    )
    year_from: int | None = Field(None, description="Earliest release year the user allows.")
    year_to: int | None = Field(None, description="Latest release year the user allows.")
    genres: list[str] = Field(
        default_factory=list, description="TMDB genre names the title must have (any of)."
    )
    exclude_genres: list[str] = Field(
        default_factory=list, description="TMDB genre names the title must not have."
    )
    exclude_ids: list[str] = Field(
        default_factory=list,
        description='Titles the user already rejected, as "movie/603" or "tv/1396".',
    )
    limit: int = Field(10, description="How many candidates to return (1-20).")


class Candidate(BaseModel):
    id: str  # "movie/603"
    title: str
    year: int | None
    genres: list[str]
    snippet: str


class SearchCatalogResult(BaseModel):
    candidates: list[Candidate]
    clues: Clues | None
    searched_text: str
    rows_after_filters: int
    reranked: bool


@dataclass(frozen=True)
class SearchConfig:
    query_mode: Literal["clues", "whole"] = "whole"
    depth: int = 100
    rerank_depth: int = 20
    rerank_chars: int = 1000


def key_str(key: tuple[str, int]) -> str:
    return f"{key[0]}/{key[1]}"


def parse_key(text: str) -> tuple[str, int] | None:
    """'movie/603' -> ('movie', 603); None for anything else."""
    kind, _, number = text.strip().partition("/")
    return (kind, int(number)) if kind in ("movie", "tv") and number.isdigit() else None


class CatalogSearch:
    """The search pipeline over one catalog snapshot."""

    def __init__(
        self,
        catalog: Catalog,
        embedder: Embedder,
        llm: LLM,
        genre_names: dict[int, str],
        config: SearchConfig = SearchConfig(),
    ) -> None:
        self.catalog = catalog
        self.embedder = embedder
        self.llm = llm
        self.config = config
        self.texts = catalog.documents("doc_wiki")
        self.index = DenseIndex(document_vectors(embedder, self.texts))
        self.genre_names = genre_names
        frame = catalog.frame
        self.years = np.array(
            [
                int(d[:4]) if isinstance(d, str) and d[:4].isdigit() else -1
                for d in frame.release_date
            ]
        )
        self.genres = [[genre_names.get(int(g), "") for g in ids] for ids in frame.genre_ids]

    def mask(self, args: SearchCatalogArgs) -> np.ndarray:
        """Rows that satisfy every filter."""
        frame = self.catalog.frame
        allowed = np.ones(len(frame), dtype=bool)
        if args.item_type:
            allowed &= (frame.item_type == args.item_type).to_numpy()
        if args.year_from:
            allowed &= self.years >= args.year_from
        if args.year_to:
            allowed &= (self.years <= args.year_to) & (self.years > 0)
        want = {g.casefold() for g in args.genres}
        avoid = {g.casefold() for g in args.exclude_genres}
        if want or avoid:
            for row, names in enumerate(self.genres):
                folded = {n.casefold() for n in names}
                if (want and not want & folded) or avoid & folded:
                    allowed[row] = False
        for text in args.exclude_ids:
            if (key := parse_key(text)) and (row := self.catalog.row(key)) is not None:
                allowed[row] = False
        return allowed

    def rank(self, args: SearchCatalogArgs) -> tuple[list[int], SearchCatalogResult]:
        """All ranked rows (up to `depth`) and the tool result built from them."""
        clues = decompose(self.llm, args.query) if self.config.query_mode == "clues" else None
        searched = (clues.search_text() if clues else "") or args.query
        allowed = self.mask(args)
        hits = self.index.search(self.embedder.encode([searched])[0], self.config.depth, allowed)
        rows = [row for row, _ in hits]
        reranked = False
        if self.config.rerank_depth and len(rows) > 1:
            texts = [self.texts[r] for r in rows]
            depth, chars = self.config.rerank_depth, self.config.rerank_chars
            order = rerank(self.llm, args.query, texts, depth, chars).order
            rows, reranked = [rows[i] for i in order], True
        limit = max(1, min(args.limit, 20))
        result = SearchCatalogResult(
            candidates=[self.candidate(row) for row in rows[:limit]],
            clues=clues,
            searched_text=searched,
            rows_after_filters=int(allowed.sum()),
            reranked=reranked,
        )
        return rows, result

    def search(self, args: SearchCatalogArgs) -> SearchCatalogResult:
        return self.rank(args)[1]

    def candidate(self, row: int) -> Candidate:
        frame = self.catalog.frame
        year = int(self.years[row]) if self.years[row] > 0 else None
        return Candidate(
            id=key_str(self.catalog.keys[row]),
            title=frame.title[row],
            year=year,
            genres=[g for g in self.genres[row] if g],
            snippet=self.texts[row][:SNIPPET_CHARS],
        )
