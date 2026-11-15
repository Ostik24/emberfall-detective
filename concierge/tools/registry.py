"""Tool registry: the single source of tool names, descriptions and JSON schemas.

Purpose: the agent (concierge/agent.py) and the MCP server (concierge/mcp_server.py) both read
`TOOLS`, so they expose identical schemas (enforced by tests/test_registry.py). Descriptions
are written for the model: when to use the tool, when not to, and what the arguments mean.
`Toolbox` holds the loaded resources and runs a tool by name.
Seven tools: six data tools (also on MCP) and `respond`, the agent's way to end a turn.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from concierge.catalog import Catalog
from concierge.config import E3_MODEL
from concierge.embed import Embedder
from concierge.llm import LLM
from concierge.tmdb import TMDB
from concierge.tools.external_lookup import ExternalLookupArgs, external_lookup
from concierge.tools.find_title import TitleFinder
from concierge.tools.get_title_details import get_title_details
from concierge.tools.search_catalog import CatalogSearch, SearchCatalogArgs, parse_key
from concierge.tools.similar_to import SimilarToArgs, similar_to
from concierge.tools.watchlist import Watchlist, WatchlistArgs


class FindTitleArgs(BaseModel):
    title: str = Field(description="The title as the user wrote it (any language, typos ok).")
    year: int | None = Field(None, description="Release year, if the user gave one.")
    item_type: Literal["movie", "tv"] | None = Field(None, description="If the user said.")


class TitleDetailsArgs(BaseModel):
    id: str = Field(description='The title, as "movie/603" or "tv/1396".')


class RespondArgs(BaseModel):
    message: str = Field(
        description="Your reply to the user, in the user's language. Short and concrete."
    )
    picks: list[str] = Field(
        default_factory=list,
        description='Titles you propose, best first (max 5), as ids like "movie/603": every '
        "title your message suggests, including ones from external_lookup that are not in "
        "the catalog. Only ids returned by a tool in this conversation.",
    )
    options: list[str] = Field(
        default_factory=list,
        description="When the message asks the user a question: 2-4 short answer options.",
    )


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args: type[BaseModel]
    run: Callable[["Toolbox", Any], BaseModel] | None  # None: handled by the agent itself
    writes: bool = False
    agent_only: bool = False

    def schema(self) -> dict[str, Any]:
        """The tool definition sent to the Messages API and to MCP clients."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": _clean({**self.args.model_json_schema(), "description": None}),
        }


def _clean(schema: Any) -> Any:
    """Drop pydantic's cosmetic `title` keys and None values (they cost tokens, say nothing)."""
    if isinstance(schema, dict):
        return {
            k: _clean(v)
            for k, v in schema.items()
            if v is not None and (k != "title" or isinstance(v, dict))
        }
    if isinstance(schema, list):
        return [_clean(v) for v in schema]
    return schema


TOOLS: list[ToolSpec] = [
    ToolSpec(
        "search_catalog",
        "Find films and TV shows in the catalog from a description of their plot, scenes, "
        "characters, setting or mood. Use it whenever the user describes a title they cannot "
        "name, or asks for titles matching a description. Pass the user's words as `query` "
        "(the whole description, including uncertain details; retrieval works best on the "
        "user's full wording, so do not shorten or rewrite it). Put only constraints the "
        "user stated as filters (type, years, genres); "
        "never invent filters. Put titles the user already rejected in `exclude_ids`. "
        "Do NOT use it for a title the user names (use find_title). Returns up to `limit` "
        "candidates, best first, each with an id and a snippet of its catalog text.",
        SearchCatalogArgs,
        lambda box, args: box.search.search(args),
    ),
    ToolSpec(
        "find_title",
        "Resolve a title the user names to its id, e.g. 'Twister', 'Interstelar' (typo), "
        "'Інтерстеллар' (Ukrainian). Use it before similar_to, get_title_details or "
        "watchlist when you only have a name. Searches the catalog, then TMDB; each match "
        "says whether it is in the catalog. Do NOT use it for plot descriptions.",
        FindTitleArgs,
        lambda box, args: box.finder.find(args.title, args.year, args.item_type),
    ),
    ToolSpec(
        "similar_to",
        "Titles similar to one known title (by plot, setting and tone), with the same "
        "filters as search_catalog. Use it for 'something like X'. Needs the id of X "
        "(from find_title or a previous result); only catalog titles can be anchors.",
        SimilarToArgs,
        lambda box, args: similar_to(box.search, args),
    ),
    ToolSpec(
        "get_title_details",
        "Facts about one title from TMDB: runtime, seasons, directors (creators for TV), "
        "cast, genres, keywords, overview. Use it to answer factual questions ('who is in "
        "it', 'is it under two hours') or to check a detail the user remembers. One title "
        "per call.",
        TitleDetailsArgs,
        lambda box, args: box.details(args.id),
    ),
    ToolSpec(
        "watchlist",
        "The user's watchlist. action=list shows it. action=add / remove only PROPOSES a "
        "change: nothing is saved until the user confirms in their next message. After "
        "proposing, ask the user to confirm and stop. Use action=confirm with the token "
        "only when the user's latest message says yes to that exact change.",
        WatchlistArgs,
        lambda box, args: box.watchlist.run(args),
        writes=True,
    ),
    ToolSpec(
        "external_lookup",
        "Look outside the catalog when search_catalog found nothing convincing. Give the "
        "user's description and, if you suspect specific titles, up to 3 `title_guesses`: "
        "they are checked on TMDB, then a web search looks for the description (it costs "
        "money, so do not call this tool when catalog candidates already fit). Returns "
        "only titles that exist on TMDB, each marked in_catalog or not.",
        ExternalLookupArgs,
        lambda box, args: external_lookup(box.tmdb, box.catalog, box.llm, args),
    ),
    ToolSpec(
        "respond",
        "End your turn: send the user your message, the titles you propose (picks), and "
        "answer options if you ask a question. Always finish with this tool. Propose only "
        "ids that a tool returned. Ask at most one question at a time, and only when the "
        "top candidates really differ on something the user can answer.",
        RespondArgs,
        None,
        agent_only=True,
    ),
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


class Toolbox:
    """Loaded resources and the dispatcher that runs data tools by name."""

    def __init__(self, catalog: Catalog, tmdb: TMDB, llm: LLM, search: CatalogSearch,
                 watchlist: Watchlist) -> None:  # fmt: skip
        self.catalog = catalog
        self.tmdb = tmdb
        self.llm = llm
        self.search = search
        self.watchlist = watchlist
        self.finder = TitleFinder(catalog, tmdb)

    @classmethod
    def create(cls, llm: LLM | None = None, watchlist_path=None) -> "Toolbox":
        """Everything loaded from local files and caches (no network unless uncached)."""
        catalog, tmdb, llm = Catalog.load(), TMDB(), llm or LLM()
        genres = {**tmdb.genres("movie"), **tmdb.genres("tv")}
        search = CatalogSearch(catalog, Embedder(E3_MODEL), llm, genres)
        watchlist = Watchlist(catalog, watchlist_path) if watchlist_path else Watchlist(catalog)
        return cls(catalog, tmdb, llm, search, watchlist)

    def run(self, name: str, arguments: dict[str, Any]) -> BaseModel:
        """Validate `arguments` against the tool's schema and run it."""
        spec = TOOLS_BY_NAME[name]
        if spec.run is None:
            raise ValueError(f"{name} is not a data tool")
        return spec.run(self, spec.args.model_validate(arguments))

    def details(self, item_id: str):
        key = parse_key(item_id)
        if key is None:
            raise ValueError(f"bad id {item_id!r}; expected 'movie/603' or 'tv/1396'")
        return get_title_details(self.tmdb, self.catalog, key[0], key[1])
