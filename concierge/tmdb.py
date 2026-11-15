"""Cached TMDB client.

Purpose: every TMDB read in the project goes through `TMDB`, so responses are cached on disk
(reproducible evals and resumable long runs, AGENTS.md section 4) and the API key stays here:
it is never part of a cache key, a log line or an error message.
Input: a media type ("movie" | "tv"), an item id or discover filters.
Output: TMDB JSON as dicts (None for an id TMDB no longer knows).
"""

from typing import Any, Literal

import httpx

from concierge import config
from concierge.cache import DiskCache
from concierge.http import HTTPError, get_json

MediaType = Literal["movie", "tv"]
BASE_URL = "https://api.themoviedb.org/3"


class TMDB:
    """TMDB v3 reads with a disk cache. The key is loaded on the first uncached request."""

    def __init__(self, client: httpx.Client | None = None, cache: DiskCache | None = None) -> None:
        self.client = client or httpx.Client(
            base_url=BASE_URL, timeout=30, transport=httpx.HTTPTransport(retries=3)
        )
        self.cache = cache or DiskCache(config.TMDB_CACHE_DIR)
        self._api_key: str | None = None

    def genres(self, media: MediaType) -> dict[int, str]:
        """Genre id -> name for movies or TV (the two lists differ)."""
        data = self._get(f"/genre/{media}/list", {})
        return {g["id"]: g["name"] for g in data["genres"]}

    def discover(self, media: MediaType, filters: dict[str, Any], page: int) -> list[dict]:
        """One page (20 items) of /discover results for `filters`."""
        params = {**filters, "page": page, "include_adult": "false"}
        return self._get(f"/discover/{media}", params)["results"]

    def search(self, query: str, language: str = "en-US") -> list[dict]:
        """First page of /search/multi, movies and TV only, in TMDB's relevance order. TMDB
        also matches translated titles, so a Ukrainian title finds the film."""
        params = {"query": query, "include_adult": "false", "language": language, "page": 1}
        results = self._get("/search/multi", params)["results"]
        return [r for r in results if r.get("media_type") in ("movie", "tv")]

    def recommendations(self, media: MediaType, item_id: int) -> list[dict]:
        """First page of TMDB's recommendations for a title (its users' co-watching signal)."""
        try:
            return self._get(f"/{media}/{item_id}/recommendations", {"page": 1})["results"]
        except HTTPError as e:
            if e.status == 404:
                return []
            raise

    def details(self, media: MediaType, item_id: int, append: tuple[str, ...] = ()) -> dict | None:
        """/movie/{id} or /tv/{id}, with sub-resources (e.g. "keywords") appended in the same
        response. None when TMDB answers 404 (title removed); a 404 is not cached."""
        params = {"append_to_response": ",".join(append)} if append else {}
        try:
            return self._get(f"/{media}/{item_id}", params)
        except HTTPError as e:
            if e.status == 404:
                return None
            raise

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        key = {"path": path, "params": params}
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if self._api_key is None:
            self._api_key = config.get_secret("TMDB_API_KEY")
        data = get_json(self.client, path, {**params, "api_key": self._api_key})
        self.cache.put(key, data)
        return data
