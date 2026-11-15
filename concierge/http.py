"""Shared GET-JSON helper for external APIs (Supabase PostgREST, TMDB, Wikimedia).

Purpose: one retry and error policy for every HTTP read. 429 and 5xx are retried with backoff,
failed connections are retried by the client's transport, anything else raises `HTTPError`.
Error messages carry host, path and status only: TMDB keys travel in the query string.
Input: an httpx.Client, a URL (or a path relative to the client's base_url), query params.
Output: the parsed JSON body.
"""

import time
from typing import Any

import httpx

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})
BACKOFF_SECONDS = (2, 4, 8)


class HTTPError(RuntimeError):
    """A failed GET that was not, or no longer, worth retrying."""

    def __init__(self, url: httpx.URL, status: int) -> None:
        super().__init__(f"GET {url.host}{url.path} -> HTTP {status}")
        self.status = status


def get_json(
    client: httpx.Client,
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> Any:
    """GET `url` and return its JSON, retrying 429/5xx up to len(BACKOFF_SECONDS) times."""
    for backoff in (*BACKOFF_SECONDS, None):
        response = client.get(url, params=params, headers=headers)
        if response.status_code in RETRYABLE_STATUS and backoff is not None:
            time.sleep(backoff)
            continue
        if response.is_error:
            raise HTTPError(response.request.url, response.status_code)
        return response.json()
