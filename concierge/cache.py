"""Disk cache for expensive, reproducible calls (LLM, TMDB, web search).

Purpose: evals and the notebook replay earlier results instead of paying for them again.
Input: any JSON-serialisable key. Output: the JSON value stored under it, or None.
Each entry is one file `<dir>/<hash[:2]>/<hash>.json` holding both key and value, so a cached
call can be inspected by hand.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def key_hash(key: Any) -> str:
    """sha256 of the canonical JSON form of `key` (sorted keys, no whitespace)."""
    canonical = json.dumps(key, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DiskCache:
    """JSON values on disk, addressed by the hash of their key."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def get(self, key: Any) -> Any | None:
        """Stored value for `key`, or None if absent."""
        path = self._path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["value"]

    def put(self, key: Any, value: Any) -> None:
        """Store `value` under `key`; the write is atomic so a crash never leaves half a file."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"key": key, "value": value}, ensure_ascii=False), "utf-8")
        os.replace(tmp, path)

    def _path(self, key: Any) -> Path:
        digest = key_hash(key)
        return self.directory / digest[:2] / f"{digest}.json"
