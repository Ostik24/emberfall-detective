"""C9 watchlist: list, add and remove titles in a local SQLite watchlist.

Purpose: let the user keep what the detective found. Writes are two-step, enforced here:
`add` / `remove` never touch the database; they return a pending action with a token.
`confirm` with that token performs it. Who may confirm is the caller's decision: the agent
(concierge/agent.py) accepts `confirm` only when the user's latest message confirmed the
pending action; an MCP host asks the user before running a tool marked as writing.
Input: `WatchlistArgs`. Output: `WatchlistResult` (items, pending action, or what changed).
"""

import hashlib
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from concierge.catalog import Catalog
from concierge.config import REPO_ROOT
from concierge.tools.search_catalog import parse_key

WATCHLIST_DB = REPO_ROOT / "watchlist.db"


class WatchlistArgs(BaseModel):
    action: Literal["list", "add", "remove", "confirm"] = Field(
        description=(
            "list: show the watchlist. add/remove: PROPOSE a change (nothing is written; the "
            "user must confirm). confirm: carry out the pending change, only after the user "
            "said yes."
        )
    )
    id: str | None = Field(None, description='Title for add/remove, e.g. "movie/603".')
    token: str | None = Field(None, description="The pending action's token, for confirm.")


class WatchlistItem(BaseModel):
    id: str
    title: str
    added_at: str


class PendingAction(BaseModel):
    action: Literal["add", "remove"]
    id: str
    title: str
    token: str


class WatchlistResult(BaseModel):
    status: Literal["listed", "pending_confirmation", "done", "error"]
    items: list[WatchlistItem] = []
    pending: PendingAction | None = None
    message: str = ""


class Watchlist:
    """The SQLite store plus the at-most-one pending write."""

    def __init__(self, catalog: Catalog, path: Path = WATCHLIST_DB) -> None:
        self.catalog = catalog
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute(
            "create table if not exists watchlist (id text primary key, title text, added_at text)"
        )
        self.pending: PendingAction | None = None

    def run(self, args: WatchlistArgs) -> WatchlistResult:
        if args.action == "list":
            return WatchlistResult(status="listed", items=self.items())
        if args.action == "confirm":
            return self._confirm(args.token)
        return self._propose(args.action, args.id)

    def items(self) -> list[WatchlistItem]:
        rows = self.db.execute("select id, title, added_at from watchlist order by added_at")
        return [WatchlistItem(id=i, title=t, added_at=a) for i, t, a in rows]

    def _propose(self, action: Literal["add", "remove"], item_id: str | None) -> WatchlistResult:
        key = parse_key(item_id or "")
        row = self.catalog.row(key) if key else None
        if row is None:
            return WatchlistResult(status="error", message=f"unknown title id {item_id!r}")
        listed = any(i.id == item_id for i in self.items())
        if action == "add" and listed:
            return WatchlistResult(status="error", message="already on the watchlist")
        if action == "remove" and not listed:
            return WatchlistResult(status="error", message="not on the watchlist")
        token = hashlib.sha256(f"{action}:{item_id}".encode()).hexdigest()[:8]
        title = self.catalog.frame.title[row]
        self.pending = PendingAction(action=action, id=item_id, title=title, token=token)
        return WatchlistResult(
            status="pending_confirmation",
            pending=self.pending,
            message=f"Ask the user to confirm: {action} {title!r}. Nothing was written.",
        )

    def _confirm(self, token: str | None) -> WatchlistResult:
        pending = self.pending
        if pending is None or token != pending.token:
            return WatchlistResult(status="error", message="no pending action with this token")
        if pending.action == "add":
            now = datetime.now(UTC).isoformat(timespec="seconds")
            self.db.execute(
                "insert into watchlist values (?, ?, ?)", (pending.id, pending.title, now)
            )
        else:
            self.db.execute("delete from watchlist where id = ?", (pending.id,))
        self.db.commit()
        self.pending = None
        return WatchlistResult(
            status="done", items=self.items(), message=f"{pending.action} {pending.title!r}"
        )
