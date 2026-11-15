"""Watchlist writes need a pending action and its token."""

import pandas as pd
import pytest

from concierge.catalog import Catalog
from concierge.tools.watchlist import Watchlist, WatchlistArgs


@pytest.fixture
def watchlist(tmp_path) -> Watchlist:
    frame = pd.DataFrame(
        {"item_type": ["movie", "tv"], "item_id": [603, 1396], "title": ["The Matrix", "BB"]}
    )
    return Watchlist(Catalog(frame), tmp_path / "w.db")


def test_add_only_proposes_and_confirm_writes(watchlist):
    proposed = watchlist.run(WatchlistArgs(action="add", id="movie/603"))
    assert proposed.status == "pending_confirmation"
    assert watchlist.items() == []  # nothing written yet
    done = watchlist.run(WatchlistArgs(action="confirm", token=proposed.pending.token))
    assert done.status == "done"
    assert [i.id for i in watchlist.items()] == ["movie/603"]


def test_confirm_needs_the_right_token_and_works_once(watchlist):
    proposed = watchlist.run(WatchlistArgs(action="add", id="movie/603"))
    assert watchlist.run(WatchlistArgs(action="confirm", token="wrong")).status == "error"
    assert watchlist.items() == []
    token = proposed.pending.token
    watchlist.run(WatchlistArgs(action="confirm", token=token))
    assert watchlist.run(WatchlistArgs(action="confirm", token=token)).status == "error"


def test_remove_and_bad_requests(watchlist):
    assert watchlist.run(WatchlistArgs(action="add", id="movie/999")).status == "error"
    assert watchlist.run(WatchlistArgs(action="remove", id="tv/1396")).status == "error"
    token = watchlist.run(WatchlistArgs(action="add", id="tv/1396")).pending.token
    watchlist.run(WatchlistArgs(action="confirm", token=token))
    assert watchlist.run(WatchlistArgs(action="add", id="tv/1396")).status == "error"
    token = watchlist.run(WatchlistArgs(action="remove", id="tv/1396")).pending.token
    watchlist.run(WatchlistArgs(action="confirm", token=token))
    assert watchlist.items() == []
