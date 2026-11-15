"""external_lookup: guesses are checked on TMDB, the web is searched as well, only TMDB-resolved
titles come back."""

import pandas as pd

from concierge.catalog import Catalog
from concierge.tools import external_lookup as ext
from concierge.tools.external_lookup import ExternalLookupArgs, TitleGuess


class FakeTMDB:
    KNOWN = {
        "Twister": {"id": 664, "media_type": "movie", "title": "Twister",
                    "release_date": "1996-05-10", "overview": "Storm chasers."},
        "Night of the Twisters": {"id": 32562, "media_type": "movie",
                                  "title": "Night of the Twisters",
                                  "release_date": "1996-02-01", "overview": "A boy."},
    }  # fmt: skip

    def search(self, query):
        return [self.KNOWN[query]] if query in self.KNOWN else []


CATALOG = Catalog(pd.DataFrame({"item_type": ["movie"], "item_id": [664], "title": ["Twister"]}))


def test_web_runs_even_when_a_guess_resolves_and_unknown_titles_are_dropped(monkeypatch):
    web_calls = []

    def fake_web(llm, description):
        web_calls.append(description)
        return [TitleGuess(title="Night of the Twisters", year=1996),
                TitleGuess(title="Made Up Film", year=None)], 2  # fmt: skip

    monkeypatch.setattr(ext, "web_titles", fake_web)
    args = ExternalLookupArgs(description="kids and tornadoes", title_guesses=["Twister"])
    result = ext.external_lookup(FakeTMDB(), CATALOG, llm=None, args=args)
    assert web_calls == ["kids and tornadoes"]
    assert [(c.id, c.source, c.in_catalog) for c in result.candidates] == [
        ("movie/664", "tmdb_guess", True),
        ("movie/32562", "web", False),
    ]
    assert result.unresolved_titles == ["Made Up Film"] and result.web_searches == 2


def test_without_web_only_guesses_are_checked(monkeypatch):
    monkeypatch.setattr(ext, "web_titles", lambda *a: (_ for _ in ()).throw(AssertionError))
    args = ExternalLookupArgs(description="d", title_guesses=["Twister", "Nope"])
    result = ext.external_lookup(FakeTMDB(), CATALOG, llm=None, args=args, web=False)
    assert [c.id for c in result.candidates] == ["movie/664"]
    assert result.unresolved_titles == ["Nope"]
