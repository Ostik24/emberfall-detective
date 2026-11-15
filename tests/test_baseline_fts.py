"""Baseline full-text search emulation; expected values were produced by PostgreSQL 16."""

import pytest

from evals.baseline.fts import FullTextIndex, parse_query, to_tsvector

PG_TSVECTOR = {  # to_tsvector('english', <the text below>) on PostgreSQL 16.2
    "twister": [1], "action": [2], "storm-chas": [3], "storm": [4, 7], "chase": [5, 11],
    "tornado": [6, 13], "chaser": [8], "1996": [15], "oklahoma": [16], "café": [18],
}  # fmt: skip
TEXT = (
    "Twister. Action. storm-chasing, tornado. Storm chasers don't chase the tornadoes "
    "in 1996 Oklahoma's café."
)


def test_tsvector_matches_postgres_on_stems_stopwords_hyphens_and_positions():
    lexemes: dict[str, list[int]] = {}
    for lexeme, position in to_tsvector(TEXT):
        lexemes.setdefault(lexeme, []).append(position)
    assert lexemes == PG_TSVECTOR


def test_query_is_or_of_words_with_phrases_and_negation():
    # PostgreSQL: 'movi' | 'storm' | 'chaser' | 'chase' | 'tornado' | !'cow' | 'fli' <-> 'cow'
    parsed = parse_query('a movie where storm chasers chase a tornado -cow "flying cow" or not')
    words = [p.items for p in parsed.phrases]
    assert words[:5] == [(("movi", 0),), (("storm", 0),), (("chaser", 0),), (("chase", 0),),
                         (("tornado", 0),)]  # fmt: skip
    assert words[5] == (("fli", 0), ("cow", 1))
    assert parsed.negated == ("cow",)


@pytest.mark.parametrize(
    ("doc", "rank"),  # ts_rank_cd for 'storm | chaser | tornado | oklahoma' on PostgreSQL 16
    [
        ("storm storm tornado x y oklahoma", 0.4),
        ("storm", 0.1),
        ("tornado storm", 0.2),
        ("storm chaser", 0.2),
    ],  # fmt: skip
)
def test_rank_counts_covers_like_ts_rank_cd(doc, rank):
    index = FullTextIndex([doc, "a b c"])
    assert index.search("storm chasers tornado oklahoma") == [(0, pytest.approx(rank))]


def test_phrase_needs_adjacent_positions_and_negation_matches_documents_without_the_word():
    index = FullTextIndex(["a flying cow", "the cow was flying", "nothing here"])
    assert dict(index.search('"flying cow"')) == {0: pytest.approx(0.1)}
    assert {d for d, _ in index.search("-cow")} == {2}
