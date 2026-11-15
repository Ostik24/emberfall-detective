"""PostgreSQL full-text search as the baseline uses it, re-implemented in Python.

Purpose: reproduce the keyword half of Emberfall's `hybrid_match_titles` (migration 0002/0003)
without a database, so the baseline runs in Colab:
    fts = to_tsvector('english', doc)
    tsq = replace(websearch_to_tsquery('english', query)::text, '&', '|')::tsquery
    order by ts_rank_cd(fts, tsq) desc   where fts @@ tsq
For this OR-query `ts_rank_cd` (no weights, no normalisation) is 0.1 per cover, and every
occurrence of a query lexeme (or phrase) in the document is one cover; checked on PostgreSQL 16.
There is no IDF (brief W5). Fidelity against a real PostgreSQL is measured by
`evals/baseline/check_fts_fidelity.py`.
Known gaps: e-mails, URLs and dotted abbreviations are tokenised as plain words.
Input: documents, a query. Output: (document index, rank) for every matching document.
"""

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import snowballstemmer

# PostgreSQL's tsearch_data/english.stop, copied verbatim
STOPWORDS = frozenset((Path(__file__).parent / "english.stop").read_text().split())
TOKEN = re.compile(r"[^\W_]+(?:-[^\W_]+)*")  # words and numbers, hyphenated compounds whole
RANK_PER_COVER = 0.1

_stemmer = snowballstemmer.stemmer("english")


def _lexeme(piece: str) -> str | None:
    """Stop word -> None; anything with a digit is kept as is; words are stemmed."""
    if piece in STOPWORDS:
        return None
    if any(c.isdigit() for c in piece):
        return piece
    return _stemmer.stemWord(piece)


def to_tsvector(text: str) -> list[tuple[str, int]]:
    """(lexeme, position) pairs. A hyphenated word yields the compound, then each part, each
    at its own position; stop words are dropped but still take a position."""
    out: list[tuple[str, int]] = []
    position = 0
    for match in TOKEN.finditer(text.lower()):
        token = match.group()
        parts = token.split("-")
        for piece in [token, *parts] if len(parts) > 1 else [token]:
            position += 1
            if (lexeme := _lexeme(piece)) is not None:
                out.append((lexeme, position))
    return out


@dataclass(frozen=True)
class Phrase:
    """Lexemes with their offsets from the first one (a single word is a one-item phrase)."""

    items: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class OrQuery:
    """websearch_to_tsquery with every AND turned into OR."""

    phrases: tuple[Phrase, ...]
    negated: tuple[str, ...]


def _phrase(text: str, compound_only: bool) -> Phrase | None:
    lexemes = to_tsvector(text)
    if compound_only and "-" in text:  # 'storm-chas' <-> 'storm' <-> 'chase' == the compound
        lexemes = lexemes[:1]
    if not lexemes:
        return None
    first = lexemes[0][1]
    return Phrase(tuple((lex, pos - first) for lex, pos in lexemes))


def parse_query(query: str) -> OrQuery:
    """ "quoted text" is a phrase, -word is negated, a bare `or` is an operator; the rest are
    words (a token holding several lexemes, e.g. "x's", is a phrase)."""
    phrases: list[Phrase] = []
    negated: list[str] = []
    for raw in re.findall(r'"[^"]*"|\S+', query):
        if raw.lower() == "or":
            continue
        if raw.startswith('"'):
            phrase = _phrase(raw.strip('"'), compound_only=False)
        elif raw.startswith("-") and len(raw) > 1:
            negated += [lex for lex, _ in to_tsvector(raw[1:])[:1]]
            continue
        else:
            phrase = _phrase(raw, compound_only=True)
        if phrase and phrase not in phrases:
            phrases.append(phrase)
    return OrQuery(tuple(phrases), tuple(dict.fromkeys(negated)))


class FullTextIndex:
    """Inverted index over documents: lexeme -> {document index: positions}."""

    def __init__(self, documents: Iterable[str]) -> None:
        self.postings: dict[str, dict[int, set[int]]] = defaultdict(dict)
        self.size = 0
        for index, text in enumerate(documents):
            for lexeme, position in to_tsvector(text):
                self.postings[lexeme].setdefault(index, set()).add(position)
            self.size += 1

    def _covers(self, phrase: Phrase) -> dict[int, int]:
        """Occurrences of `phrase` per document."""
        (head, _), *rest = phrase.items
        counts: dict[int, int] = {}
        for doc, starts in self.postings.get(head, {}).items():
            n = sum(
                all(p + off in self.postings.get(lex, {}).get(doc, ()) for lex, off in rest)
                for p in starts
            )
            if n:
                counts[doc] = n
        return counts

    def search(self, query: str) -> list[tuple[int, float]]:
        """Matching documents by ts_rank_cd, highest first (ties keep document order)."""
        parsed = parse_query(query)
        covers: dict[int, int] = defaultdict(int)
        for phrase in parsed.phrases:
            for doc, n in self._covers(phrase).items():
                covers[doc] += n
        matched = set(covers)
        for lexeme in parsed.negated:  # `| !x` matches every document without x
            with_x = self.postings.get(lexeme, {})
            matched |= {d for d in range(self.size) if d not in with_x}
        ranked = [(d, RANK_PER_COVER * covers.get(d, 0)) for d in sorted(matched)]
        return sorted(ranked, key=lambda r: -r[1])
