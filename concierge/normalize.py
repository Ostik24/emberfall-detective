"""C0 normalize: clean user text, tell English from Ukrainian, fold titles for matching.

Purpose: the first step on every user message and every title comparison.
- `normalize_text`: Unicode NFC, typographic quotes and dashes to ASCII, whitespace collapsed.
- `detect_language`: "uk" when most letters are Cyrillic, else "en" (the project's two
  languages; any other Cyrillic text is also routed as "uk").
- `fold`: a matching key for titles: case-folded, accents stripped (Latin only), "&" -> "and",
  punctuation removed, so "Amélie" == "amelie" and "Spider-Man" == "spider man".
Input: a string. Output: a string, or "en" / "uk".
"""

import re
import unicodedata
from typing import Literal

Language = Literal["en", "uk"]
_PUNCT = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
                        "\u2013": "-", "\u2014": "-", "\u00a0": " "})  # fmt: skip


def normalize_text(text: str) -> str:
    """NFC, ASCII quotes/dashes, single spaces."""
    return " ".join(unicodedata.normalize("NFC", text).translate(_PUNCT).split())


def _is_cyrillic(ch: str) -> bool:
    return "Ѐ" <= ch <= "ӿ"


def detect_language(text: str) -> Language:
    """'uk' if at least half of the letters are Cyrillic, else 'en'."""
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "en"
    return "uk" if sum(map(_is_cyrillic, letters)) * 2 >= len(letters) else "en"


def fold(title: str) -> str:
    """Matching key: lower case, Latin accents removed, '&' -> 'and', punctuation -> space."""
    text = normalize_text(title).casefold().replace("&", " and ")
    chars = []
    for ch in unicodedata.normalize("NFD", text):
        if unicodedata.combining(ch) and chars and not _is_cyrillic(chars[-1]):
            continue  # drop Latin accents, keep Cyrillic й/ї intact
        chars.append(ch)
    text = unicodedata.normalize("NFC", "".join(chars))
    return " ".join(re.sub(r"[^\w\s]|_", " ", text).split())
