"""C0 normalize."""

import pytest

from concierge.normalize import detect_language, fold, normalize_text


def test_normalize_text_quotes_dashes_spaces():
    assert normalize_text("  “It’s” —  a film ") == '"It\'s" - a film'


@pytest.mark.parametrize(
    ("text", "lang"),
    [
        ("a film about storm chasers", "en"),
        ("фільм про мисливців за торнадо", "uk"),
        ("фільм Twister з Біллом Пекстоном", "uk"),
        ("Twister (Смерч)", "en"),
        ("", "en"),
        ("1996", "en"),
    ],
)
def test_language_is_the_majority_script(text, lang):
    assert detect_language(text) == lang


@pytest.mark.parametrize(
    ("title", "key"),
    [
        ("Amélie", "amelie"),
        ("Spider-Man: No Way Home", "spider man no way home"),
        ("Fast & Furious", "fast and furious"),
        ("Ї та й", "ї та й"),  # Cyrillic letters keep their marks
        ("  WALL·E ", "wall e"),
    ],
)
def test_fold(title, key):
    assert fold(title) == key
