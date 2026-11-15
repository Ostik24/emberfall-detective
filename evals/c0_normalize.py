"""C0 normalize: language identification accuracy (no LLM).

Gold: every query of known_item (en), known_item_uk (uk), title_lookup (en / uk variants) and
agent_decision (en / uk) - their `lang` labels. Metric: `lang_correct` per query, grouped by
set and by gold language. Short Ukrainian titles and mixed-script messages are the hard cases.
"""

from dataclasses import dataclass

from concierge.normalize import detect_language, normalize_text
from evals.gold import Split, load

METRICS = ("lang_correct",)
GROUPS = ("set", "lang")
SETS = {"known_item": "query", "known_item_uk": "query", "title_lookup": "query",
        "agent_decision": "message"}  # fmt: skip


@dataclass(frozen=True)
class NormalizeConfig:
    parent: str | None = None


CONFIGS = {"default": NormalizeConfig()}


def run(config_name: str, split: Split) -> dict:
    cases = []
    for name, field in SETS.items():
        for example in load(name, split=split):
            text = normalize_text(getattr(example, field))
            predicted = detect_language(text)
            cases.append(
                {
                    "id": f"{name}:{example.id}",
                    "set": name,
                    "lang": example.lang,
                    "predicted": predicted,
                    "text": text[:120],
                    "lang_correct": float(predicted == example.lang),
                }
            )
    return {
        "config": {"name": config_name},
        "cases": cases,
        "metrics": METRICS,
        "groups": GROUPS,
        "failed": [c["id"] for c in cases if not c["lang_correct"]],
    }
