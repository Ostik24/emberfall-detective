"""C11 hypothesis verifier: does it catch a wrong detail, and stay quiet otherwise?

Gold: evals/gold/false_memory.jsonl (a real recollection + one appended false sentence about
the cast or the year). The verifier checks the clues of BOTH texts against the true answer's
evidence (catalog document + TMDB facts).
Per-case metrics:
    caught            on the altered text, a clue naming the injected actor / year is
                      `contradicted` (recall of false-memory detection)
    named             the `false_memory` field mentions the injected detail
    flags_original    on the unaltered text, any clue is `contradicted`. An upper bound on
                      false alarms: real TREC recollections contain genuine mistakes too.
Precision proxy (summary): caught / (caught + flags on originals).
"""

import re
from dataclasses import dataclass

from concierge.clues import decompose
from concierge.finalize import evidence_text
from concierge.normalize import fold
from concierge.tools.registry import Toolbox
from concierge.verify import CHECKED_KINDS, PROMPT_VERSION, verify
from evals.gold import FalseMemory, Split, load

METRICS = ("caught", "named", "flags_original")
GROUPS = ("source",)


@dataclass(frozen=True)
class VerifyConfig:
    parent: str | None = None


CONFIGS = {"default": VerifyConfig()}


def injected_key(sentence: str) -> str:
    """The checkable part of the false sentence: the actor's name or the year."""
    year = re.search(r"\b(1[89]\d\d|20\d\d)\b", sentence)
    if year:
        return year.group(1)
    return fold(sentence.removeprefix("I think it starred ").rstrip("."))


def run(config_name: str, split: Split) -> dict:
    box = Toolbox.create()
    llm = box.llm
    cases = []
    for example in load("false_memory", split=split):
        assert isinstance(example, FalseMemory)
        evidence = evidence_text(box, example.answer_id)
        key = injected_key(example.wrong_detail)
        results = {}
        for name, text in (("altered", example.altered_query), ("original", example.query)):
            clues = decompose(llm, text)
            checked = clues.of_kind(*CHECKED_KINDS)
            verification = verify(llm, clues, example.answer_title, evidence)
            contradicted = [
                checked[c.clue].text
                for c in verification.checks
                if c.verdict == "contradicted" and 0 <= c.clue < len(checked)
            ]
            results[name] = (contradicted, verification.false_memory)
        contradicted, note = results["altered"]
        cases.append(
            {
                "id": example.id,
                "source": example.source,
                "injected": example.wrong_detail,
                "contradicted_altered": contradicted,
                "false_memory_note": note,
                "contradicted_original": results["original"][0],
                "caught": float(any(key in fold(t) for t in contradicted)),
                "named": float(key in fold(note or "")),
                "flags_original": float(bool(results["original"][0])),
            }
        )
    return {
        "config": {"name": config_name, "prompt": PROMPT_VERSION},
        "cases": cases,
        "metrics": METRICS,
        "groups": GROUPS,
        "failed": [c["id"] for c in cases if not c["caught"]],
        "llm": llm.spend.model_dump(),
    }


def summarize(cases: list[dict]) -> dict:
    caught = sum(c["caught"] for c in cases)
    alarms = sum(c["flags_original"] for c in cases)
    return {"precision_proxy": round(caught / (caught + alarms), 4) if caught + alarms else 0.0}
