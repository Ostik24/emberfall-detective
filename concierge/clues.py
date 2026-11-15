"""C2 clue decomposition: split a recollection into tagged clues.

Purpose: tip-of-the-tongue memories mix film facts with noise ("I watched it at my grandma's
in 2008"), hedges ("I think it was in Italy?"), exclusions ("not the remake") and
comparisons ("like Jaws but in the desert"). Retrieval should search the film facts, and the
verifier (C11) needs each fact separately to check it against a candidate.
Kinds:
    certain         stated as fact about the film
    hedged          about the film, but the user is unsure
    episodic_noise  about the user's life (where/when/with whom they watched), not the film
    exclusion       what the film is NOT
    comparison      relative to another title
Input: the recollection (EN or UK). Output: `Clues` (clues in order + the search text built
from them in code: certain, hedged and comparison clues, in English, joined).
"""

from typing import Literal

from pydantic import BaseModel

from concierge.config import CLUES_MODEL
from concierge.llm import LLM

PROMPT_VERSION = "clues-v1"
SEARCHED_KINDS = ("certain", "hedged", "comparison")

ClueKind = Literal["certain", "hedged", "episodic_noise", "exclusion", "comparison"]

SYSTEM = """You break a person's recollection of a film or TV show into separate clues.

The person is trying to remember a title. Split everything they wrote into short, \
self-contained clues, one fact each, and tag each clue:
- certain: a detail about the film or show stated without doubt ("a girl finds a dragon egg").
- hedged: a detail about the film or show the person is unsure about ("I think it was set in \
Italy", "maybe the 90s", "the actor might have been Bill Paxton").
- episodic_noise: about the person's own life or viewing context, not the content: where, when \
or with whom they watched it, how they feel about forgetting it, requests for help \
("saw it on TV at my grandma's", "it's been bugging me for years").
- exclusion: what the title is NOT ("it's not Jurassic Park", "not animated").
- comparison: a relation to another title ("like Home Alone but with a dog").

Rules:
- Keep the person's details; do not add facts or guess the title.
- Write every clue in English, even when the recollection is in another language.
- Viewing time ("I saw it in the early 2000s") is episodic_noise, but a stated release era \
("an 80s movie") is a clue about the film (certain or hedged).
- Plot, characters, scenes, setting, people, era, genre, tone and visual details are all clues \
about the film."""

CLUES_TOOL = {
    "name": "submit_clues",
    "description": "Submit the clues of the recollection, in the order they appear.",
    "input_schema": {
        "type": "object",
        "properties": {
            "clues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "One fact, in English."},
                        "kind": {
                            "type": "string",
                            "enum": [
                                "certain",
                                "hedged",
                                "episodic_noise",
                                "exclusion",
                                "comparison",
                            ],
                        },
                    },
                    "required": ["text", "kind"],
                },
            }
        },
        "required": ["clues"],
    },
}


class Clue(BaseModel):
    text: str
    kind: ClueKind


class Clues(BaseModel):
    clues: list[Clue]

    def search_text(self) -> str:
        """The clues worth searching for, joined; empty if there are none."""
        return " ".join(c.text.rstrip(".") + "." for c in self.clues if c.kind in SEARCHED_KINDS)

    def of_kind(self, *kinds: ClueKind) -> list[Clue]:
        return [c for c in self.clues if c.kind in kinds]


def decompose(llm: LLM, recollection: str, model: str = CLUES_MODEL) -> Clues:
    """Tagged clues of `recollection` (one Haiku call, cached)."""
    return llm.call_tool(
        model=model,
        prompt_version=PROMPT_VERSION,
        system=SYSTEM,
        messages=[{"role": "user", "content": f"<recollection>\n{recollection}\n</recollection>"}],
        tool=CLUES_TOOL,
        output_model=Clues,
        max_tokens=4000,
    )
