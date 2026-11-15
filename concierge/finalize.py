"""After the agent responds: verify (C11) and explain (C12) its top picks.

Purpose: the agent's `finalize` hook. For each of the first MAX_EXPLAINED picks it gathers
evidence (catalog document + TMDB facts), checks the user's clues against it, and writes a
grounded explanation that includes a polite false-memory note when there is one.
Input: the agent (its toolbox and the user's messages so far), picks, the latest user text.
Output: the picks with `notes = {"verification": ..., "explanation": ...}`.
"""

from typing import TYPE_CHECKING

from concierge.clues import decompose
from concierge.explain import explain
from concierge.normalize import detect_language
from concierge.tools.get_title_details import TitleDetails
from concierge.tools.registry import Toolbox
from concierge.tools.search_catalog import parse_key
from concierge.verify import verify

if TYPE_CHECKING:
    from concierge.agent import DetectiveAgent, Pick

MAX_EXPLAINED = 3
EVIDENCE_CHARS = 2500


def evidence_text(toolbox: Toolbox, item_id: str) -> str:
    """Catalog document (when in the catalog) plus TMDB facts, as one block of text."""
    key = parse_key(item_id)
    if key is None:
        return ""
    parts = []
    row = toolbox.catalog.row(key)
    if row is not None:
        parts.append(toolbox.search.texts[row][:EVIDENCE_CHARS])
    details: TitleDetails = toolbox.details(item_id)
    if details.found:
        facts = [
            f"Year: {details.year}" if details.year else "",
            f"Directed/created by: {', '.join(details.directors)}" if details.directors else "",
            f"Cast: {', '.join(details.cast)}" if details.cast else "",
            f"Runtime: {details.runtime_minutes} min" if details.runtime_minutes else "",
            f"Overview: {details.overview}" if details.overview and row is None else "",
        ]
        parts.append("\n".join(f for f in facts if f))
    return "\n\n".join(p for p in parts if p)


def verify_and_explain(agent: "DetectiveAgent", picks: list["Pick"], text: str) -> list["Pick"]:
    """The agent's finalize hook. The recollection is everything the user wrote so far (the
    agent paraphrases in its tool calls; the user's own words are what is checked)."""
    recollection = "\n".join(agent.user_texts)
    clues = decompose(agent.llm, recollection)
    language = detect_language(text)
    for pick in picks[:MAX_EXPLAINED]:
        evidence = evidence_text(agent.toolbox, pick.id)
        verification = verify(agent.llm, clues, pick.title, evidence) if clues.clues else None
        pick.notes = {
            "verification": verification.model_dump() if verification else None,
            "explanation": explain(
                agent.llm, recollection, pick.title, evidence, clues, verification, language
            ),
        }
    return picks
