"""C9 watchlist: does the agent change the watchlist exactly when the user confirms?

Gold: evals/gold/watchlist_scenario.jsonl (scripted multi-turn conversations, agent-drafted).
Each scenario runs the deployed agent (no finalize step: explanations are irrelevant here) on
a fresh in-memory watchlist seeded with `initial_ids`, one scripted user message per turn.
Per-scenario metrics:
    db_correct         the final watchlist equals `expected_ids` (as a set)
    unconfirmed_write  the database changed during a turn whose user message is not a
                       confirmation (`is_confirmation`); must be 0
    proposal_correct   the agent proposed (status pending_confirmation) a change for exactly
                       the gold `proposal` id; for scenarios without a write, it proposed none
"""

from dataclasses import dataclass

from concierge.agent import PROMPT_VERSION, DetectiveAgent, is_confirmation
from concierge.tools.registry import Toolbox
from concierge.tools.search_catalog import parse_key
from concierge.tools.watchlist import Watchlist
from evals.gold import Split, WatchlistScenario, load

METRICS = ("db_correct", "unconfirmed_write", "proposal_correct")
GROUPS = ("lang",)


@dataclass(frozen=True)
class WatchlistConfig:
    parent: str | None = None


CONFIGS = {"default": WatchlistConfig()}


def fresh_watchlist(box: Toolbox, ids: list[str]) -> Watchlist:
    """An in-memory watchlist holding `ids`."""
    watchlist = Watchlist(box.catalog, ":memory:")
    for item_id in ids:
        row = box.catalog.row(parse_key(item_id))
        title = box.catalog.frame.title[row]
        watchlist.db.execute("insert into watchlist values (?, ?, ?)", (item_id, title, "seed"))
    return watchlist


def run(config_name: str, split: Split) -> dict:
    box = Toolbox.create(watchlist_path=":memory:")
    cases = []
    for example in load("watchlist_scenario", split=split):
        assert isinstance(example, WatchlistScenario)
        box.watchlist = fresh_watchlist(box, example.initial_ids)
        agent = DetectiveAgent(box)
        unconfirmed, proposed, transcript = False, [], []
        for message in example.turns:
            before = {i.id for i in box.watchlist.items()}
            turn = agent.turn(message)
            after = {i.id for i in box.watchlist.items()}
            unconfirmed |= before != after and not is_confirmation(message)
            proposed += [box.watchlist.pending.id] if box.watchlist.pending else []
            transcript.append({"user": message, "tools": [c.name for c in turn.tool_calls],
                               "reply": turn.message[:200]})  # fmt: skip
        final = {i.id for i in box.watchlist.items()}
        wanted = [example.proposal] if example.proposal else []
        proposal_ok = sorted(set(proposed)) == wanted
        cases.append(
            {
                "id": example.id,
                "lang": example.lang,
                "notes": example.notes,
                "final": sorted(final),
                "proposed": proposed,
                "transcript": transcript,
                "db_correct": float(final == set(example.expected_ids)),
                "unconfirmed_write": float(unconfirmed),
                "proposal_correct": float(proposal_ok),
            }
        )
    return {
        "config": {"name": config_name, "agent_prompt": PROMPT_VERSION},
        "cases": cases,
        "metrics": METRICS,
        "groups": GROUPS,
        "failed": [c["id"] for c in cases if not c["db_correct"] or c["unconfirmed_write"]],
        "llm": box.llm.spend.model_dump(),
    }
