"""Build the `agent_decision` set for C1 (the agent's first move on a user message).

    uv run python data/build_agent_decisions.py

Source: data/raw/agent_decisions.jsonl, written by the coding agent (source `agent_drafted`,
not yet checked by the owner): no public dataset labels tool choice for this tool set.
Classes: search_catalog, find_title, watchlist, ask_user (respond with a question), respond
(no tool). Named titles expect find_title first, also for "like X", facts and watchlist
adds, because every later tool needs the title's id. A filter is listed only when the user
states it as a constraint; hedged details ("I think from the 2010s") are not filters.
Grading rules live in evals/c1_agent.py.
Output: evals/gold/agent_decision.jsonl.
"""

import json
from collections import Counter

from concierge.config import DATA_DIR
from evals.gold import AgentDecision, assign_split, write

SOURCE = DATA_DIR / "raw" / "agent_decisions.jsonl"


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    examples = [
        AgentDecision(
            id=f"c1-{row['key']}",
            lang=row["lang"],
            source="agent_drafted",
            split=assign_split(f"c1-{row['key']}"),
            message=row["message"],
            expected_action=row["action"],
            expected_filters=row["filters"],
            notes=row.get("note", ""),
        )
        for row in rows
    ]
    path = write("agent_decision", examples)
    print(f"wrote {path}: {len(examples)} examples")
    print(dict(Counter((e.split, e.expected_action) for e in examples)))


if __name__ == "__main__":
    main()
