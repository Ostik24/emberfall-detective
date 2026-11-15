"""Build the `watchlist_scenario` set for C9 (multi-turn watchlist conversations).

    uv run python data/build_watchlist_scenarios.py

Source: data/raw/watchlist_scenarios.jsonl, written by the coding agent (`agent_drafted`):
adds and removes that are confirmed, declined, waived in the same message ("just do it"),
or interrupted by a new request; read-only listings; a removal of a title not on the list;
English and Ukrainian.
Output: evals/gold/watchlist_scenario.jsonl.
"""

import json
from collections import Counter

from concierge.config import DATA_DIR
from evals.gold import WatchlistScenario, assign_split, write

SOURCE = DATA_DIR / "raw" / "watchlist_scenarios.jsonl"


def main() -> None:
    rows = [json.loads(line) for line in SOURCE.read_text().splitlines() if line.strip()]
    examples = [
        WatchlistScenario(
            id=f"c9-{row['key']}",
            lang=row["lang"],
            source="agent_drafted",
            split=assign_split(f"c9-{row['key']}"),
            notes=row["note"],
            turns=row["turns"],
            initial_ids=row["initial"],
            expected_ids=row["expected"],
            proposal=row["proposal"],
        )
        for row in rows
    ]
    path = write("watchlist_scenario", examples)
    print(f"wrote {path}: {len(examples)} examples")
    print(dict(Counter((e.split, e.lang) for e in examples)))


if __name__ == "__main__":
    main()
