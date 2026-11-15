"""Gold sets: schema, split rule and loading.

Purpose: one definition of every gold file in `evals/gold/<set>.jsonl`, shared by the builders,
`python -m evals.validate_gold` and the eval runners (`.claude/skills/add-gold-example`).
Input: a set name. Output: validated pydantic examples.
A new set is one model plus one entry in `SCHEMAS`.
"""

import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from concierge.config import REPO_ROOT

GOLD_DIR = REPO_ROOT / "evals" / "gold"
SPLIT_SEED = "emberfall-2026"
DEV_SHARE = 0.5

Split = Literal["dev", "test"]


def assign_split(example_id: str) -> Split:
    """Deterministic split from a hash of the id: no example ever moves between splits."""
    digest = hashlib.sha256(f"{SPLIT_SEED}:{example_id}".encode()).hexdigest()
    return "dev" if int(digest[:8], 16) / 0x1_0000_0000 < DEV_SHARE else "test"


class GoldExample(BaseModel):
    """Fields every gold example has."""

    model_config = ConfigDict(extra="forbid")

    id: str
    lang: Literal["en", "uk"]
    # dataset:<name> | handwritten | llm_draft (unchecked) | llm_draft_checked | agent_drafted
    # | synthetic:<x>
    source: str
    split: Split
    notes: str = ""

    def split_key(self) -> str:
        """What the split hash is computed from; examples sharing it share a split."""
        return self.id


class KnownItem(GoldExample):
    """A tip-of-the-tongue query and the one title it describes."""

    query: str
    answer_item_id: int
    answer_item_type: Literal["movie", "tv"]
    answer_title: str
    in_catalog: bool  # against data/catalog_extended.parquet
    wikidata_id: str | None = None

    def split_key(self) -> str:
        """A translation shares the split of its English original."""
        return self.id.removeprefix("uk-")


class TitleLookup(GoldExample):
    """A title as a user might type it; all variants of one title share a split."""

    query: str
    variant: Literal["clean", "typo", "uk"]
    answer_item_id: int
    answer_item_type: Literal["movie", "tv"]
    answer_title: str

    def split_key(self) -> str:
        return f"{self.answer_item_type}-{self.answer_item_id}"


class DetailsFacts(GoldExample):
    """Facts about a catalog film from Wikidata, independent of TMDB (gold for C8)."""

    item_type: Literal["movie", "tv"]
    item_id: int
    title: str
    wikidata_id: str
    directors: list[str]
    runtime_minutes: int | None
    year: int | None


class SimilarPairs(GoldExample):
    """An anchor title and the catalog titles TMDB recommends for it (silver labels)."""

    anchor_id: str  # "movie/603"
    anchor_title: str
    expected_ids: list[str]


Action = Literal[
    "search_catalog", "find_title", "watchlist", "external_lookup", "ask_user", "respond"
]


class AgentDecision(GoldExample):
    """A user message and the agent's expected first move. `ask_user` = respond with a
    question and options; `respond` = answer without any tool. `expected_filters` lists the
    search_catalog filters the user STATED; any other filter the agent sets is a hallucinated
    constraint."""

    message: str
    expected_action: Action
    expected_filters: dict[str, Any] = {}


class FalseMemory(GoldExample):
    """A real recollection and a copy with ONE detail replaced by a wrong one (C11)."""

    query: str
    altered_query: str
    wrong_detail: str
    true_detail: str
    answer_id: str  # "movie/603"
    answer_title: str

    def split_key(self) -> str:
        """Same split as the known_item query it was made from."""
        return self.id.removeprefix("fm-")


class WatchlistScenario(GoldExample):
    """A scripted multi-turn conversation about the watchlist and the database state it must
    leave. `proposal` is the id the agent should propose to add/remove (None: no write)."""

    turns: list[str]
    initial_ids: list[str]
    expected_ids: list[str]
    proposal: str | None


SCHEMAS: dict[str, type[GoldExample]] = {
    "known_item": KnownItem,
    "known_item_uk": KnownItem,
    "title_lookup": TitleLookup,
    "details": DetailsFacts,
    "similar_pairs": SimilarPairs,
    "agent_decision": AgentDecision,
    "false_memory": FalseMemory,
    "watchlist_scenario": WatchlistScenario,
}


def gold_path(name: str) -> Path:
    return GOLD_DIR / f"{name}.jsonl"


def load(name: str, split: Split | None = None) -> list[GoldExample]:
    """Examples of set `name`, optionally one split only."""
    model = SCHEMAS[name]
    lines = gold_path(name).read_text().splitlines()
    examples = [model.model_validate_json(line) for line in lines if line.strip()]
    return [e for e in examples if split is None or e.split == split]


def write(name: str, examples: list[GoldExample]) -> Path:
    """Write set `name`, one JSON object per line, sorted by id for stable diffs."""
    path = gold_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(examples, key=lambda e: e.id)
    path.write_text("".join(e.model_dump_json() + "\n" for e in rows))
    return path
