"""Project configuration: model ids, prices, paths and secret loading.

Purpose: the only place where model ids live (AGENTS.md section 5) and where secrets are read
(AGENTS.md section 6).
Input: a secret name. Output: its value from Colab Secrets, else from the repo's `.env`.
"""

from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
CATALOG_PATH = DATA_DIR / "catalog.parquet"
CATALOG_EXTENDED_PATH = DATA_DIR / "catalog_extended.parquet"
CACHE_DIR = REPO_ROOT / ".cache"
LLM_CACHE_DIR = CACHE_DIR / "llm"
LLM_USAGE_LOG = CACHE_DIR / "llm_usage.jsonl"
TMDB_CACHE_DIR = CACHE_DIR / "tmdb"
WIKIMEDIA_CACHE_DIR = CACHE_DIR / "wikimedia"
ENV_FILE = REPO_ROOT / ".env"

# Embedding models (brief section 4). E3 is a local SentenceTransformer directory copied from
# Drive `emberfall_pilot/v2/models/E3`; base gte-small, the comparison, comes from the HF Hub.
E3_MODEL = REPO_ROOT / "models" / "E3"
GTE_SMALL = "thenlper/gte-small"
EMBED_DIM = 384

HAIKU = "claude-haiku-5-5"
SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"

# Model per role, as fixed in docs/PROJECT_BRIEF.md section 6.
AGENT_MODEL = HAIKU
CLUES_MODEL = HAIKU
CLARIFIER_MODEL = HAIKU
VERIFIER_MODEL = HAIKU
RERANK_MODEL = HAIKU
RERANK_COMPARISON_MODEL = SONNET
EXPLAIN_MODEL = SONNET
JUDGE_MODEL_DEV = SONNET
JUDGE_MODEL_TEST = OPUS

CLAUDE_KEY_NAME = "EMBERFALL_CLAUDE_KEY"

# Sonnet 5.5 and Opus 5.5 reject tool_choice {"type": "tool"} with a 400; Haiku 5.5 accepts it.
FORCED_TOOL_CHOICE_MODELS = frozenset({HAIKU})


@dataclass(frozen=True)
class ModelPrice:
    """USD per million tokens. Cache writes (5 min TTL) cost 1.25x input, cache reads 0.1x."""

    input: float
    output: float


# First-party API prices as of 2026-11-03. Haiku 5.5 charges 5x above 100K prompt tokens;
# that tier is not modelled because no prompt in this project comes near it.
PRICES: dict[str, ModelPrice] = {
    HAIKU: ModelPrice(input=0.10, output=0.50),
    SONNET: ModelPrice(input=2.00, output=10.00),
    OPUS: ModelPrice(input=4.00, output=20.00),
}


# Claude web search tool: $10 per 1,000 searches, on top of tokens.
WEB_SEARCH_USD = 0.01
WEB_SEARCH_TOOL = "web_search_20250305"  # Haiku 5.5 has no dynamic-filtering version


class MissingSecretError(RuntimeError):
    """A required secret is in neither Colab Secrets nor `.env`."""


_SECRETS: dict[str, str] = {}


def get_secret(name: str) -> str:
    """Return the secret `name`: Colab Secrets first, then `.env`. Raise if neither has it.
    A value found once is remembered for the process, so keys read in the notebook's main
    thread are reused by worker threads (Gradio) without asking Colab again."""
    if name not in _SECRETS:
        value = _colab_secret(name) or dotenv_values(ENV_FILE).get(name)
        if not value:
            raise MissingSecretError(f"{name} is not set in Colab Secrets or {ENV_FILE}")
        _SECRETS[name] = value
    return _SECRETS[name]


def _colab_secret(name: str) -> str | None:
    """Colab Secrets lookup; None outside Colab or when the secret is not defined there."""
    try:
        from google.colab import userdata
    except ImportError:
        return None
    try:
        return userdata.get(name)
    except userdata.SecretNotFoundError:
        return None
