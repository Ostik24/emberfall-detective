---
name: add-gold-example
description: Add or draft labeled gold examples for an Emberfall Movie Detective eval set (known-item, parse/router, dialog, explanation, tools) in the correct schema, with validation and split assignment. Use when creating or extending evaluation data.
---

# Add gold examples

Gold files live in `evals/gold/<set>.jsonl`, one JSON object per line. Every example has `id`, `lang` (`en`|`uk`), `source` (`dataset:<name>` | `handwritten` | `llm_draft_checked` | `agent_drafted`), `split` (`dev`|`test`), `notes`. Schemas are code: `evals/gold.py` (`SCHEMAS`); `agent_drafted` = written by the coding agent, not yet checked by the owner (say so in the report).

## Schemas (extra fields per set)
- `known_item`: `query`, `answer_item_id` (TMDB id), `answer_item_type`, `answer_title`, `in_catalog` (bool, computed against `data/catalog_extended.parquet`). In-catalog items feed C4/C5/E2E; out-of-catalog items feed the C10 fallback eval. Drop only with a reason (`ambiguous`, `no_tmdb_match`).
- `agent_decision`: `message`, `expected_action` (`ask_user`, `respond` or a tool name from `concierge/tools/registry.py`), `expected_filters` (only filters the user stated; anything else the agent adds counts as a hallucinated constraint).
- `dialog`: `opening_message`, `hidden_profile` (the slots the simulated user knows), `required_slots`.
- `explanation`: `item_id`, `query`, `explanation`, `claims` [{`text`, `supported`: bool}] - human labels, used to validate the judge.
- `tools`: `conversation`, `expected_tool_sequence` [{`tool`, `args`}], `requires_confirmation`, `expected_db_state` (for watchlist scenarios).
- `title_lookup`: `query` (clean / typo / UK transliteration, set `variant`), `answer_item_id`.
- `similar_pairs`: `anchor_item_id`, `filters`, `expected_item_ids` (from TMDB recommendations, manually checked).

## Rules
1. Check existing public datasets before generating synthetic data, and record which ones were checked in DEV_LOG.
2. LLM-drafted examples are `llm_draft_checked` only after the owner (or a stated manual check) verified them. Never mark unchecked drafts as gold.
3. Cover hard cases on purpose: negation ("nothing scary"), mixed intents, title-like vs description-like ambiguity, decades/ranges, people names, UK queries, typos.
4. Keep class balance visible: after adding, print counts per route / lang / split.
5. Split assignment: deterministic hash of `id` with the fixed seed, stratified by set type and lang; never move an example from test to dev after results were seen.
6. Validate with `uv run python -m evals.validate_gold <set>` (schema + split rule + catalog membership + duplicate check) before committing.
