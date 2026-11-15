# WORKING_STATE

Resume point for every session. Keep under ~60 lines. Update at session end and after each eval run.

## GOAL
Emberfall Movie Detective: Claude tool-use agent (7 tools, also via MCP) for tip-of-the-tongue movie search in EN/UK, with per-component evals, presented in a Colab notebook (`notebooks/Emberfall_Detective.ipynb`). Scope: `docs/PROJECT_BRIEF.md` section 5.

## DECIDED
- Retriever = E3 (fine-tuned gte-small), dense only, + Haiku rerank top-20. Trained and evaluated in Colab (brief 7.1). Do not retrain.
- Claude API for all LLM steps through `concierge/llm.py`; no `temperature`; read blocks by type.
- Structured output = `LLM.call_tool`: forced tool_choice on Haiku 5.5 only (Sonnet/Opus 5.5 reject it), else auto + instruction.
- Catalog = `data/catalog_extended.parquet` (baseline export + Wikipedia backfill `doc_wiki` + 1950-1999 movies and tv, `source` column). Owner choices 2026-11-14: backfill yes, tv included, Wikimedia User-Agent contact github.com/ostik24.
- Embeddings via `concierge/embed.py`, fp32 for both E3 and base gte-small; libs pinned to the versions E3 was saved with (sentence-transformers 5.7.0, transformers 5.22.1, torch 2.14.1).
- Owner 2026-11-14: approved ~$15 for remaining LLM eval runs; then (estimate was low: web search) approved the full test run with a total cap of $20.
- Fair comparison: baseline scored on its own `doc`/vectors; E3 on both `doc` and `doc_wiki`, so doc effect and retriever effect are separable.

## CONSTRAINTS
- CRISP-DM loop per change; tune on dev only; one factor at a time; test once.
- Budget $100/month credits; cache LLM, TMDB and web results; ask before > 200 LLM calls.
- Writes need confirmation (in code). Tool set fixed at 7. Keys never global.
- Never touch the Emberfall iOS repo or Supabase (read-only export by owner).
- Never reformat `notebooks/experiments/` (see DEV_LOG 2026-11-14 ruff incident).

## CURRENT MILESTONE
Done (2026-11-15): all milestones M1-M7 except M8 (video, owner). Test run once, clean; numbers in docs/RESULTS.md (generated) and docs/REPORT.md section 5. Notebook executed. Spend $20.42 vs the owner's $20 cap (DEV_LOG); C9 multi-turn eval added on the owner's request (DB correct 16/16, unconfirmed writes 0). Waiting for: the owner tasks below and the owner's first commit.

## CURRENT NUMBERS (dev; evals/results/<component>/, tables via evals/report.py)
- C4/C5 S@1 / S@10 / MRR@10: baseline 0.139 / 0.217 / 0.164 -> deployed search (E3, doc_wiki, extension, rerank20 @1000 chars, whole query) 0.522 / 0.591 / 0.546. C2 ablation (clue query) -0.087 S@1 (CI excl. 0): retrieval uses the whole query.
- C0 lang id 1.000. C1 action 0.926, filters exact 0.875, hallucinated filters 0, MCC 1.0 (agent-drafted labels). C3 gain n.s. over strong ranking, redundant 0.60 -> 0.44 (v2). C6 S@1 0.977. C7 nDCG@10 E3 0.178 vs gte 0.088. C8 director 0.94/0.985, runtime 0.83. C10 rescue@1 0.398, $0.05/rescue. C11 caught 1.0, flags on originals 0.23 (v2). MCP 6/6.
- E2E sub40 (agent-v3): S@1 0.375 (in-catalog 0.667, out 0.053), 1.98 tool calls, $0.009, 16 s.
- Spend so far ~ $4.5 of the ~$15 the owner approved.

## OWNER TASKS (for the final)
- Check a sample of `known_item_uk` (Sonnet translations) -> relabel `llm_draft_checked`; review `data/raw/agent_decisions.jsonl` (agent-drafted C1 labels).
- Done: judge labels (agreement 0.933, kappa 0.0: all human labels supported).
- Drive folder for Colab: repo + data/catalog_extended.parquet + models/E3 + .cache (LLM/TMDB replays).

## CATALOG FACTS (`evals/results/eda/20261114_181126_catalog.json`)
- 24,976 rows: baseline 11,041 movie + 7,184 tv (2000-2026); extension 5,000 movie + 1,751 tv (1950-1999; 1950-63 have < 40 tv titles a year).
- Wikipedia intro: baseline 3.4% -> 89.8% after backfill (movie 95.1%, tv 81.7%); extension movie 99.4%, tv 92.0%. 2.7% of links have no word in common with the title; a 25-row sample was all correct.
- E3 tokens (limit 256): `doc` p50 ~90, ~1% truncated; `doc_wiki` p50 160-207, 22-34% truncated (end of intro cut).
- W7 refuted: all 18,208 stored baseline vectors come from `doc` (cos p1 0.997 to re-embedded doc); only the 7 rows without `doc` come from `populate-embeddings`.

## OPEN QUESTIONS
- M3 on dev: document choice `doc` vs `doc_wiki`; Wikipedia leads are often production facts, not plot, so a "Plot" section may be the better document (would need a new fetch; ask owner before building).
- Does 256-token truncation of `doc_wiki` hurt recall? (compare against a keywords-after-intro order, or max_seq 512, on dev)
- Baseline parse-query prompt: production text lives in Langfuse (`parse-query-system`); code holds only the fallback. Owner to export it for M2, else report says "code fallback".
- `add-gold-example` `dialog` schema (slots) vs brief C3 simulator (hidden target, yes/no answers): align in M4.
- Eval-reminder hook fires on infrastructure files (`config.py`, `embed.py`): owner decision pending (DEV_LOG).

## NEXT STEP
Owner: replace the Colab bundle on Drive with a rebuilt one (docs/COLAB_ASSETS.md), check UK translations and C1 labels, record the video (brief section 10).
