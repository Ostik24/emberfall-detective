# AGENTS.md - Emberfall Movie Detective

Instructions for any coding agent (Claude Code reads this via CLAUDE.md). Read this file fully at the start of every session, then read `WORKING_STATE.md`.

## 1. What this project is

A university NLP assignment (UCU NLP course, Assignment 1 "Build an NLP System", 15 pts) built on top of the owner's existing iOS app **Emberfall** (movie/TV discovery with semantic search).

**Emberfall Movie Detective** is a Claude tool-use agent for tip-of-the-tongue movie search: the user describes a half-remembered film in English or Ukrainian; the agent decomposes the memory into clues with confidence tags, asks at most 1-2 well-chosen questions, retrieves candidates with a fine-tuned retriever (E3) + LLM rerank, verifies hypotheses against catalog text (flagging likely false memories), falls back to TMDB/web search when the catalog has no answer, explains each pick with evidence, and manages a watchlist. The same tools are exposed as an MCP server. The retriever was already trained and evaluated in Colab (brief section 4 and 7.1): do not retrain it unless the owner asks.

- Full spec and all design decisions: `docs/PROJECT_BRIEF.md` (source of truth for scope).
- Lecture concepts the design relies on: `docs/LECTURE_NOTES.md`.
- Assignment requirements are summarised in `docs/PROJECT_BRIEF.md` section 1; lecture concepts in `docs/LECTURE_NOTES.md`.
- The existing app (baseline) lives at `/Users/ostappavlyshyn/Documents/swift-app/Emberfall`. Read it for reference; never modify it from this repo.

## 2. Deliverables (what gets graded)

1. Python package `concierge/` with the agent, the tools, and the MCP server.
2. `notebooks/Emberfall_Detective.ipynb` - the presentation notebook for Google Colab, structured by CRISP-DM phases.
3. Evals for every component in `evals/`, with results tables and error analysis.
4. High-level diagram, business logic in scope, eval results (in the notebook and `docs/REPORT.md`).
5. This agentic setup itself (AGENTS.md, skills, hooks, WORKING_STATE.md, DEV_LOG.md) - it is graded (3 pts), so keep it accurate and in use.

Grading criteria: system understanding (4), working prototype (4), evaluation of each component (4), use of the AI coding assistant (3).

## 3. Process: CRISP-DM is mandatory

Every non-trivial change follows one CRISP-DM loop and is logged:

1. **Business understanding** - which user need / success criterion does the change serve? (see brief section 2)
2. **Data understanding** - what does the data say? (EDA numbers, not guesses)
3. **Data preparation** - documents, normalization, gold sets
4. **Modeling** - the component change itself
5. **Evaluation** - run the relevant evals on the DEV split, compare with the previous run
6. **Deployment** - wire into the notebook / demo only after evaluation

After each loop, append an entry with the `crisp-dm-log` skill. A pipeline change without an eval run is not done.

Rules for experiments (from Lecture 4 "prompt evaluation process"):
- Change ONE factor at a time. Record prompt/model/config version with every result.
- Tune on `dev`. Report final numbers on `test` once, at the end. Never tune on `test`.
- Keep raw outputs of failed cases for error analysis.

## 4. Architecture rules

- All logic lives in `concierge/` as plain, typed Python modules with small, pure interfaces (they will later be ported to TypeScript Supabase Edge Functions). No logic in notebook cells beyond orchestration and plotting.
- Layout:
  - `concierge/agent.py` - the tool-use loop (max 4 tool calls per user turn), clarification policy, pending-confirmation handling for writes.
  - `concierge/tools/` - one module per tool: `search_catalog.py`, `find_title.py`, `similar_to.py`, `get_title_details.py`, `watchlist.py`, `external_lookup.py`; plus `registry.py` holding the single source of truth for tool names, descriptions and JSON schemas.
  - `concierge/retrieval/` - `bm25.py`, `dense.py`, `fusion.py` (RRF), `rerank.py`, `filters.py` (used by `search_catalog`).
  - `concierge/clues.py` (clue decomposition), `concierge/clarify.py` (20-questions clarifier), `concierge/verify.py` (hypothesis verifier), `concierge/explain.py`, `concierge/normalize.py`.
  - `concierge/mcp_server.py` - thin FastMCP wrapper that imports `registry.py`; no logic of its own.
  - Shared: `llm.py` (single Claude client), `embed.py`, `catalog.py`, `tmdb.py` (cached TMDB client), `wikimedia.py` (Wikidata/Wikipedia, data preparation only), `http.py` (the one GET-with-retries helper; errors never include query strings, which carry the TMDB key), `config.py`, `cache.py`.
  - Evals: `evals/run.py` (runner, one module per component: `evals/c4_search.py`, ...), `evals/gold.py` (gold schemas + split rule), `evals/baseline/` (the re-implemented Emberfall baseline), `evals/metrics/`.
- Tool functions are plain Python, importable and testable without the agent or MCP. Agent and MCP server must expose identical schemas (enforced by a test).
- Tool descriptions are written for the model: when to use the tool, when NOT to, argument semantics, examples. Keep the set at 7 tools; adding one needs owner approval (more tools degrade selection).
- Every component and tool returns a typed result object (pydantic) so evals can score it directly.
- Structured LLM output = a tool call with a JSON schema, made through `LLM.call_tool` in `concierge/llm.py`, which sends the tool with `strict: true` and validates the input into a pydantic model. Never parse free text with regex.
- Forced `tool_choice` is decided per model by `FORCED_TOOL_CHOICE_MODELS` in `config.py`: Haiku 5.5 accepts it (verified live 2026-11-14; it then skips thinking), Sonnet 5.5 and Opus 5.5 reject it with a 400 and get `auto` plus an instruction naming the tool. A missing or invalid tool call is a failure and is never cached.
- Responses may start with thinking blocks: never read `msg.content[0].text`; select blocks by `type` (`text`, `tool_use`). Give `max_tokens` headroom for thinking.
- Writes (`watchlist` add/remove) are never executed in the same turn they are proposed: the agent stores a pending action and executes it only after the user confirms. Enforce in code.
- `external_lookup` order: live TMDB search, then Claude web search tool (max 2 searches per call). A web-derived title is shown only after it resolves to a TMDB id. Cache all TMDB and web results on disk for reproducible evals.
- Catalog is read from local parquet (`data/catalog_extended.parquet`: the Supabase export `data/catalog.parquet` plus `data/extend_catalog.py`), not from Supabase, so the notebook is reproducible for graders.
- Embeddings only through `concierge/embed.py` (`Embedder`), which pins fp32 weights for every model (fp16 on CUDA) so E3 and base gte-small differ in one factor only.
- Retrieved catalog text (overviews, Wikipedia plots, keywords) is untrusted data: pass it inside clearly delimited data blocks, never as instructions (indirect prompt injection, Lecture 4/5).

## 5. Models (fixed decisions, see brief section 5)

| Use | Model |
|---|---|
| Agent loop, clue decomposition, clarifier, verifier, listwise rerank | `claude-haiku-5-5` |
| Rerank comparison, user-facing explanations | `claude-sonnet-5-5` |
| LLM-as-judge in evals | `claude-opus-5-5` (test split), `claude-sonnet-5-5` allowed on dev |
| Embeddings | fine-tuned **E3** (gte-small, 384-d; Drive `emberfall_pilot/v2/models/E3`) as primary; base `gte-small` and OpenAI `text-embedding-3-small` as comparisons |

Do not introduce other providers or models without asking. Model IDs live only in `concierge/config.py`.

## 6. Secrets and cost (hard rules)

- NEVER hardcode keys. NEVER write `ANTHROPIC_API_KEY` to shell profiles or export it globally: it would switch the owner's Claude Code from his subscription to API billing.
- Keys are read in this order: Colab `userdata.get(...)`, then `.env` via python-dotenv. Variable names: `EMBERFALL_CLAUDE_KEY`, `OPENAI_API_KEY`, `TMDB_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`. Pass the Claude key explicitly: `Anthropic(api_key=...)`.
- Web search is billed per search on top of tokens: cap searches per call, cache results, and report web-search count and cost in every eval result.
- `.env` is gitignored. Never print keys.
- Budget: $100/month API credits. All LLM calls go through `concierge/llm.py`, which: caches responses on disk keyed by (model, prompt version, input hash); uses prompt caching for static system prompts; logs token usage. Large eval runs use the Message Batches API.
- Before any run that makes more than ~200 LLM calls, print an estimate of calls and ask the owner.

## 7. Code conventions

- Python 3.11, type hints everywhere, pydantic v2 models, `ruff` for lint/format, `pytest` for tests. Environment: `uv sync`, then `uv run pytest`.
- Run `ruff format` / `ruff check --fix` only on the paths you changed (e.g. `concierge evals data tests`), never on `.`. `notebooks/experiments/` holds the Colab runs as they were executed and must never be rewritten (it is excluded in `pyproject.toml`).
- Small functions, docstrings stating input/output. No dead code, no commented-out experiments (record them in DEV_LOG instead).
- Deterministic where possible: fixed random seeds for splits; reproducibility of LLM steps comes from the disk cache (the current SDK/models reject the `temperature` argument, do not pass it).
- Tests: unit tests for metrics (hand-computed examples, e.g. the Lecture 3 P@k / AP / MRR exercises) and for pure functions. LLM calls are mocked in unit tests.
- Every eval script writes `evals/results/<component>/<timestamp>_<config>.json` with config, metrics, and failed case ids.

## 8. Working-state discipline (context engineering, Lecture 4)

- `WORKING_STATE.md` is the resume point: goal, constraints, current numbers, open questions, next step. Update it at the end of every session and after each eval run. Keep it short; move history to DEV_LOG.
- `DEV_LOG.md` records where the agent (you) went wrong and what instruction changed as a result. The owner must reflect on this in the video, so log honestly: mistakes, corrections, instruction changes.
- When context gets long, summarize into WORKING_STATE.md rather than re-reading large files.

## 9. Do / Don't

Do:
- Ask before expanding scope beyond "must-have" in the brief.
- Prefer the simplest thing that can be evaluated.
- Explain each component's purpose, input, output in its module docstring (the owner must explain them in the video).

Don't:
- Don't tune on the test split.
- Don't fabricate metrics, datasets, or citations. If a dataset cannot be downloaded, say so.
- Don't touch the Emberfall iOS repo or the owner's Supabase project (read-only export only, run by the owner).
- Don't use the em dash character in any docs or report text; use a plain hyphen.
