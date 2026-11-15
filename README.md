# Emberfall Movie Detective

A Claude tool-use agent for tip-of-the-tongue film search (English and Ukrainian), built on
top of the Emberfall iOS app's search as the baseline. UCU NLP, Assignment 1.

- Presentation notebook (CRISP-DM): `notebooks/Emberfall_Detective.ipynb`
- Report: `docs/REPORT.md`; scope and decisions: `docs/PROJECT_BRIEF.md`
- How the coding agent was run: `AGENTS.md`, `.claude/skills/`, `WORKING_STATE.md`, `DEV_LOG.md`

## Data flow

```mermaid
flowchart TD
    U["User message (EN / UK)"] --> N["C0 normalize<br/>(local code)"]
    N --> A{{"C1 agent loop<br/>Claude Haiku 5.5<br/>max 4 tool calls / turn"}}
    A -->|search_catalog| S["C4 dense search<br/>E3 model, local, top 100<br/>filters applied first"]
    S --> R{{"C5 rerank top 20<br/>Claude Haiku 5.5"}}
    A -->|find_title| F["C6 fuzzy title match<br/>(local)"]
    F -.->|no catalog match / Ukrainian| T[("TMDB API")]
    A -->|similar_to| SIM["C7 E3 neighbours<br/>(local)"]
    A -->|get_title_details| T
    A -->|watchlist| W[("C9 SQLite<br/>write only after the user confirms")]
    A -->|external_lookup| T
    A -->|external_lookup| WEB{{"C10 Claude web search<br/>max 2 searches"}}
    WEB --> T
    R --> A
    A -->|respond: picks| G["Hallucination guard<br/>picks = ids a tool returned"]
    G --> C2{{"C2 clue decomposition<br/>Claude Haiku 5.5"}}
    C2 --> V{{"C11 verifier<br/>Claude Haiku 5.5"}}
    T --> V
    V --> E{{"C12 explanation<br/>Claude Sonnet 5.5"}}
    E --> OUT["Reply: picks + explanations<br/>+ false-memory note"]
    CL{{"C3 clarifier<br/>Claude Haiku 5.5"}} -.->|evaluated with a user simulator| A
    MCP["MCP server (stdio)<br/>same 6 data tools"] -.-> S
```

Legend: hexagons are Claude model calls, cylinders are external services or storage,
rectangles are local code / local models (E3 runs on the machine).

## Layout

| Path | What |
|---|---|
| `concierge/agent.py` | the agent loop (Haiku 5.5, 7 tools, rules enforced in code) |
| `concierge/tools/` | the tools; `registry.py` is the single source of names and schemas |
| `concierge/retrieval/` | dense retrieval (E3) and the Haiku reranker |
| `concierge/{clues,clarify,verify,explain,normalize}.py` | C2, C3, C11, C12, C0 |
| `concierge/mcp_server.py` | the data tools over MCP (stdio) |
| `data/` | catalog export / extension / EDA and gold-set builders |
| `evals/` | one module per component, `run.py` (runner), `gold/`, `results/`, `baseline/` |

## Run

```
uv sync --all-groups
uv run pytest                                   # unit tests, no network
uv run python -m concierge.chat                 # the demo in the terminal, with a live trace
uv run --with gradio python -m concierge.chat --web   # the same in the browser
uv run python -m evals.run --component c4 --config tool_whole   # one eval (dev split)
claude mcp add emberfall -- uv --directory "$PWD" run python -m concierge.mcp_server
```

Keys (never exported globally): `EMBERFALL_CLAUDE_KEY`, `TMDB_API_KEY` in `.env` or Colab
Secrets. Large local assets, not in git: `data/catalog_extended.parquet` (built by
`data/export_catalog.py` + `data/extend_catalog.py`), `models/E3` (the fine-tuned retriever),
`.cache/` (LLM and TMDB responses; with it, every eval and the notebook replay without keys).
