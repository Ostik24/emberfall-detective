"""Build (and optionally execute) notebooks/Emberfall_Detective.ipynb.

    uv run --group notebook python notebooks/build_notebook.py           # build only
    uv run --group notebook python notebooks/build_notebook.py --execute # build + run cells

The notebook is the presentation (brief section 9, CRISP-DM order). Cells only orchestrate and
plot: tables come from evals/report.py (saved result files), components from concierge/.
Keeping the source here, as code, makes the notebook reviewable and rebuildable.
"""

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK = Path(__file__).parent / "Emberfall_Detective.ipynb"


def md(text: str):
    return nbformat.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbformat.v4.new_code_cell(text.strip())


CELLS = [
    md("""
# Emberfall Movie Detective

A Claude tool-use agent for **tip-of-the-tongue** film search: you describe a half-remembered
film (English or Ukrainian), it finds the title, explains the match, points out details you
probably misremember, and can save it to a watchlist. The same tools are served over MCP.

UCU NLP course, Assignment 1. Built on the owner's iOS app **Emberfall**, whose search
pipeline is the baseline. Sections follow CRISP-DM. All numbers come from saved evaluation
files in `evals/results/`; LLM calls replay from the disk cache, so this notebook runs
without an API key.
"""),
    md("## 0. Setup"),
    code("""
import os, sys
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    # One zip holds the repository, the catalog, the E3 model and the caches (docs/COLAB_ASSETS.md).
    # It is downloaded from a shared Google Drive link: no Drive access is needed.
    BUNDLE_ID = "1OsDhPC5UjxfKSLCpwVF5rPZsh3sjGyA8"
    REPO = "/content/emberfall_detective"
    if not os.path.exists(REPO):
        !gdown -q {BUNDLE_ID} -O /content/bundle.zip
        !unzip -q /content/bundle.zip -d /content
    %cd {REPO}
    import importlib.metadata as md
    loaded = {m: sys.modules[m].__version__ for m in ("numpy", "pandas", "torch") if m in sys.modules}
    %pip install -q -e .
    changed = [m for m, v in loaded.items() if md.version(m) != v]
    if changed:  # Colab had older versions in memory: a restart loads the installed ones
        raise SystemExit(f"Installed {changed}: Runtime -> Restart session, then Run all again "
                         "(the second time this cell skips the unzip and the install is instant).")
else:
    while not os.path.exists("pyproject.toml"):
        os.chdir("..")
sys.path.insert(0, os.getcwd())

import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 120, "display.width", 200)
from evals import report
"""),
    md("""
## 1. Business understanding

**User need.** "I remember a movie but not its title" (detective mode) and "something like
this" (discovery). People describe plot fragments, scenes, mood and where they watched it;
they hedge, exclude ("not the remake") and misremember (wrong actor, wrong decade).

**Success criteria** (brief section 2): the intended title in the top 10, ideally first
(Success@k, MRR); no shown title violates a stated constraint; no hallucinated titles (every
title resolves to a catalog/TMDB id); explanations claim only what the title's text supports;
vague requests get at most 1-2 targeted questions; Ukrainian close to English.

**Baseline (Emberfall today).** Free tier: a Claude call parses the request, then one SQL
function fuses gte-small vector search with Postgres full-text search (RRF, popularity nudge,
40 results). Catalog: 18,225 titles from 2000-2026, documents mostly TMDB blurbs. Known
weaknesses (W1-W9) are listed in `docs/PROJECT_BRIEF.md`; several are tested below.

**Out of scope:** streaming availability, the iOS UI, accounts.
"""),
    md("## 2. Data understanding"),
    md("""
**Catalog.** The Supabase export was extended with a Wikipedia intro for every title
(`doc_wiki`) and with the most-voted films and shows of 1950-1999 (ToT queries often target
older films). Coverage, document lengths in E3 tokens and the W7 check
(`data/eda_catalog.py`):
"""),
    code("""
import json, glob
eda = json.loads(open(sorted(glob.glob("evals/results/eda/*.json"))[-1]).read())
display(pd.DataFrame({"rows": eda["coverage"]["rows"], "with Wikipedia intro": eda["coverage"]["wiki_share"]}))
tokens = {k: v for k, v in eda["tokens"].items() if k != "max_seq_length"}
display(pd.DataFrame(tokens).T.round(3))
print("W7 (were stored vectors built from doc or from title. overview?):", eda["w7"])
"""),
    md("""
**Known-item queries.** TREC ToT 2023 test and 2024 test (via the 2025 release), the splits
the retriever never saw (TREC ToT 2023 train/dev selected E3 in Colab). Answers are Wikipedia
pages, linked to TMDB through Wikidata. How many answers does the catalog contain?
"""),
    code("""
from evals.gold import load
ki = pd.DataFrame([e.model_dump() for e in load("known_item")])
display(pd.crosstab([ki.source, ki.split], ki.in_catalog, margins=True))
"""),
    md("## 3. Data preparation"),
    md("""
**Documents.** `doc` (baseline: title, genres, 15 TMDB keywords, mostly the TMDB overview) vs
`doc_wiki` (same format, English Wikipedia intro when one exists).

**Gold sets** (`evals/gold/*.jsonl`, schemas in `evals/gold.py`, split by a hash of the id so
no example ever moves between dev and test). Sources: public datasets first; then sets built
in code without an LLM; agent-drafted or LLM-drafted sets are labelled as such.
"""),
    code("""
from evals.gold import SCHEMAS
rows = []
for name in SCHEMAS:
    try:
        for e in load(name):
            rows.append({"set": name, "split": e.split, "source": e.source.split(":")[0]})
    except FileNotFoundError:
        pass
display(pd.crosstab([pd.Series([r["set"] for r in rows], name="set"),
                     pd.Series([r["source"] for r in rows], name="source")],
                    pd.Series([r["split"] for r in rows], name="split")))
"""),
    md("## 4. Modeling"),
    md("""
```
user message (EN/UK)
  -> C0 normalize            language id, Unicode clean-up
  -> C1 agent (Haiku 5.5)    tool loop, max 4 tool calls per turn, ends with `respond`
       tools (also over MCP):
       C4 search_catalog     E3 dense top-100 over doc_wiki (filters applied first)
                              -> C5 Haiku listwise rerank of the top 20
       C6 find_title         fuzzy title match (typos), TMDB for Ukrainian titles
       C7 similar_to         neighbours of the title's E3 vector
       C8 get_title_details  TMDB runtime, cast, director
       C9 watchlist          SQLite; add/remove only after the user confirms (enforced in code)
       C10 external_lookup   TMDB for guesses, then Claude web search (max 2); TMDB-resolved only
  -> picks: only ids a tool returned (hallucination guard)
  -> C2 clues (Haiku)        the user's memory as tagged clues
  -> C11 verifier (Haiku)    each clue supported / contradicted / not mentioned; false memory
  -> C12 explain (Sonnet)    1-2 sentences grounded in the evidence
C3 clarifier: one question that splits close candidates (evaluated with a user simulator).
```
"""),
    code("""
from concierge.tools.registry import TOOLS
display(pd.DataFrame([{"tool": t.name, "writes": t.writes, "agent only": t.agent_only,
                       "description (what the model reads)": t.description[:160] + "..."}
                      for t in TOOLS]).set_index("tool"))
"""),
    md("""
**Retriever (done in Colab before this repo, brief 7.1).** gte-small fine-tuned (E3) on 5,905
synthetic ToT queries from Claude Haiku with hard negatives. On TREC ToT 2023 (231k Wikipedia
pages) it lifted Recall@100 from 0.213 to 0.313; with rerank depth 20 the end-to-end gain was
not significant (+0.002, CI [-0.055, +0.054]). Experiment notebooks: `notebooks/experiments/`.

**One query, end to end** (a known false memory: Tom Hanks is not in Twister):
"""),
    code("""
from concierge.agent import DetectiveAgent
from concierge.finalize import verify_and_explain
from concierge.tools.registry import Toolbox
box = Toolbox.create(watchlist_path=":memory:")
agent = DetectiveAgent(box, finalize=verify_and_explain)
turn = agent.turn("A movie where storm chasers follow tornadoes across Oklahoma, I think Tom Hanks "
                  "was in it, and a cow flies past their truck")
print(turn.message, "\\n")
for p in turn.picks[:2]:
    print(p.id, p.title, p.year, "|", (p.notes.get("verification") or {}).get("false_memory"))
    print("   ", p.notes.get("explanation"))
print("\\ntools:", [c.name for c in turn.tool_calls], "| cost $", turn.cost_usd, "| s", turn.latency_s)
"""),
    md("""
**MCP.** `claude mcp add emberfall -- uv --directory <repo> run python -m concierge.mcp_server`
serves the six data tools with the registry's schemas (low-level `Server` of mcp SDK 2.x).
"""),
    code("""display(report.metrics_table("mcp", "dev", ["stdio"]))"""),
    md("## 5. Evaluation (dev split; the test split is run once, at the end)"),
    md("""
Each row is one configuration; `parent` is the configuration it changes by exactly one factor.
Cells show the mean and the 95% bootstrap CI over queries.

**C4/C5 search_catalog + rerank** - from the deployed baseline to the chosen pipeline:
"""),
    code("""
c4 = report.metrics_table("c4", "dev", report.C4_CHAIN)
display(c4[["parent", "success@1", "success@10", "mrr@10", "recall@100"]])
means = c4["mrr@10"].str.split().str[0].astype(float)
means.plot.barh(figsize=(7, 4), title="C4/C5 MRR@10 on dev (n=115)"); plt.gca().invert_yaxis(); plt.show()
"""),
    code(
        """display(report.group_table("c4", "dev", "tool_whole", ["success@1", "success@10", "mrr@10"]))"""
    ),
    md(
        "**C0 language id, C1 agent decisions, C3 clarifier, C6 find_title, C7 similar_to, C8 details**"
    ),
    code("""
display(report.metrics_table("c0", "dev", ["default"]))
display(report.metrics_table("c1", "dev", ["default"])); print(report.extra("c1", "dev", "default"))
display(report.metrics_table("c3", "dev", ["default"])); print(report.extra("c3", "dev", "default"))
display(report.metrics_table("c6", "dev", ["wratio", "default", "catalog_only", "tmdb_always", "accept80"]))
display(report.group_table("c6", "dev", "default", ["success@1", "success@5"]))
display(report.metrics_table("c7", "dev", ["e3_docwiki", "gte_docwiki", "stored_doc"]))
display(report.metrics_table("c8", "dev", ["default"]))
display(report.metrics_table("c9", "dev", ["default"]))
"""),
    md("**C10 external lookup, C11 verifier, C12 explanations (judge + judge validation)**"),
    code("""
display(report.metrics_table("c10", "dev", ["web"])); print(report.extra("c10", "dev", "web"))
display(report.metrics_table("c11", "dev", ["default"])); print(report.extra("c11", "dev", "default"))
display(report.metrics_table("c12", "dev", ["default"])); print(report.extra("c12", "dev", "default"))
"""),
    md("**End to end, and the English / Ukrainian gap**"),
    code("""
display(report.metrics_table("e2e", "dev", ["default", "uk"]))
display(report.en_uk_gap("dev"))
display(report.group_table("e2e", "dev", "default", ["success@1", "success@5", "tool_calls", "cost_usd"]))
"""),
    md("**Error analysis** - failed cases with the stage that lost the answer:"),
    code("""
display(report.failed_cases("c4", "dev", "tool_whole", ["answer_title", "rank", "top3"], n=8))
display(report.failed_cases("e2e", "dev", "default", ["answer_title", "catalog", "tools", "picks"], n=8))
"""),
    md("**Cost** (real API calls only; replays from the cache are free):"),
    code("""display(report.spend())"""),
    md("## 6. Demo"),
    code("""
RUN_DEMO = IN_COLAB  # Gradio UI: new questions need EMBERFALL_CLAUDE_KEY and TMDB_API_KEY in Colab Secrets
if RUN_DEMO:
    %pip install -q gradio
    import gradio as gr
    from concierge import config
    for name in ("EMBERFALL_CLAUDE_KEY", "TMDB_API_KEY"):
        config.get_secret(name)  # read here, in the main thread; Gradio's workers reuse them
    demo_agent = DetectiveAgent(box, finalize=verify_and_explain)
    def chat(message, history):
        t = demo_agent.turn(message)
        lines = [t.message] + [f"- {p.title} ({p.year}): {p.notes.get('explanation', '')}" for p in t.picks]
        if t.options:
            lines.append("Options: " + " / ".join(t.options))
        return "\\n".join(lines)
    demo = gr.ChatInterface(chat, title="Emberfall Movie Detective")
    # share=True: a public *.gradio.live link (works with any browser). Anyone with it spends
    # your API key: do not share it, and run demo.close() when done.
    demo.launch(share=True, show_error=True)
"""),
    md("""
## 7. Conclusions and what to port to the app

See `docs/REPORT.md` for the final test-split numbers, the decisions and their evidence, and
the list of changes worth porting to Emberfall (fine-tuned encoder, Wikipedia documents,
pre-2000 catalog, filters before ranking, a cheap LLM rerank, the verifier's false-memory
note).
"""),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    nb = nbformat.v4.new_notebook(cells=CELLS)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3"}
    nb.metadata["colab"] = {"provenance": []}
    if args.execute:
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(NOTEBOOK.parent)}}).execute()  # fmt: skip
    nbformat.write(nb, NOTEBOOK)
    print(f"wrote {NOTEBOOK} ({len(CELLS)} cells, executed={args.execute})")


if __name__ == "__main__":
    main()
