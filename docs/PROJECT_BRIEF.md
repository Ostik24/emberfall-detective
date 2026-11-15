# Emberfall Movie Detective - Project Brief

Source of truth for scope and design decisions. Decisions here were agreed with the owner; change them only with the owner's approval and note the change in DEV_LOG.md.

## 1. Context

- Course: UCU NLP, Assignment 1 "Build an NLP System". Requirements: working prototype, CRISP-DM reflected in the report, data-flow diagram, purpose/input/output of each component, evaluation of EVERY component with well-chosen metrics, agent instruction files + skills, video <= 10 min. "Vanilla RAG" is explicitly not graded.
- Base: the owner's iOS app Emberfall (`/Users/ostappavlyshyn/Documents/swift-app/Emberfall`). Its current search pipeline is the **baseline**. The homework contribution is the new agent, the fine-tuned retriever, and the evaluations, clearly separated from what existed before.
- Task framing: **tip-of-the-tongue (ToT) known-item retrieval** for movies, an active research problem (TREC ToT track 2023-2025). Queries are verbose and contain hedges, exclusions, relative comparisons, episodic memories and false memories.

## 2. Business understanding

**User need:** "I remember a movie but not its title" (detective mode) and "find me something like this" (discovery). Users describe plot fragments, scenes, mood, viewing context or constraints in free text (EN or UK).

**Success criteria:**
- Known-item: the intended title in the top 10 (Success@10), ideally rank 1 (MRR).
- No shown title violates a stated hard constraint.
- No hallucinated titles: every title shown resolves to a catalog/TMDB id.
- Explanations only claim what the title's catalog text supports (faithfulness); likely false memories are pointed out politely.
- Vague requests trigger at most 1-2 targeted questions.
- Ukrainian queries perform close to English ones (gap measured).

**Out of scope:** streaming availability, the iOS UI, accounts, real-time catalog updates, screenshot search (product idea for later).

## 3. Baseline (existing Emberfall, read-only reference)

Files: `supabase/functions/{parse-query,semantic-search,populate-embeddings}/index.ts`, `supabase/migrations/0001..0013`, `supabase/populate-local/index.mjs`, `Emberfall/Features/Discover/DescribeRecommender.swift`, `RecommendationEngine.swift`, `EmbeddingTasteModel.swift`, `Emberfall/Features/Watchlist/SemanticMatcher.swift`.

Current "describe" flow: Claude Haiku 4.5 `extract_intent` tool call -> free tier gte-small 384-d + Postgres FTS ('english') fused with RRF + popularity nudge, premium tier OpenAI text-embedding-3-small (halfvec) pure cosine -> client keyword recall via TMDB ids scored with `0.68 overlap + 0.20 cos + 0.08 taste + 0.04 pop` (Apple NLEmbedding) -> semantic hits first, keyword hits appended -> post-filter on type/years/min_rating only.

Catalog doc (`populate-local`): title + genres + up to 15 TMDB keywords + Wikipedia intro (<= 2500 chars) or TMDB overview; releases 2000-2026.

Known weaknesses (hypotheses unless verified): W1 `exclude`/`genres`/`people` parsed but ignored; W2 filters after top-40 retrieval; W3 free vs premium differ in two factors at once; W4 free tier English-only; W5 FTS has no IDF; W6 gte-small likely truncates long docs; W7 two populate paths may have mixed doc types; W8 no unified final ranking; W9 no evaluation.

## 4. Retriever: DECIDED (experiments done in Colab, see section 7.1)

- **Model: `E3`** = `thenlper/gte-small` fine-tuned with MultipleNegativesRankingLoss on 5,905 synthetic ToT queries (Claude Haiku, 5 query styles: pilot, short, scene, episodic, false_memory) with one hard negative per pair (mined from base-model ranks 30-100, same-title and eval docs excluded); 1 epoch, batch 32, train_len 256, lr 2e-5, trained in fp32, inference in fp16. Saved at Google Drive `emberfall_pilot/v2/models/E3`.
- **First stage: dense only** (BM25 weight 0; on TUNE any BM25 weight hurt after fine-tuning).
- **Second stage: Claude Haiku listwise rerank of the top 20** (forced tool `submit_ranking`, candidate texts passed as data).
- Rerank depth (20 vs 50) is an OPEN question to test on dev data in this project, never on the reported test split.
- Port to the app later: query encoder on-device (Core ML) or hosted; catalog vectors recomputed server-side once. Not part of the homework.

## 5. Target system (must-have scope)

A **Claude tool-use agent** (Lecture 4 "agentic data search"). Routing = the agent's tool choice; parsing = the arguments it passes.

```
user message (EN/UK)
  -> [C0 normalize]          language id, unicode normalization
  -> [C1 agent]              Haiku tool-use loop, max 4 tool calls per turn
  -> [C2 clue decomposition] memory -> clues, each tagged: certain | hedged | episodic_noise | exclusion | comparison
                             certain/exclusion -> hard filters; hedged -> soft; episodic_noise -> dropped from the search query
  -> [C3 clarifier]          only if top candidates are close: ask the ONE question that best splits them (20-questions style),
                             as tappable options; max 2 turns
  -> tools (concierge/tools/, same functions exposed via MCP):
       C4 search_catalog(query, filters)   E3 dense top-100 (pre-filtered) -> C5 Haiku rerank top-20
       C6 find_title(title, year?)         fuzzy title match (typos, UK transliteration), then TMDB search
       C7 similar_to(item_id, filters)     anchor-embedding neighbours + filters
       C8 get_title_details(item_id)       TMDB details: runtime, cast, director, keywords (catalog has no runtime)
       C9 watchlist(action, item_id?)      add | remove | list in SQLite; writes need confirmation (enforced in code)
       C10 external_lookup(query)          fallback when the catalog has no confident answer: live TMDB search, then
                                           Claude web search (max 2), result must resolve to a TMDB id
  -> [C11 hypothesis verifier] for each top candidate: which clues are supported / contradicted by its catalog text;
                               flags a likely false memory ("it was Bill Paxton, not Tom Hanks")
  -> [C12 explain]           Sonnet, 1-2 sentences per pick grounded only in the title's text, includes the verifier's notes
```

Rules:
- Fallback order is strict: catalog tools -> `external_lookup` only below a confidence threshold tuned on dev.
- Web and catalog texts are untrusted data (indirect prompt injection, Lecture 4/5).
- Tool set stays at 7 tools; adding one needs owner approval.
- Catalog: extend the Emberfall catalog with pre-2000 classics (ToT queries often target older films); use the existing decade-pass logic of `populate-embeddings`, embed docs with E3 locally.

### MCP server (thin layer)
- `concierge/mcp_server.py`: official `mcp` Python SDK (FastMCP), stdio, exposes C4, C6-C10 with the SAME functions and schemas as the notebook agent. No logic of its own. Read-only hints on read tools; `watchlist` marked as writing.
- Connect: `claude mcp add emberfall -- python -m concierge.mcp_server` (verify against current docs).
- Test: tool list + schemas identical to the agent registry, one successful call.
- Tool search (`defer_loading`) is NOT used: 7 tools are too few for it to pay off. Worth checking instead: tool use examples (better argument accuracy) and programmatic tool calling (parallel hypothesis verification); verify availability for Haiku 5.5 in docs.

## 6. Models and budget

Haiku 5.5 for agent loop, clue decomposition, clarifier, rerank; Sonnet 5.5 for explanations; Opus 5.5 as judge on the final test; Claude API only (personal Console org, $100/month Max credits, auto-reload off). Embeddings: E3 (primary), base gte-small and OpenAI text-embedding-3-small as comparisons. Do not pass `temperature`; read response blocks by type (thinking blocks come first).

## 7. Evaluation

### 7.1 Retriever experiments (DONE, report as is)

Data: TREC ToT 2023 via ir_datasets (231,852 Wikipedia movie pages; human queries: `train` 150 = TUNE, `dev` 150 = TEST). Docs truncated to 2,000 chars. TUNE evaluated on a 50k-doc subcorpus (all TUNE answers + 50k random), TEST on the full corpus, TEST run once.

Pilot (all 300 queries, no tuning, full corpus): dense MRR@10 0.040 -> 0.123 after fine-tuning (1,779 pairs, 49 s on a T4); hybrid+rerank 0.111 -> 0.201.

TUNE (dense MRR@10 / R@100; base model 0.063 / 0.407):
| exp | change vs parent | MRR@10 | R@100 | note |
|---|---|---|---|---|
| E0 | pilot repro (2k pairs, 1 style) | 0.197 | 0.527 | |
| E1 | 6k pairs, same style | 0.169 | 0.467 | significantly worse (-0.028, CI [-0.050, -0.007]) |
| E2 | + 5 styles | 0.191 | 0.553 | n.s. |
| E3 | + hard negatives | **0.207** | **0.573** | selected (pre-registered rule) |
| E4 | train_len 512 | 0.203 | 0.560 | n.s., 2.4x slower |
| E2b/E3b | styles / +hard neg at 2k pairs | 0.190 / 0.193 | 0.49 / 0.51 | n.s. |
| E6b | batch 128 | 0.192 | 0.533 | n.s. |
| E7b | gte-base | 0.188 | 0.527 | n.s., 4x slower |

TEST (full corpus, 150 queries, run once): BM25 R@100 0.173 / MRR 0.061; dense base 0.213 / 0.049; hybrid base 0.193 / 0.099; dense pilot FT 0.307 / 0.106; dense E3 0.313 / 0.091; hybrid base + rerank MRR 0.137; E3 + rerank MRR 0.139. E3+rerank vs base+rerank: +0.002, 95% CI [-0.055, +0.054].

Conclusions to report:
1. Fine-tuning on LLM-generated ToT queries robustly improves candidate recall (R@100 ~0.20 -> ~0.31 on TEST), cheaply (minutes on a T4).
2. Differences between fine-tuning variants are within noise; TUNE ranking E3 > E0 reversed on TEST. More data of a single synthetic style hurt significantly (E1).
3. With rerank depth 20 the recall gain does not reach the final top-10: no significant end-to-end gain on TEST. Hypothesis (post hoc, to be tested on dev only): deeper rerank would realise the recall gain.
4. Infrastructure lessons (also in DEV_LOG): fp16-loaded weights + AdamW -> NaN; JAX via bm25s preallocated GPU; SDK rejects `temperature`; thinking blocks precede text.

### 7.2 Agent and component evaluation (to do in this repo)

| Comp. | Metric | Why (lecture) |
|---|---|---|
| C0 | language id accuracy | sanity |
| C1 | tool-selection macro-F1 + confusion matrix; clarify-vs-act MCC; argument field F1, hard-constraint exact match | imbalance (L2) |
| C2 | clue tag accuracy vs hand-annotated clues (~60 queries, incl. MS-ToT style); ablation: decomposed query vs whole query (literature found LLM reformulations can lose to the whole query) | L4 decomposition |
| C3 | Success@1 vs number of questions with an LLM user simulator that knows the hidden target and answers only yes/no/not sure from its metadata (never names it, to avoid leakage); redundant-question rate | CRS simulator literature |
| C4/C5 | R@100, nDCG@10, MRR on the catalog known-item set (in-catalog subset); constraint violation rate | L3, L4 |
| C6 | Success@1 clean / typo / UK names | |
| C7 | nDCG@10 vs expected neighbours | |
| C8 | field accuracy; runtime-constraint satisfaction | |
| C9 | action + args exact match, DB state, unconfirmed-write rate = 0 | |
| C10 | rescue rate on out-of-catalog known items; false-fallback rate; cost per rescue | |
| C11 | hallucinated-title rate (unverified vs verified); false-memory detection precision/recall on queries with injected wrong details | |
| C12 | faithfulness, answer relevancy; Opus judge validated vs ~30 human labels (Cohen's kappa) | L2 judge benchmark, L4 RAGAS |
| MCP | schemas identical, one call per tool | |
| E2E | Success@10, MRR, violation rate, tool calls, cost, latency; EN vs UK gap (Lecture 5 technical bias) | extrinsic (L2) |

Protocol: dev/test split with fixed seed; tune on dev; report test once; change one factor at a time; log every loop.

## 8. Data

1. Emberfall catalog export -> `data/catalog.parquet` (script run by the owner; vectors as float32), extended with pre-2000 classics.
2. Known-item queries: MS-ToT / TREC ToT movie queries mapped to TMDB ids via Wikidata; in-catalog vs out-of-catalog split (the latter feeds C10).
3. Hand-annotated sets: clues (C2), agent decisions (C1), tool conversations (C9), explanations (C12), title lookups (C6).
4. Ukrainian set: ~100 translated known-item queries (LLM draft + manual check of a sample).
5. Retriever artefacts on Google Drive `emberfall_pilot/` (pilot) and `emberfall_pilot/v2/` (models E0..E7b, results, final_test_runs.json, synthetic queries `synth_v2.jsonl`, hard negatives).

## 9. Notebook (`notebooks/Emberfall_Detective.ipynb`)

0 setup (Colab secrets, Drive, `USE_CACHE=True`); 1 business understanding; 2 data understanding (catalog EDA, coverage of ToT answers, doc length/truncation); 3 data preparation (docs, synthetic queries, gold sets); 4 modeling (diagram, component/tool table, walkthrough of one query with Langfuse trace, retriever experiments summary from 7.1, MCP section); 5 evaluation (7.2 tables, EN/UK gap, error analysis, judge validation); 6 Gradio demo; 7 conclusions and what to port to the app. Retriever experiment notebooks are attached as-is in `notebooks/experiments/`.

## 10. Video outline (<= 10 min)

1:00 task and why ToT is hard; 2:00 diagram + tools; 2:30 live demo (vague memory -> clue decomposition -> one clarifying question -> hypotheses verified -> false memory flagged; out-of-catalog film rescued; save to watchlist; UK query) with trace, plus 20 s of MCP in Claude Desktop; 2:30 evals incl. the fine-tuning story (honest TEST result); 1:30 agentic setup and DEV_LOG reflection.

## 11. Milestones

M1 repo + catalog export/extension + EDA -> M2 gold sets + baseline re-implementation + metric tests -> M3 search_catalog with E3 + rerank, find_title, evals -> M4 agent loop, clue decomposition, clarifier with simulator, evals -> M5 verifier, explanations, watchlist, details, similar_to, evals -> M6 external_lookup + MCP, evals -> M7 error analysis on dev, single test run, notebook + REPORT.md -> M8 video.
