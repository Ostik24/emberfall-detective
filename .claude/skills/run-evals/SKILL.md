---
name: run-evals
description: Run the evaluation suite for one or all Emberfall Movie Detective components on the dev (default) or test split, compare with the previous run, and update WORKING_STATE.md. Use after any change to concierge/ or prompts, or when asked for current metrics.
---

# Run evals

## Inputs to confirm
- Component(s): `c0`..`c12`, `mcp`, `e2e`, `ablation`, or `all` (numbering as in brief sections 5 and 7.2). Default: the components touched by the last change.
- Split: `dev` (default). `test` ONLY when the owner explicitly says this is the final run; note it in DEV_LOG.
- Config name (e.g. `openai_hybrid_rerank-haiku`).

## Steps
1. Estimate LLM calls (cases x LLM steps per case, minus cached). If > 200, show the estimate and ask before running. Prefer the Batches API for large runs.
2. Run: `python -m evals.run --component <c> --split <split> --config <config>`.
3. The runner writes `evals/results/<component>/<timestamp>_<config>.json` with: config + prompt versions + model ids, metrics, per-case outputs, failed case ids, token usage and cost estimate.
4. Compare with the latest previous result of the same component/split: print a table `metric | previous | current | delta`.
5. For every metric that got worse, or failed cases count > 0, list the 5 most informative failed cases with a one-line diagnosis each (which stage lost the evidence: parse, filter, retrieve, rank, explain).
6. Update `WORKING_STATE.md` -> CURRENT NUMBERS (one line per component, dev split) and NEXT STEP.
7. If this run closes a CRISP-DM loop, invoke the `crisp-dm-log` skill.

## Metric reference (do not swap metrics without updating the brief)
- C0 normalize: language id accuracy.
- C1 agent: tool-selection macro-F1 + confusion matrix; clarify-vs-act MCC; argument per-field P/R/F1, hard-constraint exact match, stated/inferred accuracy, BERTScore(query arg).
- C2 clue decomposition: clue tag accuracy vs hand-annotated clues; ablation decomposed query vs whole query.
- C3 clarifier: Success@1 vs number of questions with the LLM user simulator; redundant-question rate.
- C4 search_catalog: Recall@100, nDCG@10, MRR, Success@10; per language.
- C5 rerank: delta nDCG@10, delta MRR vs C4 order.
- Constraint safety: violation rate in top-10 per constraint type.
- C6 find_title: Success@1 (clean / typo / UK).
- C7 similar_to: nDCG@10 vs expected neighbours.
- C8 get_title_details: field accuracy; runtime-constraint satisfaction.
- C9 watchlist: action + argument exact match, final DB state, unconfirmed-write rate (must be 0).
- C10 external_lookup: rescue rate (out-of-catalog), false-fallback rate, web searches and cost per rescue.
- C11 hypothesis verifier: hallucinated-title rate (unverified vs verified); false-memory detection precision/recall.
- C12 explain: faithfulness, answer relevancy; judge kappa vs human labels.
- MCP: schemas identical to agent registry, one successful call per tool.
- E2E: Success@10, MRR (in- and out-of-catalog), violation rate, tool calls, cost and latency per query.

## Never
- Never tune anything based on test results.
- Never report a metric without the config and split it came from.
