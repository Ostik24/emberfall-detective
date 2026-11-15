# Kickoff prompt for the first Claude Code session

Paste this as the first message in Claude Code, opened in the repo root.

---

You are the coding agent for Emberfall Movie Detective, my NLP course Assignment 1.

1. Read `AGENTS.md`, then `WORKING_STATE.md`, then `docs/PROJECT_BRIEF.md` and `docs/LECTURE_NOTES.md`. Lecture concepts are in `docs/LECTURE_NOTES.md`.
2. My existing app (the baseline) is at `/Users/ostappavlyshyn/Documents/swift-app/Emberfall`. Read the files listed in brief section 3 to confirm the baseline description is accurate. Report any mismatch before writing code. Do not modify that repo.
3. Then do milestone M1 only:
   - scaffold the repo (pyproject with ruff + pytest, package layout from AGENTS.md section 4, `concierge/config.py` with model ids, `concierge/llm.py` with disk cache + prompt caching + usage logging, key loading per AGENTS.md section 6);
   - write `data/export_catalog.py` (paged export of `public.title_embeddings` to `data/catalog.parquet`, vectors parsed to float32) - I will run it myself;
   - write unit tests for IR metrics (P@k, AP, MRR, nDCG@k, Recall@k) using the worked examples from Lecture 3.
4. Before writing code, give me a short plan (files + what each does) and wait for my OK.
5. At the end, update `WORKING_STATE.md` and add a DEV_LOG entry.

Answer me in Ukrainian; write code, comments and docs in English.
