---
name: crisp-dm-log
description: Append a CRISP-DM iteration entry to DEV_LOG.md and docs/REPORT.md notes after a change has been evaluated. Use at the end of each change-evaluate loop so the report shows how CRISP-DM was applied.
---

# CRISP-DM log entry

Append to the top of the entries in `DEV_LOG.md`:

```
### YYYY-MM-DD - CRISP-DM loop: <component> - <one-line change>
- Business understanding: which success criterion (brief section 2) this targets.
- Data understanding: the observation that motivated it (numbers, failed case ids).
- Data preparation: data/gold/document changes, if any.
- Modeling: the single factor changed (old -> new), prompt/model/config version.
- Evaluation: metric | before | after (dev split), result file path.
- Decision: keep / revert / follow-up, and why.
- Deployment: wired into notebook/demo? (yes/no)
```

Then add one line to the "Iterations" table in `docs/REPORT.md` (create the table if missing):
`| date | component | change | key metric before -> after | decision |`

Rules: numbers must come from a saved result file; if the change was reverted, still log it (negative results are results).
