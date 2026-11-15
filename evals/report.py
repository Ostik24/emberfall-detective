"""Tables for the notebook and docs/REPORT.md, built from saved result files only.

Purpose: every number shown in the presentation comes from evals/results/<component>/*.json
(no recomputation, no API calls), so the notebook runs without keys.
Input: component, split, config names. Output: pandas DataFrames (one row per config).
"""

import json
from pathlib import Path

import pandas as pd

from evals.run import RESULTS_DIR

C4_CHAIN = ["baseline", "gte_doc", "e3_doc", "e3_docwiki", "e3_docwiki_ext",
            "e3_docwiki_ext_rr20", "e3_docwiki_ext_rr50", "e3_docwiki_ext_rr20_c1000",
            "tool_whole", "tool_clues", "baseline_ext"]  # fmt: skip


def result(component: str, split: str, config: str) -> dict | None:
    """The latest saved result of one config, or None."""
    paths = sorted((RESULTS_DIR / component).glob(f"*_{split}_{config}.json"))
    return json.loads(paths[-1].read_text()) if paths else None


def metrics_table(component: str, split: str, configs: list[str]) -> pd.DataFrame:
    """Mean and 95% CI of every metric, one row per config (missing configs skipped)."""
    rows = []
    for config in configs:
        r = result(component, split, config)
        if r is None:
            continue
        row = {"config": config, "n": r["summary"]["n"], "parent": r["config"].get("parent")}
        for metric, value in r["summary"]["overall"].items():
            low, high = value["ci95"]
            row[metric] = f"{value['mean']:.3f} [{low:.2f}, {high:.2f}]"
        rows.append(row)
    return pd.DataFrame(rows).set_index("config") if rows else pd.DataFrame()


def group_table(component: str, split: str, config: str, metrics: list[str]) -> pd.DataFrame:
    """Metric means per group value (e.g. source dataset, language, in/out of catalog)."""
    r = result(component, split, config)
    if r is None:
        return pd.DataFrame()
    groups = r["summary"]["by_group"]
    return pd.DataFrame(
        {g: {"n": v["n"], **{m: v.get(m) for m in metrics}} for g, v in groups.items()}
    ).T


def extra(component: str, split: str, config: str) -> dict:
    """Component-specific summary (macro-F1, kappa, cost per rescue, ...)."""
    r = result(component, split, config)
    return (r or {}).get("summary", {}).get("extra", {})


def failed_cases(component: str, split: str, config: str, fields: list[str], n: int = 5):
    """The first `n` failed cases with the chosen fields, for error analysis."""
    r = result(component, split, config)
    if r is None:
        return pd.DataFrame()
    failed = set(r["failed"])
    rows = [{f: c.get(f) for f in ["id", *fields]} for c in r["cases"] if c["id"] in failed]
    return pd.DataFrame(rows[:n])


def spend() -> pd.DataFrame:
    """API calls and cost by model and prompt version, from the usage log."""
    from concierge.config import LLM_USAGE_LOG

    log = Path(LLM_USAGE_LOG)
    if not log.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    frame = pd.DataFrame([r for r in rows if not r["from_cache"]])
    return (
        frame.groupby(["model", "prompt_version"])
        .agg(calls=("cost_usd", "size"), cost_usd=("cost_usd", "sum"))
        .sort_values("cost_usd", ascending=False)
    )


def en_uk_gap(split: str, metrics=("success@1", "success@5", "rr")) -> pd.DataFrame:
    """Paired EN vs UK on the same queries (each UK query is a translation of an EN one)."""
    from evals.metrics.stats import paired_difference

    en, uk = result("e2e", split, "default"), result("e2e", split, "uk")
    if en is None or uk is None:
        return pd.DataFrame()
    english = {c["id"]: c for c in en["cases"]}
    pairs = [(english[c["id"].removeprefix("uk-")], c) for c in uk["cases"]
             if c["id"].removeprefix("uk-") in english]  # fmt: skip
    rows = {}
    for m in metrics:
        a, b = [p[0][m] for p in pairs], [p[1][m] for p in pairs]
        delta, low, high = paired_difference(a, b)
        rows[m] = {"n": len(pairs), "en": sum(a) / len(a), "uk": sum(b) / len(b),
                   "uk - en": delta, "ci95": f"[{low:+.3f}, {high:+.3f}]"}  # fmt: skip
    return pd.DataFrame(rows).T
