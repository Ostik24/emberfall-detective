"""Eval runner (the `run-evals` skill).

    uv run python -m evals.run --component c4 --config e3_doc [--split dev]

Purpose: run one config of one component, aggregate, save, and compare.
A component module exposes `CONFIGS` (name -> config with an optional `parent`) and
`run(config_name, split) -> {"config", "cases", "metrics", "groups", "failed", ...}`; each case
carries the per-query value of every metric and the group fields.
Output: evals/results/<component>/<timestamp>_<split>_<config>.json with config, metrics
(mean and 95% bootstrap CI), metrics per group, per-case outputs and failed ids; printed
comparison with the previous run of the same config and with the parent config (paired CI).
"""

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path

from concierge.config import REPO_ROOT
from evals.metrics.stats import bootstrap_ci, paired_difference

RESULTS_DIR = REPO_ROOT / "evals" / "results"
COMPONENTS = {
    "c0": "evals.c0_normalize",
    "c1": "evals.c1_agent",
    "c3": "evals.c3_clarify",
    "c4": "evals.c4_search",
    "c6": "evals.c6_find_title",
    "c7": "evals.c7_similar",
    "c8": "evals.c8_details",
    "c9": "evals.c9_watchlist",
    "c10": "evals.c10_external",
    "c11": "evals.c11_verify",
    "c12": "evals.c12_explain",
    "mcp": "evals.mcp_check",
    "e2e": "evals.e2e",
}


def aggregate(result: dict) -> dict:
    """Mean + CI per metric overall, and mean per group value. A case whose metric is None
    (not applicable) is left out of that metric's mean."""
    cases, metrics = result["cases"], result["metrics"]
    overall = {}
    for metric in metrics:
        values = [c[metric] for c in cases if c[metric] is not None]
        mean, low, high = bootstrap_ci(values)
        overall[metric] = {
            "mean": round(mean, 4),
            "ci95": [round(low, 4), round(high, 4)],
            "n": len(values),
        }
    groups: dict[str, dict] = {}
    for field in result.get("groups", ()):
        for value in sorted({c[field] for c in cases}):
            members = [c for c in cases if c[field] == value]
            groups[f"{field}={value}"] = {
                "n": len(members),
                **{m: _mean(members, m) for m in metrics},
            }
    return {"n": len(cases), "overall": overall, "by_group": groups}


def _mean(cases: list[dict], metric: str) -> float | None:
    values = [c[metric] for c in cases if c[metric] is not None]
    return round(sum(values) / len(values), 4) if values else None


def latest(component: str, split: str, config: str, before: str | None = None) -> dict | None:
    """The most recent saved result of a config (older than `before`, if given)."""
    paths = sorted((RESULTS_DIR / component).glob(f"*_{split}_{config}.json"))
    paths = [p for p in paths if before is None or p.name < before]
    return json.loads(paths[-1].read_text()) if paths else None


def compare(current: dict, other: dict, label: str) -> None:
    """Paired comparison on the cases both runs share."""
    theirs = {c["id"]: c for c in other["cases"]}
    shared = [c for c in current["cases"] if c["id"] in theirs]
    print(f"\n{label}: {other['config']['name']} ({len(shared)} shared cases)")
    print(f"{'metric':12} {'other':>8} {'this':>8} {'delta':>8}  95% CI")
    for metric in current["metrics"]:
        pairs = [(theirs[c["id"]].get(metric), c[metric]) for c in shared]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            continue
        a, b = [x for x, _ in pairs], [y for _, y in pairs]
        delta, low, high = paired_difference(a, b)
        mark = "*" if low > 0 or high < 0 else ""
        print(
            f"{metric:12} {sum(a) / len(a):8.3f} {sum(b) / len(b):8.3f} {delta:+8.3f}  "
            f"[{low:+.3f}, {high:+.3f}]{mark}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--component", required=True, choices=sorted(COMPONENTS))
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    args = parser.parse_args()
    if args.split == "test":
        print("TEST split: only for the single final run (AGENTS.md section 3).")

    module = importlib.import_module(COMPONENTS[args.component])
    result = module.run(args.config, args.split)
    result["split"] = args.split
    result["summary"] = aggregate(result)
    if hasattr(module, "summarize"):
        result["summary"]["extra"] = module.summarize(result["cases"])

    out_dir: Path = RESULTS_DIR / args.component
    out_dir.mkdir(parents=True, exist_ok=True)
    name = f"{datetime.now():%Y%m%d_%H%M%S}_{args.split}_{args.config}.json"
    (out_dir / name).write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str))

    summary = result["summary"]
    print(f"{args.component} {args.config} {args.split}: n={summary['n']}")
    for metric, value in summary["overall"].items():
        print(f"  {metric:16} {value['mean']:.3f}  CI {value['ci95']}  n={value['n']}")
    for group, values in summary["by_group"].items():
        shown = " ".join(
            f"{m}={values[m]:.3f}" for m in result["metrics"][:3] if values[m] is not None
        )
        print(f"  {group:40} n={values['n']:3} {shown}")
    if extra := summary.get("extra"):
        print("  extra:", json.dumps(extra, ensure_ascii=False))
    if previous := latest(args.component, args.split, args.config, before=name):
        compare(result, previous, "vs previous run")
    parent = getattr(module.CONFIGS[args.config], "parent", None)
    if parent and (parent_result := latest(args.component, args.split, parent)):
        compare(result, parent_result, "vs parent")
    print(f"\nwrote {out_dir / name}; failed cases: {len(result['failed'])}")


if __name__ == "__main__":
    main()
