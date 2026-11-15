"""Catalog EDA for notebook section 2 (data understanding).

Purpose: the numbers behind the data decisions, reproducible from the parquet files.
- coverage: rows and Wikipedia-intro share per source (baseline / pre-2000 extension) and type;
- document length in E3 tokens, `doc` vs `doc_wiki`, and the share E3 would truncate;
- Wikipedia match check: articles whose title shares no word with the TMDB title;
- W7 check: was each stored baseline `embedding` computed from `doc` or from `title. overview`?
  Both texts are re-embedded with base gte-small and compared to the stored vector.
Input: data/catalog_extended.parquet, data/catalog.parquet (stored vectors), models/E3.
Output: printed summary and evals/results/eda/<timestamp>_catalog.json.

    uv run python data/eda_catalog.py            # about 6 min on a laptop, W7 dominates
    uv run python data/eda_catalog.py --skip-w7
"""

import argparse
import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from concierge import config
from concierge.embed import Embedder

RESULTS_DIR = config.REPO_ROOT / "evals" / "results" / "eda"
STOPWORDS = {"the", "a", "an", "of", "and", "in", "to"}


def percentiles(values: pd.Series) -> dict[str, float]:
    """p50 / p95 / max of the non-null values."""
    v = values.dropna().to_numpy()
    return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}  # fmt: skip


def coverage(df: pd.DataFrame) -> dict:
    """Rows and share with a Wikipedia intro per (source, item_type); extension tv per year."""
    groups = df.groupby(["source", "item_type"])
    extension_tv = df[(df.source == "extension") & (df.item_type == "tv")]
    return {
        "rows": {f"{s}/{t}": len(g) for (s, t), g in groups},
        "wiki_share": {f"{s}/{t}": round(g.wiki_title.notna().mean(), 4) for (s, t), g in groups},
        "extension_tv_per_year": extension_tv.release_date.str[:4].value_counts().sort_index()
        .to_dict(),
    }  # fmt: skip


def token_lengths(df: pd.DataFrame, embedder: Embedder) -> dict:
    """E3 token counts of `doc` and `doc_wiki` per (source, item_type), share over the limit."""
    limit = embedder.max_seq_length
    out = {"max_seq_length": limit}
    for column in ("doc", "doc_wiki"):
        tokens = df[column].map(embedder.count_tokens, na_action="ignore")
        for (source, item_type), t in tokens.groupby([df.source, df.item_type]):
            if t.notna().any():
                out[f"{column}/{source}/{item_type}"] = {
                    **percentiles(t),
                    f"share_over_{limit}": round(float((t.dropna() > limit).mean()), 4),
                    "share_over_512": round(float((t.dropna() > 512).mean()), 4),
                }
    return out


def title_words(text: str) -> set[str]:
    """Lower-case ASCII words of a title, without parentheses and stopwords."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return set(re.findall(r"[a-z0-9]+", re.sub(r"\(.*?\)", " ", text))) - STOPWORDS


def wiki_title_mismatches(df: pd.DataFrame, sample: int = 25) -> dict:
    """Articles sharing no word with the TMDB title (translations, alternative titles, or
    wrong links); a fixed-seed sample is returned for manual review."""
    linked = df[df.wiki_title.notna()]
    disjoint = [
        bool(title_words(t)) and not (title_words(t) & title_words(w))
        for t, w in zip(linked.title, linked.wiki_title, strict=True)
    ]
    mismatched = linked[disjoint]
    rows = mismatched.sample(min(sample, len(mismatched)), random_state=1)
    return {
        "linked": len(linked),
        "no_shared_word": len(mismatched),
        "sample": rows[["item_type", "title", "release_date", "wiki_title"]].to_dict("records"),
    }


def w7_check(catalog_path: Path, embedder: Embedder) -> dict:
    """Cosine of each stored baseline vector to re-embedded `doc` and `title. overview`."""
    t = pq.read_table(catalog_path, columns=["title", "overview", "doc", "embedding"]).to_pandas()
    t = t[t.doc.notna()]
    stored = np.stack(t.embedding.to_numpy()).astype(np.float32)
    # populate-embeddings builds `title. overview`, or the bare title if there is no overview.
    short = [f"{a}. {b}" if b else a for a, b in zip(t.title, t.overview.fillna(""), strict=True)]
    differs = np.array([d != s for d, s in zip(t.doc, short, strict=True)])
    cos_doc = (stored * embedder.encode(t.doc.tolist())).sum(1)[differs]
    cos_short = (stored * embedder.encode(short)).sum(1)[differs]
    return {
        "model": embedder.name,
        "rows_compared": int(differs.sum()),
        "cos_to_doc_p1_p50": np.percentile(cos_doc, [1, 50]).round(4).tolist(),
        "cos_to_title_overview_p50_p99": np.percentile(cos_short, [50, 99]).round(4).tolist(),
        "closer_to_title_overview": int((cos_short > cos_doc).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extended", type=Path, default=config.CATALOG_EXTENDED_PATH)
    parser.add_argument("--catalog", type=Path, default=config.CATALOG_PATH)
    parser.add_argument("--skip-w7", action="store_true")
    args = parser.parse_args()

    df = pq.read_table(args.extended).to_pandas()
    result = {
        "inputs": {"extended": args.extended.name, "catalog": args.catalog.name},
        "coverage": coverage(df),
        "tokens": token_lengths(df, Embedder()),
        "wiki_title_mismatch": wiki_title_mismatches(df),
    }
    if not args.skip_w7:
        result["w7"] = w7_check(args.catalog, Embedder(config.GTE_SMALL))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_catalog.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    print(json.dumps({k: v for k, v in result.items() if k != "wiki_title_mismatch"}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
