"""How faithful is `evals/baseline/fts.py` to PostgreSQL? (evaluation of the re-implementation)

    uv run --with pgserver --with "psycopg[binary]" python -m evals.baseline.check_fts_fidelity [N]

pgserver ships a real PostgreSQL 16 as a Python wheel; it is a one-off check tool, not a project
dependency. The baseline catalog's `doc` column is loaded into a temporary database and N dev
known-item queries (fixed hash order over both sources, default 60: PostgreSQL's ts_rank_cd
takes seconds to minutes per long OR-query) run through the baseline's SQL and through the
Python index.
Output: per-query agreement (same matched set, same rank for every matched document, same
top-80 set as the SQL keeps) and evals/results/baseline_fts/<timestamp>_pg16.json.
Ties: PostgreSQL orders equal ranks arbitrarily, so top-80 sets can differ at the cut even when
every rank agrees; the per-document rank agreement is the strict measure.
"""

import hashlib
import json
import sys
import tempfile
from datetime import datetime

import pgserver
import psycopg
import pyarrow.parquet as pq

from concierge.config import CATALOG_PATH, REPO_ROOT
from evals.baseline.fts import FullTextIndex
from evals.gold import load

TOP = 80  # least(match_count, 40) * 2 in hybrid_match_titles
SQL = """
with q as (
  select nullif(replace(websearch_to_tsquery('english', %s)::text, '&', '|'), '')::tsquery tsq
)
select d.ix, ts_rank_cd(d.fts, q.tsq)::float8
from docs d cross join q
where q.tsq is not null and d.fts @@ q.tsq
"""


def main() -> None:
    docs = pq.read_table(CATALOG_PATH, columns=["doc"])["doc"].to_pylist()
    docs = [d for d in docs if d]
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    dev = load("known_item", split="dev")
    queries = sorted(dev, key=lambda e: hashlib.sha256(e.id.encode()).hexdigest())[:limit]
    index = FullTextIndex(docs)

    with tempfile.TemporaryDirectory() as pgdata:
        server = pgserver.get_server(pgdata, cleanup_mode="stop")
        with psycopg.connect(server.get_uri()) as conn:
            conn.execute(
                "create table docs (ix int, doc text, fts tsvector "
                "generated always as (to_tsvector('english', doc)) stored)"
            )
            with conn.cursor().copy("copy docs (ix, doc) from stdin") as copy:
                for ix, doc in enumerate(docs):
                    copy.write_row((ix, doc))
            rows = []
            for n, example in enumerate(queries, start=1):
                print(f"{n}/{len(queries)} {example.id}", flush=True)
                pg = dict(conn.execute(SQL, (example.query,)).fetchall())
                py = dict(index.search(example.query))
                same_rank = sum(abs(pg[d] - py.get(d, -1)) < 1e-6 for d in pg)
                top_pg = {d for d, _ in sorted(pg.items(), key=lambda r: -r[1])[:TOP]}
                top_py = {d for d, _ in sorted(py.items(), key=lambda r: -r[1])[:TOP]}
                rows.append(
                    {
                        "id": example.id,
                        "pg_matched": len(pg),
                        "py_matched": len(py),
                        "same_set": set(pg) == set(py),
                        "rank_agreement": same_rank / len(pg) if pg else 1.0,
                        "top80_overlap": len(top_pg & top_py) / max(len(top_pg), 1),
                    }
                )

    n = len(rows)
    summary = {
        "queries": n,
        "docs": len(docs),
        "postgres": "16.2 (pgserver)",
        "same_matched_set": sum(r["same_set"] for r in rows) / n,
        "mean_rank_agreement": sum(r["rank_agreement"] for r in rows) / n,
        "mean_top80_overlap": sum(r["top80_overlap"] for r in rows) / n,
    }
    out_dir = REPO_ROOT / "evals" / "results" / "baseline_fts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{datetime.now():%Y%m%d_%H%M%S}_pg16.json"
    worst = sorted(rows, key=lambda r: r["rank_agreement"])[:10]
    out.write_text(json.dumps({"summary": summary, "worst": worst, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))
    print("worst:", [(r["id"], round(r["rank_agreement"], 3)) for r in worst[:5]])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
