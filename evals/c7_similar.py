"""C7 similar_to: do the neighbours match what TMDB recommends? (silver labels)

Gold: evals/gold/similar_pairs.jsonl (TMDB recommendations that are in the catalog).
Configs (one factor each):
    e3_docwiki     deployed: E3 vectors of `doc_wiki`, extended catalog (the tool's index)
    gte_docwiki    base gte-small instead of E3                         parent e3_docwiki
    stored_doc     the baseline's stored vectors of `doc` (baseline rows only; anchors from
                   the extension cannot be found there)                 parent gte_docwiki
Metrics: nDCG@10 and Recall@10 against the expected ids. Recommendations are co-watching
signals, so absolute values are low by design; the comparison between configs is the point.
"""

from dataclasses import dataclass
from typing import Literal

from concierge.catalog import Catalog
from concierge.config import E3_MODEL, GTE_SMALL
from concierge.embed import Embedder
from concierge.retrieval.dense import DenseIndex, document_vectors
from concierge.tools.search_catalog import key_str, parse_key
from evals.gold import SimilarPairs, Split, load
from evals.metrics.ir import ndcg_at_k, recall_at_k

METRICS = ("ndcg@10", "recall@10")
GROUPS = ("source",)


@dataclass(frozen=True)
class SimilarConfig:
    vectors: Literal["e3", "gte-small", "stored"]
    parent: str | None = None


CONFIGS = {
    "e3_docwiki": SimilarConfig("e3"),
    "gte_docwiki": SimilarConfig("gte-small", parent="e3_docwiki"),
    "stored_doc": SimilarConfig("stored", parent="gte_docwiki"),
}


def run(config_name: str, split: Split) -> dict:
    config = CONFIGS[config_name]
    catalog = Catalog.load(source="emberfall" if config.vectors == "stored" else None)
    if config.vectors == "stored":
        vectors = catalog.stored_vectors()
    else:
        model = E3_MODEL if config.vectors == "e3" else GTE_SMALL
        vectors = document_vectors(Embedder(model), catalog.documents("doc_wiki"))
    index = DenseIndex(vectors)
    cases = []
    for example in load("similar_pairs", split=split):
        assert isinstance(example, SimilarPairs)
        row = catalog.row(parse_key(example.anchor_id))
        ranked: list[str] = []
        if row is not None:
            hits = index.search(vectors[row], 11)
            ranked = [key_str(catalog.keys[r]) for r, _ in hits if r != row][:10]
        expected = set(example.expected_ids)
        cases.append(
            {
                "id": example.id,
                "source": example.source,
                "anchor": example.anchor_title,
                "found": row is not None,
                "top3": ranked[:3],
                "ndcg@10": ndcg_at_k(ranked, expected, 10),
                "recall@10": recall_at_k(ranked, expected, 10),
            }
        )
    return {
        "config": {"name": config_name, **config.__dict__},
        "cases": cases,
        "metrics": METRICS,
        "groups": GROUPS,
        "failed": [c["id"] for c in cases if c["ndcg@10"] == 0],
    }
