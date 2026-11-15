"""C4 search_catalog + C5 rerank: known-item retrieval over the catalog.

Purpose: does the right title come back, and how high? Gold: `evals/gold/known_item.jsonl`,
queries whose answer is in `data/catalog_extended.parquet`. A config changes ONE factor
against its parent (`parent` field), so every step of the comparison is attributable:
    baseline        deployed Emberfall: hybrid (gte-small + FTS), baseline rows, `doc`
    gte_doc         its semantic half alone (stored gte-small vectors)       parent baseline
    e3_doc          E3 instead of gte-small                                   parent gte_doc
    e3_docwiki      `doc_wiki` (Wikipedia intro) instead of `doc`             parent e3_doc
    e3_docwiki_ext  + the 1950-1999 extension rows                            parent e3_docwiki
    baseline_ext    the baseline algorithm on the extended catalog, `doc_wiki` parent baseline
    *_rr20          + Haiku listwise rerank of the top 20 (C5; the parent is the C4 order)
    *_rr50          rerank depth 50 instead of 20 (brief open question, dev only)
    *_rr20_c1000    the reranker sees 1000 characters per candidate instead of 350
    tool_whole      the deployed `search_catalog` pipeline with the whole query; must
                    reproduce e3_docwiki_ext_rr20_c1000 (same factors, different code path)
    tool_clues      C2 ablation: dense query = film-related clues only     parent tool_whole
Answers missing from a config's catalog count as misses: that is what a user of it would see.
Input: config name, split. Output: per-query rank of the answer and the metrics below.
"""

from dataclasses import dataclass
from typing import Literal

from concierge.catalog import Catalog, DocField
from concierge.config import E3_MODEL, GTE_SMALL
from concierge.embed import Embedder
from concierge.llm import LLM
from concierge.retrieval.dense import DenseIndex, document_vectors
from concierge.retrieval.rerank import PROMPT_VERSION, rerank
from concierge.tmdb import TMDB
from concierge.tools.search_catalog import CatalogSearch, SearchCatalogArgs, SearchConfig
from evals.baseline.hybrid import EmberfallHybrid
from evals.gold import KnownItem, Split, load
from evals.metrics.ir import ndcg_at_k, recall_at_k, reciprocal_rank, success_at_k

DEPTH = 100  # candidates handed to the reranker (brief section 5)
METRICS = ("success@1", "success@10", "mrr@10", "ndcg@10", "recall@100")
GROUPS = ("source", "answer_catalog")


@dataclass(frozen=True)
class RetrievalConfig:
    retriever: Literal["hybrid", "dense", "tool"]
    model: Literal["stored", "gte-small", "e3"]  # stored = the exported Supabase vectors
    doc: DocField
    extended: bool  # False: baseline (emberfall) rows only
    parent: str | None = None
    rerank_depth: int = 0  # 0: no rerank
    rerank_chars: int = 350  # candidate text shown to the reranker, as in the Colab runs
    query_mode: Literal["whole", "clues"] = "whole"  # only for retriever="tool"


CONFIGS = {
    "baseline": RetrievalConfig("hybrid", "stored", "doc", extended=False),
    "gte_doc": RetrievalConfig("dense", "stored", "doc", extended=False, parent="baseline"),
    "e3_doc": RetrievalConfig("dense", "e3", "doc", extended=False, parent="gte_doc"),
    "e3_docwiki": RetrievalConfig("dense", "e3", "doc_wiki", extended=False, parent="e3_doc"),
    "e3_docwiki_ext": RetrievalConfig(
        "dense", "e3", "doc_wiki", extended=True, parent="e3_docwiki"
    ),
    "baseline_ext": RetrievalConfig(
        "hybrid", "gte-small", "doc_wiki", extended=True, parent="baseline"
    ),
    "e3_docwiki_ext_rr20": RetrievalConfig(
        "dense", "e3", "doc_wiki", extended=True, parent="e3_docwiki_ext", rerank_depth=20
    ),
    "e3_docwiki_ext_rr20_c1000": RetrievalConfig(
        "dense",
        "e3",
        "doc_wiki",
        extended=True,
        parent="e3_docwiki_ext_rr20",
        rerank_depth=20,
        rerank_chars=1000,
    ),  # fmt: skip
    "tool_whole": RetrievalConfig(
        "tool",
        "e3",
        "doc_wiki",
        extended=True,
        parent="e3_docwiki_ext_rr20_c1000",
        rerank_depth=20,
        rerank_chars=1000,
        query_mode="whole",
    ),
    "tool_clues": RetrievalConfig(
        "tool",
        "e3",
        "doc_wiki",
        extended=True,
        parent="tool_whole",
        rerank_depth=20,
        rerank_chars=1000,
        query_mode="clues",
    ),  # fmt: skip
    "e3_docwiki_ext_rr50": RetrievalConfig(
        "dense", "e3", "doc_wiki", extended=True, parent="e3_docwiki_ext_rr20", rerank_depth=50
    ),
}


def build(config: RetrievalConfig, catalog: Catalog, llm: LLM):
    """A function query -> ranked catalog rows for this config (first stage only, except
    for the deployed tool, which reranks itself)."""
    if config.retriever == "tool":
        tmdb = TMDB()
        engine = CatalogSearch(
            catalog,
            Embedder(E3_MODEL),
            llm,
            {**tmdb.genres("movie"), **tmdb.genres("tv")},
            SearchConfig(config.query_mode, DEPTH, config.rerank_depth, config.rerank_chars),
        )
        return lambda query: engine.rank(SearchCatalogArgs(query=query))[0]
    embedder = Embedder(E3_MODEL if config.model == "e3" else GTE_SMALL)
    if config.model == "stored":
        vectors = catalog.stored_vectors()
    else:
        vectors = document_vectors(embedder, catalog.documents(config.doc))
    if config.retriever == "hybrid":
        popularity = catalog.frame.popularity.to_numpy()
        hybrid = EmberfallHybrid(catalog.documents(config.doc), vectors, popularity, embedder)
        return hybrid.search
    index = DenseIndex(vectors)

    def dense(query: str) -> list[int]:
        return [row for row, _ in index.search(embedder.encode([query])[0], DEPTH)]

    return dense


def case_metrics(ranked: list, answer: tuple[str, int]) -> dict[str, float]:
    """The METRICS for one query with a single relevant title."""
    relevant = {answer}
    values = (
        success_at_k(ranked, relevant, 1),
        success_at_k(ranked, relevant, 10),
        reciprocal_rank(ranked[:10], relevant),
        ndcg_at_k(ranked, relevant, 10),
        recall_at_k(ranked, relevant, 100),
    )
    return dict(zip(METRICS, values, strict=True))


def run(config_name: str, split: Split) -> dict:
    """Per-case ranks and metrics; evals.run aggregates them."""
    config = CONFIGS[config_name]
    catalog = Catalog.load(source=None if config.extended else "emberfall")
    llm = LLM()
    search = build(config, catalog, llm)
    texts = catalog.documents(config.doc)
    separate_rerank = config.rerank_depth and config.retriever != "tool"
    extended = Catalog.load()
    cases = []
    for example in load("known_item", split=split):
        assert isinstance(example, KnownItem)
        if not example.in_catalog:
            continue
        answer = (example.answer_item_type, example.answer_item_id)
        rows = search(example.query)
        rerank_errors = 0
        if separate_rerank:
            reranked = rerank(
                llm, example.query, [texts[r] for r in rows], config.rerank_depth,
                config.rerank_chars,
            )  # fmt: skip
            rows = [rows[i] for i in reranked.order]
            rerank_errors = reranked.invalid + reranked.missing
        ranked = [catalog.keys[r] for r in rows]
        rank = ranked.index(answer) + 1 if answer in ranked else None
        cases.append(
            {
                "id": example.id,
                "source": example.source,
                "answer": f"{answer[0]}/{answer[1]}",
                "answer_title": example.answer_title,
                "answer_catalog": extended.frame.source[extended.row(answer)],
                "rank": rank,
                "top3": [f"{t}/{i}" for t, i in ranked[:3]],
                "rerank_errors": rerank_errors,
                **case_metrics(ranked, answer),
            }
        )
    return {
        "config": {
            "name": config_name,
            **config.__dict__,
            "depth": DEPTH,
            "rerank_prompt": PROMPT_VERSION if config.rerank_depth else None,
        },
        "cases": cases,
        "metrics": METRICS,
        "groups": GROUPS,
        "failed": [c["id"] for c in cases if c["success@10"] == 0],
        "llm": llm.spend.model_dump(),
    }
