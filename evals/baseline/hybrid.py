"""The Emberfall baseline's free-tier search, re-implemented: `hybrid_match_titles`.

Purpose: the comparison system for C4/C5 and end-to-end evals (brief section 3). Mirrors
supabase/migrations/0002_hybrid_search.sql (+ 0003: the keyword side reads `doc`):
- semantic: top 80 by cosine of base gte-small vectors;
- full text: top 80 by ts_rank_cd of an OR-query over `doc` (evals/baseline/fts.py);
- score = 0.35 / (50 + fts rank) + 1.0 / (50 + semantic rank)
          + 0.02 * popularity / (popularity + 100); top 40 returned.
The app then shows these semantic-search hits first and appends TMDB keyword hits after them,
so the first 40 positions, which every @10 metric reads, come from this function alone.
Input: catalog rows (keys, keyword documents, vectors, popularity), a query.
Output: up to 40 row indices, best first.
"""

from dataclasses import dataclass

import numpy as np

from concierge.embed import Embedder
from concierge.retrieval.dense import DenseIndex
from evals.baseline.fts import FullTextIndex


@dataclass(frozen=True)
class HybridConfig:
    """Defaults of `hybrid_match_titles` as deployed."""

    match_count: int = 40
    full_text_weight: float = 0.35
    semantic_weight: float = 1.0
    rrf_k: int = 50
    popularity_weight: float = 0.02
    popularity_scale: float = 100.0

    @property
    def depth(self) -> int:
        """Each list is cut at least(match_count, 40) * 2 before fusion."""
        return min(self.match_count, 40) * 2


class EmberfallHybrid:
    """Hybrid search over one catalog snapshot."""

    def __init__(
        self,
        keyword_docs: list[str],
        vectors: np.ndarray,
        popularity: np.ndarray,
        embedder: Embedder,
        config: HybridConfig = HybridConfig(),
    ) -> None:
        self.fts = FullTextIndex(keyword_docs)
        self.dense = DenseIndex(vectors)
        self.popularity = np.nan_to_num(popularity.astype(np.float64))
        self.embedder = embedder
        self.config = config

    def search(self, query: str, allowed: np.ndarray | None = None) -> list[int]:
        """Row indices of the fused top `match_count`. `allowed` is the SQL's type filter."""
        c = self.config
        semantic = self.dense.search(self.embedder.encode([query])[0], c.depth, allowed)
        keyword = [row for row, _ in self.fts.search(query) if allowed is None or allowed[row]]
        scores: dict[int, float] = {}
        for rank, row in enumerate(keyword[: c.depth], start=1):
            scores[row] = scores.get(row, 0.0) + c.full_text_weight / (c.rrf_k + rank)
        for rank, (row, _) in enumerate(semantic, start=1):
            scores[row] = scores.get(row, 0.0) + c.semantic_weight / (c.rrf_k + rank)
        for row in scores:
            pop = self.popularity[row]
            scores[row] += c.popularity_weight * pop / (pop + c.popularity_scale)
        ranked = sorted(scores, key=lambda row: (-scores[row], row))
        return ranked[: c.match_count]
