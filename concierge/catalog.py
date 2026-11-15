"""The title catalog, read from local parquet (AGENTS.md section 4).

Purpose: one place that knows the catalog's columns and the (item_type, item_id) keys every tool,
eval and baseline uses.
Input: `data/catalog_extended.parquet` (baseline export + Wikipedia backfill + 1950-1999).
Output: a `Catalog` with row lookup by key, document texts and boolean row masks.
Rows with `source == "emberfall"` are the baseline's catalog; `doc` is the baseline's document
(null for the pre-2000 extension), `doc_wiki` the populate-local style document with a
Wikipedia intro, present for every row.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from concierge.config import CATALOG_EXTENDED_PATH

Key = tuple[str, int]
DocField = Literal["doc", "doc_wiki"]


class Catalog:
    """Catalog rows in a fixed order; row numbers index every vector array built from it."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame.reset_index(drop=True)
        self.keys: list[Key] = list(
            zip(self.frame.item_type, self.frame.item_id.astype(int), strict=True)
        )
        self._rows = {key: row for row, key in enumerate(self.keys)}

    @classmethod
    def load(cls, path: Path = CATALOG_EXTENDED_PATH, source: str | None = None) -> "Catalog":
        """The whole catalog, or only the rows of one `source` ("emberfall" or "extension")."""
        frame = pq.read_table(path).to_pandas()
        if source is not None:
            frame = frame[frame.source == source]
        return cls(frame)

    def __len__(self) -> int:
        return len(self.keys)

    def row(self, key: Key) -> int | None:
        return self._rows.get(key)

    def documents(self, field: DocField) -> list[str]:
        """Texts of `field`; rows without one fall back to "title. overview", which is what
        the baseline embedded for them (populate-embeddings)."""
        overview = self.frame.overview.fillna("")
        fallback = self.frame.title.where(overview == "", self.frame.title + ". " + overview)
        return self.frame[field].fillna(fallback).tolist()

    def stored_vectors(self) -> np.ndarray:
        """The baseline's gte-small vectors as exported from Supabase (emberfall rows only)."""
        column = self.frame.embedding
        if column.isna().any():
            raise ValueError("stored vectors exist only for the baseline (emberfall) rows")
        return np.stack(column.to_numpy()).astype(np.float32)

    def mask(self, item_type: str | None = None) -> np.ndarray | None:
        """Rows of one type, or None for no restriction."""
        return None if item_type is None else (self.frame.item_type == item_type).to_numpy()
