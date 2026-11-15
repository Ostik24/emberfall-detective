"""Text embeddings for dense retrieval.

Purpose: turn queries and catalog documents into vectors compared by dot product. The primary
model is E3 (gte-small fine-tuned on synthetic tip-of-the-tongue queries, brief section 4); base
gte-small is the comparison. Both mean-pool and L2-normalise, and neither uses a query or
document prefix, so the same `encode` serves both sides.
Input: texts. Output: float32 array (n, 384) with unit rows.
Texts longer than the model's `max_seq_length` are truncated (E3: 256 tokens, its train length).
"""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from concierge.config import E3_MODEL, EMBED_DIM, REPO_ROOT


def _portable_name(model: str | Path) -> str:
    """Hub ids as they are; local directories relative to the repository when inside it."""
    if isinstance(model, Path) and model.resolve().is_relative_to(REPO_ROOT):
        return model.resolve().relative_to(REPO_ROOT).as_posix()
    return str(model)


class EmbeddingError(RuntimeError):
    """The model produced vectors that cannot be used (wrong size, NaN or inf)."""


class Embedder:
    """A loaded sentence embedding model. `model` is a local directory (Path) or a Hub id."""

    def __init__(self, model: str | Path = E3_MODEL, device: str | None = None) -> None:
        if isinstance(model, Path) and not model.is_dir():
            raise FileNotFoundError(f"embedding model directory not found: {model}")
        # A repo-relative name ("models/E3"), so caches keyed by it work on any machine.
        self.name = _portable_name(model)
        # Without the pin, transformers 5 loads the Hub gte-small in its fp16 checkpoint dtype.
        self.model = SentenceTransformer(
            str(model), device=device, model_kwargs={"dtype": torch.float32}
        )
        if self.model.device.type == "cuda":
            self.model.half()  # the Colab runs used fp16 inference on GPU
        dim = self.model.get_embedding_dimension()
        if dim != EMBED_DIM:
            raise EmbeddingError(f"{self.name} gives {dim}-d vectors, expected {EMBED_DIM}")

    @property
    def max_seq_length(self) -> int:
        """Token limit; longer texts are cut from the end."""
        return self.model.max_seq_length

    def count_tokens(self, text: str) -> int:
        """Tokens the model would see for `text` before truncation, special tokens included."""
        return len(self.model.tokenizer(text, truncation=False)["input_ids"])

    def encode(
        self, texts: Sequence[str], batch_size: int = 64, progress: bool = False
    ) -> np.ndarray:
        """Embed `texts` into a float32 (n, 384) array of unit vectors."""
        vectors = self.model.encode(
            list(texts),
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        ).astype(np.float32)
        if not np.isfinite(vectors).all():
            raise EmbeddingError(f"{self.name} returned non-finite values")
        return vectors
