"""Embedder on real models: E3 from models/E3 (skipped if absent), gte-small from the Hub."""

import numpy as np
import pytest
import torch

from concierge.config import E3_MODEL, GTE_SMALL
from concierge.embed import Embedder, EmbeddingError

needs_e3 = pytest.mark.skipif(not E3_MODEL.is_dir(), reason="models/E3 not downloaded")


@pytest.fixture(scope="module")
def e3() -> Embedder:
    return Embedder(device="cpu")


@needs_e3
def test_e3_gives_unit_float32_vectors_and_keeps_its_train_length(e3):
    vectors = e3.encode(["Twister", "a film about storm chasers", ""])
    assert vectors.shape == (3, 384) and vectors.dtype == np.float32
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert e3.max_seq_length == 256
    assert e3.count_tokens("hello world") == 4  # [CLS] hello world [SEP]


@needs_e3
def test_e3_places_a_paraphrased_memory_nearest_its_film(e3):
    docs = [
        "Twister. Action. tornado, storm chaser. Scientists chase tornadoes across Oklahoma.",
        "The Shootist. Western. gunfighter, cancer. A dying gunfighter rents a room.",
        "Finding Nemo. Animation. clownfish, ocean. A clownfish searches the sea for his son.",
    ]
    query = "movie where people drive into twisters to put sensors inside them"
    scores = e3.encode(docs) @ e3.encode([query])[0]
    assert scores.argmax() == 0


def test_hub_gte_small_is_loaded_in_fp32_like_e3():
    base = Embedder(GTE_SMALL, device="cpu")
    assert next(base.model.parameters()).dtype == torch.float32


def test_missing_model_directory_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="embedding model directory not found"):
        Embedder(tmp_path / "E3")


@needs_e3
def test_non_finite_vectors_are_rejected(e3, monkeypatch):
    monkeypatch.setattr(e3.model, "encode", lambda texts, **_: np.full((len(texts), 384), np.nan))
    with pytest.raises(EmbeddingError):
        e3.encode(["x"])
