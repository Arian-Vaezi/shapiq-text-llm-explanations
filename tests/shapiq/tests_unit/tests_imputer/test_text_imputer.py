"""Minimal tests for the TextImputer."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.imputer import TextImputer

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeTokenizer:
    mask_token_id = 103

    def __call__(self, text, **kwargs):
        word_ids = list(range(1, len(text.split()) + 1))
        return {"input_ids": [0] + word_ids + [2]}

    def decode(self, token_ids):
        return " ".join(str(t) for t in token_ids)


class FakeClassifier:
    tokenizer = FakeTokenizer()

    def __call__(self, texts, **kwargs):
        return [{"label": "POSITIVE", "score": 0.8} for _ in texts]


@pytest.fixture(autouse=True)
def mock_pipeline(monkeypatch):
    monkeypatch.setattr("transformers.pipeline", lambda *args, **kwargs: FakeClassifier())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_invalid_config():
    with pytest.raises(ValueError):
        TextImputer("dummy", "test", mask_strategy="invalid")
    with pytest.raises(ValueError):
        TextImputer("dummy", "test", segmentation="invalid")


def test_value_function():
    imputer = TextImputer("dummy", "I love machine learning")
    n = imputer.n_features
    coalitions = np.ones((3, n), dtype=bool)
    values = imputer.value_function(coalitions)
    assert values.shape == (3,)
    assert np.all(np.isfinite(values))


def test_empty_prediction():
    imputer = TextImputer("dummy", "I love machine learning")
    n = imputer.n_features
    empty = np.zeros((1, n), dtype=bool)
    values = imputer.value_function(empty)
    assert values[0] == pytest.approx(imputer.normalization_value)
