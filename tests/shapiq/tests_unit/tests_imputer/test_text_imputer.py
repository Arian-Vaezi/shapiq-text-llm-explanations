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
        return {"input_ids": [0, *word_ids, 2]}

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


# invalid mask strategy + segmentation
def test_invalid_config():
    with pytest.raises(ValueError):
        TextImputer("dummy", "test", mask_strategy="invalid")
    with pytest.raises(ValueError):
        TextImputer("dummy", "test", segmentation="invalid")


# output shape check
def test_value_function():
    imputer = TextImputer("dummy", "I love machine learning")
    n = imputer.n_features
    coalitions = np.ones((3, n), dtype=bool)
    values = imputer.value_function(coalitions)
    assert values.shape == (3,)
    assert np.all(np.isfinite(values))


# normalization
def test_empty_prediction():
    imputer = TextImputer("dummy", "I love machine learning")
    n = imputer.n_features
    empty = np.zeros((1, n), dtype=bool)
    values = imputer.value_function(empty)
    assert values[0] == pytest.approx(imputer.normalization_value)


# fake model test
def test_fake_model_negative_label():
    imputer = TextImputer("dummy", "I hate this")
    imputer._classifier = lambda texts: [{"label": "NEGATIVE", "score": 0.9} for _ in texts]
    values = imputer.value_function(np.ones((1, imputer.n_features), dtype=bool))
    assert values[0] < 0
