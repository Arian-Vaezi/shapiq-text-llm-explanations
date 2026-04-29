"""This test module contains all tests for the text imputer module of the shapiq package."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq.imputer import TextImputer


def test_text_imputer_init_token():
    """Test initialization of the text imputer with token segmentation."""

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    text = "I love machine learning"

    imputer = TextImputer(
        model_name=model_name,
        input_text=text,
        segmentation="token",
    )

    assert imputer.n_features == len(imputer.tokens)
    assert np.array_equal(imputer.tokens, imputer._tokens)
    assert imputer.sample_size == 1


def test_text_imputer_invalid_config():
    """Test invalid initialization arguments."""

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    with pytest.raises(ValueError):
        TextImputer(
            model_name=model_name,
            input_text="test",
            mask_strategy="invalid",
        )

    with pytest.raises(ValueError):
        TextImputer(
            model_name=model_name,
            input_text="test",
            segmentation="invalid",
        )


def test_text_imputer_mask_strategy():
    """Test masking strategy replaces tokens correctly."""

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    text = "I love NLP"

    imputer = TextImputer(
        model_name=model_name,
        input_text=text,
        segmentation="token",
        mask_strategy="mask",
    )

    # full coalition (all features present)
    coalition = np.ones(imputer.n_features, dtype=bool)
    tokens = imputer._coalition_to_tokens(coalition)

    assert len(tokens) == len(imputer.tokens)

    # empty coalition (all masked)
    coalition = np.zeros(imputer.n_features, dtype=bool)
    tokens = imputer._coalition_to_tokens(coalition)

    assert np.all(tokens == imputer._mask_token_id)


def test_text_imputer_remove_strategy():
    """Test remove strategy removes tokens correctly."""

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    text = "I love NLP"

    imputer = TextImputer(
        model_name=model_name,
        input_text=text,
        segmentation="token",
        mask_strategy="remove",
    )

    coalition_full = np.ones(imputer.n_features, dtype=bool)
    full_tokens = imputer._coalition_to_tokens(coalition_full)

    coalition_partial = np.zeros(imputer.n_features, dtype=bool)
    partial_tokens = imputer._coalition_to_tokens(coalition_partial)

    assert len(partial_tokens) <= len(full_tokens)


def test_text_imputer_value_function_shape():
    """Test value_function returns correct shape."""

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    text = "I love machine learning"

    imputer = TextImputer(
        model_name=model_name,
        input_text=text,
        segmentation="token",
    )

    coalitions = np.array([
        np.ones(imputer.n_features, dtype=bool),
        np.zeros(imputer.n_features, dtype=bool),
    ])

    values = imputer.value_function(coalitions)

    assert len(values) == 2
    assert np.all(np.isfinite(values))


def test_text_imputer_empty_prediction():
    """Test empty prediction is computed."""

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    text = "I love machine learning"

    imputer = TextImputer(
        model_name=model_name,
        input_text=text,
    )

    assert imputer.empty_prediction is not None


def test_text_imputer_decode():
    """Test token decoding returns string."""

    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    text = "I love AI"

    imputer = TextImputer(
        model_name=model_name,
        input_text=text,
    )

    decoded = imputer._decode(imputer.tokens)

    assert isinstance(decoded, str)
    assert len(decoded) > 0