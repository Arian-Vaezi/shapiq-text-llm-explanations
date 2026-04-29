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
        token_ids = []
        next_id = 1

        for word in text.split():
            # Simulate subword tokenization
            if word == "unbelievable":
                token_ids.extend([next_id, next_id + 1])
                next_id += 2
            else:
                token_ids.append(next_id)
                next_id += 1

        return {"input_ids": [0, *token_ids, 2]}

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


def test_empty_prediction_uses_real_empty_model_output(monkeypatch):
    class EmptyAwareClassifier:
        tokenizer = FakeTokenizer()

        def __call__(self, texts, **kwargs):
            outputs = []
            for text in texts:
                if "103" in text:
                    outputs.append({"label": "POSITIVE", "score": 0.42})
                else:
                    outputs.append({"label": "POSITIVE", "score": 0.8})
            return outputs

    monkeypatch.setattr(
        "transformers.pipeline",
        lambda *args, **kwargs: EmptyAwareClassifier(),
    )

    imputer = TextImputer("dummy", "I love machine learning")

    assert imputer.empty_prediction == pytest.approx(0.42)
    assert imputer.normalization_value == pytest.approx(0.42)


# Word segmentation should use whitespace-split words as players.
def test_word_segmentation_sets_word_players():
    imputer = TextImputer(
        "dummy", "I love machine learning", segmentation="word")

    assert imputer.players.tolist() == ["I", "love", "machine", "learning"]
    assert imputer.n_features == 4


def test_players_returns_copy():
    imputer = TextImputer("dummy", "I love machine learning", segmentation="word")

    players = imputer.players
    players[0] = "changed"

    assert imputer.players.tolist() == ["I", "love", "machine", "learning"]


def test_word_segmentation_can_differ_from_token_segmentation():
    imputer = TextImputer("dummy", "the story of RBG is unbelievable", segmentation="word")

# A single word can map to multiple tokenizer tokens.
def test_word_segmentation_can_differ_from_token_segmentation():
    imputer = TextImputer(
        "dummy", "the story of RBG is unbelievable", segmentation="word")

    assert imputer.players.tolist() == [
        "the", "story", "of", "RBG", "is", "unbelievable"]
    assert imputer.tokens.tolist() == [1, 2, 3, 4, 5, 6, 7]


def test_word_and_token_segmentation_have_different_feature_counts():
    text = "the story of RBG is unbelievable"

    word_imputer = TextImputer("dummy", text, segmentation="word")
    token_imputer = TextImputer("dummy", text, segmentation="token")

    assert word_imputer.n_features == 6
    assert token_imputer.n_features == 7


def test_token_segmentation_sets_token_players():
    imputer = TextImputer("dummy", "I love machine learning", segmentation="token")

    assert imputer.players.tolist() == [1, 2, 3, 4]
    assert imputer.tokens.tolist() == [1, 2, 3, 4]
    assert imputer.n_features == 4


def test_word_coalition_masks_words():
    imputer = TextImputer("dummy", "I love NLP", segmentation="word")

    coalition = np.array([True, False, True])
    text = imputer._coalition_to_text(coalition)

    assert text == "I [MASK] NLP"


def test_token_coalition_masks_token_ids():
    imputer = TextImputer("dummy", "I love NLP", segmentation="token")

    coalition = np.array([True, False, True])
    text = imputer._coalition_to_text(coalition)

    assert text == "1 103 3"


def test_word_coalition_removes_words():
    imputer = TextImputer("dummy", "I love NLP", segmentation="word", mask_strategy="remove")

    coalition = np.array([True, False, True])
    text = imputer._coalition_to_text(coalition)

    assert text == "I NLP"


def test_token_coalition_removes_token_ids():
    imputer = TextImputer("dummy", "I love NLP", segmentation="token", mask_strategy="remove")

    coalition = np.array([True, False, True])
    text = imputer._coalition_to_text(coalition)

    assert text == "1 3"


def test_coalition_with_wrong_length_raises():
    imputer = TextImputer("dummy", "I love NLP", segmentation="word")

    coalition = np.array([True, False])

    with pytest.raises(ValueError, match="Coalition must have shape"):
        imputer._coalition_to_text(coalition)


def test_mask_strategy_requires_mask_token(monkeypatch):
    class TokenizerWithoutMaskToken(FakeTokenizer):
        mask_token_id = None

    class ClassifierWithoutMaskToken:
        tokenizer = TokenizerWithoutMaskToken()

        def __call__(self, texts, **kwargs):
            return [{"label": "POSITIVE", "score": 0.8} for _ in texts]

    monkeypatch.setattr(
        "transformers.pipeline",
        lambda *args, **kwargs: ClassifierWithoutMaskToken(),
    )

    with pytest.raises(ValueError, match="requires tokenizer.mask_token_id"):
        TextImputer("dummy", "I love NLP", mask_strategy="mask")
