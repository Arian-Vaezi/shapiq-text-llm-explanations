"""Tests for the agentic tool-use semantic segmenter."""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

from semantic_segmenter import SemanticSegmenter, validate_partition  # noqa: E402


def make_uninitialized_segmenter(min_segment_words: int = 3) -> SemanticSegmenter:
    segmenter = SemanticSegmenter.__new__(SemanticSegmenter)
    segmenter.threshold = 0.5
    segmenter.window = 6
    segmenter.min_segment_words = min_segment_words
    return segmenter


class FakeEmbeddingModel:
    def encode(
        self,
        texts: list[str],
        *,
        batch_size: int,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        del texts, batch_size, convert_to_numpy, normalize_embeddings, show_progress_bar
        return np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 1.0],
            ],
            dtype=float,
        )


def test_semantic_segmenter_default_configuration_without_model_load() -> None:
    defaults = {field.name: field.default for field in fields(SemanticSegmenter)}

    assert defaults["window"] == 6
    assert defaults["min_segment_words"] == 3
    assert defaults["threshold"] == 0.5


def test_merge_short_segments_keeps_short_leading_block_separate() -> None:
    segmenter = make_uninitialized_segmenter()

    segments = segmenter._merge_short_segments(["Will", "it rain today", "in Siberia tomorrow?"])

    assert segments == ["Will", "it rain today", "in Siberia tomorrow?"]


def test_merge_short_segments_merges_later_short_block_backward() -> None:
    segmenter = make_uninitialized_segmenter()

    segments = segmenter._merge_short_segments(["What have been", "the weather trends", "recent"])

    assert segments == ["What have been", "the weather trends recent"]


def test_merge_short_segments_merges_short_trailing_block_backward() -> None:
    segmenter = make_uninitialized_segmenter()

    segments = segmenter._merge_short_segments(["Calculate the average", "of"])

    assert segments == ["Calculate the average of"]


def test_segment_with_debug_returns_segments_and_boundary_rows_without_model_load() -> None:
    segmenter = make_uninitialized_segmenter(min_segment_words=1)
    segmenter.model = FakeEmbeddingModel()

    segments, debug_rows = segmenter.segment_with_debug("alpha beta gamma delta")

    assert segments == ["alpha beta", "gamma delta"]
    assert debug_rows == [
        {
            "left_word": "alpha",
            "right_word": "beta",
            "similarity": 1.0,
            "threshold": 0.5,
            "cut_before_merge": False,
        },
        {
            "left_word": "beta",
            "right_word": "gamma",
            "similarity": 0.0,
            "threshold": 0.5,
            "cut_before_merge": True,
        },
        {
            "left_word": "gamma",
            "right_word": "delta",
            "similarity": 1.0,
            "threshold": 0.5,
            "cut_before_merge": False,
        },
    ]


def test_segment_returns_same_segments_as_segment_with_debug_without_model_load() -> None:
    segmenter = make_uninitialized_segmenter(min_segment_words=1)
    segmenter.model = FakeEmbeddingModel()

    assert segmenter.segment("alpha beta gamma delta") == segmenter.segment_with_debug(
        "alpha beta gamma delta"
    )[0]


def test_validate_partition_accepts_correct_ordered_partition() -> None:
    validate_partition(
        "Will it rain in Siberia tomorrow?",
        ["Will it rain", "in Siberia tomorrow?"],
    )


def test_validate_partition_rejects_missing_word() -> None:
    with pytest.raises(ValueError, match="preserve every whitespace-tokenized original word"):
        validate_partition(
            "Will it rain in Siberia tomorrow?",
            ["Will it rain", "Siberia tomorrow?"],
        )


def test_validate_partition_rejects_reordered_word() -> None:
    with pytest.raises(ValueError, match="preserve every whitespace-tokenized original word"):
        validate_partition(
            "Will it rain in Siberia tomorrow?",
            ["it Will rain", "in Siberia tomorrow?"],
        )
