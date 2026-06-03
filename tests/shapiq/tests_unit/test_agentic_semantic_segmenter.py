"""Tests for the agentic tool-use semantic segmenter."""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

from semantic_segmenter import SemanticSegmenter, validate_partition  # noqa: E402


def make_uninitialized_segmenter(min_segment_words: int = 2) -> SemanticSegmenter:
    segmenter = SemanticSegmenter.__new__(SemanticSegmenter)
    segmenter.min_segment_words = min_segment_words
    return segmenter


def test_semantic_segmenter_default_configuration_without_model_load() -> None:
    defaults = {field.name: field.default for field in fields(SemanticSegmenter)}

    assert defaults["window"] == 2
    assert defaults["min_segment_words"] == 2
    assert defaults["threshold"] == 0.5


def test_merge_short_segments_merges_short_leading_weather_block_forward() -> None:
    segmenter = make_uninitialized_segmenter()

    segments = segmenter._merge_short_segments(["Will", "it rain", "in Siberia tomorrow?"])

    assert segments == ["Will it rain", "in Siberia tomorrow?"]


def test_merge_short_segments_merges_short_leading_why_block_forward() -> None:
    segmenter = make_uninitialized_segmenter()

    segments = segmenter._merge_short_segments(["Why", "is Siberia generally", "so cold?"])

    assert segments == ["Why is Siberia generally", "so cold?"]


def test_merge_short_segments_merges_short_trailing_block_backward() -> None:
    segmenter = make_uninitialized_segmenter()

    segments = segmenter._merge_short_segments(["Calculate the average", "of"])

    assert segments == ["Calculate the average of"]


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
