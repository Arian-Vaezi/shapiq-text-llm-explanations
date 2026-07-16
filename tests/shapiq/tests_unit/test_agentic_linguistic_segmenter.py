"""Tests for the agentic tool-use linguistic segmenter."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

from linguistic_segmenter import (  # noqa: E402
    CandidateSpan,
    CharacterCandidate,
    LinguisticSegmenter,
    assemble_segments,
    resolve_overlaps,
    snap_candidates_to_words,
    whitespace_tokens,
)
from semantic_segmenter import validate_partition  # noqa: E402


def _model_is_available() -> bool:
    try:
        spacy = importlib.import_module("spacy")
        spacy.load("en_core_web_sm")
    except (ImportError, OSError):
        return False
    return True


requires_model = pytest.mark.skipif(
    not _model_is_available(),
    reason="en_core_web_sm not installed",
)


def test_overlap_resolution_prefers_preposition_phrase_over_inner_noun_chunk() -> None:
    accepted = resolve_overlaps(
        [
            CandidateSpan(2, 4, 1, "PREP_PHRASE"),
            CandidateSpan(3, 4, 3, "NOUN_CHUNK"),
        ]
    )

    assert accepted == [CandidateSpan(2, 4, 1, "PREP_PHRASE")]


def test_character_boundaries_snap_to_whole_whitespace_tokens() -> None:
    tokens = whitespace_tokens("weather in berlin.")

    snapped = snap_candidates_to_words(
        tokens,
        [CharacterCandidate(11, 17, 3, "NOUN_CHUNK")],
    )

    assert snapped == [CandidateSpan(2, 2, 3, "NOUN_CHUNK")]


def test_function_shell_stray_becomes_its_own_segment() -> None:
    text = "what is the weather"
    tokens = whitespace_tokens(text)

    segments, debug_rows = assemble_segments(
        text,
        tokens,
        [CandidateSpan(2, 3, 3, "NOUN_CHUNK")],
        [True, True, False, False],
    )

    assert segments == ["what is", "the weather"]
    assert debug_rows[0]["rule"] == "FUNCTION_SHELL"
    validate_partition(text, segments)


def test_shell_only_candidate_coalesces_with_adjacent_shell_stray() -> None:
    text = "what is the weather"
    tokens = whitespace_tokens(text)

    segments, debug_rows = assemble_segments(
        text,
        tokens,
        [
            CandidateSpan(0, 0, 3, "NOUN_CHUNK"),
            CandidateSpan(2, 3, 3, "NOUN_CHUNK"),
        ],
        [True, True, False, False],
    )

    assert segments == ["what is", "the weather"]
    assert debug_rows[0]["rule"] == "FUNCTION_SHELL"
    validate_partition(text, segments)


@pytest.mark.parametrize(
    ("text", "accepted", "expected"),
    [
        (
            "find the weather",
            [CandidateSpan(1, 2, 3, "NOUN_CHUNK")],
            ["find the weather"],
        ),
        (
            "the weather please",
            [CandidateSpan(0, 1, 3, "NOUN_CHUNK")],
            ["the weather please"],
        ),
        (
            "weather quickly berlin",
            [
                CandidateSpan(0, 0, 3, "NOUN_CHUNK"),
                CandidateSpan(2, 2, 3, "NOUN_CHUNK"),
            ],
            ["weather quickly", "berlin"],
        ),
    ],
)
def test_content_strays_merge_in_deterministic_directions(
    text: str,
    accepted: list[CandidateSpan],
    expected: list[str],
) -> None:
    tokens = whitespace_tokens(text)

    segments, debug_rows = assemble_segments(
        text,
        tokens,
        accepted,
        [False] * len(tokens),
    )

    assert segments == expected
    assert any(row["rule"] == "STRAY_MERGE" for row in debug_rows)
    validate_partition(text, segments)


def test_zero_candidates_falls_back_to_one_segment_per_word() -> None:
    text = "no linguistic candidates here"
    tokens = whitespace_tokens(text)

    segments, debug_rows = assemble_segments(text, tokens, [], [False] * len(tokens))

    assert segments == ["no", "linguistic", "candidates", "here"]
    assert all(row["rule"] == "FALLBACK" for row in debug_rows)
    validate_partition(text, segments)


def test_missing_model_falls_back_without_raising(monkeypatch) -> None:
    spacy = importlib.import_module("spacy")

    def fail_model_load(model_id: str) -> None:
        del model_id
        msg = "model unavailable"
        raise OSError(msg)

    monkeypatch.setattr(spacy, "load", fail_model_load)

    segmenter = LinguisticSegmenter()
    segments, debug_rows = segmenter.segment_with_debug("what is the weather")

    assert segmenter.nlp is None
    assert segments == ["what", "is", "the", "weather"]
    assert all(row["rule"] == "FALLBACK" for row in debug_rows)
    validate_partition("what is the weather", segments)


@requires_model
def test_real_model_segments_weather_request_into_signal_units() -> None:
    segmenter = LinguisticSegmenter()

    segments = segmenter.segment("what is the weather in berlin tomorrow morning")

    assert segments == ["what is", "the weather", "in berlin", "tomorrow morning"]
