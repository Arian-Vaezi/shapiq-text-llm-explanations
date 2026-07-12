"""Tests for the demo's controlled artifact-only stability analysis."""

from __future__ import annotations

import pytest
from demos.rag_retrieval_explanation.evals.reporting.analyze_stability import (
    _absolute_attribution_scores,
    _compare_pair,
    _spearman,
)


def test_absolute_attribution_scores_use_stable_passage_ids() -> None:
    artifact = {
        "gold_attributions": [
            {
                "chunk": "Retrieved 1: passage_a | Evidence A",
                "attribution": -0.5,
            },
            {
                "chunk": "Retrieved 2: passage_b | Evidence B",
                "attribution": 0.25,
            },
        ]
    }

    assert _absolute_attribution_scores(artifact, target="gold") == {
        "passage_a": 0.5,
        "passage_b": 0.25,
    }


def test_spearman_handles_missing_chunks_as_zero() -> None:
    left = {"a": 3.0, "b": 2.0, "c": 1.0}
    right = {"a": 9.0, "b": 4.0}

    assert _spearman(left, right, ids={"a", "b"}) == pytest.approx(1.0)
    result = _spearman(left, right, ids={"a", "b", "c"})
    assert result is not None
    assert result > 0.8


def test_empty_attributions_do_not_count_as_top_agreement() -> None:
    """Two missing top attributions are undefined rather than an agreement."""
    artifact = {"retrieved": [], "generated_attributions": []}

    rows = _compare_pair(
        left_name="left",
        left_artifacts={"case": artifact},
        right_name="right",
        right_artifacts={"case": artifact},
        target="generated",
    )

    assert rows[0]["top_attribution_agrees"] is None


def test_stability_rejects_different_case_sets() -> None:
    """Stability summaries must not silently compare incomplete runs."""
    artifact = {"retrieved": [], "gold_attributions": []}

    with pytest.raises(ValueError, match="different case sets"):
        _compare_pair(
            left_name="left",
            left_artifacts={"case-a": artifact},
            right_name="right",
            right_artifacts={"case-b": artifact},
            target="gold",
        )
