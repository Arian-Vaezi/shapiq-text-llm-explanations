"""Tests for the demo's controlled artifact-only stability analysis."""

from __future__ import annotations

import pytest
from demos.rag_retrieval_explanation.evals.reporting.analyze_stability import (
    _absolute_attribution_scores,
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
    assert _spearman(left, right, ids={"a", "b", "c"}) > 0.8
