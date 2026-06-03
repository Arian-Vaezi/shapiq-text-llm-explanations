"""Backward-compatible shim — re-exports from core.evaluation."""

from __future__ import annotations

from core.evaluation import (  # noqa: F401
    FaithfulnessSummary,
    attribution_order,
    deletion_evaluation,
    deletion_evaluation_from_order,
    insertion_evaluation,
    player_title,
    random_deletion_baseline,
    random_insertion_baseline,
    retrieval_score_deletion_baseline,
    retrieval_score_order,
    score_mask,
    summarize_faithfulness,
)
