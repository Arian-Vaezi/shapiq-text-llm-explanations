"""Focused tests for the demo's evidence-evaluation protocol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from demos.rag_retrieval_explanation.core.evaluation import (
    deletion_evaluation_from_order,
    insertion_evaluation,
    leave_one_out_order,
)
from demos.rag_retrieval_explanation.core.rag_game import RAGRetrievalGame
from demos.rag_retrieval_explanation.core.schemas import RetrievedChunk
from demos.rag_retrieval_explanation.evals.experiments.metrics import (
    exact_match,
    interaction_diagnostics,
    normalized_auc,
)
from demos.rag_retrieval_explanation.evals.reporting.analyze_interactions import (
    aggregate_interaction_runs,
)

import shapiq


def _chunk(title: str) -> RetrievedChunk:
    return RetrievedChunk(title=title, text=title)


def test_exact_match_uses_normalized_text() -> None:
    assert exact_match("The answer is: Paris!", "the answer is paris") == 1.0
    assert exact_match("Paris", "Paris France") == 0.0


def test_interaction_aggregation_rejects_missing_summary(tmp_path) -> None:
    """A missing requested run must fail instead of producing an empty report."""
    with pytest.raises(FileNotFoundError, match="Missing interaction input summary"):
        aggregate_interaction_runs({"missing": tmp_path / "missing-run"})


def test_exact_shapley_on_additive_two_player_game() -> None:
    chunks = [_chunk("A"), _chunk("B")]

    def scorer(question: str, target: str, selected: list[RetrievedChunk]) -> float:
        del question, target
        return float(len(selected))

    game = RAGRetrievalGame("q", "a", chunks, scorer=scorer)
    values = shapiq.KernelSHAP(n=2, random_state=42).approximate(budget=4, game=game)
    first_order = values.get_n_order(order=1).dict_values

    assert first_order[(0,)] == pytest.approx(1.0)
    assert first_order[(1,)] == pytest.approx(1.0)


def test_interaction_diagnostics_expose_denominator_and_skips() -> None:
    rows: list[dict[str, object]] = [
        {
            "relation": "complementary",
            "interaction_sign_correct": True,
        },
        {
            "relation": "complementary",
            "interaction_sign_correct": None,
        },
        {
            "relation": "redundant",
            "interaction_sign_correct": False,
        },
    ]

    diagnostics = interaction_diagnostics(rows)

    assert diagnostics["interaction_total_pairs"] == 3
    assert diagnostics["interaction_evaluable_pairs"] == 2
    assert diagnostics["interaction_skipped_pairs"] == 1
    assert diagnostics["interaction_sign_accuracy"] == 0.5
    assert diagnostics["interaction_complementary_evaluable_pairs"] == 1
    assert diagnostics["interaction_complementary_skipped_pairs"] == 1


def test_interaction_summary_keeps_gold_and_generated_targets_separate(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame(
        [
            {
                "interaction_total_pairs": 2,
                "interaction_evaluable_pairs": 1,
                "interaction_skipped_pairs": 1,
                "interaction_correct_signs": 1,
                "interaction_complementary_total_pairs": 2,
                "interaction_complementary_evaluable_pairs": 1,
                "interaction_complementary_skipped_pairs": 1,
                "interaction_complementary_correct_signs": 1,
                "generated_interaction_total_pairs": 2,
                "generated_interaction_evaluable_pairs": 2,
                "generated_interaction_skipped_pairs": 0,
                "generated_interaction_correct_signs": 1,
                "generated_interaction_complementary_total_pairs": 2,
                "generated_interaction_complementary_evaluable_pairs": 2,
                "generated_interaction_complementary_skipped_pairs": 0,
                "generated_interaction_complementary_correct_signs": 1,
            }
        ]
    ).to_csv(run_dir / "summary.csv", index=False)

    rows = aggregate_interaction_runs({"fixture": run_dir})
    by_target = {(row.target, row.relation_type): row for row in rows}

    assert by_target[("Gold/reference answer", "overall")].coverage == 0.5
    assert by_target[("Generated answer", "overall")].coverage == 1.0
    assert by_target[("Generated answer", "complementary")].sign_recovery == 0.5


def test_leave_one_out_order_matches_largest_score_drop() -> None:
    chunks = [_chunk("big"), _chunk("small")]

    def scorer(question: str, target: str, selected: list[RetrievedChunk]) -> float:
        del question, target
        return sum(2.0 if chunk.title == "big" else 0.5 for chunk in selected)

    game = RAGRetrievalGame("q", "a", chunks, scorer=scorer)

    assert leave_one_out_order(game) == [0, 1]


def test_deletion_and_insertion_curves_are_auc_ready() -> None:
    chunks = [_chunk("A"), _chunk("B")]

    def scorer(question: str, target: str, selected: list[RetrievedChunk]) -> float:
        del question, target
        return float(len(selected)) / 2.0

    game = RAGRetrievalGame("q", "a", chunks, scorer=scorer)
    frame = pd.DataFrame(
        [
            {"player": 1, "chunk": chunks[0].title, "attribution": 0.5},
            {"player": 2, "chunk": chunks[1].title, "attribution": 0.5},
        ]
    )
    deletion = deletion_evaluation_from_order(game, [0, 1])
    insertion = insertion_evaluation(game, frame)

    assert deletion["score"].to_list() == [1.0, 0.5, 0.0]
    assert insertion["score"].to_list() == [0.0, 0.5, 1.0]
    assert normalized_auc(np.asarray(deletion["score"])) == pytest.approx(0.5)
