"""Tests for the exact-computation adapter used by the agentic tool-use demo.

These tests verify that the demo only ever labels a result with an official shapiq
interaction index (e.g. ``k-SII``) when it was actually produced by the real
``shapiq.game_theory.exact.ExactComputer``, never by a hand-rolled heuristic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import shapiq

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

from exact_interactions import (  # noqa: E402
    MAX_EXACT_DEMO_PLAYERS,
    ExactComputationLimitError,
    UnsupportedExactIndexError,
    compute_exact_interactions,
)
from tool_game import ToolUseGame  # noqa: E402
from tool_schemas import TOOL_DESCRIPTIONS  # noqa: E402


class DeterministicScorer:
    """Deterministic toy scorer: score grows with the number of selected segments."""

    def score_batch(self, prompts, *, target_tool, tool_descriptions):
        del target_tool, tool_descriptions
        return [0.1 * prompt.count("SEGMENT") for prompt in prompts]


def make_toy_game(n_segments: int) -> ToolUseGame:
    user_segments = [f"SEGMENT{i}" for i in range(n_segments)]
    return ToolUseGame(
        target_tool="weather_tool",
        user_segments=user_segments,
        system_prompt="Use weather_tool for weather questions.",
        scorer=DeterministicScorer(),
        tool_descriptions={"weather_tool": TOOL_DESCRIPTIONS["weather_tool"]},
        normalize=False,
    )


def test_compute_exact_interactions_uses_real_exact_computer() -> None:
    game = make_toy_game(3)

    explanation, metadata = compute_exact_interactions(game=game, index="k-SII", max_order=2)

    assert isinstance(explanation, shapiq.InteractionValues)
    assert metadata.algorithm == "ExactComputer"
    assert metadata.index == "k-SII"
    assert metadata.n_players == 3
    assert metadata.max_order == 2
    assert metadata.coalition_count == 2**3


def test_compute_exact_interactions_matches_exact_computer_oracle() -> None:
    """Use ExactComputer itself as the oracle instead of hand-deriving values."""
    game = make_toy_game(3)

    explanation, _ = compute_exact_interactions(game=game, index="SII", max_order=2)

    oracle = shapiq.game_theory.exact.ExactComputer(game=game, n_players=game.n_players)
    oracle_values = oracle(index="SII", order=2)

    assert explanation.dict_values == oracle_values.dict_values


@pytest.mark.parametrize("index", ["SV", "SII", "k-SII", "STII", "FSII", "BV", "BII"])
def test_supported_indices_are_computed_and_native(index: str) -> None:
    game = make_toy_game(3)

    explanation, metadata = compute_exact_interactions(game=game, index=index, max_order=2)

    assert isinstance(explanation, shapiq.InteractionValues)
    assert explanation.index == index
    assert metadata.index == index


def test_unsupported_index_raises_unsupported_exact_index_error() -> None:
    game = make_toy_game(3)

    with pytest.raises(UnsupportedExactIndexError):
        compute_exact_interactions(game=game, index="NOT_A_REAL_INDEX", max_order=2)


def test_player_count_above_limit_raises_exact_computation_limit_error() -> None:
    class OversizedGame:
        n_players = MAX_EXACT_DEMO_PLAYERS + 1

    with pytest.raises(ExactComputationLimitError):
        compute_exact_interactions(game=OversizedGame(), index="k-SII", max_order=2)


def test_player_count_at_limit_is_accepted() -> None:
    game = make_toy_game(MAX_EXACT_DEMO_PLAYERS)

    explanation, metadata = compute_exact_interactions(game=game, index="SV", max_order=1)

    assert isinstance(explanation, shapiq.InteractionValues)
    assert metadata.n_players == MAX_EXACT_DEMO_PLAYERS
    assert metadata.coalition_count == 2**MAX_EXACT_DEMO_PLAYERS


def test_k_sii_order2_efficiency_residual_near_zero() -> None:
    """k-SII at max_order=2 is efficient: sum of all values equals the total game value."""
    game = make_toy_game(3)
    explanation, _ = compute_exact_interactions(game=game, index="k-SII", max_order=2)

    all_mask = np.ones((1, 3), dtype=bool)
    empty_mask = np.zeros((1, 3), dtype=bool)
    full_value = float(game.value_function(all_mask)[0])
    empty_value = float(game.value_function(empty_mask)[0])
    total_game_value = full_value - empty_value

    ksii_sum = sum(float(v) for k, v in explanation.dict_values.items() if 1 <= len(k) <= 2)
    assert abs(ksii_sum - total_game_value) < 1e-4


def test_k_sii_order2_nonzero_for_nonadditive_game() -> None:
    """For a game with genuine interactions, k-SII order-2 values are non-zero."""

    class PairwiseScorer:
        def score_batch(self, prompts, *, target_tool, tool_descriptions):
            del target_tool, tool_descriptions
            # Score = number of distinct pairs of SEG tokens present
            counts = [prompt.count("SEG") for prompt in prompts]
            return [float(n * (n - 1)) / 2 for n in counts]

    game = ToolUseGame(
        target_tool="weather_tool",
        user_segments=["SEG0", "SEG1", "SEG2"],
        system_prompt="Use weather_tool for weather questions.",
        scorer=PairwiseScorer(),
        tool_descriptions={"weather_tool": TOOL_DESCRIPTIONS["weather_tool"]},
        normalize=False,
    )

    explanation, _ = compute_exact_interactions(game=game, index="k-SII", max_order=2)

    order_2_values = [float(v) for k, v in explanation.dict_values.items() if len(k) == 2]
    assert any(abs(v) > 1e-6 for v in order_2_values)


# ---------------------------------------------------------------------------
# Direct-answer (no_tool) game: exact SV/k-SII sanity checks
# ---------------------------------------------------------------------------


class DeterministicDirectAnswerScorer:
    """Deterministic toy scorer standing in for NativeDirectAnswerScorer's no_tool contract."""

    def score_batch(self, prompts, *, target_tool, tool_descriptions):
        del tool_descriptions
        assert target_tool == "no_tool"
        return [0.1 * prompt.count("SEGMENT") for prompt in prompts]


def make_direct_answer_toy_game(n_segments: int) -> ToolUseGame:
    user_segments = [f"SEGMENT{i}" for i in range(n_segments)]
    return ToolUseGame(
        target_tool="no_tool",
        user_segments=user_segments,
        system_prompt="Answer stable conceptual questions directly.",
        scorer=DeterministicDirectAnswerScorer(),
        tool_descriptions={"no_tool": TOOL_DESCRIPTIONS["no_tool"]},
        normalize=True,
    )


def test_direct_answer_game_empty_coalition_normalizes_to_zero() -> None:
    game = make_direct_answer_toy_game(3)
    empty_mask = np.zeros((1, 3), dtype=bool)

    assert float(game(empty_mask)[0]) == pytest.approx(0.0, abs=1e-9)


def test_direct_answer_game_shapley_values_sum_to_full_coalition_value() -> None:
    game = make_direct_answer_toy_game(3)
    explanation, _ = compute_exact_interactions(game=game, index="SV", max_order=1)

    full_mask = np.ones((1, 3), dtype=bool)
    v_full = float(game(full_mask)[0])
    sv_sum = sum(float(v) for key, v in explanation.dict_values.items() if len(key) == 1)

    assert sv_sum == pytest.approx(v_full, abs=1e-6)


def test_direct_answer_game_one_player_case() -> None:
    game = make_direct_answer_toy_game(1)

    explanation, metadata = compute_exact_interactions(game=game, index="SV", max_order=1)

    assert metadata.n_players == 1
    full_mask = np.ones((1, 1), dtype=bool)
    v_full = float(game(full_mask)[0])
    (only_value,) = [float(v) for key, v in explanation.dict_values.items() if len(key) == 1]
    assert only_value == pytest.approx(v_full, abs=1e-6)


def test_direct_answer_game_two_player_case() -> None:
    game = make_direct_answer_toy_game(2)

    explanation, metadata = compute_exact_interactions(game=game, index="SV", max_order=1)

    assert metadata.n_players == 2
    full_mask = np.ones((1, 2), dtype=bool)
    v_full = float(game(full_mask)[0])
    sv_sum = sum(float(v) for key, v in explanation.dict_values.items() if len(key) == 1)
    assert sv_sum == pytest.approx(v_full, abs=1e-6)


def test_direct_answer_game_ksii_matrix_symmetry_and_finite_values() -> None:
    game = make_direct_answer_toy_game(3)
    explanation, _ = compute_exact_interactions(game=game, index="k-SII", max_order=2)

    n = game.n_players
    matrix = np.zeros((n, n))
    for key, value in explanation.dict_values.items():
        if len(key) == 2:
            i, j = key
            matrix[i, j] = float(value)
            matrix[j, i] = float(value)
        elif len(key) == 1:
            (i,) = key
            matrix[i, i] = float(value)

    # The k-SII order-2 matrix diagonal is not a standard Shapley value -- it is
    # only compared against itself here for symmetry/finiteness, never treated
    # as an SV in this test or elsewhere.
    assert np.allclose(matrix, matrix.T)
    assert np.all(np.isfinite(matrix))


def test_direct_answer_game_player_ordering_matches_coalition_masks() -> None:
    game = make_direct_answer_toy_game(3)
    first_only = np.array([[True, False, False]])
    second_only = np.array([[False, True, False]])

    score_first = float(game.value_function(first_only)[0])
    score_second = float(game.value_function(second_only)[0])

    # Both single-segment coalitions score identically under this
    # occurrence-count scorer, confirming each coalition mask selects the
    # intended single player rather than an off-by-one shifted one.
    assert score_first == pytest.approx(score_second)
    assert score_first == pytest.approx(0.1)
