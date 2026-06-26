"""Tests for the exact-computation adapter used by the agentic tool-use demo.

These tests verify that the demo only ever labels a result with an official shapiq
interaction index (e.g. ``k-SII``) when it was actually produced by the real
``shapiq.game_theory.exact.ExactComputer``, never by a hand-rolled heuristic.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
