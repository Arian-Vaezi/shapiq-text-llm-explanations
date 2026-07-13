"""Tests for agentic tool-use demo plotting fallbacks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import shapiq
from shapiq.plot.sentence import interaction_matrix_from_explanation

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

import app  # noqa: E402
import sample_data  # noqa: E402
from _app_impl import plotting, ui  # noqa: E402
from tool_game import ToolUseGame, ToolUseSegment  # noqa: E402
from tool_schemas import TOOL_DESCRIPTIONS  # noqa: E402


def test_load_text_plotters_handles_import_error(monkeypatch) -> None:
    def fail_load_sentence_plot_module() -> None:
        msg = "optional compiled extension unavailable"
        raise ImportError(msg)

    monkeypatch.setattr(plotting, "load_sentence_plot_module", fail_load_sentence_plot_module)

    bar_plot, heatmap_plot, error = app.load_text_plotters()

    assert bar_plot is None
    assert heatmap_plot is None
    assert error == "optional compiled extension unavailable"


def test_fallback_attribution_chart_uses_streamlit_bar_chart(monkeypatch) -> None:
    calls = []

    def fake_bar_chart(frame: pd.DataFrame, *, use_container_width: bool) -> None:
        calls.append((frame, use_container_width))

    monkeypatch.setattr(app.st, "bar_chart", fake_bar_chart)

    app.show_fallback_attribution_chart(
        pd.DataFrame(
            [
                {"segment": "S1", "attribution": 0.2},
                {"segment": "U1", "attribution": -0.1},
            ]
        )
    )

    assert len(calls) == 1
    chart_frame, use_container_width = calls[0]
    assert list(chart_frame.index) == ["U1", "S1"]
    assert list(chart_frame.columns) == ["attribution"]
    assert use_container_width is True


class RecordingScorer:
    """Small scorer that records every batch it receives."""

    def __init__(self) -> None:
        self.calls = []

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        self.calls.append(
            {
                "prompts": prompts,
                "target_tool": target_tool,
                "tool_descriptions": tool_descriptions,
            }
        )
        if target_tool == "calculator_tool":
            score = 2.0
        elif target_tool == "no_tool":
            score = -1.0
        else:
            score = 0.0
        return [score for _ in prompts]


def test_app_no_longer_has_invalid_heuristic_fallback() -> None:
    """The hand-rolled finite-difference heuristic must not exist as a normal code path."""
    assert not hasattr(app, "ExactFallbackApproximator")
    assert not hasattr(app, "DemoInteractionValues")


def test_compute_interaction_explanation_uses_real_exact_computer_below_limit() -> None:
    scorer = RecordingScorer()
    game = ToolUseGame(
        target_tool="calculator_tool",
        user_segments=["Calculate", "238 times 47"],
        system_prompt="Use calculator_tool for arithmetic.",
        scorer=scorer,
        tool_descriptions=app.TOOLS,
        normalize=False,
    )

    explanation, algorithm_label = app.compute_interaction_explanation(
        game=game,
        index="k-SII",
        max_order=2,
        budget=None,
    )

    assert isinstance(explanation, shapiq.InteractionValues)
    assert explanation.index == "k-SII"
    assert "ExactComputer" in algorithm_label
    assert "4 / 4 coalitions" in algorithm_label  # 2 ** n_players == 4


def test_values_to_frame_accepts_real_interaction_values() -> None:
    explanation = shapiq.InteractionValues(
        values=np.asarray([0.0, 0.3, -0.1, 0.2]),
        index="SV",
        max_order=1,
        min_order=0,
        n_players=2,
        interaction_lookup={(): 0, (0,): 1, (1,): 2},
        estimated=False,
        baseline_value=0.0,
    )
    segments = [
        ToolUseSegment(source="system", label="S1", text="Use weather_tool."),
        ToolUseSegment(source="user", label="U1", text="rain tomorrow"),
    ]

    frame = app.values_to_frame(explanation.get_n_order(order=1), segments)

    assert list(frame["segment"]) == ["S1", "U1"]
    assert list(frame["direction"]) == ["positive", "negative"]


def test_app_build_coalition_prompt_keeps_fixed_context_and_masks_user_segments() -> None:
    user_segments = [
        app.ToolUseSegment(source="user", label="U1", text="What is the weather"),
        app.ToolUseSegment(source="user", label="U2", text="in Berlin tomorrow?"),
    ]

    empty_prompt = app.build_coalition_prompt(
        [],
        system_prompt="You are a tool router.",
        tool_context="- weather_tool: Forecasts",
    )
    full_prompt = app.build_coalition_prompt(
        user_segments,
        system_prompt="You are a tool router.",
        tool_context="- weather_tool: Forecasts",
    )

    assert "You are a tool router." in empty_prompt
    assert "weather_tool: Forecasts" in empty_prompt
    assert "What is the weather" not in empty_prompt
    assert "in Berlin tomorrow?" not in empty_prompt
    assert "What is the weather in Berlin tomorrow?" in full_prompt


def test_router_prompt_matches_game_grand_coalition() -> None:
    user_segments = [
        app.ToolUseSegment(source="user", label="U1", text="Calculate"),
        app.ToolUseSegment(source="user", label="U2", text="238 times 47"),
    ]
    system_prompt = "Use calculator_tool for arithmetic."
    tool_context = app.format_tool_context(app.TOOLS)

    router_prompt = app.build_coalition_prompt(
        user_segments,
        system_prompt=system_prompt,
        tool_context=tool_context,
    )
    game = ToolUseGame(
        target_tool="calculator_tool",
        user_segments=user_segments,
        system_prompt=system_prompt,
        tool_context=tool_context,
        scorer=RecordingScorer(),
        tool_descriptions=app.TOOLS,
        normalize=False,
    )

    assert router_prompt == game.build_prompt(game.segments)


def test_target_selection_receives_fixed_context_prompt() -> None:
    user_segments = [
        app.ToolUseSegment(source="user", label="U1", text="Calculate"),
        app.ToolUseSegment(source="user", label="U2", text="238 times 47"),
    ]
    prompt = app.build_coalition_prompt(
        user_segments,
        system_prompt="Use calculator_tool for arithmetic.",
        tool_context=app.format_tool_context(app.TOOLS),
    )
    scorer = RecordingScorer()

    choice = app.choose_tool_with_scorer(
        scorer,
        prompt,
        tool_descriptions=app.TOOLS,
    )

    assert choice.tool == "calculator_tool"
    assert len(scorer.calls) == len(app.TOOLS)
    for call in scorer.calls:
        seen_prompt = call["prompts"][0]
        assert "Use calculator_tool for arithmetic." in seen_prompt
        assert "Available tools:" in seen_prompt
        assert "User request:" in seen_prompt
        assert "Calculate 238 times 47" in seen_prompt
        assert "Assistant:" in seen_prompt


def test_scoring_prompt_preview_returns_none_for_lexical_backend() -> None:
    preview = app.build_scoring_prompt_preview(
        app.LexicalToolScorer(),
        "Use calculator_tool for arithmetic.",
        target_tool="calculator_tool",
        tool_descriptions=app.TOOLS,
    )

    assert preview is None


def test_scoring_prompt_preview_uses_actual_prompt_builder() -> None:
    scorer = app.LLMToolScorer(llm=app.MockLLM())

    preview = app.build_scoring_prompt_preview(
        scorer,
        "Use calculator_tool for arithmetic.",
        target_tool="calculator_tool",
        tool_descriptions=app.TOOLS,
    )

    assert preview is not None
    assert "Target tool:" in preview
    assert "calculator_tool" in preview


def test_app_main_no_longer_references_stale_llm_scorer_name() -> None:
    assert "llm_scorer" not in app.main.__code__.co_names


def test_same_scorer_selects_and_explains_full_prompt() -> None:
    user_segments = [
        app.ToolUseSegment(source="user", label="U1", text="Calculate"),
        app.ToolUseSegment(source="user", label="U2", text="238 times 47"),
    ]
    system_prompt = "Use calculator_tool for arithmetic."
    tool_context = app.format_tool_context(app.TOOLS)
    full_prompt = app.build_coalition_prompt(
        user_segments,
        system_prompt=system_prompt,
        tool_context=tool_context,
    )
    scorer = RecordingScorer()

    choice = app.choose_tool_with_scorer(
        scorer,
        full_prompt,
        tool_descriptions=app.TOOLS,
    )
    game = ToolUseGame(
        target_tool=choice.tool,
        user_segments=user_segments,
        system_prompt=system_prompt,
        tool_context=tool_context,
        scorer=scorer,
        tool_descriptions=app.TOOLS,
        normalize=False,
    )

    assert game.scorer is scorer
    assert choice.tool == "calculator_tool"
    assert scorer.calls[0]["prompts"] == [game.build_prompt(game.segments)]
    scorer.calls.clear()

    values = game.value_function(np.asarray([[True, True]], dtype=bool))

    assert values.shape == (1,)
    assert scorer.calls == [
        {
            "prompts": [game.build_prompt(game.segments)],
            "target_tool": "calculator_tool",
            "tool_descriptions": app.TOOLS,
        }
    ]


def test_build_mock_trace_does_not_require_selected_tool() -> None:
    trace = app.build_mock_trace("Explain photosynthesis")

    assert "target_tool" not in trace
    assert trace["system_segments"] == app.MOCK_SYSTEM_SEGMENTS
    assert trace["user_segments"] == ["Explain photosynthesis"]


def test_segment_user_request_passes_only_user_request_to_segmenter() -> None:
    class FakeSegmenter:
        def __init__(self) -> None:
            self.seen_text = None

        def segment_with_debug(self, text: str):
            self.seen_text = text
            return ["weather in Berlin"], []

    segmenter = FakeSegmenter()

    segments, debug_rows = app.segment_user_request(segmenter, "weather in Berlin")

    assert segmenter.seen_text == "weather in Berlin"
    assert segments == ["weather in Berlin"]
    assert debug_rows == []


def test_sample_data_uses_derived_tool_descriptions() -> None:
    assert sample_data.TOOLS == TOOL_DESCRIPTIONS
    assert app.TOOLS == TOOL_DESCRIPTIONS
    for trace in sample_data.SAMPLE_TRACES.values():
        assert trace["target_tool"] in app.TOOLS


def test_top_pairwise_interactions_sorted_by_abs_value() -> None:
    matrix = pd.DataFrame([[0.0, 0.1, 0.5], [0.1, 0.0, -0.3], [0.5, -0.3, 0.0]])
    segments = [
        app.ToolUseSegment(source="user", label="U1", text="first"),
        app.ToolUseSegment(source="user", label="U2", text="second"),
        app.ToolUseSegment(source="user", label="U3", text="third"),
    ]

    result = app.top_pairwise_interactions(matrix, segments)

    values = [row["value"] for row in result]
    assert values == sorted(values, key=abs, reverse=True)
    assert abs(float(result[0]["value"]) - 0.5) < 1e-9


def test_top_pairwise_interactions_has_segment_text_and_labels() -> None:
    matrix = pd.DataFrame([[0.0, 0.4], [0.4, 0.0]])
    segments = [
        app.ToolUseSegment(source="user", label="U1", text="rain in Berlin"),
        app.ToolUseSegment(source="user", label="U2", text="tomorrow morning"),
    ]

    result = app.top_pairwise_interactions(matrix, segments)

    assert len(result) == 1
    row = result[0]
    assert row["segment_i"] == "U1"
    assert row["segment_j"] == "U2"
    assert row["text_i"] == "rain in Berlin"
    assert row["text_j"] == "tomorrow morning"


def test_top_pairwise_interactions_complementary_for_positive_value() -> None:
    matrix = pd.DataFrame([[0.0, 0.3], [0.3, 0.0]])
    segments = [
        app.ToolUseSegment(source="user", label="U1", text="a"),
        app.ToolUseSegment(source="user", label="U2", text="b"),
    ]

    result = app.top_pairwise_interactions(matrix, segments)

    assert result[0]["type"] == "complementary"


def test_top_pairwise_interactions_redundant_for_negative_value() -> None:
    matrix = pd.DataFrame([[0.0, -0.2], [-0.2, 0.0]])
    segments = [
        app.ToolUseSegment(source="user", label="U1", text="a"),
        app.ToolUseSegment(source="user", label="U2", text="b"),
    ]

    result = app.top_pairwise_interactions(matrix, segments)

    assert result[0]["type"] == "redundant"


def test_top_pairwise_interactions_respects_n_limit() -> None:
    matrix = pd.DataFrame([[0.0, 0.1, 0.2], [0.1, 0.0, 0.3], [0.2, 0.3, 0.0]])
    segments = [
        app.ToolUseSegment(source="user", label=f"U{i}", text=f"text {i}") for i in range(3)
    ]

    result = app.top_pairwise_interactions(matrix, segments, n=2)

    assert len(result) == 2


def test_attribution_tab_mentions_interactions_tab() -> None:
    """Interactions now live in the XAI tab instead of a separate main tab."""
    assert ui.MAIN_TAB_LABELS == ("1. Agent Result", "2. XAI Explanation")
    assert "MAIN_TAB_LABELS" in ui.main.__code__.co_names


def _three_player_ksii_explanation() -> shapiq.InteractionValues:
    """A small k-SII explanation with distinct singleton and pairwise values."""
    return shapiq.InteractionValues(
        n_players=3,
        values=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        index="k-SII",
        min_order=1,
        max_order=2,
        estimated=False,
        baseline_value=0.0,
        interaction_lookup={(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5},
    )


def test_pairwise_matrix_from_explanation_includes_main_effects_on_diagonal() -> None:
    """The combined matrix used for display must show order-1 main effects on the diagonal."""
    explanation = _three_player_ksii_explanation()

    matrix = app.pairwise_matrix_from_explanation(explanation, 3)

    assert list(np.diag(matrix.to_numpy())) == [1.0, 2.0, 3.0]


def test_pairwise_matrix_from_explanation_pairwise_values_symmetric() -> None:
    explanation = _three_player_ksii_explanation()

    matrix = app.pairwise_matrix_from_explanation(explanation, 3).to_numpy()

    assert matrix[0, 1] == matrix[1, 0] == 4.0
    assert matrix[0, 2] == matrix[2, 0] == 5.0
    assert matrix[1, 2] == matrix[2, 1] == 6.0


def test_pairwise_matrix_from_explanation_missing_interactions_stay_zero() -> None:
    """Interactions absent from the lookup (never computed/zero) must read as 0.0, not crash."""
    explanation = shapiq.InteractionValues(
        n_players=3,
        values=np.array([1.0]),
        index="k-SII",
        min_order=1,
        max_order=2,
        estimated=False,
        baseline_value=0.0,
        interaction_lookup={(0,): 0},
    )

    matrix = app.pairwise_matrix_from_explanation(explanation, 3).to_numpy()

    assert matrix[0, 0] == 1.0
    assert matrix[1, 1] == 0.0
    assert matrix[2, 2] == 0.0
    assert matrix[0, 1] == matrix[1, 0] == 0.0
    assert matrix[0, 2] == matrix[2, 0] == 0.0
    assert matrix[1, 2] == matrix[2, 1] == 0.0


def test_pairwise_matrix_from_explanation_player_count_mismatch_raises() -> None:
    explanation = _three_player_ksii_explanation()

    with pytest.raises(ValueError, match="n_players must match"):
        app.pairwise_matrix_from_explanation(explanation, 4)


def test_pairwise_only_matrix_from_explanation_has_zero_diagonal() -> None:
    """The diagonal-free variant (internal ranking use) never shows main effects."""
    explanation = _three_player_ksii_explanation()

    matrix = app.pairwise_only_matrix_from_explanation(explanation, 3).to_numpy()

    assert list(np.diag(matrix)) == [0.0, 0.0, 0.0]
    # off-diagonal values are unaffected by dropping the main effects
    assert matrix[0, 1] == matrix[1, 0] == 4.0


def test_fallback_matrix_matches_heatmap_matrix() -> None:
    """The fallback/full matrix table must show identical values to the visual heatmap."""
    explanation = _three_player_ksii_explanation()

    fallback_matrix = app.pairwise_matrix_from_explanation(explanation, 3).to_numpy()
    heatmap_matrix = interaction_matrix_from_explanation(explanation, 3)

    assert np.allclose(fallback_matrix, heatmap_matrix)


def test_strongest_pair_ignores_diagonal_main_effects() -> None:
    """A large main effect on the diagonal must never be reported as the strongest pair."""
    explanation = shapiq.InteractionValues(
        n_players=3,
        values=np.array([100.0, 100.0, 100.0, 0.1, 0.2, -0.3]),
        index="k-SII",
        min_order=1,
        max_order=2,
        estimated=False,
        baseline_value=0.0,
        interaction_lookup={(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5},
    )
    matrix = app.pairwise_matrix_from_explanation(explanation, 3)
    labels = ["U1", "U2", "U3"]

    pair_label, pair_value = app.strongest_pair(matrix, labels)

    assert pair_label == "U2 + U3"
    assert pair_value == -0.3


def test_top_pairwise_interactions_ignores_diagonal_main_effects() -> None:
    """Ranking must never surface a diagonal main effect as a pairwise interaction."""
    explanation = shapiq.InteractionValues(
        n_players=3,
        values=np.array([100.0, 100.0, 100.0, 0.1, 0.2, -0.3]),
        index="k-SII",
        min_order=1,
        max_order=2,
        estimated=False,
        baseline_value=0.0,
        interaction_lookup={(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5},
    )
    matrix = app.pairwise_matrix_from_explanation(explanation, 3)
    segments = [
        app.ToolUseSegment(source="user", label=f"U{i + 1}", text=f"text {i}") for i in range(3)
    ]

    result = app.top_pairwise_interactions(matrix, segments)

    assert len(result) == 3
    assert all(abs(row["value"]) <= 0.3 for row in result)


def test_ksii_heatmap_card_uses_corrected_terminology() -> None:
    """The heatmap title/caption must use the main-and-pairwise wording, not the old
    pairwise-only phrasing that mischaracterized the diagonal as redundant.
    """
    consts = app.main.__code__.co_consts
    assert any(
        isinstance(c, str) and "Main and pairwise effects" in c and "k-SII" in c for c in consts
    )
    assert any(isinstance(c, str) and "Diagonal:" in c and "Off-diagonal:" in c for c in consts)
