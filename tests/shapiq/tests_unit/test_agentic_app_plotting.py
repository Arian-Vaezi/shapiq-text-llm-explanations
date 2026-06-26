"""Tests for agentic tool-use demo plotting fallbacks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

import shapiq

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

import app  # noqa: E402
import sample_data  # noqa: E402
from tool_game import ToolUseGame, ToolUseSegment  # noqa: E402
from tool_schemas import TOOL_DESCRIPTIONS  # noqa: E402


def test_load_text_plotters_handles_import_error(monkeypatch) -> None:
    def fail_load_sentence_plot_module() -> None:
        msg = "optional compiled extension unavailable"
        raise ImportError(msg)

    monkeypatch.setattr(app, "load_sentence_plot_module", fail_load_sentence_plot_module)

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
