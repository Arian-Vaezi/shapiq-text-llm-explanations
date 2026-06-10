"""Tests for agentic tool-use demo plotting fallbacks."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

import app  # noqa: E402
from tool_game import ToolUseSegment  # noqa: E402


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


class FakeGame:
    n_players = 2

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        return np.asarray([0.2 + 0.3 * row[0] + 0.5 * row[1] for row in coalitions])


def test_exact_fallback_approximator_returns_demo_interaction_values() -> None:
    approximator = app.ExactFallbackApproximator(n=2, index="SV", max_order=2)

    explanation = approximator.approximate(budget=4, game=FakeGame())
    first_order = explanation.get_n_order(order=1)

    assert first_order.dict_values == {
        (0,): pytest.approx(0.3),
        (1,): pytest.approx(0.5),
        (0, 1): pytest.approx(0.0),
    }
    assert explanation.get_n_order_values(order=2).shape == (2, 2)


def test_values_to_frame_accepts_demo_interaction_values() -> None:
    explanation = app.DemoInteractionValues(
        first_order=[0.3, -0.1],
        second_order=pd.DataFrame([[0.0, 0.2], [0.2, 0.0]]),
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
