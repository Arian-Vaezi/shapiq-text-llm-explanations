"""Tests for agentic tool-use demo scorers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

from scorers import LexicalToolRouter, LexicalToolScorer, LLMToolScorer, MockLLM  # noqa: E402
from tool_game import ToolUseGame, ToolUseSegment  # noqa: E402

TOOL_DESCRIPTIONS = {
    "weather_tool": "Fetch current weather or forecasts for a place and date.",
    "calculator_tool": "Compute exact arithmetic and numeric expressions.",
    "web_search_tool": "Search the web for current, recent, or external facts.",
    "no_tool": "Answer directly without calling an external tool.",
}


def test_llm_tool_scorer_parses_numeric_output() -> None:
    scorer = LLMToolScorer(llm=MockLLM("0.75"))

    scores = scorer.score_batch(
        ["User asks whether it will rain in Berlin tomorrow."],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [0.75]


def test_llm_tool_scorer_clamps_numeric_output() -> None:
    scorer = LLMToolScorer(llm=MockLLM("1.7"))

    scores = scorer.score_batch(
        ["User asks whether it will rain in Berlin tomorrow."],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [1.0]


def test_llm_tool_scorer_falls_back_on_invalid_output() -> None:
    fallback = LexicalToolScorer()
    scorer = LLMToolScorer(llm=MockLLM("not a score"), fallback_scorer=fallback)
    prompt = "Use weather_tool for forecast questions. Will it rain in Berlin tomorrow?"

    scores = scorer.score_batch(
        [prompt],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == fallback.score_batch(
        [prompt],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )


def test_lexical_tool_scorer_returns_one_score_per_prompt() -> None:
    scorer = LexicalToolScorer()

    scores = scorer.score_batch(
        [
            "Will it rain in Berlin tomorrow?",
            "Explain photosynthesis in simple terms.",
        ],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert len(scores) == 2
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_lexical_tool_router_returns_mock_llm_choice() -> None:
    router = LexicalToolRouter()

    choice = router.choose_tool(
        "Will it rain in Berlin tomorrow morning?",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert choice.tool == "weather_tool"
    assert 0.0 <= choice.score <= 1.0
    assert set(choice.scores) == set(TOOL_DESCRIPTIONS)
    assert "weather" in choice.reason or "rain" in choice.reason


def test_tool_use_game_accepts_llm_scorer() -> None:
    game = ToolUseGame(
        target_tool="weather_tool",
        segments=[
            ToolUseSegment("system", "S1", "Use weather_tool for forecasts."),
            ToolUseSegment("user", "U1", "Will it rain tomorrow?"),
        ],
        scorer=LLMToolScorer(llm=MockLLM("0.75")),
        tool_descriptions=TOOL_DESCRIPTIONS,
        normalize=False,
    )
    coalitions = np.array([[False, False], [True, False], [True, True]])

    scores = game.value_function(coalitions)

    assert scores.tolist() == [0.75, 0.75, 0.75]


def test_tool_use_game_builds_coalition_prompt() -> None:
    game = ToolUseGame(
        target_tool="weather_tool",
        segments=[
            ToolUseSegment("system", "S1", "Use weather_tool for forecasts."),
            ToolUseSegment("user", "U1", "Will it rain tomorrow?"),
        ],
        normalize=False,
    )

    prompt = game.build_prompt([game.segments[0]])

    assert "Use weather_tool for forecasts." in prompt
    assert "User request:\n(none)" in prompt
