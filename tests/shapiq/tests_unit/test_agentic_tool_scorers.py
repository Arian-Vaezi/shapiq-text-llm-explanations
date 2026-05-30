"""Tests for agentic tool-use demo scorers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

from scorers import (  # noqa: E402
    LexicalToolRouter,
    LexicalToolScorer,
    LLMToolScorer,
    LogProbToolScorer,
    MockLLM,
)
from tool_game import ToolUseGame, ToolUseSegment  # noqa: E402

TOOL_DESCRIPTIONS = {
    "weather_tool": "Fetch current weather or forecasts for a place and date.",
    "calculator_tool": "Compute exact arithmetic and numeric expressions.",
    "web_search_tool": "Search the web for current, recent, or external facts.",
    "no_tool": "Answer directly without calling an external tool.",
}


def make_fake_logprob_scorer(
    logprobs: dict[str, float],
    candidate_texts: dict[str, str] | None = None,
) -> LogProbToolScorer:
    scorer = LogProbToolScorer.__new__(LogProbToolScorer)
    scorer.candidate_template = "The correct tool is {tool_name}."
    scorer.candidate_texts = candidate_texts or {}
    scorer.normalize_by_length = True
    scorer.last_debug_outputs = []

    def fake_sequence_logprob(prompt: str, continuation: str) -> float:
        del prompt
        for tool_name, score in logprobs.items():
            if tool_name in continuation:
                return score
        return -10.0

    scorer._sequence_logprob = fake_sequence_logprob
    return scorer


class FakeGenerator:
    """Small deterministic generator for scorer tests."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        del prompt
        return self.response


def test_llm_tool_scorer_parse_score_accepts_plain_number() -> None:
    scorer = LLMToolScorer(llm=FakeGenerator("0.7"))

    assert scorer.parse_score("0.7") == 0.7


def test_llm_tool_scorer_parse_score_accepts_labeled_number() -> None:
    scorer = LLMToolScorer(llm=FakeGenerator("Score: 0.7"))

    assert scorer.parse_score("Score: 0.7") == 0.7
    assert scorer.parse_score("The score is 0.7\n") == 0.7


def test_llm_tool_scorer_parse_score_rejects_out_of_range_number() -> None:
    scorer = LLMToolScorer(llm=FakeGenerator("1.5"))

    with pytest.raises(ValueError):
        scorer.parse_score("1.5")


def test_llm_tool_scorer_parse_score_rejects_tool_index() -> None:
    scorer = LLMToolScorer(llm=FakeGenerator("tool 3"))

    with pytest.raises(ValueError):
        scorer.parse_score("tool 3")


def test_llm_tool_scorer_parse_score_rejects_text_without_number() -> None:
    scorer = LLMToolScorer(llm=FakeGenerator("not a number"))

    with pytest.raises(ValueError):
        scorer.parse_score("not a number")


def test_llm_tool_scorer_parses_numeric_output() -> None:
    scorer = LLMToolScorer(llm=MockLLM("0.75"))

    scores = scorer.score_batch(
        ["User asks whether it will rain in Berlin tomorrow."],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [0.75]


def test_llm_tool_scorer_falls_back_on_out_of_range_numeric_output() -> None:
    fallback = LexicalToolScorer()
    scorer = LLMToolScorer(llm=MockLLM("1.7"), fallback_scorer=fallback)
    prompt = "User asks whether it will rain in Berlin tomorrow."

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
    assert scorer.last_debug_outputs[0]["used_fallback"] is True


def test_llm_tool_scorer_score_batch_uses_fake_generator() -> None:
    scorer = LLMToolScorer(llm=FakeGenerator("0.8"))

    scores = scorer.score_batch(
        [
            "Use weather_tool for forecasts.",
            "Will it rain in Berlin tomorrow?",
        ],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [0.8, 0.8]
    assert scorer.last_debug_outputs == [
        {
            "target_tool": "weather_tool",
            "raw_output": "0.8",
            "parsed_score": 0.8,
            "used_fallback": False,
            "fallback_score": None,
            "final_score": 0.8,
        },
        {
            "target_tool": "weather_tool",
            "raw_output": "0.8",
            "parsed_score": 0.8,
            "used_fallback": False,
            "fallback_score": None,
            "final_score": 0.8,
        },
    ]


def test_llm_tool_scorer_falls_back_on_invalid_output() -> None:
    fallback = LexicalToolScorer()
    scorer = LLMToolScorer(llm=FakeGenerator("not a score"), fallback_scorer=fallback)
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
    assert len(scorer.last_debug_outputs) == 1
    debug_output = scorer.last_debug_outputs[0]
    assert debug_output["raw_output"] == "not a score"
    assert debug_output["parsed_score"] is None
    assert debug_output["used_fallback"] is True
    assert debug_output["fallback_score"] == scores[0]
    assert debug_output["final_score"] == scores[0]


def test_logprob_tool_scorer_returns_array_for_prompts() -> None:
    scorer = make_fake_logprob_scorer(
        {
            "weather_tool": 2.0,
            "calculator_tool": 0.0,
            "web_search_tool": -1.0,
            "no_tool": -2.0,
        }
    )

    scores = scorer.score_batch(
        ["Prompt one", "Prompt two"],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert isinstance(scores, np.ndarray)
    assert scores.shape == (2,)
    assert len(scorer.last_debug_outputs) == 2


def test_logprob_tool_scorer_prefers_highest_target_logprob() -> None:
    scorer = make_fake_logprob_scorer(
        {
            "weather_tool": 4.0,
            "calculator_tool": 0.0,
            "web_search_tool": -1.0,
            "no_tool": -2.0,
        }
    )

    scores = scorer.score_batch(
        ["Will it rain tomorrow?"],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores[0] > 0.5


def test_logprob_tool_scorer_debug_probabilities_sum_to_one() -> None:
    scorer = make_fake_logprob_scorer(
        {
            "weather_tool": 1.5,
            "calculator_tool": 0.5,
            "web_search_tool": -0.5,
            "no_tool": -1.5,
        }
    )

    scorer.score_batch(
        ["Will it rain tomorrow?"],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    debug_output = scorer.last_debug_outputs[0]
    assert debug_output["target_tool"] == "weather_tool"
    assert set(debug_output["candidate_tools"]) == set(TOOL_DESCRIPTIONS)
    assert np.isclose(sum(debug_output["candidate_probs"]), 1.0)
    assert 0.0 <= debug_output["final_score"] <= 1.0


def test_logprob_tool_scorer_uses_candidate_text_override_for_no_tool() -> None:
    no_tool_text = "The assistant should answer directly without using an external tool."
    scorer = make_fake_logprob_scorer(
        {"weather_tool": 1.0, "no_tool": 0.5},
        candidate_texts={"no_tool": no_tool_text},
    )

    scorer.score_batch(
        ["Explain photosynthesis."],
        target_tool="no_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    debug_output = scorer.last_debug_outputs[0]
    no_tool_index = debug_output["candidate_tools"].index("no_tool")
    assert debug_output["candidate_continuations"][no_tool_index] == no_tool_text


def test_logprob_tool_scorer_falls_back_to_template_for_missing_candidate_text() -> None:
    scorer = make_fake_logprob_scorer(
        {"weather_tool": 1.0, "no_tool": 0.5},
        candidate_texts={"no_tool": "Answer directly."},
    )

    scorer.score_batch(
        ["Will it rain tomorrow?"],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    debug_output = scorer.last_debug_outputs[0]
    weather_index = debug_output["candidate_tools"].index("weather_tool")
    assert (
        debug_output["candidate_continuations"][weather_index]
        == "The correct tool is weather_tool."
    )


def test_logprob_tool_scorer_debug_contains_candidate_continuations() -> None:
    scorer = make_fake_logprob_scorer({"weather_tool": 1.0})

    scorer.score_batch(
        ["Will it rain tomorrow?"],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    debug_output = scorer.last_debug_outputs[0]
    assert "candidate_continuations" in debug_output
    assert len(debug_output["candidate_continuations"]) == len(TOOL_DESCRIPTIONS)


def test_logprob_tool_scorer_requires_candidate_tools() -> None:
    scorer = make_fake_logprob_scorer({"weather_tool": 1.0})

    with pytest.raises(ValueError):
        scorer.score_batch(
            ["Will it rain tomorrow?"],
            target_tool="weather_tool",
            tool_descriptions={},
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
