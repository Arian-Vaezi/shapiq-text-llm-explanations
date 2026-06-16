"""Tests for the Groq deterministic router coalition value-function scorer."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

from router_scorers import GroqDeterministicRouterScorer  # noqa: E402
from tool_schemas import TOOL_DESCRIPTIONS  # noqa: E402

FIXED_PROMPT = (
    "- Use weather_tool for weather, rain, temperature, forecast, or city-date questions.\n\n"
    "Available tools:\n"
    "- weather_tool: Get current weather conditions or a forecast.\n\n"
    "User request:\nWill it rain in Berlin tomorrow?\n\n"
    "Assistant:"
)


def make_recording_client_factory(*, selected_tool: str | None, raises_for_count: int = 0):
    """Build a fake Groq client factory that records each call and returns selected_tool."""
    calls: list[dict[str, object]] = []
    state = {"count": 0}

    class FakeCompletions:
        def create(self, *, model, messages, temperature, response_format):
            state["count"] += 1
            calls.append(
                {
                    "model": model,
                    "messages": messages,
                    "temperature": temperature,
                    "response_format": response_format,
                }
            )
            if selected_tool is None:
                content = "not json"
            else:
                content = f'{{"selected_tool":"{selected_tool}"}}'
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            )

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    factory = lambda api_key: FakeClient()  # noqa: E731
    return factory, calls


def test_score_batch_returns_one_when_selected_tool_matches_target(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, _calls = make_recording_client_factory(selected_tool="weather_tool")
    scorer = GroqDeterministicRouterScorer(client_factory=factory)

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [1.0]


def test_score_batch_returns_zero_when_selected_tool_differs_from_target(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, _calls = make_recording_client_factory(selected_tool="calculator_tool")
    scorer = GroqDeterministicRouterScorer(client_factory=factory)

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [0.0]


def test_score_batch_accepts_no_tool_as_target_and_candidate(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, _calls = make_recording_client_factory(selected_tool="no_tool")
    scorer = GroqDeterministicRouterScorer(client_factory=factory)

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="no_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [1.0]


def test_score_batch_preserves_prompt_order(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    class OrderedCompletions:
        def __init__(self) -> None:
            self.responses = iter(["weather_tool", "calculator_tool", "no_tool"])

        def create(self, *, model, messages, temperature, response_format):
            del model, messages, temperature, response_format
            tool = next(self.responses)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=f'{{"selected_tool":"{tool}"}}'),
                    ),
                ],
            )

    class OrderedChat:
        completions = OrderedCompletions()

    class OrderedClient:
        chat = OrderedChat()

    scorer = GroqDeterministicRouterScorer(client_factory=lambda api_key: OrderedClient())

    scores = scorer.score_batch(
        ["prompt one\n\nUser request:\nA\n\nAssistant:",
         "prompt two\n\nUser request:\nB\n\nAssistant:",
         "prompt three\n\nUser request:\nC\n\nAssistant:"],
        target_tool="calculator_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [0.0, 1.0, 0.0]


def test_score_batch_caches_repeated_prompts_and_avoids_extra_api_calls(monkeypatch) -> None:
    """choose_tool_with_scorer calls score_batch once per candidate tool with the same prompt."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, calls = make_recording_client_factory(selected_tool="weather_tool")
    scorer = GroqDeterministicRouterScorer(client_factory=factory)

    for target_tool in TOOL_DESCRIPTIONS:
        scorer.score_batch(
            [FIXED_PROMPT],
            target_tool=target_tool,
            tool_descriptions=TOOL_DESCRIPTIONS,
        )

    assert len(calls) == 1


def test_score_batch_treats_unparseable_output_as_no_match(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, _calls = make_recording_client_factory(selected_tool=None)
    scorer = GroqDeterministicRouterScorer(client_factory=factory)

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [0.0]
    assert scorer.last_debug_outputs[0]["selected_tool"] is None


def test_score_batch_rejects_unknown_target_tool(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, _calls = make_recording_client_factory(selected_tool="weather_tool")
    scorer = GroqDeterministicRouterScorer(client_factory=factory)

    with pytest.raises(ValueError, match="not a known decision candidate"):
        scorer.score_batch(
            [FIXED_PROMPT],
            target_tool="calendar_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )


def test_missing_api_key_raises_clear_error(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    scorer = GroqDeterministicRouterScorer()

    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        scorer.score_batch(
            [FIXED_PROMPT],
            target_tool="weather_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )


def test_router_prompt_requests_no_tool_arguments_or_final_answer(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, calls = make_recording_client_factory(selected_tool="weather_tool")
    scorer = GroqDeterministicRouterScorer(client_factory=factory)

    scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert len(calls) == 1
    call = calls[0]
    assert call["temperature"] == 0
    assert call["response_format"] == {"type": "json_object"}
    user_message = call["messages"][1]["content"]
    assert "tool_arguments" not in user_message
    assert "assistant_answer" not in user_message
    assert "no_tool" in user_message


def test_build_scoring_prompt_matches_actual_router_prompt(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, calls = make_recording_client_factory(selected_tool="weather_tool")
    scorer = GroqDeterministicRouterScorer(client_factory=factory)

    preview = scorer.build_scoring_prompt(
        FIXED_PROMPT,
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )
    scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert preview == calls[0]["messages"][1]["content"]
