"""Tests for the Groq deterministic router coalition value-function scorer."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

import router_scorers  # noqa: E402
from agent_failure import AgentFailureKind  # noqa: E402
from groq_agent import GroqInferenceResult  # noqa: E402
from router_scorers import (  # noqa: E402
    GroqDeterministicRouterScorer,
    GroqSoftVoteToolScorer,
    ToolTrajectory,
    TrajectoryArgumentMatchScorer,
    build_groq_inference_trajectory_provider,
)
from tool_schemas import EXECUTABLE_TOOL_SCHEMAS, TOOL_DESCRIPTIONS  # noqa: E402

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
        [
            "prompt one\n\nUser request:\nA\n\nAssistant:",
            "prompt two\n\nUser request:\nB\n\nAssistant:",
            "prompt three\n\nUser request:\nC\n\nAssistant:",
        ],
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


def make_soft_vote_client_factory(outputs: list[str]):
    """Build a fake Groq client factory that returns queued raw JSON/text outputs."""
    calls: list[dict[str, object]] = []

    class SoftVoteCompletions:
        def __init__(self) -> None:
            self.outputs = iter(outputs)

        def create(self, **kwargs):
            calls.append(kwargs)
            content = next(self.outputs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            )

    class SoftVoteChat:
        completions = SoftVoteCompletions()

    class SoftVoteClient:
        chat = SoftVoteChat()

    return lambda api_key: SoftVoteClient(), calls


def test_soft_vote_score_single_returns_one_for_all_target_votes(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, _calls = make_soft_vote_client_factory(
        ['{"best_tool":"weather_tool"}'] * 5,
    )
    scorer = GroqSoftVoteToolScorer(n_samples=5, client_factory=factory)

    score, votes, _raw_outputs = scorer.score_single(
        FIXED_PROMPT,
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert score == 1.0
    assert votes == ["weather_tool"] * 5


def test_soft_vote_score_single_returns_zero_for_no_target_votes(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, _calls = make_soft_vote_client_factory(
        [
            '{"best_tool":"calculator_tool"}',
            '{"best_tool":"no_tool"}',
            '{"best_tool":"web_search_tool"}',
        ],
    )
    scorer = GroqSoftVoteToolScorer(n_samples=3, client_factory=factory)

    score, votes, _raw_outputs = scorer.score_single(
        FIXED_PROMPT,
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert score == 0.0
    assert votes == ["calculator_tool", "no_tool", "web_search_tool"]


def test_soft_vote_score_single_returns_fractional_target_frequency(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, _calls = make_soft_vote_client_factory(
        [
            '{"best_tool":"weather_tool"}',
            '{"best_tool":"calculator_tool"}',
            '{"best_tool":"weather_tool"}',
            '{"best_tool":"no_tool"}',
            '{"best_tool":"weather_tool"}',
        ],
    )
    scorer = GroqSoftVoteToolScorer(n_samples=5, client_factory=factory)

    score, _votes, _raw_outputs = scorer.score_single(
        FIXED_PROMPT,
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert score == 0.6


def test_soft_vote_score_batch_returns_one_score_per_prompt(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, _calls = make_soft_vote_client_factory(
        [
            '{"best_tool":"weather_tool"}',
            '{"best_tool":"calculator_tool"}',
            '{"best_tool":"calculator_tool"}',
            '{"best_tool":"weather_tool"}',
        ],
    )
    scorer = GroqSoftVoteToolScorer(n_samples=2, client_factory=factory)

    scores = scorer.score_batch(
        [FIXED_PROMPT, FIXED_PROMPT.replace("Berlin", "Paris")],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [0.5, 0.5]
    assert len(scorer.last_debug_outputs) == 2


def test_soft_vote_debug_output_uses_soft_vote_wording(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, _calls = make_soft_vote_client_factory(
        ['{"best_tool":"weather_tool"}', '{"best_tool":"calculator_tool"}'],
    )
    scorer = GroqSoftVoteToolScorer(n_samples=2, client_factory=factory)

    scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    debug_output = scorer.last_debug_outputs[0]
    assert debug_output["score_kind"] == "soft-vote score"
    assert debug_output["score_description"] == "empirical target-tool selection frequency"
    assert "reference_tool" not in debug_output
    assert "margin" not in debug_output


def test_soft_vote_retries_invalid_outputs_then_counts_valid_vote(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, calls = make_soft_vote_client_factory(
        [
            "not json",
            '{"best_tool":"calendar_tool"}',
            '{"best_tool":"weather_tool"}',
        ],
    )
    scorer = GroqSoftVoteToolScorer(n_samples=1, max_retries=2, client_factory=factory)

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [1.0]
    assert len(calls) == 3


def test_soft_vote_invalid_outputs_fall_back_to_no_match(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    factory, calls = make_soft_vote_client_factory(
        ["not json", '{"best_tool":"calendar_tool"}'],
    )
    scorer = GroqSoftVoteToolScorer(n_samples=1, max_retries=1, client_factory=factory)

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [0.0]
    assert scorer.last_debug_outputs[0]["selected_tools"] == [None]
    assert len(calls) == 2


def make_recording_provider(trajectories_by_prompt: dict[str, ToolTrajectory]):
    """Build a fake trajectory_provider that records each prompt it is asked about."""
    calls: list[str] = []

    def provider(prompt: str) -> ToolTrajectory:
        calls.append(prompt)
        return trajectories_by_prompt[prompt]

    return provider, calls


def test_trajectory_score_returns_zero_for_different_selected_tool() -> None:
    reference = ToolTrajectory(selected_tool="weather_tool", tool_arguments={"location": "Berlin"})
    provider, _calls = make_recording_provider(
        {FIXED_PROMPT: ToolTrajectory(selected_tool="calculator_tool", tool_arguments={})},
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [0.0]


def test_trajectory_score_returns_one_when_tool_matches_and_reference_has_no_arguments() -> None:
    reference = ToolTrajectory(selected_tool="no_tool", tool_arguments={})
    provider, _calls = make_recording_provider(
        {FIXED_PROMPT: ToolTrajectory(selected_tool="no_tool", tool_arguments={})},
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="no_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [1.0]


def test_trajectory_score_full_argument_match_uses_both_weights() -> None:
    reference = ToolTrajectory(
        selected_tool="weather_tool",
        tool_arguments={"location": "Berlin", "date": "tomorrow"},
    )
    provider, _calls = make_recording_provider(
        {
            FIXED_PROMPT: ToolTrajectory(
                selected_tool="weather_tool",
                tool_arguments={"location": "Berlin", "date": "tomorrow"},
            ),
        },
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
        tool_match_weight=0.5,
        arg_match_weight=0.5,
    )

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == pytest.approx([1.0])


def test_trajectory_score_partial_argument_match_ratio() -> None:
    reference = ToolTrajectory(
        selected_tool="weather_tool",
        tool_arguments={"location": "Berlin", "date": "tomorrow"},
    )
    provider, _calls = make_recording_provider(
        {
            FIXED_PROMPT: ToolTrajectory(
                selected_tool="weather_tool",
                tool_arguments={"location": "Berlin"},
            ),
        },
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
        tool_match_weight=0.5,
        arg_match_weight=0.5,
    )

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == pytest.approx([0.5 + 0.5 * 0.5])
    assert scorer.last_debug_outputs[0]["argument_match_ratio"] == pytest.approx(0.5)


def test_trajectory_score_normalizes_alias_keys() -> None:
    reference = ToolTrajectory(
        selected_tool="weather_tool",
        tool_arguments={"location": "Berlin", "date": "tomorrow"},
    )
    provider, _calls = make_recording_provider(
        {
            FIXED_PROMPT: ToolTrajectory(
                selected_tool="weather_tool",
                # uses aliases instead of the canonical "location"/"date" keys
                tool_arguments={"city": "Berlin", "date_or_time": "tomorrow"},
            ),
        },
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == pytest.approx([1.0])


def test_trajectory_score_value_normalization_ignores_case_and_punctuation() -> None:
    reference = ToolTrajectory(
        selected_tool="web_search_tool", tool_arguments={"query": "Apple's newest product"}
    )
    provider, _calls = make_recording_provider(
        {
            FIXED_PROMPT: ToolTrajectory(
                selected_tool="web_search_tool",
                tool_arguments={"query": "  APPLES NEWEST PRODUCT!! "},
            ),
        },
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="web_search_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == pytest.approx([1.0])


def test_trajectory_score_strict_text_equality_different_text_scores_tool_weight_only() -> None:
    # "Berlin" != "weather in Berlin tomorrow" under strict equality → arg ratio 0
    reference = ToolTrajectory(selected_tool="web_search_tool", tool_arguments={"query": "Berlin"})
    provider, _calls = make_recording_provider(
        {
            FIXED_PROMPT: ToolTrajectory(
                selected_tool="web_search_tool",
                tool_arguments={"query": "weather in Berlin tomorrow"},
            ),
        },
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="web_search_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    # tool match (0.5) + arg_match_weight * 0 = 0.5
    assert scores == pytest.approx([0.5])


def test_trajectory_score_calculator_expression_arithmetic_normalization() -> None:
    reference = ToolTrajectory(
        selected_tool="calculator_tool",
        tool_arguments={"expression": "238 * 47"},
    )
    provider, _calls = make_recording_provider(
        {
            FIXED_PROMPT: ToolTrajectory(
                selected_tool="calculator_tool",
                tool_arguments={"expression": "238*47"},
            ),
        },
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="calculator_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == pytest.approx([1.0])


def test_trajectory_score_calculator_expression_mismatch_scores_zero_ratio() -> None:
    reference = ToolTrajectory(
        selected_tool="calculator_tool",
        tool_arguments={"expression": "238 * 47"},
    )
    provider, _calls = make_recording_provider(
        {
            FIXED_PROMPT: ToolTrajectory(
                selected_tool="calculator_tool",
                tool_arguments={"expression": "1 + 1"},
            ),
        },
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="calculator_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == pytest.approx([0.5])


def test_trajectory_score_rejects_target_tool_not_matching_reference() -> None:
    reference = ToolTrajectory(selected_tool="weather_tool", tool_arguments={"location": "Berlin"})
    provider, _calls = make_recording_provider({})
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    with pytest.raises(ValueError, match="does not match the reference trajectory"):
        scorer.score_batch(
            [FIXED_PROMPT],
            target_tool="calculator_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )


def test_trajectory_score_rejects_unknown_target_tool() -> None:
    reference = ToolTrajectory(selected_tool="weather_tool", tool_arguments={"location": "Berlin"})
    provider, _calls = make_recording_provider({})
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    with pytest.raises(ValueError, match="not a known decision candidate"):
        scorer.score_batch(
            [FIXED_PROMPT],
            target_tool="calendar_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )


def test_trajectory_score_caches_provider_calls_per_prompt() -> None:
    reference = ToolTrajectory(selected_tool="weather_tool", tool_arguments={"location": "Berlin"})
    provider, calls = make_recording_provider(
        {
            FIXED_PROMPT: ToolTrajectory(
                selected_tool="weather_tool", tool_arguments={"location": "Berlin"}
            )
        },
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )
    scorer.score_batch(
        [FIXED_PROMPT, FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert calls == [FIXED_PROMPT]


def test_trajectory_score_rejects_negative_weights() -> None:
    reference = ToolTrajectory(selected_tool="weather_tool", tool_arguments={})
    provider, _calls = make_recording_provider({})

    with pytest.raises(ValueError, match="non-negative"):
        TrajectoryArgumentMatchScorer(
            reference_trajectory=reference,
            trajectory_provider=provider,
            tool_match_weight=-0.1,
        )


# ---------------------------------------------------------------------------
# New tests: trajectory fidelity edge cases
# ---------------------------------------------------------------------------


def test_trajectory_score_alias_on_both_sides_still_scores_one() -> None:
    # Reference uses alias "city"; coalition uses alias "date_or_time". Both canonicalize.
    reference = ToolTrajectory(
        selected_tool="weather_tool",
        tool_arguments={"city": "Berlin", "date_or_time": "tomorrow"},
    )
    provider, _calls = make_recording_provider(
        {
            FIXED_PROMPT: ToolTrajectory(
                selected_tool="weather_tool",
                tool_arguments={"location": "Berlin", "date": "tomorrow"},
            ),
        },
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == pytest.approx([1.0])


def test_trajectory_score_conflicting_aliases_raises_value_error() -> None:
    # "city" and "place" both map to "location" but carry different values.
    reference = ToolTrajectory(selected_tool="weather_tool", tool_arguments={"location": "Berlin"})
    provider, _calls = make_recording_provider(
        {
            FIXED_PROMPT: ToolTrajectory(
                selected_tool="weather_tool",
                tool_arguments={"city": "Berlin", "place": "Paris"},
            ),
        },
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    with pytest.raises(ValueError, match="Conflicting values"):
        scorer.score_batch(
            [FIXED_PROMPT],
            target_tool="weather_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )


def test_trajectory_score_weights_not_summing_to_one_raises_value_error() -> None:
    reference = ToolTrajectory(selected_tool="weather_tool", tool_arguments={"location": "Berlin"})
    provider, _calls = make_recording_provider({})

    with pytest.raises(ValueError, match=r"sum to 1\.0"):
        TrajectoryArgumentMatchScorer(
            reference_trajectory=reference,
            trajectory_provider=provider,
            tool_match_weight=0.6,
            arg_match_weight=0.6,
        )


def test_trajectory_score_unicode_multiply_matches_ascii() -> None:
    # "238 * 47" should evaluate to the same value as "238 * 47".
    reference = ToolTrajectory(
        selected_tool="calculator_tool",
        tool_arguments={"expression": "238 \u00d7 47"},
    )
    provider, _calls = make_recording_provider(
        {
            FIXED_PROMPT: ToolTrajectory(
                selected_tool="calculator_tool",
                tool_arguments={"expression": "238 * 47"},
            ),
        },
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="calculator_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == pytest.approx([1.0])


def test_trajectory_score_unicode_multiply_commutative_match() -> None:
    # "238 * 47" and "47 * 238" both evaluate to 11186.
    reference = ToolTrajectory(
        selected_tool="calculator_tool",
        tool_arguments={"expression": "238 \u00d7 47"},
    )
    provider, _calls = make_recording_provider(
        {
            FIXED_PROMPT: ToolTrajectory(
                selected_tool="calculator_tool",
                tool_arguments={"expression": "47 * 238"},
            ),
        },
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="calculator_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == pytest.approx([1.0])


def test_trajectory_score_york_does_not_match_new_york() -> None:
    # Strict text equality: "york" must not match "new york".
    reference = ToolTrajectory(selected_tool="web_search_tool", tool_arguments={"query": "York"})
    provider, _calls = make_recording_provider(
        {
            FIXED_PROMPT: ToolTrajectory(
                selected_tool="web_search_tool",
                tool_arguments={"query": "New York"},
            ),
        },
    )
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="web_search_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    # tool match (0.5) + arg_match_weight * 0 = 0.5
    assert scores == pytest.approx([0.5])


def test_trajectory_score_reference_unknown_tool_raises_at_construction() -> None:
    reference = ToolTrajectory(selected_tool="calendar_tool", tool_arguments={"date": "tomorrow"})
    provider, _calls = make_recording_provider({})

    with pytest.raises(ValueError, match="not a known decision"):
        TrajectoryArgumentMatchScorer(
            reference_trajectory=reference,
            trajectory_provider=provider,
        )


def test_trajectory_score_reference_missing_required_arg_raises_at_construction() -> None:
    # weather_tool requires "location"; providing only "date" is invalid.
    reference = ToolTrajectory(selected_tool="weather_tool", tool_arguments={"date": "tomorrow"})
    provider, _calls = make_recording_provider({})

    with pytest.raises(ValueError, match="missing required arguments"):
        TrajectoryArgumentMatchScorer(
            reference_trajectory=reference,
            trajectory_provider=provider,
        )


def test_groq_trajectory_provider_transient_failure_raises_coalition_transient_error(
    monkeypatch,
) -> None:
    # Arrange: Groq returns a transient failure (rate limit).
    def fake_groq_inference(user_request, tool_schemas, model_name, **kwargs):
        return GroqInferenceResult(
            available=False,
            error="429 rate limit",
            failure_kind=AgentFailureKind.RATE_LIMIT,
        )

    monkeypatch.setattr(router_scorers, "run_groq_tool_inference", fake_groq_inference)
    provider = build_groq_inference_trajectory_provider(
        model_name="llama-3.1-8b-instant",
        tool_schemas=EXECUTABLE_TOOL_SCHEMAS,
        tool_context="dummy context",
    )

    # Act / Assert — use the class from router_scorers to avoid dual-path identity issues
    with pytest.raises(router_scorers.CoalitionTransientFailureError, match="rate limit"):
        provider(FIXED_PROMPT)


def test_groq_trajectory_provider_malformed_response_raises_value_error(
    monkeypatch,
) -> None:
    # Arrange: Groq returns a non-transient failure (malformed JSON).
    def fake_groq_inference(user_request, tool_schemas, model_name, **kwargs):
        return GroqInferenceResult(
            error="Groq returned malformed JSON.",
            failure_kind=AgentFailureKind.INVALID_REQUEST,
        )

    monkeypatch.setattr(router_scorers, "run_groq_tool_inference", fake_groq_inference)
    provider = build_groq_inference_trajectory_provider(
        model_name="llama-3.1-8b-instant",
        tool_schemas=EXECUTABLE_TOOL_SCHEMAS,
        tool_context="dummy context",
    )

    # Act / Assert: non-transient → ValueError, not a silent empty trajectory
    with pytest.raises(ValueError, match="malformed JSON"):
        provider(FIXED_PROMPT)


def test_groq_trajectory_provider_failure_not_cached_second_call_retries(
    monkeypatch,
) -> None:
    # Arrange: first call raises, second call succeeds.
    call_count = {"n": 0}

    def failing_then_succeeding(user_request, tool_schemas, model_name, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return GroqInferenceResult(
                error="Groq returned malformed JSON.",
                failure_kind=AgentFailureKind.INVALID_REQUEST,
            )
        return GroqInferenceResult(
            selected_tool="no_tool",
            tool_arguments={},
        )

    monkeypatch.setattr(router_scorers, "run_groq_tool_inference", failing_then_succeeding)
    provider = build_groq_inference_trajectory_provider(
        model_name="llama-3.1-8b-instant",
        tool_schemas=EXECUTABLE_TOOL_SCHEMAS,
        tool_context="dummy context",
    )
    reference = ToolTrajectory(selected_tool="no_tool", tool_arguments={})
    scorer = TrajectoryArgumentMatchScorer(
        reference_trajectory=reference,
        trajectory_provider=provider,
    )

    # First call raises (provider failure, not cached).
    with pytest.raises(ValueError):
        scorer.score_batch(
            [FIXED_PROMPT],
            target_tool="no_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )

    # Second call invokes provider again (not returning a cached failure).
    scores = scorer.score_batch(
        [FIXED_PROMPT],
        target_tool="no_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert call_count["n"] == 2
    assert scores == pytest.approx([1.0])
