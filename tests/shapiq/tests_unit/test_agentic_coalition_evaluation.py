"""Tests for typed, retrying coalition evaluation in the agentic tool-use demo.

These tests use fake evaluators/scorers and an injected ``sleep_fn`` -- no real
provider calls, no HF downloads, no Streamlit runtime, and no real sleeping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import shapiq

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

import app  # noqa: E402
from _app_impl import shapley  # noqa: E402
from coalition_evaluation import (  # noqa: E402
    CoalitionAttemptResult,
    CoalitionEvaluationIncompleteError,
    CoalitionEvaluationStatus,
    RetryPolicy,
    aggregate_metrics,
    ensure_all_real,
    evaluate_coalition_with_retry,
    evaluate_coalitions,
    evaluate_game_exactly,
)
from exact_interactions import MAX_EXACT_DEMO_PLAYERS, compute_exact_interactions  # noqa: E402
from tool_game import ToolUseGame  # noqa: E402
from tool_schemas import TOOL_DESCRIPTIONS  # noqa: E402

RETRY_POLICY = RetryPolicy(initial_delay_seconds=0.5, max_delay_seconds=8.0, backoff_multiplier=2.0)


class TransientError(Exception):
    """Stand-in for a simulated rate-limit/timeout-style transient provider failure."""


def make_toy_game(scorer: object, n_segments: int = 3) -> ToolUseGame:
    """Build a toy game opted into the deferred (exact-demo) empty-coalition path.

    ``defer_empty_coalition_evaluation=True`` is required here: it is the only
    supported way for evaluate_game_exactly to evaluate the empty coalition without
    construction having already (eagerly, unretried) scored it once itself.
    """
    return ToolUseGame(
        target_tool="weather_tool",
        user_segments=[f"SEGMENT{i}" for i in range(n_segments)],
        system_prompt="Use weather_tool for weather questions.",
        scorer=scorer,
        tool_descriptions={"weather_tool": TOOL_DESCRIPTIONS["weather_tool"]},
        normalize=False,
        defer_empty_coalition_evaluation=True,
    )


def classify_only_transient_error(exc: BaseException) -> str:
    """Classifier recognizing only the test's own TransientError as transient."""
    return "transient" if isinstance(exc, TransientError) else "semantic"


# --- Test 1: transient failure then success -------------------------------------------


def test_transient_failure_then_success() -> None:
    calls = {"n": 0}

    def attempt() -> CoalitionAttemptResult:
        calls["n"] += 1
        if calls["n"] == 1:
            msg = "rate limited"
            raise TransientError(msg)
        return CoalitionAttemptResult(score=0.42)

    sleeps: list[float] = []
    outcome = evaluate_coalition_with_retry(
        attempt,
        retry_policy=RETRY_POLICY,
        sleep_fn=sleeps.append,
        classify_exception_fn=classify_only_transient_error,
    )

    assert outcome.status is CoalitionEvaluationStatus.REAL
    assert outcome.score == pytest.approx(0.42)
    assert outcome.attempts == 2
    assert outcome.retry_count == 1
    assert sleeps == [0.5]  # initial_delay_seconds * multiplier**0

    metrics = aggregate_metrics([outcome])
    assert metrics.retry_triggered_count == 1
    assert metrics.retry_success_count == 1
    assert metrics.real_count == 1


# --- Test 2: transient failure exhausts retries ----------------------------------------


def test_transient_failure_exhausts_retries() -> None:
    def attempt() -> CoalitionAttemptResult:
        msg = "still rate limited"
        raise TransientError(msg)

    sleeps: list[float] = []
    outcome = evaluate_coalition_with_retry(
        attempt,
        retry_policy=RETRY_POLICY,
        sleep_fn=sleeps.append,
        classify_exception_fn=classify_only_transient_error,
    )

    assert outcome.status is CoalitionEvaluationStatus.RETRY_EXHAUSTED
    assert outcome.score is None

    metrics = aggregate_metrics([outcome])
    assert metrics.retry_exhausted_count == 1
    assert metrics.real_count == 0

    with pytest.raises(CoalitionEvaluationIncompleteError):
        ensure_all_real([outcome], metrics)


# --- Test 3: semantic failure is not retried --------------------------------------------


def test_semantic_failure_is_not_retried() -> None:
    def attempt() -> CoalitionAttemptResult:
        msg = "no final answer"
        raise ValueError(msg)

    def fail_if_called(_delay: float) -> None:
        msg = "must not sleep for a non-retried semantic failure"
        raise AssertionError(msg)

    outcome = evaluate_coalition_with_retry(
        attempt,
        retry_policy=RETRY_POLICY,
        sleep_fn=fail_if_called,
        classify_exception_fn=classify_only_transient_error,
    )

    assert outcome.status is CoalitionEvaluationStatus.SEMANTIC_FAILURE
    assert outcome.attempts == 1
    assert outcome.retry_count == 0

    metrics = aggregate_metrics([outcome])
    assert metrics.semantic_failure_count == 1

    with pytest.raises(CoalitionEvaluationIncompleteError):
        ensure_all_real([outcome], metrics)


# --- Test 4: one bad coalition aborts the whole exact explanation -----------------------


def test_one_bad_coalition_aborts_whole_explanation(monkeypatch) -> None:
    class MostlyGoodScorer:
        def score_batch(self, prompts, *, target_tool, tool_descriptions):
            del target_tool, tool_descriptions
            out = []
            for prompt in prompts:
                if "SEGMENT1" in prompt and "SEGMENT0" not in prompt and "SEGMENT2" not in prompt:
                    msg = "no final answer"
                    raise ValueError(msg)
                out.append(0.1 * prompt.count("SEGMENT"))
            return out

    game = make_toy_game(MostlyGoodScorer())

    # First, confirm evaluate_game_exactly itself raises and leaves the game untouched.
    with pytest.raises(CoalitionEvaluationIncompleteError) as exc_info:
        evaluate_game_exactly(game, retry_policy=RETRY_POLICY, sleep_fn=lambda _d: None)

    metrics = exc_info.value.metrics
    assert metrics.coalition_total == 8
    assert metrics.real_count == 7
    assert metrics.semantic_failure_count == 1
    assert not game.precomputed

    # Now confirm the full app-facing integration path never reaches ExactComputer either:
    # patch app.compute_exact_interactions (the name actually bound in app.py's namespace)
    # with a spy that fails the test if it is ever invoked.
    def spy_compute_exact_interactions(**kwargs):
        msg = "compute_exact_interactions must not be called"
        raise AssertionError(msg)

    monkeypatch.setattr(shapley, "compute_exact_interactions", spy_compute_exact_interactions)

    second_game = make_toy_game(MostlyGoodScorer())
    with pytest.raises(CoalitionEvaluationIncompleteError):
        app.compute_interaction_explanation(
            game=second_game,
            index="k-SII",
            max_order=2,
            budget=None,
        )


# --- Test 5: all-real outcome still reaches ExactComputer -------------------------------


def test_all_real_outcome_reaches_exact_computer() -> None:
    class DeterministicScorer:
        def score_batch(self, prompts, *, target_tool, tool_descriptions):
            del target_tool, tool_descriptions
            return [0.1 * prompt.count("SEGMENT") for prompt in prompts]

    game = make_toy_game(DeterministicScorer())

    metrics = evaluate_game_exactly(game, retry_policy=RETRY_POLICY, sleep_fn=lambda _d: None)

    assert metrics.real_count == 8
    assert metrics.fallback_count == 0
    assert game.precomputed

    result, computed_metadata = compute_exact_interactions(game=game, index="k-SII", max_order=2)
    assert isinstance(result, shapiq.InteractionValues)
    assert computed_metadata.coalition_count == 8


# --- Test 6: retry metrics distinguish path health ---------------------------------------


def test_retry_metrics_distinguish_path_health() -> None:
    def make_always_succeeds() -> list:
        return [lambda: CoalitionAttemptResult(score=0.1) for _ in range(8)]

    def make_two_retry_then_succeed() -> list:
        attempt_fns = []
        for index in range(8):
            state = {"n": 0}

            def attempt(state=state, index=index) -> CoalitionAttemptResult:
                state["n"] += 1
                if index < 2 and state["n"] == 1:
                    msg = "flaky"
                    raise TransientError(msg)
                return CoalitionAttemptResult(score=0.1)

            attempt_fns.append(attempt)
        return attempt_fns

    _, metrics_a = evaluate_coalitions(
        make_always_succeeds(),
        retry_policy=RETRY_POLICY,
        sleep_fn=lambda _d: None,
        classify_exception_fn=classify_only_transient_error,
    )
    _, metrics_b = evaluate_coalitions(
        make_two_retry_then_succeed(),
        retry_policy=RETRY_POLICY,
        sleep_fn=lambda _d: None,
        classify_exception_fn=classify_only_transient_error,
    )

    # Both scenarios are fully real with zero fallbacks -- but they are not equivalent.
    assert metrics_a.real_count == metrics_b.real_count == 8
    assert metrics_a.fallback_count == metrics_b.fallback_count == 0

    assert metrics_a.retry_triggered_count == 0
    assert metrics_a.retry_success_count == 0
    assert metrics_b.retry_triggered_count == 2
    assert metrics_b.retry_success_count == 2
    assert metrics_a != metrics_b


# --- Gap 1: empty coalition must go through the same retry/fail-loud protocol -----------


def test_three_player_exact_run_makes_exactly_eight_scorer_calls() -> None:
    """No double evaluation: construction must not separately score the empty coalition."""

    class CountingScorer:
        def __init__(self) -> None:
            self.calls = 0

        def score_batch(self, prompts, *, target_tool, tool_descriptions):
            del target_tool, tool_descriptions
            self.calls += 1
            return [0.1 * prompt.count("SEGMENT") for prompt in prompts]

    scorer = CountingScorer()
    game = make_toy_game(scorer)
    assert scorer.calls == 0  # construction itself must not call the scorer

    evaluate_game_exactly(game, retry_policy=RETRY_POLICY, sleep_fn=lambda _d: None)

    assert scorer.calls == 8  # one call per coalition, including the empty one -- not 9


def test_empty_coalition_succeeds_after_transient_failure_and_is_counted() -> None:
    calls = {"n": 0}

    class FlakyOnEmptyScorer:
        def score_batch(self, prompts, *, target_tool, tool_descriptions):
            del target_tool, tool_descriptions
            out = []
            for prompt in prompts:
                if prompt.count("SEGMENT") == 0:
                    calls["n"] += 1
                    if calls["n"] == 1:
                        msg = "rate limited on empty coalition"
                        raise TransientError(msg)
                out.append(0.1 * prompt.count("SEGMENT"))
            return out

    game = make_toy_game(FlakyOnEmptyScorer())

    metrics = evaluate_game_exactly(
        game,
        retry_policy=RETRY_POLICY,
        sleep_fn=lambda _d: None,
        classify_exception_fn=classify_only_transient_error,
    )

    assert metrics.coalition_total == 8
    assert metrics.real_count == 8
    assert metrics.retry_triggered_count == 1
    assert metrics.retry_success_count == 1
    assert game.precomputed
    assert game.normalization_value == pytest.approx(0.0)
    assert float(game(game.empty_coalition)[0]) == pytest.approx(0.0)


def test_empty_coalition_retry_exhaustion_aborts_whole_explanation() -> None:
    class AlwaysFailsOnEmptyScorer:
        def score_batch(self, prompts, *, target_tool, tool_descriptions):
            del target_tool, tool_descriptions
            out = []
            for prompt in prompts:
                if prompt.count("SEGMENT") == 0:
                    msg = "still rate limited on empty coalition"
                    raise TransientError(msg)
                out.append(0.1 * prompt.count("SEGMENT"))
            return out

    game = make_toy_game(AlwaysFailsOnEmptyScorer())

    with pytest.raises(CoalitionEvaluationIncompleteError) as exc_info:
        evaluate_game_exactly(
            game,
            retry_policy=RETRY_POLICY,
            sleep_fn=lambda _d: None,
            classify_exception_fn=classify_only_transient_error,
        )

    metrics = exc_info.value.metrics
    assert metrics.coalition_total == 8
    assert metrics.real_count == 7
    assert metrics.retry_exhausted_count == 1
    # No precomputed table and no ExactComputer call may happen after this failure.
    assert not game.precomputed
    assert game.game_values == {}


def test_normalized_empty_coalition_remains_correct_after_successful_precomputation() -> None:
    class OffsetScorer:
        def score_batch(self, prompts, *, target_tool, tool_descriptions):
            del target_tool, tool_descriptions
            return [0.1 * prompt.count("SEGMENT") + 5.0 for prompt in prompts]

    game = ToolUseGame(
        target_tool="weather_tool",
        user_segments=[f"SEGMENT{i}" for i in range(3)],
        system_prompt="Use weather_tool for weather questions.",
        scorer=OffsetScorer(),
        tool_descriptions={"weather_tool": TOOL_DESCRIPTIONS["weather_tool"]},
        normalize=True,
        defer_empty_coalition_evaluation=True,
    )  # raw empty-coalition score is 5.0, not 0.0

    evaluate_game_exactly(game, retry_policy=RETRY_POLICY, sleep_fn=lambda _d: None)

    assert game.normalization_value == pytest.approx(5.0)
    assert float(game(game.empty_coalition)[0]) == pytest.approx(0.0)
    # Normalization happens exactly once: full coalition's normalized value is the raw
    # value function output minus the (single) normalization baseline, not double-shifted.
    raw_full = float(game.value_function(game.grand_coalition.reshape(1, -1))[0])
    assert float(game(game.grand_coalition)[0]) == pytest.approx(raw_full - 5.0)


# --- Approximate-path safety: a deferred game must never run uninitialized ---------------


def test_deferred_game_called_before_evaluation_raises_instead_of_using_placeholder() -> None:
    """A deferred, unresolved game must never silently normalize against 0.0."""

    class AnyScorer:
        def score_batch(self, prompts, *, target_tool, tool_descriptions):
            del target_tool, tool_descriptions
            return [0.5 for _ in prompts]

    game = ToolUseGame(
        target_tool="weather_tool",
        user_segments=[f"SEGMENT{i}" for i in range(3)],
        system_prompt="Use weather_tool for weather questions.",
        scorer=AnyScorer(),
        tool_descriptions={"weather_tool": TOOL_DESCRIPTIONS["weather_tool"]},
        defer_empty_coalition_evaluation=True,
    )

    with pytest.raises(RuntimeError, match="defer_empty_coalition_evaluation=True"):
        game(game.empty_coalition)


def test_deferred_game_cannot_enter_approximate_path_with_uninitialized_baseline() -> None:
    """compute_interaction_explanation's approximate branch must reject an unresolved game.

    Above MAX_EXACT_DEMO_PLAYERS the approximate (shapiq approximator) path calls
    game(...) many times during sampling; it must never receive a deferred game
    whose normalization baseline was never resolved.
    """

    class AnyScorer:
        def score_batch(self, prompts, *, target_tool, tool_descriptions):
            del target_tool, tool_descriptions
            return [0.5 for _ in prompts]

    # n_players above MAX_EXACT_DEMO_PLAYERS so compute_interaction_explanation
    # routes to the approximate branch, not the exact (evaluate_game_exactly) one.
    n_players = MAX_EXACT_DEMO_PLAYERS + 1
    game = ToolUseGame(
        target_tool="weather_tool",
        user_segments=[f"SEGMENT{i}" for i in range(n_players)],
        system_prompt="Use weather_tool for weather questions.",
        scorer=AnyScorer(),
        tool_descriptions={"weather_tool": TOOL_DESCRIPTIONS["weather_tool"]},
        defer_empty_coalition_evaluation=True,
    )
    assert game._normalization_resolved is False  # asserting internal guard state

    with pytest.raises(RuntimeError, match="uninitialized placeholder"):
        app.compute_interaction_explanation(
            game=game,
            index="SV",
            max_order=1,
            budget=8,
        )
