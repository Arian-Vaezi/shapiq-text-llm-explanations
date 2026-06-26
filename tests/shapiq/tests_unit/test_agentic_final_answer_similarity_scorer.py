"""Tests for the end-to-end final-answer semantic similarity coalition scorer."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

try:
    # Must match how final_answer_similarity_scorer.py itself resolves this module
    # (package path preferred), so the raised exception class identity matches what
    # this test imports -- a bare import here would bind a second, distinct class.
    from demos.agentic_tool_use_explanation.agent_failure import (
        AgentFailureKind,
        CoalitionTransientFailureError,
    )
except ModuleNotFoundError:
    from agent_failure import AgentFailureKind, CoalitionTransientFailureError
from final_answer_similarity_scorer import (  # noqa: E402
    CoalitionSemanticFailureError,
    FinalAnswerSimilarityScorer,
    extract_final_answer,
)
from scorers import build_coalition_prompt, join_user_request_segments  # noqa: E402
from tool_game import ToolUseGame  # noqa: E402
from tool_schemas import TOOL_DESCRIPTIONS  # noqa: E402

SYSTEM_PROMPT = "SYSTEM"
TOOL_CONTEXT = "- weather_tool: Get current weather conditions or a forecast."


def make_prompt(user_request: str) -> str:
    """Build a coalition prompt using the real, canonical builder (no naive joining)."""
    return build_coalition_prompt(
        user_request,
        system_prompt=SYSTEM_PROMPT,
        tool_context=TOOL_CONTEXT,
    )


def make_segment_prompt(selected_segments: list[str]) -> str:
    """Build a coalition prompt from segments via the real join + build helpers."""
    return make_prompt(join_user_request_segments(selected_segments))


FULL_REQUEST = "Will it rain in Berlin tomorrow morning?"
EMPTY_PROMPT = make_prompt("")
FULL_PROMPT = make_prompt(FULL_REQUEST)


def make_fake_agent(
    answers: dict[str, str],
    *,
    failing_requests: frozenset[str] = frozenset(),
    erroring_requests: dict[str, str] | None = None,
):
    """Build a fake complete-agent callable that returns predefined final answers.

    ``failing_requests`` simulates the agent raising (e.g. a network exception).
    ``erroring_requests`` simulates the agent returning a result object with
    ``.error`` set and no usable answer text (e.g. a missing API key), which is
    how the real Groq/Gemini/HF-local backends report failure without raising.
    """
    erroring_requests = erroring_requests or {}
    calls: list[str] = []

    def agent(user_request: str) -> object:
        calls.append(user_request)
        if user_request in failing_requests:
            msg = "simulated agent execution failure"
            raise RuntimeError(msg)
        if user_request in erroring_requests:
            return SimpleNamespace(
                selected_tool=None,
                tool_arguments={},
                assistant_answer="",
                final_answer="",
                error=erroring_requests[user_request],
            )
        answer = answers.get(user_request, "")
        return SimpleNamespace(
            selected_tool="weather_tool" if answer else None,
            tool_arguments={},
            assistant_answer=answer,
            final_answer=answer,
            error=None,
        )

    return agent, calls


def make_fake_embedder(vectors: dict[str, list[float]], *, dimension: int = 3):
    """Build a fake batch embedder mapping known texts to fixed vectors."""
    calls: list[list[str]] = []
    default_vector = [0.0] * dimension

    def embedder(texts):
        calls.append(list(texts))
        return np.array([vectors.get(text, default_vector) for text in texts], dtype=float)

    return embedder, calls


def test_full_coalition_compared_with_itself_is_approximately_one() -> None:
    agent, _calls = make_fake_agent(
        {FULL_REQUEST: "It will be sunny.", "": "I have no information."}
    )
    embedder, _embed_calls = make_fake_embedder(
        {
            "It will be sunny.": [1.0, 0.0, 0.0],
            "I have no information.": [0.0, 1.0, 0.0],
        }
    )
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will be sunny.",
        empty_prompt=EMPTY_PROMPT,
    )

    scores = scorer.score_batch(
        [FULL_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scorer.last_debug_outputs[0]["raw_similarity"] == pytest.approx(1.0)
    # The full coalition's own answer matches the reference exactly here, and the empty
    # coalition's (real, dissimilar) answer is what the score is normalized against.
    assert scores[0] == pytest.approx(1.0, abs=1e-6)


def test_empty_coalition_normalizes_to_zero() -> None:
    agent, _calls = make_fake_agent(
        {FULL_REQUEST: "It will be sunny.", "": "I have no information."}
    )
    embedder, _embed_calls = make_fake_embedder(
        {
            "It will be sunny.": [1.0, 0.0, 0.0],
            "I have no information.": [0.0, 1.0, 0.0],
        }
    )
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will be sunny.",
        empty_prompt=EMPTY_PROMPT,
    )

    scores = scorer.score_batch(
        [EMPTY_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [0.0]
    assert scorer.last_empty_raw_similarity == pytest.approx(0.0)


def test_raw_and_normalized_values_differ_only_by_a_constant_offset() -> None:
    """v(S) = r(S) - r(empty): normalization is a single constant shift, not per-coalition."""
    requests = {
        "Will it rain in Berlin?": "Light rain expected in Berlin.",
        "Will it rain?": "Unclear without a location.",
        "": "I have no information.",
    }
    vectors = {
        "Light rain expected in Berlin.": [0.9, 0.1],
        "Unclear without a location.": [0.1, 0.9],
        "I have no information.": [0.0, 1.0],
    }
    agent, _calls = make_fake_agent(requests)
    embedder, _embed_calls = make_fake_embedder(vectors)
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="Light rain expected in Berlin.",
        empty_prompt=EMPTY_PROMPT,
    )

    prompts = [make_prompt(text) for text in ("Will it rain in Berlin?", "Will it rain?")]
    normalized_scores = scorer.score_batch(
        prompts,
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )
    raw_scores = [row["raw_similarity"] for row in scorer.last_debug_outputs]
    offset = scorer.last_empty_raw_similarity

    assert raw_scores[0] - normalized_scores[0] == pytest.approx(offset)
    assert raw_scores[1] - normalized_scores[1] == pytest.approx(offset)
    # The offset is identical for every coalition -- a single constant shift.
    assert (raw_scores[0] - normalized_scores[0]) == pytest.approx(
        raw_scores[1] - normalized_scores[1]
    )


def test_marginal_differences_unchanged_by_empty_coalition_normalization() -> None:
    """Attribution-relevant quantities (differences between coalition values) are invariant."""
    closer_request = "Will it rain in Berlin?"
    farther_request = "Will it rain?"
    agent, _calls = make_fake_agent(
        {
            closer_request: "Light rain expected in Berlin.",
            farther_request: "Unclear without a location.",
            "": "I have no information.",
        }
    )
    embedder, _embed_calls = make_fake_embedder(
        {
            "Light rain expected in Berlin.": [0.9, 0.1],
            "Unclear without a location.": [0.1, 0.9],
            "I have no information.": [0.0, 1.0],
        }
    )
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="Light rain expected in Berlin.",
        empty_prompt=EMPTY_PROMPT,
    )

    prompts = [make_prompt(closer_request), make_prompt(farther_request)]
    normalized_scores = scorer.score_batch(
        prompts,
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )
    raw_scores = [row["raw_similarity"] for row in scorer.last_debug_outputs]

    raw_marginal = raw_scores[0] - raw_scores[1]
    normalized_marginal = normalized_scores[0] - normalized_scores[1]
    assert raw_marginal == pytest.approx(normalized_marginal)


def test_coalition_ordering_reflects_similarity_to_reference() -> None:
    closer_request = "Will it rain in Berlin?"
    farther_request = "Will it rain?"
    agent, _calls = make_fake_agent(
        {
            FULL_REQUEST: "Light rain expected in Berlin tomorrow morning.",
            closer_request: "Light rain expected in Berlin.",
            farther_request: "Unclear without a location.",
            "": "I have no information.",
        }
    )
    embedder, _embed_calls = make_fake_embedder(
        {
            "Light rain expected in Berlin tomorrow morning.": [1.0, 0.0],
            "Light rain expected in Berlin.": [0.9, 0.1],
            "Unclear without a location.": [0.1, 0.9],
            "I have no information.": [0.0, 1.0],
        }
    )
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="Light rain expected in Berlin tomorrow morning.",
        empty_prompt=EMPTY_PROMPT,
    )

    closer_prompt = make_prompt(closer_request)
    farther_prompt = make_prompt(farther_request)
    scores = scorer.score_batch(
        [closer_prompt, farther_prompt],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores[0] > scores[1]


def test_agent_called_with_reconstructed_masked_request() -> None:
    agent, calls = make_fake_agent({FULL_REQUEST: "It will be sunny.", "": "No info."})
    embedder, _embed_calls = make_fake_embedder(
        {"It will be sunny.": [1.0, 0.0], "No info.": [0.0, 1.0]}
    )
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will be sunny.",
        empty_prompt=EMPTY_PROMPT,
    )

    scorer.score_batch(
        [FULL_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert FULL_REQUEST in calls
    assert "" in calls


def test_coalition_prompt_uses_canonical_builder_not_naive_joining() -> None:
    """Coalitions are rebuilt via scorers.build_coalition_prompt, matching ToolUseGame."""
    game = ToolUseGame(
        target_tool="weather_tool",
        user_segments=["Will it rain", "in Berlin"],
        system_prompt=SYSTEM_PROMPT,
        tool_context=TOOL_CONTEXT,
    )
    game_prompt = game.build_prompt(["Will it rain", "in Berlin"])
    naive_join_prompt = make_prompt("Will it rain in Berlin")

    assert game_prompt == naive_join_prompt == make_segment_prompt(["Will it rain", "in Berlin"])


def test_repeated_coalition_does_not_rerun_agent() -> None:
    agent, calls = make_fake_agent({FULL_REQUEST: "It will be sunny.", "": "No info."})
    embedder, embed_calls = make_fake_embedder(
        {"It will be sunny.": [1.0, 0.0], "No info.": [0.0, 1.0]}
    )
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will be sunny.",
        empty_prompt=EMPTY_PROMPT,
    )

    scorer.score_batch(
        [FULL_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )
    calls_after_first_run = len(calls)
    scorer.score_batch(
        [FULL_PROMPT, FULL_PROMPT],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert len(calls) == calls_after_first_run
    # The embedder is also not asked to re-embed already-cached answer text.
    total_embedded_texts = sum(len(batch) for batch in embed_calls)
    assert total_embedded_texts == len({"It will be sunny.", "No info."})


def test_agent_exception_propagates_and_never_reaches_embedder() -> None:
    """A failed coalition must raise, never silently fall back to a placeholder score."""
    failing_request = FULL_REQUEST
    agent, _calls = make_fake_agent(
        {"": "No info."},
        failing_requests=frozenset({failing_request}),
    )
    embedder, embed_calls = make_fake_embedder({"No info.": [0.0, 1.0]}, dimension=2)
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will be sunny.",
        empty_prompt=EMPTY_PROMPT,
    )

    with pytest.raises(RuntimeError, match="simulated agent execution failure"):
        scorer.score_batch(
            [FULL_PROMPT],
            target_tool="weather_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )

    # No placeholder or failure-derived text ever reaches the embedder.
    embedded_texts = {text for batch in embed_calls for text in batch}
    assert embedded_texts == set()


def test_backend_reported_error_raises_semantic_failure_and_never_reaches_embedder() -> None:
    """A result object with .error set and no answer text (e.g. missing API key)."""
    agent, _calls = make_fake_agent(
        {"": "No info."},
        erroring_requests={FULL_REQUEST: "GROQ_API_KEY is not set."},
    )
    embedder, embed_calls = make_fake_embedder({"No info.": [0.0, 1.0]}, dimension=2)
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will be sunny.",
        empty_prompt=EMPTY_PROMPT,
    )

    with pytest.raises(CoalitionSemanticFailureError, match=r"GROQ_API_KEY is not set\."):
        scorer.score_batch(
            [FULL_PROMPT],
            target_tool="weather_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )

    embedded_texts = {text for batch in embed_calls for text in batch}
    assert "GROQ_API_KEY is not set." not in embedded_texts


def test_no_final_answer_raises_semantic_failure_and_never_reaches_embedder() -> None:
    agent, _calls = make_fake_agent({FULL_REQUEST: "", "": "No info."})
    embedder, embed_calls = make_fake_embedder({"No info.": [0.0, 1.0]}, dimension=2)
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will be sunny.",
        empty_prompt=EMPTY_PROMPT,
    )

    with pytest.raises(CoalitionSemanticFailureError, match="no usable final-answer text"):
        scorer.score_batch(
            [FULL_PROMPT],
            target_tool="weather_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )

    embedded_texts = {text for batch in embed_calls for text in batch}
    assert "" not in embedded_texts


def test_failed_coalition_is_not_cached_and_can_be_retried() -> None:
    """A failed attempt is never cached, so calling again re-runs the agent (retry-friendly)."""
    request_state = {"should_fail": True}
    calls: list[str] = []

    def agent(user_request: str) -> object:
        calls.append(user_request)
        if user_request == FULL_REQUEST and request_state["should_fail"]:
            msg = "simulated transient failure"
            raise RuntimeError(msg)
        answer = {"": "No info.", FULL_REQUEST: "It will be sunny."}[user_request]
        return SimpleNamespace(
            selected_tool="weather_tool" if answer else None,
            tool_arguments={},
            assistant_answer=answer,
            final_answer=answer,
            error=None,
        )

    embedder, _embed_calls = make_fake_embedder(
        {"It will be sunny.": [1.0, 0.0], "No info.": [0.0, 1.0]}
    )
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will be sunny.",
        empty_prompt=EMPTY_PROMPT,
    )

    with pytest.raises(RuntimeError, match="simulated transient failure"):
        scorer.score_batch(
            [FULL_PROMPT], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS
        )

    request_state["should_fail"] = False
    scores = scorer.score_batch(
        [FULL_PROMPT], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS
    )

    assert len(scores) == 1
    assert np.isfinite(scores[0])
    assert calls.count(FULL_REQUEST) == 2  # re-run, not served from a poisoned cache


def test_missing_reference_answer_raises_immediately() -> None:
    agent, _calls = make_fake_agent({})
    embedder, _embed_calls = make_fake_embedder({})

    with pytest.raises(ValueError, match="non-empty reference_answer"):
        FinalAnswerSimilarityScorer(
            agent_callable=agent,
            embedder=embedder,
            reference_answer="",
            empty_prompt=EMPTY_PROMPT,
        )

    with pytest.raises(ValueError, match="non-empty reference_answer"):
        FinalAnswerSimilarityScorer(
            agent_callable=agent,
            embedder=embedder,
            reference_answer="   ",
            empty_prompt=EMPTY_PROMPT,
        )


def test_batch_coalition_values_return_one_dimensional_array_in_order() -> None:
    """Integration through the real ToolUseGame: batch order must match input order."""
    segment_a, segment_b = "Will it rain", "in Berlin"
    answers = {
        "": "I have no information.",
        segment_a: "Maybe rain.",
        segment_b: "Somewhere in Berlin.",
        f"{segment_a} {segment_b}": "Light rain expected in Berlin tomorrow.",
    }
    vectors = {
        "I have no information.": [0.0, 1.0, 0.0],
        "Maybe rain.": [0.4, 0.6, 0.0],
        "Somewhere in Berlin.": [0.2, 0.3, 0.5],
        "Light rain expected in Berlin tomorrow.": [1.0, 0.0, 0.0],
    }
    agent, _calls = make_fake_agent(answers)
    embedder, _embed_calls = make_fake_embedder(vectors)

    game = ToolUseGame(
        target_tool="weather_tool",
        user_segments=[segment_a, segment_b],
        system_prompt=SYSTEM_PROMPT,
        tool_context=TOOL_CONTEXT,
    )
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="Light rain expected in Berlin tomorrow.",
        empty_prompt=game.build_prompt([]),
    )
    game.scorer = scorer

    coalitions = np.array(
        [
            [False, False],
            [True, False],
            [False, True],
            [True, True],
        ],
        dtype=bool,
    )
    values = game.value_function(coalitions)

    assert isinstance(values, np.ndarray)
    assert values.shape == (4,)
    # Coalition (True, True) reproduces the reference answer exactly -> highest raw value.
    assert values[3] > values[1]
    assert values[3] > values[2]
    assert values[3] > values[0]


def test_rejects_unknown_target_tool() -> None:
    agent, _calls = make_fake_agent({FULL_REQUEST: "It will be sunny.", "": "No info."})
    embedder, _embed_calls = make_fake_embedder(
        {"It will be sunny.": [1.0, 0.0], "No info.": [0.0, 1.0]}
    )
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will be sunny.",
        empty_prompt=EMPTY_PROMPT,
    )

    with pytest.raises(ValueError, match="not a known decision candidate"):
        scorer.score_batch(
            [FULL_PROMPT],
            target_tool="calendar_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )


def test_extract_final_answer_prefers_agent_response_then_assistant_then_final() -> None:
    assert extract_final_answer(SimpleNamespace(agent_response="A")) == "A"
    assert extract_final_answer(SimpleNamespace(agent_response="", assistant_answer="B")) == "B"
    assert (
        extract_final_answer(
            SimpleNamespace(agent_response="", assistant_answer="", final_answer="C")
        )
        == "C"
    )
    assert extract_final_answer(SimpleNamespace()) == ""


def test_existing_scorer_imports_are_unaffected() -> None:
    """Adding this scorer module must not break imports of the existing scorers."""
    import router_scorers  # noqa: F401
    import scorers  # noqa: F401


def make_wrapped_failure_agent(failure_kind: AgentFailureKind | None, error_text: str):
    """Build an agent_callable returning an opaque result with .error/.failure_kind set.

    Mirrors the shape groq_agent.run_groq_tool_inference / gemini_agent.run_gemini_tool_inference
    actually return: they catch the real provider exception internally and embed it as
    .error text plus a structured .failure_kind, rather than letting it propagate.
    """

    def agent(user_request: str) -> object:
        if user_request == "":
            return SimpleNamespace(
                selected_tool="weather_tool",
                assistant_answer="No info.",
                final_answer="No info.",
                error=None,
                failure_kind=None,
            )
        return SimpleNamespace(
            selected_tool=None,
            tool_arguments={},
            assistant_answer="",
            final_answer="",
            error=error_text,
            failure_kind=failure_kind,
        )

    return agent


def make_fake_embedder_for_empty_only():
    def embedder(texts):
        return np.array([[0.0, 1.0] for _ in texts], dtype=float)

    return embedder


def test_wrapped_rate_limit_result_enters_retry_path_and_succeeds_later() -> None:
    """A wrapped 429/rate-limit result must raise the retryable transient error type."""
    attempts = {"n": 0}

    def agent(user_request: str) -> object:
        if user_request == "":
            return SimpleNamespace(
                selected_tool="weather_tool",
                assistant_answer="No info.",
                final_answer="No info.",
                error=None,
                failure_kind=None,
            )
        attempts["n"] += 1
        if attempts["n"] == 1:
            return SimpleNamespace(
                selected_tool=None,
                tool_arguments={},
                assistant_answer="",
                final_answer="",
                error="429 rate limit exceeded",
                failure_kind=AgentFailureKind.RATE_LIMIT,
            )
        return SimpleNamespace(
            selected_tool="weather_tool",
            assistant_answer="It will be sunny.",
            final_answer="It will be sunny.",
            error=None,
            failure_kind=None,
        )

    embedder, _embed_calls = make_fake_embedder(
        {"It will be sunny.": [1.0, 0.0], "No info.": [0.0, 1.0]}
    )
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will be sunny.",
        empty_prompt=EMPTY_PROMPT,
    )

    with pytest.raises(CoalitionTransientFailureError):
        scorer.score_batch(
            [FULL_PROMPT], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS
        )

    scores = scorer.score_batch(
        [FULL_PROMPT], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS
    )
    assert len(scores) == 1
    assert np.isfinite(scores[0])
    assert attempts["n"] == 2  # the failed attempt was never cached, so it really retried


def test_wrapped_timeout_result_enters_retry_path() -> None:
    agent = make_wrapped_failure_agent(AgentFailureKind.TIMEOUT, "request timed out")
    embedder = make_fake_embedder_for_empty_only()
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will be sunny.",
        empty_prompt=EMPTY_PROMPT,
    )

    with pytest.raises(CoalitionTransientFailureError, match="request timed out"):
        scorer.score_batch(
            [FULL_PROMPT], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS
        )


@pytest.mark.parametrize(
    "failure_kind",
    [AgentFailureKind.AUTHENTICATION, AgentFailureKind.INVALID_REQUEST],
)
def test_wrapped_permanent_failure_result_is_semantic_and_not_retried(failure_kind) -> None:
    agent = make_wrapped_failure_agent(failure_kind, "permanent failure")
    embedder = make_fake_embedder_for_empty_only()
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will be sunny.",
        empty_prompt=EMPTY_PROMPT,
    )

    with pytest.raises(CoalitionSemanticFailureError, match="permanent failure"):
        scorer.score_batch(
            [FULL_PROMPT], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS
        )


def test_wrapped_no_final_answer_result_is_semantic_and_not_retried() -> None:
    """No .error at all, just an empty answer: must stay the original semantic path."""
    agent, _calls = make_fake_agent({FULL_REQUEST: "", "": "No info."})
    embedder, _embed_calls = make_fake_embedder({"No info.": [0.0, 1.0]}, dimension=2)
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will be sunny.",
        empty_prompt=EMPTY_PROMPT,
    )

    with pytest.raises(CoalitionSemanticFailureError, match="no usable final-answer text"):
        scorer.score_batch(
            [FULL_PROMPT], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS
        )
