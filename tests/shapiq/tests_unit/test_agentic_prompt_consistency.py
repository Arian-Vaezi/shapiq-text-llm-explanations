"""Tests for the canonical coalition-prompt builder shared by app.py and ToolUseGame.

These guard against a regression where ``app.py``'s prompt builder and
``tool_game.ToolUseGame.build_prompt`` independently formatted the empty
coalition's "no user text" case differently (one produced an extra blank line).
That textual mismatch made the empty coalition map to two distinct cache keys
in value-function scorers (e.g. ``FinalAnswerSimilarityScorer``), forcing a
second, independently-sampled agent call for what should be a single cached
"empty coalition" value -- silently breaking Shapley efficiency for any
backend that is not perfectly deterministic across calls.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

from app import ToolUseSegment, build_coalition_prompt  # noqa: E402
from final_answer_similarity_scorer import FinalAnswerSimilarityScorer  # noqa: E402
from scorers import split_coalition_prompt  # noqa: E402
from tool_game import ToolUseGame  # noqa: E402

SYSTEM_PROMPT = "Use weather_tool for weather questions."
TOOL_CONTEXT = "- weather_tool: Get current weather conditions or a forecast."
USER_TEXT = "Will it rain in Berlin tomorrow morning?"


def _make_game() -> ToolUseGame:
    return ToolUseGame(
        target_tool="weather_tool",
        user_segments=[USER_TEXT],
        system_prompt=SYSTEM_PROMPT,
        tool_context=TOOL_CONTEXT,
    )


def test_app_and_game_empty_prompts_are_byte_identical() -> None:
    app_empty = build_coalition_prompt([], system_prompt=SYSTEM_PROMPT, tool_context=TOOL_CONTEXT)
    game_empty = _make_game().build_prompt([])

    assert app_empty == game_empty


def test_app_and_game_full_prompts_are_byte_identical() -> None:
    segment = ToolUseSegment(source="user", label="U1", text=USER_TEXT)
    app_full = build_coalition_prompt(
        [segment],
        system_prompt=SYSTEM_PROMPT,
        tool_context=TOOL_CONTEXT,
    )
    game_full = _make_game().build_prompt([USER_TEXT])

    assert app_full == game_full


def test_split_coalition_prompt_recovers_same_user_request_for_both_builders() -> None:
    app_empty = build_coalition_prompt([], system_prompt=SYSTEM_PROMPT, tool_context=TOOL_CONTEXT)
    game_empty = _make_game().build_prompt([])

    assert split_coalition_prompt(app_empty)[1] == split_coalition_prompt(game_empty)[1] == ""


def _make_fake_scorer_components(answers: dict[str, str], vectors: dict[str, list[float]]):
    calls: list[str] = []

    def agent(user_request: str) -> object:
        calls.append(user_request)
        answer = answers[user_request]
        return SimpleNamespace(
            selected_tool="weather_tool",
            assistant_answer=answer,
            final_answer=answer,
            error=None,
        )

    def embedder(texts):
        return np.array([vectors[text] for text in texts], dtype=float)

    return agent, embedder, calls


def test_game_call_on_empty_coalition_is_exactly_zero() -> None:
    """game(empty_coalition) must return 0.0: the scorer's own normalization is already
    anchored to the (now-canonical) empty prompt, so shapiq's Game-level normalization_value
    must be an exact, structural no-op rather than an accidental near-zero cancellation."""
    answers = {USER_TEXT: "It will rain.", "": "I have no information."}
    vectors = {"It will rain.": [1.0, 0.0], "I have no information.": [0.0, 1.0]}
    agent, embedder, calls = _make_fake_scorer_components(answers, vectors)

    game = _make_game()
    empty_prompt = game.build_prompt([])
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will rain.",
        empty_prompt=empty_prompt,
    )
    game.scorer = scorer
    # Mirror ToolUseGame's own normalization_value computation (done in __init__ before the
    # scorer was attached above) using the now-attached scorer, to check Game-level wiring.
    game.normalization_value = scorer.score_batch(
        [empty_prompt],
        target_tool="weather_tool",
        tool_descriptions=game.tool_descriptions,
    )[0]

    empty_coalition = np.zeros((1, game.n_players), dtype=bool)
    result = game(empty_coalition)

    assert result[0] == pytest.approx(0.0, abs=1e-12)
    # The empty coalition must hit the cache once, not trigger two independent agent calls
    # (one via the scorer's own empty_prompt anchor, one via the game's internal lookup).
    assert calls.count("") == 1


def test_game_normalization_is_applied_exactly_once() -> None:
    """normalization_value must come from a single source and be subtracted exactly once.

    Constructing ToolUseGame already calls score_segments([]) once to derive
    normalization_value; this test checks that value_function itself (the raw,
    un-normalized layer) does NOT also subtract that baseline, so Game.__call__ is
    the only place normalization happens.
    """
    answers = {USER_TEXT: "It will rain.", "": "I have no information."}
    vectors = {"It will rain.": [1.0, 0.0], "I have no information.": [0.0, 1.0]}
    agent, embedder, _calls = _make_fake_scorer_components(answers, vectors)

    game = _make_game()
    empty_prompt = game.build_prompt([])
    scorer = FinalAnswerSimilarityScorer(
        agent_callable=agent,
        embedder=embedder,
        reference_answer="It will rain.",
        empty_prompt=empty_prompt,
    )
    game.scorer = scorer
    game.normalization_value = scorer.score_batch(
        [empty_prompt],
        target_tool="weather_tool",
        tool_descriptions=game.tool_descriptions,
    )[0]

    full_coalition = np.ones((1, game.n_players), dtype=bool)
    raw_full = game.value_function(full_coalition)[0]
    normalized_full = game(full_coalition)[0]

    # value_function is raw (no subtraction); __call__ subtracts normalization_value exactly
    # once on top of it.
    assert normalized_full == pytest.approx(raw_full - game.normalization_value)
