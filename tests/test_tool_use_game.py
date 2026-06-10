from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def load_tool_game_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "src" / "demos" / "agentic_tool_use_explanation" / "tool_game.py"
    spec = importlib.util.spec_from_file_location(
        "agentic_tool_use_explanation.tool_game", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeScorer:
    def __init__(self):
        self.prompts = []
        self.calls = []
        self.target_tool = None
        self.tool_descriptions = None

    def score_batch(
        self, prompts: list[str], *, target_tool: str, tool_descriptions: dict[str, str]
    ) -> list[float]:
        self.prompts = prompts
        self.calls.append(prompts)
        self.target_tool = target_tool
        self.tool_descriptions = tool_descriptions
        return [0.25 for _ in prompts]


class MismatchedBatchScorer:
    def score_batch(
        self, prompts: list[str], *, target_tool: str, tool_descriptions: dict[str, str]
    ) -> list[float]:
        if len(prompts) == 1:
            return [0.0]
        return [0.5]


class NonFiniteScorer:
    def score_batch(
        self, prompts: list[str], *, target_tool: str, tool_descriptions: dict[str, str]
    ) -> list[float]:
        return [float("nan") for _ in prompts]


TOOL_DESCRIPTIONS = {
    "weather_tool": "Fetch current weather or forecasts.",
    "calculator_tool": "Compute exact arithmetic.",
}


def make_game(module, scorer=None, normalize=False):
    return module.ToolUseGame(
        target_tool="weather_tool",
        user_segments=[
            "What is the weather",
            "in Berlin tomorrow?",
        ],
        system_prompt="You are a tool router.",
        scorer=scorer or FakeScorer(),
        tool_descriptions=TOOL_DESCRIPTIONS,
        normalize=normalize,
    )


def test_tool_use_game_uses_only_user_segments_as_players():
    module = load_tool_game_module()
    game = make_game(module)

    assert game.n_players == 2
    assert game.user_segments == ["What is the weather", "in Berlin tomorrow?"]
    assert [segment.label for segment in game.segments] == ["U1", "U2"]


def test_empty_coalition_keeps_fixed_context_and_removes_user_segments():
    module = load_tool_game_module()
    game = make_game(module)

    prompt = game.build_prompt([])

    assert "You are a tool router." in prompt
    assert "weather_tool: Fetch current weather or forecasts." in prompt
    assert "calculator_tool: Compute exact arithmetic." in prompt
    assert "What is the weather" not in prompt
    assert "in Berlin tomorrow?" not in prompt
    assert "User request:\n\nAssistant:" in prompt


def test_full_coalition_keeps_fixed_context_and_includes_all_user_segments():
    module = load_tool_game_module()
    game = make_game(module)

    prompt = game.build_prompt(game.selected_segments(np.array([True, True])))

    assert "You are a tool router." in prompt
    assert "weather_tool: Fetch current weather or forecasts." in prompt
    assert "What is the weather in Berlin tomorrow?" in prompt


def test_value_function_batches_multiple_coalitions_in_one_scorer_call():
    module = load_tool_game_module()
    scorer = FakeScorer()
    game = make_game(module, scorer=scorer)
    scorer.calls.clear()

    coalitions = np.array([[False, False], [True, False], [True, True]], dtype=bool)
    values = game.value_function(coalitions)

    assert values.shape == (3,)
    assert np.allclose(values, [0.25, 0.25, 0.25])
    assert len(scorer.calls) == 1
    assert len(scorer.calls[0]) == 3
    assert scorer.target_tool == "weather_tool"
    assert scorer.tool_descriptions == TOOL_DESCRIPTIONS


def test_legacy_segments_constructor_extracts_only_user_players():
    module = load_tool_game_module()
    ToolUseSegment = module.ToolUseSegment
    scorer = FakeScorer()
    game = module.ToolUseGame(
        target_tool="weather_tool",
        segments=[
            ToolUseSegment(source="system", label="S1", text="Use weather_tool for forecasts."),
            ToolUseSegment(source="user", label="U1", text="Will it rain tomorrow?"),
        ],
        scorer=scorer,
        tool_descriptions=TOOL_DESCRIPTIONS,
        normalize=False,
    )

    assert game.n_players == 1
    assert game.player_name_lookup == {"U1": 0}
    assert "Use weather_tool for forecasts." in game.build_prompt([])
    assert "Will it rain tomorrow?" not in game.build_prompt([])


def test_game_accepts_segment_like_user_objects_from_app_layer():
    module = load_tool_game_module()

    class AppSegment:
        source = "user"
        label = "U1"
        text = "Will it rain tomorrow?"

    game = module.ToolUseGame(
        target_tool="weather_tool",
        user_segments=[AppSegment()],
        system_prompt="You are a tool router.",
        scorer=FakeScorer(),
        tool_descriptions=TOOL_DESCRIPTIONS,
        normalize=False,
    )

    assert game.n_players == 1
    assert game.segments[0].label == "U1"
    assert "Will it rain tomorrow?" in game.build_prompt(game.segments)


def test_tool_use_game_default_lexical_scorer_handles_empty_and_full_coalitions():
    module = load_tool_game_module()
    game = module.ToolUseGame(
        target_tool="weather_tool",
        user_segments=["What is the forecast?"],
        system_prompt="You are a tool router.",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    empty_value = float(game(game.empty_coalition)[0])
    full_value = float(game(game.grand_coalition)[0])

    assert 0.0 <= empty_value <= 1.0
    assert 0.0 <= full_value <= 1.0


def test_tool_use_game_rejects_score_length_mismatch():
    module = load_tool_game_module()
    game = make_game(module, scorer=MismatchedBatchScorer())

    coalitions = np.array([[False, False], [True, True]], dtype=bool)
    with pytest.raises(ValueError, match="one score per prompt"):
        game.value_function(coalitions)


def test_tool_use_game_rejects_non_finite_scores():
    module = load_tool_game_module()

    with pytest.raises(ValueError, match="finite numeric scores"):
        module.ToolUseGame(
            target_tool="weather_tool",
            user_segments=["What is the forecast?"],
            scorer=NonFiniteScorer(),
            normalize=False,
        )
