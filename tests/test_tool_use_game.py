from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def load_tool_game_module():
    root = Path(__file__).resolve().parents[1]
    module_path = root / "src" / "demos" / "agentic_tool_use_explanation" / "tool_game.py"
    spec = importlib.util.spec_from_file_location("agentic_tool_use_explanation.tool_game", module_path)
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

    def score_batch(self, prompts: list[str], *, target_tool: str, tool_descriptions: dict[str, str]) -> list[float]:
        self.prompts = prompts
        self.calls.append(prompts)
        self.target_tool = target_tool
        self.tool_descriptions = tool_descriptions
        return [0.25 for _ in prompts]


class MismatchedBatchScorer:
    def score_batch(self, prompts: list[str], *, target_tool: str, tool_descriptions: dict[str, str]) -> list[float]:
        if len(prompts) == 1:
            return [0.0]
        return [0.5]


class NonFiniteScorer:
    def score_batch(self, prompts: list[str], *, target_tool: str, tool_descriptions: dict[str, str]) -> list[float]:
        return [float("nan") for _ in prompts]


def test_tool_use_game_accepts_custom_scorer_and_batch_values():
    module = load_tool_game_module()
    ToolUseGame = module.ToolUseGame
    ToolUseSegment = module.ToolUseSegment

    segments = [
        ToolUseSegment(source="system", label="S1", text="Use the weather tool."),
        ToolUseSegment(source="user", label="U1", text="What is the forecast for tomorrow?"),
    ]
    scorer = FakeScorer()
    game = ToolUseGame(target_tool="weather_tool", segments=segments, scorer=scorer)

    coalitions = np.array([[False, False], [True, True]], dtype=bool)
    values = game.value_function(coalitions)

    assert values.shape == (2,)
    assert np.allclose(values, [0.25, 0.25])
    assert scorer.target_tool == "weather_tool"
    assert scorer.tool_descriptions == {}
    assert len(scorer.prompts) == 2
    assert scorer.calls[-1] == scorer.prompts


def test_tool_use_game_default_lexical_scorer_handles_empty_and_full_coalitions():
    module = load_tool_game_module()
    ToolUseGame = module.ToolUseGame
    ToolUseSegment = module.ToolUseSegment

    segments = [
        ToolUseSegment(source="system", label="S1", text="Use the weather tool."),
        ToolUseSegment(source="user", label="U1", text="What is the forecast?"),
    ]
    game = ToolUseGame(target_tool="weather_tool", segments=segments)

    empty_value = float(game(game.empty_coalition)[0])
    full_value = float(game(game.grand_coalition)[0])

    assert empty_value == 0.0
    assert 0.0 <= full_value <= 1.0


def test_tool_use_game_builds_prompt_from_selected_segments():
    module = load_tool_game_module()
    ToolUseGame = module.ToolUseGame
    ToolUseSegment = module.ToolUseSegment

    segments = [
        ToolUseSegment(source="system", label="S1", text="Use weather_tool for forecasts."),
        ToolUseSegment(source="system", label="S2", text="Avoid tools for stable facts."),
        ToolUseSegment(source="user", label="U1", text="Will it rain tomorrow?"),
    ]
    game = ToolUseGame(
        target_tool="weather_tool",
        segments=segments,
        scorer=FakeScorer(),
        normalize=False,
    )

    selected = game.selected_segments(np.array([True, False, True]))
    prompt = game.build_prompt(selected)

    assert "Use weather_tool for forecasts." in prompt
    assert "Avoid tools for stable facts." not in prompt
    assert "Will it rain tomorrow?" in prompt


def test_tool_use_game_rejects_score_length_mismatch():
    module = load_tool_game_module()
    ToolUseGame = module.ToolUseGame
    ToolUseSegment = module.ToolUseSegment

    game = ToolUseGame(
        target_tool="weather_tool",
        segments=[
            ToolUseSegment(source="system", label="S1", text="Use the weather tool."),
            ToolUseSegment(source="user", label="U1", text="What is the forecast?"),
        ],
        scorer=MismatchedBatchScorer(),
        normalize=False,
    )

    coalitions = np.array([[False, False], [True, True]], dtype=bool)
    with pytest.raises(ValueError, match="one score per prompt"):
        game.value_function(coalitions)


def test_tool_use_game_rejects_non_finite_scores():
    module = load_tool_game_module()
    ToolUseGame = module.ToolUseGame
    ToolUseSegment = module.ToolUseSegment

    with pytest.raises(ValueError, match="finite numeric scores"):
        ToolUseGame(
            target_tool="weather_tool",
            segments=[
                ToolUseSegment(source="system", label="S1", text="Use the weather tool."),
            ],
            scorer=NonFiniteScorer(),
            normalize=False,
        )
