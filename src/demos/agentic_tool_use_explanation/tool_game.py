"""Cooperative game for explaining agentic tool-use decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from shapiq.game import Game

if TYPE_CHECKING:
    from scorers import ToolScorerProtocol

SegmentSource = Literal["system", "user"]


@dataclass(frozen=True)
class ToolUseSegment:
    """A system or user text segment used as one player in the tool-use game."""

    source: SegmentSource
    label: str
    text: str


class ToolUseGame(Game):
    """Game where players are system/user prompt segments and value is target-tool support."""

    def __init__(
        self,
        *,
        target_tool: str,
        segments: list[ToolUseSegment],
        scorer: "ToolScorerProtocol | None" = None,
        tool_descriptions: dict[str, str] | None = None,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        if not segments:
            msg = "ToolUseGame requires at least one segment."
            raise ValueError(msg)
        self.target_tool = target_tool
        self.segments = segments
        if scorer is not None:
            self.scorer = scorer
        else:
            # Use the lexical baseline scorer implemented in scorers.py
            try:
                from demos.agentic_tool_use_explanation.scorers import LexicalToolScorer
            except ModuleNotFoundError:
                from scorers import LexicalToolScorer

            self.scorer = LexicalToolScorer()
        self.tool_descriptions = tool_descriptions or {}
        empty_score = self.score_segments([])
        super().__init__(
            n_players=len(segments),
            normalize=normalize,
            normalization_value=empty_score,
            verbose=verbose,
            player_names=[segment.label for segment in segments],
        )

    def selected_segments(self, coalition: np.ndarray) -> list[ToolUseSegment]:
        """Return prompt segments selected by a boolean coalition."""
        coalition = np.asarray(coalition, dtype=bool)
        return [segment for keep, segment in zip(coalition, self.segments, strict=True) if keep]

    def score_segments(self, selected_segments: list[ToolUseSegment]) -> float:
        """Score support for the target tool from selected prompt segments."""
        prompt = self.build_prompt(selected_segments)
        scores = self.scorer.score_batch(
            [prompt],
            target_tool=self.target_tool,
            tool_descriptions=self.tool_descriptions,
        )
        if len(scores) != 1:
            msg = "ToolScorerProtocol.score_batch must return one score per prompt."
            raise ValueError(msg)
        score = float(scores[0])
        if not math.isfinite(score):
            msg = "ToolScorerProtocol.score_batch must return finite numeric scores."
            raise ValueError(msg)
        return score

    def build_prompt(self, selected_segments: list[ToolUseSegment]) -> str:
        """Build the coalition prompt from selected system/user segments."""
        system_lines = [
            f"- {segment.text}" for segment in selected_segments if segment.source == "system"
        ]
        user_lines = [
            f"- {segment.text}" for segment in selected_segments if segment.source == "user"
        ]
        return (
            "System rules:\n"
            + ("\n".join(system_lines) if system_lines else "(none)")
            + "\n\nUser request:\n"
            + ("\n".join(user_lines) if user_lines else "(none)")
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate support for each coalition of prompt segments."""
        prompts = [self.build_prompt(self.selected_segments(coalition)) for coalition in coalitions]
        scores = self.scorer.score_batch(
            prompts,
            target_tool=self.target_tool,
            tool_descriptions=self.tool_descriptions,
        )
        if len(scores) != len(prompts):
            msg = "ToolScorerProtocol.score_batch must return one score per prompt."
            raise ValueError(msg)
        values = np.asarray(scores, dtype=float)
        if values.shape != (len(prompts),):
            msg = "ToolScorerProtocol.score_batch must return a one-dimensional score list."
            raise ValueError(msg)
        if not np.all(np.isfinite(values)):
            msg = "ToolScorerProtocol.score_batch must return finite numeric scores."
            raise ValueError(msg)
        return values


def budget_for_demo(n_players: int) -> int:
    """Small interactive default budget."""
    return int(min(2**n_players, max(48, 8 * n_players * np.log2(n_players + 1))))
