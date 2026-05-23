"""Cooperative game for explaining agentic tool-use decisions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from shapiq.game import Game

SegmentSource = Literal["system", "user"]


@dataclass(frozen=True)
class ToolUseSegment:
    """A system or user text segment used as one player in the tool-use game."""

    source: SegmentSource
    label: str
    text: str


TOOL_KEYWORDS = {
    "weather_tool": {
        "weather",
        "rain",
        "forecast",
        "temperature",
        "snow",
        "wind",
        "berlin",
        "tomorrow",
        "morning",
    },
    "calculator_tool": {
        "calculate",
        "compute",
        "times",
        "multiply",
        "plus",
        "minus",
        "divide",
        "percent",
        "number",
        "final",
    },
    "web_search_tool": {
        "latest",
        "newest",
        "current",
        "recent",
        "today",
        "weekend",
        "won",
        "race",
        "product",
        "search",
        "web",
    },
    "no_tool": {
        "explain",
        "what",
        "simple",
        "terms",
        "conceptual",
        "stable",
        "knowledge",
        "directly",
    },
}


def normalize_tokens(text: str) -> set[str]:
    """Return lowercase alphanumeric tokens."""
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def lexical_tool_score(
    selected_segments: list[ToolUseSegment],
    target_tool: str,
) -> float:
    """Lightweight local scorer for a tool-call decision.

    This is scaffolding: it scores how strongly the visible system/user segments
    point to the target tool. Replace it with target-tool log-likelihood from a
    real tool-calling model for a final demo.
    """
    # TODO(final-demo): Replace this lexical heuristic with a model-backed scorer.
    # Good options:
    # - target tool-name log-likelihood from a tool-calling LLM,
    # - probability from a small trained tool-router classifier,
    # - structured tool-call probability from a real agent trace,
    # - contrastive log-odds between `target_tool` and `no_tool`.
    if not selected_segments:
        return 0.0

    target_keywords = TOOL_KEYWORDS[target_tool]
    user_tokens = set()
    system_tokens = set()
    for segment in selected_segments:
        if segment.source == "user":
            user_tokens |= normalize_tokens(segment.text)
        else:
            system_tokens |= normalize_tokens(segment.text)

    user_hits = len(user_tokens & target_keywords)
    system_hits = len(system_tokens & target_keywords)
    explicit_tool_rule = target_tool.lower() in " ".join(
        segment.text.lower() for segment in selected_segments if segment.source == "system"
    )

    competing_hits = 0
    for tool, keywords in TOOL_KEYWORDS.items():
        if tool == target_tool:
            continue
        competing_hits += len(user_tokens & keywords) * 0.35

    raw_score = 0.85 * user_hits + 0.65 * system_hits + (1.25 if explicit_tool_rule else 0.0)
    raw_score -= competing_hits

    # Smooth into [0, 1] while keeping empty/weak coalitions low.
    return float(1 / (1 + math.exp(-(raw_score - 1.3))))


class ToolUseGame(Game):
    """Game where players are system/user prompt segments and value is target-tool support."""

    def __init__(
        self,
        *,
        target_tool: str,
        segments: list[ToolUseSegment],
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        if not segments:
            msg = "ToolUseGame requires at least one segment."
            raise ValueError(msg)
        self.target_tool = target_tool
        self.segments = segments
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
        # TODO(final-demo): Keep this method as the stable integration point.
        # Swap `lexical_tool_score` for an LLM/tool-router scorer without changing
        # the shapiq game or Streamlit visualization code.
        return lexical_tool_score(selected_segments, self.target_tool)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate support for each coalition of prompt segments."""
        values = np.zeros(coalitions.shape[0], dtype=float)
        for row_idx, coalition in enumerate(coalitions):
            values[row_idx] = self.score_segments(self.selected_segments(coalition))
        return values


def budget_for_demo(n_players: int) -> int:
    """Small interactive default budget."""
    return int(min(2**n_players, max(48, 8 * n_players * math.log2(n_players + 1))))
