"""Cooperative game for explaining agentic tool-use decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

try:
    from demos.agentic_tool_use_explanation.scorers import (
        build_coalition_prompt as _build_canonical_coalition_prompt,
        join_user_request_segments as _join_user_request_segments,
    )
except ModuleNotFoundError:
    from scorers import (
        build_coalition_prompt as _build_canonical_coalition_prompt,
        join_user_request_segments as _join_user_request_segments,
    )

try:
    from shapiq.game import Game
except Exception:  # noqa: BLE001

    class Game:  # type: ignore[no-redef]
        """Small fallback base when optional shapiq C extensions are unavailable."""

        def __init__(
            self,
            *,
            n_players: int,
            normalize: bool = True,
            normalization_value: float = 0.0,
            verbose: bool = False,
            player_names: list[str] | None = None,
        ) -> None:
            self.n_players = n_players
            self.normalize = normalize
            self.normalization_value = normalization_value
            self.verbose = verbose
            self.player_names = player_names or [str(idx) for idx in range(n_players)]


if TYPE_CHECKING:
    from scorers import ToolScorerProtocol

SegmentSource = Literal["system", "user"]


@dataclass(frozen=True)
class ToolUseSegment:
    """A labeled text segment used by the demo UI and tests."""

    source: SegmentSource
    label: str
    text: str


class ToolUseGame(Game):
    """Game where players are user-request segments and fixed context is always present."""

    def __init__(
        self,
        *,
        target_tool: str,
        user_segments: list[str | ToolUseSegment] | None = None,
        system_prompt: str = "",
        tool_context: str | None = None,
        segments: list[ToolUseSegment] | None = None,
        scorer: ToolScorerProtocol | None = None,
        tool_descriptions: dict[str, str] | None = None,
        normalize: bool = True,
        verbose: bool = False,
        defer_empty_coalition_evaluation: bool = False,
    ) -> None:
        """Build a ``ToolUseGame``.

        Args:
            defer_empty_coalition_evaluation: If ``False`` (default), construction
                behaves exactly as before: the empty coalition is scored once,
                synchronously, here, and ``normalization_value`` is correct
                immediately -- ``game(...)`` can be called right away. If ``True``,
                construction makes *no* scorer call at all; the empty coalition must
                instead be resolved together with every other coalition by
                ``coalition_evaluation.evaluate_game_exactly``, which goes through
                the same typed-outcome/retry protocol and sets
                ``normalization_value`` once that succeeds. Calling ``game(...)``
                on a deferred game before that resolution raises ``RuntimeError``
                rather than silently normalizing against a placeholder. Only the
                exact-computation demo path should opt into this.
        """
        self.target_tool = target_tool
        self.tool_descriptions = (
            dict(tool_descriptions)
            if tool_descriptions is not None
            else _default_tool_descriptions()
        )
        if user_segments is None:
            if segments is None:
                msg = "ToolUseGame requires at least one user segment."
                raise ValueError(msg)
            user_segments = [segment for segment in segments if segment.source == "user"]
            if not system_prompt:
                system_prompt = "\n".join(
                    f"- {segment.text}" for segment in segments if segment.source == "system"
                )
        self.segments = self._coerce_user_segments(user_segments)
        if not self.segments:
            msg = "ToolUseGame requires at least one user segment."
            raise ValueError(msg)
        self.user_segments = [segment.text for segment in self.segments]
        self.system_prompt = system_prompt.strip()
        self.tool_context = (
            self._format_tool_context(self.tool_descriptions)
            if tool_context is None
            else tool_context.strip()
        )
        if scorer is not None:
            self.scorer = scorer
        else:
            # Use the lexical baseline scorer implemented in scorers.py
            try:
                from demos.agentic_tool_use_explanation.scorers import LexicalToolScorer
            except ModuleNotFoundError:
                from scorers import LexicalToolScorer

            self.scorer = LexicalToolScorer()

        self.normalize_requested = normalize
        self.empty_coalition_evaluation_deferred = defer_empty_coalition_evaluation
        if defer_empty_coalition_evaluation:
            # Opt-in only: the empty coalition is itself a real coalition value and
            # must go through the same typed-outcome and retry protocol as every
            # other coalition, not an untracked, unretried call made here.
            # normalization_value starts as an explicit placeholder and __call__
            # refuses to use it until something resolves it -- see __call__ below.
            empty_score = 0.0
            self._normalization_resolved = False
        else:
            empty_score = self.score_segments([])
            self._normalization_resolved = True
        super().__init__(
            n_players=len(self.segments),
            normalize=normalize,
            normalization_value=empty_score,
            verbose=verbose,
            player_names=[segment.label for segment in self.segments],
        )

    def __call__(self, coalitions: np.ndarray, *, verbose: bool = False) -> np.ndarray:
        """Evaluate and normalize coalitions, guarding against an unresolved deferred baseline."""
        if not self._normalization_resolved:
            msg = (
                "ToolUseGame was constructed with defer_empty_coalition_evaluation=True "
                "and its normalization baseline has not been resolved yet. Resolve it "
                "first -- e.g. coalition_evaluation.evaluate_game_exactly(game, ...) -- "
                "before calling the game; otherwise normalized values would be computed "
                "against an uninitialized placeholder baseline instead of the real "
                "empty-coalition score."
            )
            raise RuntimeError(msg)
        return super().__call__(coalitions, verbose=verbose)

    @staticmethod
    def _format_tool_context(tool_descriptions: dict[str, str]) -> str:
        """Render tool definitions as fixed prompt context."""
        return "\n".join(
            f"- {tool_name}: {description}" for tool_name, description in tool_descriptions.items()
        )

    @staticmethod
    def _coerce_user_segments(user_segments: list[str | ToolUseSegment]) -> list[ToolUseSegment]:
        """Return labeled user segments regardless of caller input shape."""
        segments: list[ToolUseSegment] = []
        for idx, segment in enumerate(user_segments):
            if hasattr(segment, "text"):
                if getattr(segment, "source", "user") != "user":
                    continue
                text = str(segment.text).strip()
                label = str(getattr(segment, "label", f"U{idx + 1}"))
            else:
                text = str(segment).strip()
                label = f"U{idx + 1}"
            if text:
                segments.append(ToolUseSegment(source="user", label=label, text=text))
        return segments

    def selected_segments(self, coalition: np.ndarray) -> list[ToolUseSegment]:
        """Return user-request segments selected by a boolean coalition."""
        coalition = np.asarray(coalition, dtype=bool)
        return [segment for keep, segment in zip(coalition, self.segments, strict=True) if keep]

    def score_segments(self, selected_segments: list[ToolUseSegment]) -> float:
        """Score support for the target tool from selected user-request segments."""
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

    def build_prompt(self, selected_segments: list[str | ToolUseSegment]) -> str:
        """Build a coalition prompt with fixed system/tool context and selected user text.

        Delegates to ``scorers.build_coalition_prompt``, the single canonical prompt
        format shared with ``app.py``, so the same coalition always produces the
        identical prompt text regardless of call site. This matters most for the
        empty coalition, which both this game's normalization baseline and
        value-function scorers (e.g. ``FinalAnswerSimilarityScorer``) treat as a
        cache key; a textually different "empty" prompt would trigger a second,
        independent agent call for what should be a single cached value and could
        break Shapley efficiency under non-deterministic backends.
        """
        user_request = _join_user_request_segments(selected_segments)
        return _build_canonical_coalition_prompt(
            user_request,
            system_prompt=self.system_prompt,
            tool_context=self.tool_context,
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate support for each coalition of user-request segments."""
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


def _default_tool_descriptions() -> dict[str, str]:
    """Return derived tool descriptions from the canonical schema module."""
    try:
        from demos.agentic_tool_use_explanation.tool_schemas import TOOL_DESCRIPTIONS
    except ModuleNotFoundError:
        from tool_schemas import TOOL_DESCRIPTIONS

    return dict(TOOL_DESCRIPTIONS)
