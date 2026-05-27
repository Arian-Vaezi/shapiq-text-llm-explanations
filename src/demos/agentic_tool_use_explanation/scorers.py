"""Scorers for agentic tool-use coalition value functions."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Protocol

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


class ToolScorerProtocol(Protocol):
    """Common interface for coalition value-function scorers."""

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        """Score how strongly each coalition prompt supports the target tool."""


class TextGeneratorProtocol(Protocol):
    """Minimal interface expected by LLMToolScorer."""

    def generate(self, prompt: str) -> str:
        """Generate a text response for one prompt."""


def normalize_tokens(text: str) -> set[str]:
    """Return lowercase alphanumeric tokens."""
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def clamp_score(score: float) -> float:
    """Clamp a numeric score to the value-function range."""
    if not math.isfinite(score):
        msg = "Score must be finite."
        raise ValueError(msg)
    return float(min(1.0, max(0.0, score)))


@dataclass
class LexicalToolScorer:
    """Fast keyword baseline for target-tool support."""

    tool_keywords: dict[str, set[str]] = field(default_factory=lambda: TOOL_KEYWORDS)

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        """Score prompts using lightweight keyword evidence."""
        return [
            self.score_prompt(
                prompt,
                target_tool=target_tool,
                tool_descriptions=tool_descriptions,
            )
            for prompt in prompts
        ]

    def score_prompt(
        self,
        prompt: str,
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> float:
        """Score one prompt with lexical evidence for the target tool."""
        del tool_descriptions
        if not prompt.strip():
            return 0.0

        target_keywords = self.tool_keywords[target_tool]
        tokens = normalize_tokens(prompt)
        target_hits = len(tokens & target_keywords)
        explicit_tool_name = target_tool.lower() in prompt.lower()

        competing_hits = 0.0
        for tool, keywords in self.tool_keywords.items():
            if tool == target_tool:
                continue
            competing_hits += len(tokens & keywords) * 0.35

        raw_score = 0.85 * target_hits + (1.25 if explicit_tool_name else 0.0)
        raw_score -= competing_hits
        return float(1 / (1 + math.exp(-(raw_score - 1.3))))


@dataclass
class MockLLM:
    """Fake LLM for tests and local wiring."""

    response: str | None = None

    def generate(self, prompt: str) -> str:
        """Return a fixed response or a deterministic prompt-aware score."""
        if self.response is not None:
            return self.response
        target_tool = self._extract_block(prompt, "Target tool:", "Available tools:").strip()
        coalition_prompt = self._extract_block(prompt, "Prompt:", "Return only one number")
        return f"{self._score_prompt(coalition_prompt, target_tool):.3f}"

    def _score_prompt(self, prompt: str, target_tool: str) -> float:
        if not prompt.strip():
            return 0.0
        target_keywords = TOOL_KEYWORDS[target_tool]
        tokens = normalize_tokens(prompt)
        target_hits = len(tokens & target_keywords)
        explicit_tool_name = target_tool.lower() in prompt.lower()
        raw_score = 0.95 * target_hits + (1.5 if explicit_tool_name else 0.0)
        return float(1 / (1 + math.exp(-(raw_score - 1.1))))

    @staticmethod
    def _extract_block(text: str, start_marker: str, end_marker: str) -> str:
        _, _, after_start = text.partition(start_marker)
        block, _, _ = after_start.partition(end_marker)
        return block.strip()


@dataclass
class LLMToolScorer:
    """Experimental LLM-as-a-judge scorer for target-tool support."""

    llm: TextGeneratorProtocol
    fallback_scorer: ToolScorerProtocol | None = None

    def __post_init__(self) -> None:
        if self.fallback_scorer is None:
            self.fallback_scorer = LexicalToolScorer()

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        """Score prompts with the LLM, falling back per prompt when needed."""
        scores = []
        for prompt in prompts:
            scoring_prompt = self.build_scoring_prompt(
                prompt,
                target_tool=target_tool,
                tool_descriptions=tool_descriptions,
            )
            try:
                output = self.llm.generate(scoring_prompt)
                scores.append(self.parse_score(output))
            except (RuntimeError, TypeError, ValueError):
                scores.append(
                    self._fallback_score(
                        prompt,
                        target_tool=target_tool,
                        tool_descriptions=tool_descriptions,
                    )
                )
        return scores

    def build_scoring_prompt(
        self,
        prompt: str,
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> str:
        """Build the LLM-as-a-judge prompt for one coalition prompt."""
        tool_lines = "\n".join(
            f"- {tool_name}: {description}" for tool_name, description in tool_descriptions.items()
        )
        return (
            "You are evaluating whether an assistant should call a specific tool.\n\n"
            f"Target tool:\n{target_tool}\n\n"
            f"Available tools:\n{tool_lines}\n\n"
            f"Prompt:\n{prompt}\n\n"
            "Return only one number between 0 and 1."
        )

    def parse_score(self, output: str) -> float:
        """Parse and validate one LLM score."""
        score = float(output.strip())
        if not math.isfinite(score):
            msg = "LLM score must be finite."
            raise ValueError(msg)
        return clamp_score(score)

    def _fallback_score(
        self,
        prompt: str,
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> float:
        """Return a safe fallback score for one prompt."""
        if self.fallback_scorer is None:
            return 0.0
        scores = self.fallback_scorer.score_batch(
            [prompt],
            target_tool=target_tool,
            tool_descriptions=tool_descriptions,
        )
        if len(scores) != 1:
            msg = "Fallback scorer must return one score per prompt."
            raise ValueError(msg)
        return clamp_score(float(scores[0]))
