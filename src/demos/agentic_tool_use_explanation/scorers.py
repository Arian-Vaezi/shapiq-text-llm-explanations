"""Scorers for agentic tool-use coalition value functions."""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

DEFAULT_HF_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

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


@dataclass(frozen=True)
class ToolChoice:
    """A lightweight tool-router decision."""

    tool: str
    score: float
    reason: str
    scores: dict[str, float]


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


@dataclass
class HuggingFaceTextGenerator:
    """Adapt the shared HuggingFace wrapper to TextGeneratorProtocol."""

    model_id: str = DEFAULT_HF_MODEL_ID
    device: str = "auto"
    hf_token: str | None = None
    max_new_tokens: int = 8
    use_chat_template: bool = True

    def __post_init__(self) -> None:
        wrapper_device = "cuda" if self.device == "auto" else self.device
        try:
            from demos.shared.hf_model import HFModelWrapper
        except ModuleNotFoundError:
            src_dir = Path(__file__).resolve().parents[2]
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            from demos.shared.hf_model import HFModelWrapper

        self._model = HFModelWrapper(
            model_name=self.model_id,
            device=wrapper_device,
            hf_token=self.hf_token or None,
        )

    def generate(self, prompt: str) -> str:
        """Generate one scoring response for an LLM-as-a-judge prompt."""
        return self._model.generate_text(
            prompt,
            max_new_tokens=self.max_new_tokens,
            chat=self.use_chat_template,
        ).strip()


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
class LexicalToolRouter:
    """Small local router used when no real LLM backend is loaded."""

    scorer: LexicalToolScorer = field(default_factory=LexicalToolScorer)

    def choose_tool(self, prompt: str, tool_descriptions: dict[str, str]) -> ToolChoice:
        """Choose the most supported tool for a user prompt."""
        scores = {
            tool_name: self.scorer.score_prompt(
                prompt,
                target_tool=tool_name,
                tool_descriptions=tool_descriptions,
            )
            for tool_name in tool_descriptions
        }
        selected_tool = max(scores, key=scores.get)
        return ToolChoice(
            tool=selected_tool,
            score=scores[selected_tool],
            reason=self._build_reason(prompt, selected_tool),
            scores=scores,
        )

    def _build_reason(self, prompt: str, selected_tool: str) -> str:
        tokens = normalize_tokens(prompt)
        hits = sorted(tokens & self.scorer.tool_keywords[selected_tool])
        if selected_tool == "weather_tool":
            purpose = "the question asks about weather, forecast, place, or time."
        elif selected_tool == "calculator_tool":
            purpose = "the question asks for exact arithmetic or a numeric result."
        elif selected_tool == "web_search_tool":
            purpose = "the question depends on current, recent, latest, or external facts."
        else:
            purpose = "the question can be answered directly from stable knowledge."

        if hits:
            return f"Matched {', '.join(hits[:5])}; {purpose}"
        return purpose


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
    last_debug_outputs: list[dict[str, object]] = field(default_factory=list, init=False)

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
        self.last_debug_outputs = []
        scores = []
        for prompt in prompts:
            scoring_prompt = self.build_scoring_prompt(
                prompt,
                target_tool=target_tool,
                tool_descriptions=tool_descriptions,
            )
            raw_output = None
            parsed_score = None
            fallback_score = None
            used_fallback = False
            try:
                raw_output = self.llm.generate(scoring_prompt)
                parsed_score = self.parse_score(raw_output)
                final_score = parsed_score
            except (RuntimeError, TypeError, ValueError):
                used_fallback = True
                fallback_score = self._fallback_score(
                    prompt,
                    target_tool=target_tool,
                    tool_descriptions=tool_descriptions,
                )
                final_score = fallback_score
            scores.append(final_score)
            self.last_debug_outputs.append(
                {
                    "target_tool": target_tool,
                    "raw_output": raw_output,
                    "parsed_score": parsed_score,
                    "used_fallback": used_fallback,
                    "fallback_score": fallback_score,
                    "final_score": final_score,
                }
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
        match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", output)
        if match is None:
            msg = "LLM output did not contain a numeric score."
            raise ValueError(msg)
        score = float(match.group(0))
        if not math.isfinite(score):
            msg = "LLM score must be finite."
            raise ValueError(msg)
        if not 0.0 <= score <= 1.0:
            msg = "LLM score must be between 0 and 1."
            raise ValueError(msg)
        return score

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
