"""Scorers for agentic tool-use coalition value functions."""

from __future__ import annotations

import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

DEFAULT_HF_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_LOGPROB_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_CANDIDATE_TEMPLATE = "The correct tool is {tool_name}."

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


class LogProbToolScorer:
    """Score tool decisions from local LM continuation likelihoods."""

    def __init__(
        self,
        model_id: str = DEFAULT_LOGPROB_MODEL_ID,
        candidate_template: str = DEFAULT_CANDIDATE_TEMPLATE,
        candidate_texts: dict[str, str] | None = None,
        device: str | None = None,
        dtype: str = "auto",
        normalize_by_length: bool = True,
        max_pairs_per_batch: int | None = 32,
    ) -> None:
        self.model_id = model_id
        self.candidate_template = candidate_template
        self.candidate_texts = candidate_texts or {}
        self.device = device
        self.dtype = dtype
        self.normalize_by_length = normalize_by_length
        self.max_pairs_per_batch = max_pairs_per_batch
        self.last_debug_outputs: list[dict[str, object]] = []
        if max_pairs_per_batch is not None and max_pairs_per_batch < 1:
            msg = "max_pairs_per_batch must be positive or None."
            raise ValueError(msg)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        if self.device is None or self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        model_kwargs: dict[str, Any] = {}
        if dtype != "auto":
            model_kwargs["torch_dtype"] = getattr(torch, dtype)
        elif self.device == "cuda":
            model_kwargs["torch_dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _candidate_continuation(self, tool_name: str) -> str:
        """Return the continuation used to score one candidate tool."""
        if tool_name in self.candidate_texts:
            return self.candidate_texts[tool_name]
        return self.candidate_template.format(tool_name=tool_name)

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        """Return a contrastive tool-decision score for each coalition prompt.

        For ordinary tools this is the target-vs-reference log-score difference
        ``log P(target_tool) - log P(no_tool)`` using the scorer's candidate
        continuation log scores directly. When ``target_tool == "no_tool"``,
        the reference is the strongest available non-no-tool candidate.
        """
        candidate_tools = self._validate_candidate_tools(target_tool, tool_descriptions)
        self.last_debug_outputs = []
        candidate_score_rows = self._candidate_log_scores(prompts, candidate_tools)
        if len(candidate_score_rows) != len(prompts):
            msg = "Candidate scoring must return one score dictionary per prompt."
            raise ValueError(msg)

        scores = []
        candidate_continuations = {
            tool_name: self._candidate_continuation(tool_name) for tool_name in candidate_tools
        }
        for prompt, candidate_log_scores in zip(prompts, candidate_score_rows, strict=True):
            final_score = self._contrastive_score(
                candidate_log_scores,
                target_tool=target_tool,
                candidate_tools=candidate_tools,
            )
            scores.append(final_score)
            self.last_debug_outputs.append(
                {
                    "target_tool": target_tool,
                    "candidate_tools": candidate_tools.copy(),
                    "candidate_continuations": [
                        candidate_continuations[tool_name] for tool_name in candidate_tools
                    ],
                    "candidate_log_scores": [
                        candidate_log_scores[tool_name] for tool_name in candidate_tools
                    ],
                    "candidate_logprobs": [
                        candidate_log_scores[tool_name] for tool_name in candidate_tools
                    ],
                    "reference_tool": self._reference_tool(
                        candidate_log_scores,
                        target_tool=target_tool,
                        candidate_tools=candidate_tools,
                    ),
                    "final_score": final_score,
                    "prompt_preview": prompt[:240],
                }
            )
        return scores

    def _validate_candidate_tools(
        self,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[str]:
        """Return available decision candidates after validating required tools."""
        candidate_tools = list(tool_descriptions)
        if len(candidate_tools) < 2:
            msg = "LogProbToolScorer requires at least two decision candidates."
            raise ValueError(msg)
        if target_tool not in candidate_tools:
            msg = f"Target tool {target_tool!r} is not available."
            raise ValueError(msg)
        if "no_tool" not in candidate_tools:
            msg = "LogProbToolScorer requires a 'no_tool' decision candidate."
            raise ValueError(msg)
        return candidate_tools

    def _candidate_log_scores(
        self,
        prompts: list[str],
        candidate_tools: list[str],
    ) -> list[dict[str, float]]:
        """Score all prompt/candidate continuations in batched model calls."""
        pair_prompts = []
        pair_continuations = []
        pair_tools = []
        for prompt in prompts:
            for tool_name in candidate_tools:
                pair_prompts.append(prompt)
                pair_continuations.append(self._candidate_continuation(tool_name))
                pair_tools.append(tool_name)

        pair_scores = self._sequence_logprobs_batched(pair_prompts, pair_continuations)
        if len(pair_scores) != len(pair_tools):
            msg = "Candidate scoring must return one log score per prompt/candidate pair."
            raise ValueError(msg)

        rows = [{tool_name: math.nan for tool_name in candidate_tools} for _ in prompts]
        for pair_index, (tool_name, score) in enumerate(zip(pair_tools, pair_scores, strict=True)):
            prompt_index = pair_index // len(candidate_tools)
            rows[prompt_index][tool_name] = self._validate_log_score(tool_name, score)
        return rows

    def _contrastive_score(
        self,
        candidate_log_scores: dict[str, float],
        *,
        target_tool: str,
        candidate_tools: list[str],
    ) -> float:
        """Return the target-vs-reference log-score difference."""
        reference_tool = self._reference_tool(
            candidate_log_scores,
            target_tool=target_tool,
            candidate_tools=candidate_tools,
        )
        target_score = self._require_candidate_score(candidate_log_scores, target_tool)
        reference_score = self._require_candidate_score(candidate_log_scores, reference_tool)
        return target_score - reference_score

    def _reference_tool(
        self,
        candidate_log_scores: dict[str, float],
        *,
        target_tool: str,
        candidate_tools: list[str],
    ) -> str:
        """Choose the reference candidate for the contrastive score."""
        self._validate_score_dict(candidate_log_scores, candidate_tools)
        if target_tool != "no_tool":
            return "no_tool"
        alternatives = [tool_name for tool_name in candidate_tools if tool_name != "no_tool"]
        if not alternatives:
            msg = "no_tool contrastive scoring requires a non-no-tool reference candidate."
            raise ValueError(msg)
        return max(alternatives, key=lambda tool_name: candidate_log_scores[tool_name])

    def _validate_score_dict(
        self,
        candidate_log_scores: dict[str, float],
        candidate_tools: list[str],
    ) -> None:
        """Validate candidate score coverage and finiteness."""
        for tool_name in candidate_tools:
            self._require_candidate_score(candidate_log_scores, tool_name)

    def _require_candidate_score(
        self,
        candidate_log_scores: dict[str, float],
        tool_name: str,
    ) -> float:
        """Return one finite candidate score or raise a clear validation error."""
        if tool_name not in candidate_log_scores:
            msg = f"Candidate scoring did not return a score for {tool_name!r}."
            raise ValueError(msg)
        return self._validate_log_score(tool_name, candidate_log_scores[tool_name])

    def _validate_log_score(self, tool_name: str, score: float) -> float:
        """Return a finite log score as float."""
        score = float(score)
        if not math.isfinite(score):
            msg = f"Candidate score for {tool_name!r} must be finite."
            raise ValueError(msg)
        return score

    def _sequence_logprobs_batched(
        self,
        prompts: list[str],
        continuations: list[str],
    ) -> list[float]:
        """Score continuation likelihoods for prompt/candidate pairs in batches."""
        if len(prompts) != len(continuations):
            msg = "Prompts and continuations must have the same length."
            raise ValueError(msg)
        if not prompts:
            return []

        max_pairs_per_batch = getattr(self, "max_pairs_per_batch", None)
        if max_pairs_per_batch is None:
            return self._sequence_logprobs_batch(prompts, continuations)

        scores: list[float] = []
        for start in range(0, len(prompts), max_pairs_per_batch):
            stop = start + max_pairs_per_batch
            scores.extend(
                self._sequence_logprobs_batch(
                    prompts[start:stop],
                    continuations[start:stop],
                )
            )
        return scores

    def _sequence_logprobs_batch(
        self,
        prompts: list[str],
        continuations: list[str],
    ) -> list[float]:
        """Score continuation token likelihood under a causal LM in one batch."""
        torch = self._torch
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        full_inputs = self.tokenizer(
            [
                prompt + continuation
                for prompt, continuation in zip(prompts, continuations, strict=True)
            ],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=-1).tolist()
        full_lengths = full_inputs["attention_mask"].sum(dim=-1).tolist()
        continuation_lengths = [
            int(full_len - prompt_len)
            for prompt_len, full_len in zip(prompt_lengths, full_lengths, strict=True)
        ]
        if any(length <= 0 for length in continuation_lengths):
            msg = "Continuation must add at least one token."
            raise ValueError(msg)

        input_ids = full_inputs["input_ids"].to(self.device)
        attention_mask = full_inputs["attention_mask"].to(self.device)
        with torch.inference_mode():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = input_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)

        scores = []
        for row_index, (prompt_len, continuation_len) in enumerate(
            zip(prompt_lengths, continuation_lengths, strict=True)
        ):
            start = int(prompt_len - 1)
            stop = start + int(continuation_len)
            score = float(token_log_probs[row_index, start:stop].sum().item())
            if self.normalize_by_length:
                score /= int(continuation_len)
            scores.append(score)
        return scores

    def _sequence_logprob(self, prompt: str, continuation: str) -> float:
        """Score continuation token likelihood under a causal LM."""
        torch = self._torch
        prompt_inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        full_inputs = self.tokenizer(
            prompt + continuation,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_len = int(prompt_inputs["input_ids"].shape[-1])
        input_ids = full_inputs["input_ids"].to(self.device)
        continuation_len = int(input_ids.shape[-1] - prompt_len)
        if continuation_len <= 0:
            msg = "Continuation must add at least one token."
            raise ValueError(msg)

        with torch.inference_mode():
            logits = self.model(input_ids=input_ids).logits
            log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = input_ids[:, 1:]
            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            continuation_log_probs = token_log_probs[:, prompt_len - 1 :]
            score = float(continuation_log_probs.sum().item())
        if self.normalize_by_length:
            score /= continuation_len
        return score
