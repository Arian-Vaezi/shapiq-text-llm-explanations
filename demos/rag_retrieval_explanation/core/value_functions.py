"""Value functions for retrieved-context attribution experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from .model_backends import HuggingFaceCausalLMBackend
from .rag_game import RetrievedChunk, lexical_grounding_score, normalize_tokens


class RetrievalValueFunction(Protocol):
    """Callable value function used by `RAGRetrievalGame`."""

    name: str
    description: str

    def __call__(
        self,
        question: str,
        target_answer: str,
        selected_chunks: list[RetrievedChunk],
    ) -> float:
        """Score a coalition of retrieved chunks."""

    def warmup(self) -> None:
        """Load any resources needed before scoring."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks for a model-backed RAG prompt."""
    if not chunks:
        return "(no retrieved context)"
    return "\n\n".join(
        f"[{idx}] {chunk.title}\n{chunk.text}" for idx, chunk in enumerate(chunks, start=1)
    )


@dataclass(frozen=True)
class LexicalGroundingValue:
    """Fast local scorer based on target-answer term coverage."""

    name: str = "Lexical grounding"
    description: str = "Answer-term coverage with a small question-overlap bonus."

    def __call__(
        self,
        question: str,
        target_answer: str,
        selected_chunks: list[RetrievedChunk],
    ) -> float:
        """Score selected chunks with the local lexical baseline."""
        return lexical_grounding_score(question, target_answer, selected_chunks)

    def warmup(self) -> None:
        """Lexical scoring has no model resources to load."""


@dataclass(frozen=True)
class TargetLikelihoodValue:
    """Average target-answer likelihood under a Hugging Face causal LM."""

    backend: HuggingFaceCausalLMBackend
    temperature: float = 4.0
    name: str = "HF target likelihood"
    description: str = (
        "Scores how likely the model finds the exact target answer text after the selected "
        "context. This measures prompt-conditioned likelihood, not factual correctness by itself."
    )

    def __call__(
        self,
        question: str,
        target_answer: str,
        selected_chunks: list[RetrievedChunk],
    ) -> float:
        """Score selected chunks by target-answer likelihood."""
        context = format_context(selected_chunks)
        prompt = self.backend.build_chat_prompt(question, context)
        avg_log_likelihood = self.backend.target_log_likelihood(prompt, target_answer)
        return float(1.0 / (1.0 + math.exp(-avg_log_likelihood / self.temperature)))

    def warmup(self) -> None:
        """Load tokenizer and model before the first score call."""
        _ = self.backend.tokenizer
        _ = self.backend.model


@dataclass(frozen=True)
class ContrastiveLikelihoodValue:
    """Likelihood gain from adding retrieved context over no retrieved context."""

    backend: HuggingFaceCausalLMBackend
    temperature: float = 4.0
    name: str = "HF contrastive likelihood"
    description: str = "Target-answer likelihood relative to the no-context prompt."

    def __call__(
        self,
        question: str,
        target_answer: str,
        selected_chunks: list[RetrievedChunk],
    ) -> float:
        """Score selected chunks by likelihood gain over empty context."""
        selected_context = format_context(selected_chunks)
        empty_context = format_context([])
        selected_prompt = self.backend.build_chat_prompt(question, selected_context)
        empty_prompt = self.backend.build_chat_prompt(question, empty_context)

        selected_ll = self.backend.target_log_likelihood(selected_prompt, target_answer)
        empty_ll = self.backend.target_log_likelihood(empty_prompt, target_answer)
        return float(1.0 / (1.0 + math.exp(-(selected_ll - empty_ll) / self.temperature)))

    def warmup(self) -> None:
        """Load tokenizer and model before the first score call."""
        _ = self.backend.tokenizer
        _ = self.backend.model


@dataclass(frozen=True)
class GeneratedAnswerOverlapValue:
    """Generate an answer and compare it with the target answer by token overlap."""

    backend: HuggingFaceCausalLMBackend
    name: str = "HF generated answer overlap"
    description: str = "Generates an answer, then scores token overlap with the target answer."

    def __call__(
        self,
        question: str,
        target_answer: str,
        selected_chunks: list[RetrievedChunk],
    ) -> float:
        """Score selected chunks by generated-answer target overlap."""
        context = format_context(selected_chunks)
        generated = self.backend.generate(question, context).answer
        target_terms = set(normalize_tokens(target_answer))
        generated_terms = set(normalize_tokens(generated))
        if not target_terms:
            return 0.0
        return len(target_terms & generated_terms) / len(target_terms)

    def warmup(self) -> None:
        """Load tokenizer and model before the first generated-answer score."""
        _ = self.backend.tokenizer
        _ = self.backend.model


def make_value_function(
    mode: str,
    *,
    model_id: str,
    device_map: str = "auto",
    torch_dtype: str = "auto",
    max_new_tokens: int = 96,
) -> RetrievalValueFunction:
    """Create a configured retrieval value function."""
    if mode == "Lexical grounding":
        return LexicalGroundingValue()

    backend = HuggingFaceCausalLMBackend(
        model_id=model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
    )
    if mode == "HF target likelihood":
        return TargetLikelihoodValue(backend=backend)
    if mode == "HF contrastive likelihood":
        return ContrastiveLikelihoodValue(backend=backend)
    if mode == "HF generated answer overlap":
        return GeneratedAnswerOverlapValue(backend=backend)

    msg = f"Unknown value function mode: {mode}"
    raise ValueError(msg)
