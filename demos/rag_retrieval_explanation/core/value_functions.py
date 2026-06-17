"""Value functions for retrieved-context attribution experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from .model_backends import LlamaCppBackend, sigmoid
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
    """Average target-answer likelihood under a local GGUF model."""

    backend: LlamaCppBackend
    temperature: float = 4.0
    name: str = "Local target likelihood"
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
        prompt = self.backend.build_prompt(question, context)
        avg_log_likelihood = self.backend.target_log_likelihood(prompt, target_answer)
        return sigmoid(avg_log_likelihood / self.temperature)

    def warmup(self) -> None:
        """Load tokenizer and model before the first score call."""
        _ = self.backend.model


@dataclass(frozen=True)
class ContrastiveLikelihoodValue:
    """Likelihood gain from adding retrieved context over no retrieved context."""

    backend: LlamaCppBackend
    temperature: float = 4.0
    _empty_ll_cache: dict[tuple[str, str], float] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )
    name: str = "Local contrastive likelihood"
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
        selected_prompt = self.backend.build_prompt(question, selected_context)

        selected_ll = self.backend.target_log_likelihood(selected_prompt, target_answer)
        empty_ll = self._empty_log_likelihood(question, target_answer, empty_context)
        return sigmoid((selected_ll - empty_ll) / self.temperature)

    def _empty_log_likelihood(
        self,
        question: str,
        target_answer: str,
        empty_context: str,
    ) -> float:
        """Cache the coalition-independent no-context likelihood."""
        cache_key = (question, target_answer)
        cached = self._empty_ll_cache.get(cache_key)
        if cached is not None:
            return cached
        empty_prompt = self.backend.build_prompt(question, empty_context)
        value = self.backend.target_log_likelihood(empty_prompt, target_answer)
        self._empty_ll_cache[cache_key] = value
        return value

    def warmup(self) -> None:
        """Load tokenizer and model before the first score call."""
        _ = self.backend.model


@dataclass(frozen=True)
class GeneratedAnswerOverlapValue:
    """Generate an answer and compare it with the target answer by token overlap."""

    backend: LlamaCppBackend
    name: str = "Local generated answer overlap"
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
        _ = self.backend.model


def make_value_function(
    mode: str,
    *,
    model_path: str,
    n_ctx: int = 4096,
    n_gpu_layers: int = -1,
    n_threads: int = 0,
    max_new_tokens: int = 96,
) -> RetrievalValueFunction:
    """Create a configured retrieval value function."""
    if mode == "Lexical grounding":
        return LexicalGroundingValue()

    backend = LlamaCppBackend(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        max_new_tokens=max_new_tokens,
    )
    if mode == "Local target likelihood":
        return TargetLikelihoodValue(backend=backend)
    if mode == "Local contrastive likelihood":
        return ContrastiveLikelihoodValue(backend=backend)
    if mode == "Local generated answer overlap":
        return GeneratedAnswerOverlapValue(backend=backend)

    msg = f"Unknown value function mode: {mode}"
    raise ValueError(msg)
