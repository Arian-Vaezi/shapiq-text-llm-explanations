"""Reusable RAG retrieval attribution game for the Streamlit demo.

The game treats retrieved chunks as players. A coalition selects which chunks
are visible to the answer scorer.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from shapiq.game import Game

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "what",
    "which",
    "who",
    "with",
}


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved context chunk shown to the RAG model."""

    title: str
    text: str
    page_number: int | None = None
    chunk_type: str = "body_text"
    section_title: str = ""
    text_length: int = 0
    retrieval_score: float | None = None
    dense_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    flags: tuple[str, ...] = ()


ScoreCallable = Callable[[str, str, list[RetrievedChunk]], float]


def normalize_tokens(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric words without common stopwords."""
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [token for token in tokens if token not in STOPWORDS and len(token) > 1]


def lexical_grounding_score(
    question: str,
    target_answer: str,
    selected_chunks: list[RetrievedChunk],
) -> float:
    """Lightweight support score for local demos.

    DEMO SCAFFOLD: this intentionally avoids model downloads/API keys. It
    scores how much of the target answer is supported by the selected context,
    with a small question-overlap bonus and a length penalty for noisy context.

    FINAL DEMO: replace this with a real value function, such as target-answer
    log-likelihood, entailment/groundedness score, or answer confidence from a
    RAG pipeline.
    """
    if not selected_chunks:
        return 0.0

    context = " ".join(chunk.text for chunk in selected_chunks)
    context_terms = set(normalize_tokens(context))
    answer_terms = normalize_tokens(target_answer)
    question_terms = set(normalize_tokens(question))

    if not answer_terms:
        return 0.0

    answer_hits = sum(1 for term in answer_terms if term in context_terms)
    answer_coverage = answer_hits / len(answer_terms)

    question_bonus = 0.0
    if question_terms:
        question_bonus = len(question_terms & context_terms) / len(question_terms)

    # A mildly convex support curve makes complementary evidence easier to see:
    # partial chunks receive modest scores, while near-complete answer support
    # rises sharply. This is only the local demo scorer, not the final method.
    noise_penalty = 0.035 * max(0, len(selected_chunks) - 2)
    support_score = answer_coverage**1.35
    score = 0.88 * support_score + 0.12 * question_bonus - noise_penalty
    return float(max(0.0, min(1.0, score)))


def budget_for_exactish_demo(n_players: int, max_budget: int = 512) -> int:
    """Exact 2^n budget for small games, capped at max_budget to force sampling for larger ones."""
    return min(2**n_players, max_budget)


class RAGRetrievalGame(Game):
    """Coalition game for RAG retrieval attribution."""

    def __init__(
        self,
        question: str,
        target_answer: str,
        chunks: list[RetrievedChunk],
        *,
        scorer: ScoreCallable | None = None,
        normalize: bool = True,
        verbose: bool = False,
    ) -> None:
        """Initialize the retrieval coalition game."""
        if not chunks:
            msg = "RAGRetrievalGame requires at least one retrieved chunk."
            raise ValueError(msg)

        self.question = question
        self.target_answer = target_answer
        self.chunks = chunks
        self.scorer = scorer or lexical_grounding_score
        self._score_cache: dict[tuple[int, ...], float] = {}
        empty_score = self.score_context([])

        super().__init__(
            n_players=len(chunks),
            normalize=normalize,
            normalization_value=empty_score,
            verbose=verbose,
            player_names=[chunk.title for chunk in chunks],
        )

    def selected_chunks(self, coalition: np.ndarray) -> list[RetrievedChunk]:
        """Return chunks selected by a boolean coalition vector."""
        coalition = np.asarray(coalition, dtype=bool)
        return [chunk for keep, chunk in zip(coalition, self.chunks, strict=True) if keep]

    def score_context(self, selected_chunks: list[RetrievedChunk]) -> float:
        """Score how strongly selected chunks support the target answer.

        Replace this method or pass a custom `scorer` to connect the framework
        to a real RAG model or LLM log-likelihood target.
        """
        return self.scorer(self.question, self.target_answer, selected_chunks)

    def build_prompt(self, selected_chunks: list[RetrievedChunk]) -> str:
        """Build the prompt that a future model-backed scorer could evaluate.

        The current lexical scorer does not call this prompt. It is shown in the
        Streamlit UI so the demo audience can see what each coalition would mean
        in a real RAG prompt: only the selected chunks are visible.
        """
        context_blocks = [
            f"[{idx}] {chunk.title}\n{chunk.text}"
            for idx, chunk in enumerate(selected_chunks, start=1)
        ]
        context = "\n\n".join(context_blocks) if context_blocks else "(no retrieved context)"
        return (
            "Answer the question using only the retrieved context. If the context does not "
            "contain the answer, say that the provided context is insufficient.\n\n"
            f"Question:\n{self.question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer:"
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate each coalition of retrieved chunks.

        This is the method shapiq calls repeatedly. Each row is a boolean mask
        over retrieved chunks; the returned value is the answer-support score for
        that selected context. Results are cached so repeated evaluations of the
        same coalition (e.g. by random baselines) do not re-invoke the scorer.
        """
        values = np.zeros(coalitions.shape[0], dtype=float)
        for row_idx, coalition in enumerate(coalitions):
            key = tuple(int(i) for i in np.where(coalition)[0])
            if key not in self._score_cache:
                self._score_cache[key] = self.score_context(self.selected_chunks(coalition))
            values[row_idx] = self._score_cache[key]
        return values
