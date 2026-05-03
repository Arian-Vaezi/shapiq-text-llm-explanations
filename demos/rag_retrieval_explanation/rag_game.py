"""Reusable RAG retrieval attribution game for the Streamlit demo.

The game treats retrieved chunks as players. A coalition selects which chunks
are visible to the answer scorer.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Callable

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

    This intentionally avoids model downloads. It scores how much of the target
    answer is supported by the selected context, with a small question-overlap
    bonus and a length penalty for noisy context.
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

    noise_penalty = 0.035 * max(0, len(selected_chunks) - 2)
    # A mildly convex support curve makes complementary evidence easier to see:
    # partial chunks receive modest scores, while near-complete answer support
    # rises sharply. This is only the local demo scorer, not the final method.
    support_score = answer_coverage**1.35
    score = 0.88 * support_score + 0.12 * question_bonus - noise_penalty
    return float(max(0.0, min(1.0, score)))


class RAGRetrievalGame(Game):
    """Coalition game for RAG retrieval attribution.

    Args:
        question: User question sent to the RAG system.
        target_answer: Answer whose grounding/support should be attributed.
        chunks: Retrieved chunks. Each chunk is one player.
        scorer: Optional scoring function. If omitted, a lexical grounding
            scorer is used.
        normalize: Whether to center values at the empty context score.
    """

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
        if not chunks:
            msg = "RAGRetrievalGame requires at least one retrieved chunk."
            raise ValueError(msg)

        self.question = question
        self.target_answer = target_answer
        self.chunks = chunks
        self.scorer = scorer or lexical_grounding_score
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
        """Build the prompt used by a model-backed scorer."""
        context_blocks = [
            f"[{idx}] {chunk.title}\n{chunk.text}"
            for idx, chunk in enumerate(selected_chunks, start=1)
        ]
        context = "\n\n".join(context_blocks) if context_blocks else "(no retrieved context)"
        return (
            "Answer the question using only the retrieved context.\n\n"
            f"Question:\n{self.question}\n\n"
            f"Retrieved context:\n{context}\n\n"
            "Answer:"
        )

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Evaluate each coalition of retrieved chunks."""
        values = np.zeros(coalitions.shape[0], dtype=float)
        for row_idx, coalition in enumerate(coalitions):
            values[row_idx] = self.score_context(self.selected_chunks(coalition))
        return values


def budget_for_exactish_demo(n_players: int) -> int:
    """Reasonable default budget for a small interactive demo."""
    return int(min(2**n_players, max(32, 8 * n_players * math.log2(n_players + 1))))
