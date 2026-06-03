"""Evaluation metrics for retrieval, answer quality, and Shapley attribution.

All functions are pure (no I/O, no global state) and return typed dataclasses
so runners and the grid eval can log results without knowing each other's schema.

Metric categories
-----------------
Retrieval:
  - hit_rate           expected evidence chunk present in top-k results
  - recall_at_k        fraction of expected evidence chunks in top-k
  - mrr                mean reciprocal rank of first expected chunk
  - avg_retrieval_score  mean retrieval score of expected chunks

Answer:
  - answer_contains_expected  keyword/substring match in generated answer
  - citation_hit              answer references an expected chunk by title
  - answer_supported          answer is not flagged as unsupported

Attribution (Shapley):
  - top_shapley_is_expected   highest-attributed chunk matches expected evidence
  - shapley_mass_on_expected  fraction of total |φ| mass on expected chunks
  - rank_corr_retrieval_shapley  Spearman ρ between retrieval and Shapley rankings
  - deletion_auc, insertion_auc  area under step curves from evaluation.py
  - shapley_ranking_kendall_tau  cross-config rank stability (requires multiple runs)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RetrievalMetrics:
    """Metrics measuring retrieval quality against expected evidence."""

    hit_rate: float
    recall_at_k: float
    mrr: float | None
    avg_retrieval_score: float | None


@dataclass
class AnswerMetrics:
    """Metrics measuring generated answer quality."""

    answer_contains_expected: bool
    citation_hit: bool
    answer_supported: bool


@dataclass
class AttributionMetrics:
    """Metrics measuring Shapley attribution quality."""

    top_shapley_is_expected: bool | None
    shapley_mass_on_expected: float | None
    rank_corr_retrieval_shapley: float | None
    deletion_auc: float | None
    insertion_auc: float | None


@dataclass
class EvalMetrics:
    """Combined metrics for one (case × config) eval result."""

    case_id: str
    config_id: str
    retrieval: RetrievalMetrics
    answer: AnswerMetrics
    attribution: AttributionMetrics
    extra: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Flatten all metrics into one row dict for summary.csv."""
        return {
            "case_id": self.case_id,
            "config_id": self.config_id,
            "hit_rate": self.retrieval.hit_rate,
            "recall_at_k": self.retrieval.recall_at_k,
            "mrr": self.retrieval.mrr,
            "avg_retrieval_score": self.retrieval.avg_retrieval_score,
            "answer_contains_expected": self.answer.answer_contains_expected,
            "citation_hit": self.answer.citation_hit,
            "answer_supported": self.answer.answer_supported,
            "top_shapley_is_expected": self.attribution.top_shapley_is_expected,
            "shapley_mass_on_expected": self.attribution.shapley_mass_on_expected,
            "rank_corr_retrieval_shapley": self.attribution.rank_corr_retrieval_shapley,
            "deletion_auc": self.attribution.deletion_auc,
            "insertion_auc": self.attribution.insertion_auc,
            **self.extra,
        }


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


def compute_retrieval_metrics(
    retrieved_titles: list[str],
    expected_titles: list[str],
    retrieval_scores: list[float] | None = None,
) -> RetrievalMetrics:
    """Compute retrieval quality metrics against known expected evidence.

    Args:
        retrieved_titles: Titles of chunks returned by the retriever (in rank order).
        expected_titles:  Titles of chunks that should appear in the top-k.
        retrieval_scores: Retrieval scores parallel to retrieved_titles.
    """
    if not expected_titles:
        return RetrievalMetrics(
            hit_rate=1.0,
            recall_at_k=1.0,
            mrr=None,
            avg_retrieval_score=None,
        )

    retrieved_set = set(retrieved_titles)
    hit_count = sum(1 for title in expected_titles if title in retrieved_set)
    hit_rate = float(hit_count > 0)
    recall = hit_count / len(expected_titles)

    mrr: float | None = None
    for rank, title in enumerate(retrieved_titles, start=1):
        if title in set(expected_titles):
            mrr = 1.0 / rank
            break

    avg_score: float | None = None
    if retrieval_scores:
        score_by_title = dict(zip(retrieved_titles, retrieval_scores, strict=False))
        expected_scores = [
            score_by_title[t] for t in expected_titles if t in score_by_title
        ]
        if expected_scores:
            avg_score = float(np.mean(expected_scores))

    return RetrievalMetrics(
        hit_rate=hit_rate,
        recall_at_k=recall,
        mrr=mrr,
        avg_retrieval_score=avg_score,
    )


# ---------------------------------------------------------------------------
# Answer metrics
# ---------------------------------------------------------------------------

_UNSUPPORTED_PATTERNS = [
    r"\b(context (does not|doesn't) (contain|have|provide))\b",
    r"\b(insufficient (context|information))\b",
    r"\b(cannot (answer|find|determine))\b",
    r"\b(not (mentioned|stated|found) in the (context|retrieved))\b",
]
_COMPILED_UNSUPPORTED = [re.compile(p, re.IGNORECASE) for p in _UNSUPPORTED_PATTERNS]


def _answer_is_unsupported(answer: str) -> bool:
    """Heuristically detect answers that admit lacking evidence."""
    return any(pattern.search(answer) for pattern in _COMPILED_UNSUPPORTED)


def compute_answer_metrics(
    answer: str,
    expected_answer_keywords: list[str],
    retrieved_titles: list[str],
) -> AnswerMetrics:
    """Compute answer quality metrics.

    Args:
        answer:                   Generated answer text.
        expected_answer_keywords: Keywords or substrings that should appear in answer.
        retrieved_titles:         Titles of retrieved chunks (for citation hit check).
    """
    answer_lower = answer.lower()

    contains_expected = (
        all(kw.lower() in answer_lower for kw in expected_answer_keywords)
        if expected_answer_keywords
        else True
    )

    citation_hit = any(
        title.lower() in answer_lower or answer_lower in title.lower()
        for title in retrieved_titles
    )

    return AnswerMetrics(
        answer_contains_expected=contains_expected,
        citation_hit=citation_hit,
        answer_supported=not _answer_is_unsupported(answer),
    )


# ---------------------------------------------------------------------------
# Attribution metrics
# ---------------------------------------------------------------------------


def normalized_auc(values: np.ndarray) -> float:
    """Area under a step curve, normalized to the number of intervals."""
    if len(values) <= 1:
        return float(values[0]) if len(values) == 1 else 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.trapezoid(values, x) / (len(values) - 1))


def spearman_rho(ranks_a: list[float], ranks_b: list[float]) -> float | None:
    """Spearman rank correlation between two equal-length rank lists."""
    if len(ranks_a) != len(ranks_b) or len(ranks_a) < 2:
        return None
    a = np.array(ranks_a, dtype=float)
    b = np.array(ranks_b, dtype=float)
    n = len(a)
    d_sq = np.sum((a - b) ** 2)
    return float(1.0 - 6.0 * d_sq / (n * (n**2 - 1)))


def compute_attribution_metrics(
    shapley_frame: pd.DataFrame,
    expected_titles: list[str],
    retrieval_scores: list[float] | None = None,
    deletion_curve: np.ndarray | None = None,
    insertion_curve: np.ndarray | None = None,
) -> AttributionMetrics:
    """Compute Shapley attribution quality metrics.

    Args:
        shapley_frame:    DataFrame with columns ['chunk', 'attribution'] sorted
                          by |attribution| descending (first_order_frame output).
        expected_titles:  Titles that should receive high attribution.
        retrieval_scores: Retrieval scores in the same order as shapley_frame rows.
        deletion_curve:   Score array from deletion_evaluation (step 0..n).
        insertion_curve:  Score array from insertion_evaluation (step 0..n).
    """
    if shapley_frame.empty or not expected_titles:
        return AttributionMetrics(
            top_shapley_is_expected=None,
            shapley_mass_on_expected=None,
            rank_corr_retrieval_shapley=None,
            deletion_auc=normalized_auc(deletion_curve) if deletion_curve is not None else None,
            insertion_auc=normalized_auc(insertion_curve) if insertion_curve is not None else None,
        )

    expected_set = set(expected_titles)
    top_chunk = str(shapley_frame.iloc[0]["chunk"])
    top_is_expected = top_chunk in expected_set

    attributions = shapley_frame["attribution"].to_numpy(dtype=float)
    abs_total = float(np.sum(np.abs(attributions)))
    if abs_total > 0:
        expected_mass = sum(
            abs(float(row["attribution"]))
            for _, row in shapley_frame.iterrows()
            if str(row["chunk"]) in expected_set
        )
        mass_fraction = expected_mass / abs_total
    else:
        mass_fraction = None

    rank_corr: float | None = None
    if retrieval_scores and len(retrieval_scores) == len(shapley_frame):
        shapley_ranks = list(range(1, len(shapley_frame) + 1))
        retrieval_ranks = [
            sorted(range(len(retrieval_scores)), key=lambda i: retrieval_scores[i], reverse=True).index(j) + 1
            for j in range(len(retrieval_scores))
        ]
        rank_corr = spearman_rho(retrieval_ranks, shapley_ranks)

    return AttributionMetrics(
        top_shapley_is_expected=top_is_expected,
        shapley_mass_on_expected=mass_fraction,
        rank_corr_retrieval_shapley=rank_corr,
        deletion_auc=normalized_auc(deletion_curve) if deletion_curve is not None else None,
        insertion_auc=normalized_auc(insertion_curve) if insertion_curve is not None else None,
    )


# ---------------------------------------------------------------------------
# Cross-config stability (used by grid runner)
# ---------------------------------------------------------------------------


def kendall_tau(ranking_a: list[str], ranking_b: list[str]) -> float | None:
    """Kendall's τ between two orderings of the same items (by title).

    Returns None if the rankings have fewer than 2 common items.
    """
    common = [t for t in ranking_a if t in set(ranking_b)]
    if len(common) < 2:
        return None

    pos_b = {title: idx for idx, title in enumerate(ranking_b)}
    order_a = [pos_b[t] for t in common if t in pos_b]

    concordant = discordant = 0
    for i in range(len(order_a)):
        for j in range(i + 1, len(order_a)):
            diff = order_a[i] - order_a[j]
            if diff < 0:
                concordant += 1
            elif diff > 0:
                discordant += 1
    n = len(order_a)
    pairs = n * (n - 1) / 2
    if pairs == 0:
        return None
    return float((concordant - discordant) / pairs)


def compute_rank_stability(
    rankings: dict[str, list[str]],
) -> pd.DataFrame:
    """Return a pairwise Kendall-τ matrix across configs.

    Args:
        rankings: {config_id: [chunk_title in rank order]} — one per config.

    Returns:
        DataFrame with config_ids as both index and columns; values are τ.
    """
    config_ids = list(rankings.keys())
    n = len(config_ids)
    matrix = np.full((n, n), np.nan)
    for i, ci in enumerate(config_ids):
        for j, cj in enumerate(config_ids):
            tau = kendall_tau(rankings[ci], rankings[cj])
            if tau is not None:
                matrix[i, j] = tau
    return pd.DataFrame(matrix, index=config_ids, columns=config_ids)
