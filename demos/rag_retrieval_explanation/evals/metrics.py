"""Metrics for end-to-end controlled retrieval and attribution evaluation."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

from .benchmark import passage_id_from_title

if TYPE_CHECKING:
    import pandas as pd


def normalized_auc(values: np.ndarray) -> float:
    """Compute trapezoidal AUC normalized by the number of intervals."""
    if values.size == 0:
        return 0.0
    if values.size == 1:
        return float(values[0])
    return float(np.trapezoid(values, dx=1.0) / (values.size - 1))


def retrieval_metrics(
    retrieved_ids: list[str],
    gold_ids: tuple[str, ...],
) -> dict[str, float | None]:
    """Measure evidence retrieval against stable passage IDs."""
    gold = set(gold_ids)
    if not gold:
        return {
            "retrieval_recall_at_k": None,
            "retrieval_precision_at_k": None,
            "retrieval_mrr": None,
        }
    hits = [passage_id in gold for passage_id in retrieved_ids]
    first_hit = next((index for index, hit in enumerate(hits, start=1) if hit), None)
    return {
        "retrieval_recall_at_k": sum(hits) / len(gold),
        "retrieval_precision_at_k": sum(hits) / len(retrieved_ids) if retrieved_ids else 0.0,
        "retrieval_mrr": 1.0 / first_hit if first_hit else 0.0,
    }


def attribution_metrics(
    frame: pd.DataFrame,
    gold_ids: tuple[str, ...],
) -> dict[str, float | bool | None]:
    """Measure whether attribution mass is concentrated on labelled evidence."""
    if frame.empty or not gold_ids:
        return {
            "top_attribution_is_gold": None,
            "gold_attribution_mass": None,
        }
    ids = [passage_id_from_title(str(title)) for title in frame["chunk"]]
    values = frame["attribution"].to_numpy(dtype=float)
    total_mass = float(np.abs(values).sum())
    gold = set(gold_ids)
    gold_mass = sum(
        abs(value) for passage_id, value in zip(ids, values, strict=True) if passage_id in gold
    )
    return {
        "top_attribution_is_gold": ids[0] in gold,
        "gold_attribution_mass": gold_mass / total_mass if total_mass else None,
    }


def token_f1(prediction: str, reference: str) -> float:
    """Return bag-of-token F1 after lowercase alphanumeric normalization."""
    predicted = re.findall(r"[a-z0-9]+", prediction.lower())
    expected = re.findall(r"[a-z0-9]+", reference.lower())
    if not predicted or not expected:
        return float(predicted == expected)
    predicted_counts = {token: predicted.count(token) for token in set(predicted)}
    expected_counts = {token: expected.count(token) for token in set(expected)}
    overlap = sum(
        min(predicted_counts.get(token, 0), expected_counts.get(token, 0))
        for token in predicted_counts.keys() | expected_counts.keys()
    )
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, reference: str) -> float:
    """Return exact-match after lowercase alphanumeric normalization."""
    predicted = " ".join(re.findall(r"[a-z0-9]+", prediction.lower()))
    expected = " ".join(re.findall(r"[a-z0-9]+", reference.lower()))
    return float(predicted == expected)


def classify_knowledge_source(
    *,
    closed_book_f1: float,
    gold_context_f1: float,
    retrieved_context_f1: float,
    threshold: float = 0.65,
) -> str:
    """Classify an answer outcome without overstating unobservable model reasoning."""
    closed_correct = closed_book_f1 >= threshold
    gold_correct = gold_context_f1 >= threshold
    retrieved_correct = retrieved_context_f1 >= threshold
    if not gold_correct:
        return "generation_or_model_failure"
    if not retrieved_correct:
        return "retrieval_failure"
    if not closed_correct:
        return "retrieval_enabled"
    return "source_not_identifiable"


def interaction_diagnostics(
    rows: list[dict[str, object]],
) -> dict[str, int | float | None]:
    """Summarize labelled interaction recovery with denominator visibility."""
    total = len(rows)
    evaluable = [
        row for row in rows if row.get("interaction_sign_correct") is not None
    ]
    correct = sum(bool(row["interaction_sign_correct"]) for row in evaluable)
    diagnostics: dict[str, int | float | None] = {
        "interaction_total_pairs": total,
        "interaction_evaluable_pairs": len(evaluable),
        "interaction_skipped_pairs": total - len(evaluable),
        "interaction_coverage": len(evaluable) / total if total else None,
        "interaction_correct_signs": correct,
        "interaction_sign_recovery": correct / len(evaluable) if evaluable else None,
        "interaction_sign_accuracy": correct / len(evaluable) if evaluable else None,
    }
    relations = sorted({str(row.get("relation")) for row in rows if row.get("relation")})
    for relation in relations:
        labelled = [row for row in rows if row.get("relation") == relation]
        relation_evaluable = [
            row for row in labelled if row.get("interaction_sign_correct") is not None
        ]
        relation_correct = sum(
            bool(row["interaction_sign_correct"]) for row in relation_evaluable
        )
        diagnostics[f"interaction_{relation}_total_pairs"] = len(labelled)
        diagnostics[f"interaction_{relation}_evaluable_pairs"] = len(relation_evaluable)
        diagnostics[f"interaction_{relation}_skipped_pairs"] = (
            len(labelled) - len(relation_evaluable)
        )
        diagnostics[f"interaction_{relation}_coverage"] = (
            len(relation_evaluable) / len(labelled) if labelled else None
        )
        diagnostics[f"interaction_{relation}_correct_signs"] = relation_correct
        diagnostics[f"interaction_{relation}_sign_recovery"] = (
            relation_correct / len(relation_evaluable) if relation_evaluable else None
        )
        diagnostics[f"interaction_{relation}_sign_accuracy"] = (
            relation_correct / len(relation_evaluable) if relation_evaluable else None
        )
    return diagnostics
