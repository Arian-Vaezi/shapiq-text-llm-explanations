"""Verify that paper tables are traceable to saved evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

RUNS_DIR = Path(__file__).parent / "runs"
DEFAULT_OUTPUT = RUNS_DIR / "report_verification.json"
TOLERANCE = 0.0005

CONTROLLED_RUNS = {
    "BGE-base + lexical": RUNS_DIR / "20260616_controlled_interactions_bge_lexical",
    "BGE-base + target LL": (
        RUNS_DIR / "20260616_controlled_bge_target_ll_e4b_layers20"
    ),
    "BGE-base + contrastive LL": (
        RUNS_DIR / "20260616_controlled_bge_contrastive_ll_e4b_layers20"
    ),
    "TF--IDF + contrastive LL": (
        RUNS_DIR / "20260616_controlled_tfidf_contrastive_ll_e4b_layers20"
    ),
}
QASPER_RUN = RUNS_DIR / "20260615_qasper_30_bge_e4b_contrastive"
CONTROLLED_STABILITY_SUMMARY = RUNS_DIR / "20260616_controlled_stability"
CONTROLLED_INTERACTION_SUMMARY = RUNS_DIR / "20260616_controlled_interaction_summary"


@dataclass(frozen=True)
class ReportMetric:
    """One reported value expected in the paper."""

    table: str
    row: str
    column: str
    value: float


EXPECTED = [
    ReportMetric("Table 1", "BGE-base + lexical / Gold", "Cases", 27),
    ReportMetric("Table 1", "BGE-base + lexical / Gold", "Gold hit", 23),
    ReportMetric("Table 1", "BGE-base + lexical / Gold", "Recall@6", 0.957),
    ReportMetric("Table 1", "BGE-base + lexical / Gold", "MRR", 0.967),
    ReportMetric("Table 1", "BGE-base + lexical / Gold", "Hit top gold", 0.957),
    ReportMetric("Table 1", "BGE-base + lexical / Gold", "Hit gold mass", 0.697),
    ReportMetric("Table 1", "BGE-base + lexical / Gold", "Deletion AUC", 0.255),
    ReportMetric("Table 1", "BGE-base + target LL / Gold", "Cases", 27),
    ReportMetric("Table 1", "BGE-base + target LL / Gold", "Gold hit", 23),
    ReportMetric("Table 1", "BGE-base + target LL / Gold", "Recall@6", 0.957),
    ReportMetric("Table 1", "BGE-base + target LL / Gold", "MRR", 0.967),
    ReportMetric("Table 1", "BGE-base + target LL / Gold", "Hit top gold", 0.957),
    ReportMetric("Table 1", "BGE-base + target LL / Gold", "Hit gold mass", 0.679),
    ReportMetric("Table 1", "BGE-base + target LL / Gold", "Deletion AUC", 0.038),
    ReportMetric("Table 1", "BGE-base + target LL / Generated", "Hit top gold", 0.957),
    ReportMetric("Table 1", "BGE-base + target LL / Generated", "Hit gold mass", 0.812),
    ReportMetric("Table 1", "BGE-base + contrastive LL / Gold", "Cases", 27),
    ReportMetric("Table 1", "BGE-base + contrastive LL / Gold", "Gold hit", 23),
    ReportMetric("Table 1", "BGE-base + contrastive LL / Gold", "Recall@6", 0.957),
    ReportMetric("Table 1", "BGE-base + contrastive LL / Gold", "MRR", 0.967),
    ReportMetric(
        "Table 1", "BGE-base + contrastive LL / Gold", "Hit top gold", 0.957
    ),
    ReportMetric(
        "Table 1", "BGE-base + contrastive LL / Gold", "Hit gold mass", 0.666
    ),
    ReportMetric("Table 1", "BGE-base + contrastive LL / Gold", "Deletion AUC", 0.046),
    ReportMetric(
        "Table 1", "BGE-base + contrastive LL / Generated", "Hit top gold", 0.957
    ),
    ReportMetric(
        "Table 1", "BGE-base + contrastive LL / Generated", "Hit gold mass", 0.800
    ),
    ReportMetric("Table 1", "TF--IDF + contrastive LL / Gold", "Cases", 27),
    ReportMetric("Table 1", "TF--IDF + contrastive LL / Gold", "Gold hit", 22),
    ReportMetric("Table 1", "TF--IDF + contrastive LL / Gold", "Recall@6", 0.891),
    ReportMetric("Table 1", "TF--IDF + contrastive LL / Gold", "MRR", 0.935),
    ReportMetric("Table 1", "TF--IDF + contrastive LL / Gold", "Hit top gold", 1.000),
    ReportMetric("Table 1", "TF--IDF + contrastive LL / Gold", "Hit gold mass", 0.673),
    ReportMetric("Table 1", "TF--IDF + contrastive LL / Gold", "Deletion AUC", 0.041),
    ReportMetric(
        "Table 1", "TF--IDF + contrastive LL / Generated", "Hit top gold", 0.955
    ),
    ReportMetric(
        "Table 1", "TF--IDF + contrastive LL / Generated", "Hit gold mass", 0.780
    ),
    ReportMetric("Table 2", "Contrastive LL / gold", "Cases", 30),
    ReportMetric("Table 2", "Contrastive LL / gold", "Gold hit", 21),
    ReportMetric("Table 2", "Contrastive LL / gold", "Recall@6", 0.550),
    ReportMetric("Table 2", "Contrastive LL / gold", "MRR", 0.451),
    ReportMetric("Table 2", "Contrastive LL / gold", "Hit top gold", 0.857),
    ReportMetric("Table 2", "Contrastive LL / gold", "Hit gold mass", 0.592),
    ReportMetric("Table 2", "Contrastive LL / gold", "Deletion AUC", 0.033),
    ReportMetric("Table 2", "Contrastive LL / generated", "Cases", 30),
    ReportMetric("Table 2", "Contrastive LL / generated", "Gold hit", 21),
    ReportMetric("Table 2", "Contrastive LL / generated", "Recall@6", 0.550),
    ReportMetric("Table 2", "Contrastive LL / generated", "MRR", 0.451),
    ReportMetric("Table 2", "Contrastive LL / generated", "Hit top gold", 0.667),
    ReportMetric("Table 2", "Contrastive LL / generated", "Hit gold mass", 0.529),
    ReportMetric("Table 3", "BGE lexical vs BGE target LL / Gold", "Retrieved Jaccard", 1.000),
    ReportMetric("Table 3", "BGE lexical vs BGE target LL / Gold", "Top agreement", 0.704),
    ReportMetric("Table 3", "BGE lexical vs BGE target LL / Gold", "Spearman", 0.523),
    ReportMetric(
        "Table 3",
        "BGE target LL vs BGE contrastive LL / Gold",
        "Retrieved Jaccard",
        1.000,
    ),
    ReportMetric(
        "Table 3", "BGE target LL vs BGE contrastive LL / Gold", "Top agreement", 1.000
    ),
    ReportMetric("Table 3", "BGE target LL vs BGE contrastive LL / Gold", "Spearman", 0.979),
    ReportMetric(
        "Table 3",
        "BGE target LL vs BGE contrastive LL / Generated",
        "Retrieved Jaccard",
        1.000,
    ),
    ReportMetric(
        "Table 3",
        "BGE target LL vs BGE contrastive LL / Generated",
        "Top agreement",
        1.000,
    ),
    ReportMetric(
        "Table 3", "BGE target LL vs BGE contrastive LL / Generated", "Spearman", 0.992
    ),
    ReportMetric(
        "Table 3", "BGE lexical vs BGE contrastive LL / Gold", "Retrieved Jaccard", 1.000
    ),
    ReportMetric("Table 3", "BGE lexical vs BGE contrastive LL / Gold", "Top agreement", 0.704),
    ReportMetric("Table 3", "BGE lexical vs BGE contrastive LL / Gold", "Spearman", 0.514),
    ReportMetric(
        "Table 3",
        "BGE contrastive LL vs TF--IDF contrastive LL / Gold",
        "Retrieved Jaccard",
        0.592,
    ),
    ReportMetric(
        "Table 3",
        "BGE contrastive LL vs TF--IDF contrastive LL / Gold",
        "Top agreement",
        0.741,
    ),
    ReportMetric(
        "Table 3", "BGE contrastive LL vs TF--IDF contrastive LL / Gold", "Spearman", 0.428
    ),
    ReportMetric(
        "Table 3",
        "BGE contrastive LL vs TF--IDF contrastive LL / Generated",
        "Retrieved Jaccard",
        0.592,
    ),
    ReportMetric(
        "Table 3",
        "BGE contrastive LL vs TF--IDF contrastive LL / Generated",
        "Top agreement",
        0.778,
    ),
    ReportMetric(
        "Table 3",
        "BGE contrastive LL vs TF--IDF contrastive LL / Generated",
        "Spearman",
        0.467,
    ),
    ReportMetric("Table 4", "BGE-base + lexical / Gold / Overall", "Correct", 24),
    ReportMetric("Table 4", "BGE-base + lexical / Gold / Overall", "Recovery", 0.857),
    ReportMetric("Table 4", "BGE-base + contrastive LL / Gold / Overall", "Correct", 16),
    ReportMetric("Table 4", "BGE-base + contrastive LL / Gold / Overall", "Recovery", 0.571),
    ReportMetric(
        "Table 4", "BGE-base + contrastive LL / Generated / Overall", "Correct", 9
    ),
    ReportMetric(
        "Table 4", "BGE-base + contrastive LL / Generated / Overall", "Recovery", 0.321
    ),
]


def _mean(frame: pd.DataFrame, column: str) -> float:
    return float(pd.to_numeric(frame[column], errors="coerce").dropna().mean())


def _mean_bool(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].dropna().astype(bool).mean())


def _round3(value: float) -> float:
    return round(value + 0.0, 3)


def _controlled_metrics() -> dict[tuple[str, str, str], float]:
    metrics = {}
    for label, run_dir in CONTROLLED_RUNS.items():
        frame = pd.read_csv(run_dir / "summary.csv")
        gold_hit = frame[pd.to_numeric(frame["retrieval_recall_at_k"], errors="coerce") > 0]
        base_values = {
            "Cases": float(len(frame)),
            "Gold hit": float(len(gold_hit)),
            "Recall@6": _round3(_mean(frame, "retrieval_recall_at_k")),
            "MRR": _round3(_mean(frame, "retrieval_mrr")),
        }
        row = f"{label} / Gold"
        metrics.update(
            {("Table 1", row, column): value for column, value in base_values.items()}
        )
        metrics[("Table 1", row, "Hit top gold")] = _round3(
            _mean_bool(gold_hit, "top_attribution_is_gold")
        )
        metrics[("Table 1", row, "Hit gold mass")] = _round3(
            _mean(gold_hit, "gold_attribution_mass")
        )
        metrics[("Table 1", row, "Deletion AUC")] = _round3(
            _mean(frame, "gold_deletion_auc")
        )
        if "generated_top_attribution_is_gold" in frame:
            generated_row = f"{label} / Generated"
            metrics.update(
                {
                    ("Table 1", generated_row, column): value
                    for column, value in base_values.items()
                }
            )
            metrics[("Table 1", generated_row, "Hit top gold")] = _round3(
                _mean_bool(gold_hit, "generated_top_attribution_is_gold")
            )
            metrics[("Table 1", generated_row, "Hit gold mass")] = _round3(
                _mean(gold_hit, "generated_gold_attribution_mass")
            )
    return metrics


def _qasper_metrics() -> dict[tuple[str, str, str], float]:
    frame = pd.read_csv(QASPER_RUN / "summary.csv")
    gold_hit = frame[pd.to_numeric(frame["retrieval_recall_at_k"], errors="coerce") > 0]
    return {
        ("Table 2", "Contrastive LL / gold", "Cases"): float(len(frame)),
        ("Table 2", "Contrastive LL / gold", "Gold hit"): float(len(gold_hit)),
        ("Table 2", "Contrastive LL / gold", "Recall@6"): _round3(
            _mean(frame, "retrieval_recall_at_k")
        ),
        ("Table 2", "Contrastive LL / gold", "MRR"): _round3(
            _mean(frame, "retrieval_mrr")
        ),
        ("Table 2", "Contrastive LL / gold", "Hit top gold"): _round3(
            _mean_bool(gold_hit, "top_attribution_is_gold")
        ),
        ("Table 2", "Contrastive LL / gold", "Hit gold mass"): _round3(
            _mean(gold_hit, "gold_attribution_mass")
        ),
        ("Table 2", "Contrastive LL / gold", "Deletion AUC"): _round3(
            _mean(frame, "gold_deletion_auc")
        ),
        ("Table 2", "Contrastive LL / generated", "Cases"): float(len(frame)),
        ("Table 2", "Contrastive LL / generated", "Gold hit"): float(len(gold_hit)),
        ("Table 2", "Contrastive LL / generated", "Recall@6"): _round3(
            _mean(frame, "retrieval_recall_at_k")
        ),
        ("Table 2", "Contrastive LL / generated", "MRR"): _round3(
            _mean(frame, "retrieval_mrr")
        ),
        ("Table 2", "Contrastive LL / generated", "Hit top gold"): _round3(
            _mean_bool(gold_hit, "generated_top_attribution_is_gold")
        ),
        ("Table 2", "Contrastive LL / generated", "Hit gold mass"): _round3(
            _mean(gold_hit, "generated_gold_attribution_mass")
        ),
    }


def _stability_metrics() -> dict[tuple[str, str, str], float]:
    frame = pd.read_csv(CONTROLLED_STABILITY_SUMMARY / "summary.csv")
    name_lookup = {
        "bge_lexical": "BGE lexical",
        "bge_target_ll": "BGE target LL",
        "bge_contrastive_ll": "BGE contrastive LL",
        "tfidf_contrastive_ll": "TF--IDF contrastive LL",
    }
    metrics = {}
    for (left, right, target), group in frame.groupby(["left_run", "right_run", "target"]):
        target_label = "Generated" if target == "generated" else "Gold"
        row = f"{name_lookup[left]} vs {name_lookup[right]} / {target_label}"
        metrics[("Table 3", row, "Retrieved Jaccard")] = _round3(
            _mean(group, "retrieved_jaccard")
        )
        metrics[("Table 3", row, "Top agreement")] = _round3(
            _mean_bool(group, "top_attribution_agrees")
        )
        metrics[("Table 3", row, "Spearman")] = _round3(
            _mean(group, "spearman_union_zero_filled")
        )
    return metrics


def _interaction_metrics() -> dict[tuple[str, str, str], float]:
    frame = pd.read_csv(CONTROLLED_INTERACTION_SUMMARY / "summary.csv")
    relation_lookup = {
        "complementary": "Complementary",
        "redundant": "Redundant",
        "conflicting": "Conflicting",
        "overall": "Overall",
    }
    metrics = {}
    for _, row in frame.iterrows():
        target = "Gold" if row["target"] == "Gold/reference answer" else "Generated"
        relation = relation_lookup[str(row["relation_type"])]
        label = f"{row['setting']} / {target} / {relation}"
        metrics[("Table 4", label, "Labelled")] = float(row["total_labelled_pairs"])
        metrics[("Table 4", label, "Evaluable")] = float(row["evaluable_pairs"])
        metrics[("Table 4", label, "Coverage")] = _round3(float(row["coverage"]))
        metrics[("Table 4", label, "Correct")] = float(row["correct_signs"])
        metrics[("Table 4", label, "Recovery")] = _round3(float(row["sign_recovery"]))
    return metrics


def _curve_warnings() -> list[str]:
    warnings = []
    for run_dir in [*CONTROLLED_RUNS.values(), QASPER_RUN]:
        artifact_files = sorted((run_dir / "artifacts").glob("*.json"))
        if not artifact_files:
            warnings.append(f"No artifacts found in {run_dir}")
            continue
        missing = 0
        for path in artifact_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "faithfulness_curves" not in payload:
                missing += 1
        if missing:
            warnings.append(
                f"{missing}/{len(artifact_files)} artifacts in {run_dir.name} lack raw "
                "faithfulness_curves; rerun with the current runner to populate them."
            )
    return warnings


def verify_report_results() -> dict[str, Any]:
    """Recompute paper table values and compare them with reported numbers."""
    computed = {}
    computed.update(_controlled_metrics())
    computed.update(_qasper_metrics())
    computed.update(_stability_metrics())
    computed.update(_interaction_metrics())
    checks = []
    for expected in EXPECTED:
        key = (expected.table, expected.row, expected.column)
        actual = computed.get(key)
        checks.append(
            {
                "table": expected.table,
                "row": expected.row,
                "column": expected.column,
                "reported": expected.value,
                "computed": actual,
                "status": (
                    "missing"
                    if actual is None
                    else "ok"
                    if abs(actual - expected.value) <= TOLERANCE
                    else "mismatch"
                ),
            }
        )
    return {
        "status": (
            "ok"
            if all(check["status"] == "ok" for check in checks)
            else "needs_attention"
        ),
        "checks": checks,
        "warnings": _curve_warnings(),
    }


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = verify_report_results()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    mismatches = [check for check in result["checks"] if check["status"] != "ok"]
    print(f"Wrote report verification to {args.output}")  # noqa: T201
    for warning in result["warnings"]:
        print(f"WARNING: {warning}")  # noqa: T201
    if mismatches:
        print(json.dumps(mismatches, indent=2))  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
