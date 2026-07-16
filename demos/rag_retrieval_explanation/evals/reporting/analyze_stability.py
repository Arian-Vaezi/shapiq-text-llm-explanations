"""Analyze attribution stability across completed controlled evaluation runs."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from demos.rag_retrieval_explanation.evals.experiments.benchmark import (
    passage_id_from_title,
)

REPOSITORY_ROOT = Path(__file__).parents[4]
RESULTS_DIR = Path(__file__).parents[1] / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "derived" / "stability"
DEFAULT_RUNS = {
    "bge_lexical": RESULTS_DIR / "controlled" / "bge_lexical",
    "bge_target_ll": RESULTS_DIR / "controlled" / "bge_target_likelihood",
    "bge_contrastive_ll": RESULTS_DIR / "controlled" / "bge_contrastive_likelihood",
    "tfidf_contrastive_ll": RESULTS_DIR / "controlled" / "tfidf_contrastive_likelihood",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifacts(run_dir: Path) -> dict[str, dict[str, Any]]:
    artifacts_dir = run_dir / "artifacts"
    return {
        path.stem: _load_json(path)
        for path in sorted(artifacts_dir.glob("*.json"))
        if path.is_file()
    }


def _retrieved_ids(artifact: dict[str, Any]) -> list[str]:
    return [str(item["passage_id"]) for item in artifact["retrieved"]]


def _absolute_attribution_scores(
    artifact: dict[str, Any],
    *,
    target: str,
) -> dict[str, float]:
    key = "generated_attributions" if target == "generated" else "gold_attributions"
    return {
        passage_id_from_title(str(item["chunk"])): abs(float(item["attribution"]))
        for item in artifact.get(key, [])
    }


def _top_id(scores: dict[str, float]) -> str | None:
    if not scores:
        return None
    return max(scores, key=lambda passage_id: scores[passage_id])


def _spearman(
    left: dict[str, float],
    right: dict[str, float],
    *,
    ids: set[str],
) -> float | None:
    if len(ids) < 2:
        return None
    frame = pd.DataFrame(
        {
            "left": [left.get(passage_id, 0.0) for passage_id in sorted(ids)],
            "right": [right.get(passage_id, 0.0) for passage_id in sorted(ids)],
        }
    )
    if frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return None
    return float(frame["left"].rank(ascending=False).corr(frame["right"].rank(ascending=False)))


def _compare_pair(
    *,
    left_name: str,
    left_artifacts: dict[str, dict[str, Any]],
    right_name: str,
    right_artifacts: dict[str, dict[str, Any]],
    target: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    left_cases = set(left_artifacts)
    right_cases = set(right_artifacts)
    if left_cases != right_cases:
        msg = (
            f"Stability inputs have different case sets: "
            f"missing from {left_name}={sorted(right_cases - left_cases)}, "
            f"missing from {right_name}={sorted(left_cases - right_cases)}"
        )
        raise ValueError(msg)
    common_cases = sorted(left_cases)
    for case_id in common_cases:
        left = left_artifacts[case_id]
        right = right_artifacts[case_id]
        left_scores = _absolute_attribution_scores(left, target=target)
        right_scores = _absolute_attribution_scores(right, target=target)
        left_ids = set(_retrieved_ids(left))
        right_ids = set(_retrieved_ids(right))
        common_ids = set(left_scores) & set(right_scores)
        union_ids = set(left_scores) | set(right_scores)
        left_top = _top_id(left_scores)
        right_top = _top_id(right_scores)
        rows.append(
            {
                "case_id": case_id,
                "target": target,
                "left_run": left_name,
                "right_run": right_name,
                "retrieved_jaccard": (
                    len(left_ids & right_ids) / len(left_ids | right_ids)
                    if left_ids or right_ids
                    else None
                ),
                "common_retrieved_count": len(left_ids & right_ids),
                "top_attribution_agrees": (
                    left_top == right_top
                    if left_top is not None and right_top is not None
                    else None
                ),
                "spearman_common": _spearman(left_scores, right_scores, ids=common_ids),
                "spearman_union_zero_filled": _spearman(left_scores, right_scores, ids=union_ids),
            }
        )
    return rows


def _mean(frame: pd.DataFrame, column: str) -> str:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return f"{values.mean():.3f}" if not values.empty else "n/a"


def _mean_boolean(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].dropna()
    return f"{values.astype(bool).mean():.3f}" if not values.empty else "n/a"


def _write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Controlled Attribution Stability",
        "",
        "Stability is computed from completed controlled-run artifacts without",
        "re-running retrieval, generation, or likelihood scoring. Correlations rank",
        "chunks by absolute first-order Shapley attribution.",
        "",
    ]
    for _, group in summary.groupby(["left_run", "right_run", "target"]):
        left = str(group["left_run"].iloc[0])
        right = str(group["right_run"].iloc[0])
        target = str(group["target"].iloc[0])
        lines.extend(
            [
                f"## {left} vs {right} ({target})",
                "",
                f"- Cases: {len(group)}",
                f"- Mean retrieved-set Jaccard: {_mean(group, 'retrieved_jaccard')}",
                f"- Top-attribution agreement: {_mean_boolean(group, 'top_attribution_agrees')}",
                f"- Mean Spearman on common retrieved chunks: {_mean(group, 'spearman_common')}",
                f"- Mean Spearman on union with missing chunks as zero: "
                f"{_mean(group, 'spearman_union_zero_filled')}",
                "",
            ]
        )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_stability(output_dir: Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    """Write stability metrics for the recorded controlled experiment matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = {name: _artifacts(path) for name, path in DEFAULT_RUNS.items()}
    rows: list[dict[str, object]] = []
    comparisons = [
        ("bge_lexical", "bge_target_ll", "gold"),
        ("bge_lexical", "bge_contrastive_ll", "gold"),
        ("bge_target_ll", "bge_contrastive_ll", "gold"),
        ("bge_target_ll", "bge_contrastive_ll", "generated"),
        ("bge_contrastive_ll", "tfidf_contrastive_ll", "gold"),
        ("bge_contrastive_ll", "tfidf_contrastive_ll", "generated"),
    ]
    for left_name, right_name, target in comparisons:
        rows.extend(
            _compare_pair(
                left_name=left_name,
                left_artifacts=loaded[left_name],
                right_name=right_name,
                right_artifacts=loaded[right_name],
                target=target,
            )
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source_runs": {
            name: str(path.relative_to(REPOSITORY_ROOT)) for name, path in DEFAULT_RUNS.items()
        },
        "metric": "Spearman rank correlation over absolute first-order Shapley values",
        "note": "No model calls are made; this is an artifact-only analysis.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    _write_report(summary, output_dir)
    return summary


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    summary = run_stability(args.output_dir)
    print(f"Wrote {len(summary)} stability rows to {args.output_dir}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
