"""Aggregate RQ2 interaction coverage and sign-recovery metrics from runs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).parents[1] / "results"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "derived" / "interactions"
RELATIONS = ("complementary", "redundant", "conflicting", "overall")


@dataclass(frozen=True)
class InteractionSummaryRow:
    """One report-table row for interaction coverage and conditional recovery."""

    setting: str
    target: str
    relation_type: str
    total_labelled_pairs: int
    evaluable_pairs: int
    skipped_pairs: int
    coverage: float | None
    correct_signs: int
    sign_recovery: float | None
    run_dir: str


def _sum_column(frame: pd.DataFrame, column: str) -> int:
    if column not in frame:
        return 0
    return int(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def _aggregate_relation(
    frame: pd.DataFrame,
    *,
    setting: str,
    target: str,
    prefix: str,
    relation: str,
    run_dir: Path,
) -> InteractionSummaryRow | None:
    base = f"{prefix}interaction" if relation == "overall" else f"{prefix}interaction_{relation}"
    total = _sum_column(frame, f"{base}_total_pairs")
    if total == 0:
        return None
    evaluable = _sum_column(frame, f"{base}_evaluable_pairs")
    skipped = _sum_column(frame, f"{base}_skipped_pairs")
    correct = _sum_column(frame, f"{base}_correct_signs")
    return InteractionSummaryRow(
        setting=setting,
        target=target,
        relation_type=relation,
        total_labelled_pairs=total,
        evaluable_pairs=evaluable,
        skipped_pairs=skipped,
        coverage=evaluable / total if total else None,
        correct_signs=correct,
        sign_recovery=correct / evaluable if evaluable else None,
        run_dir=str(run_dir),
    )


def aggregate_interaction_runs(
    runs: dict[str, Path],
) -> list[InteractionSummaryRow]:
    """Build RQ2 rows from controlled run summary files."""
    rows: list[InteractionSummaryRow] = []
    for setting, run_dir in runs.items():
        summary_path = run_dir / "summary.csv"
        if not summary_path.exists():
            msg = f"Missing interaction input summary: {summary_path}"
            raise FileNotFoundError(msg)
        frame = pd.read_csv(summary_path)
        target_specs = [("Gold/reference answer", "")]
        if "generated_interaction_total_pairs" in frame.columns:
            target_specs.append(("Generated answer", "generated_"))
        for target, prefix in target_specs:
            for relation in RELATIONS:
                row = _aggregate_relation(
                    frame,
                    setting=setting,
                    target=target,
                    prefix=prefix,
                    relation=relation,
                    run_dir=run_dir,
                )
                if row is not None:
                    rows.append(row)
    return rows


def _parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        msg = "--run must have the form LABEL=PATH"
        raise argparse.ArgumentTypeError(msg)
    label, path = value.split("=", maxsplit=1)
    return label, Path(path)


def _write_markdown(rows: list[InteractionSummaryRow], output_path: Path) -> None:
    lines = [
        "# RQ2 Interaction Recovery",
        "",
        "Interaction coverage is the fraction of labelled passage pairs where both "
        "passages were retrieved in the top-k context. Sign recovery is computed "
        "only over those evaluable pairs.",
        "",
        "| Setting | Target | Type | Total labelled pairs | Evaluable pairs | "
        "Coverage | Correct signs | Sign recovery |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        coverage = "n/a" if row.coverage is None else f"{row.coverage:.3f}"
        recovery = "n/a" if row.sign_recovery is None else f"{row.sign_recovery:.3f}"
        lines.append(
            f"| {row.setting} | {row.target} | {row.relation_type} | "
            f"{row.total_labelled_pairs} | {row.evaluable_pairs} | {coverage} | "
            f"{row.correct_signs} | {recovery} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        type=_parse_run,
        default=[],
        help="Setting label and run directory, formatted as LABEL=PATH.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    runs = dict(args.run)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = aggregate_interaction_runs(runs)
    if not rows:
        msg = "No interaction rows were produced; provide at least one complete --run input."
        raise ValueError(msg)
    dict_rows = [asdict(row) for row in rows]
    pd.DataFrame(dict_rows).to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(dict_rows, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(rows, args.output_dir / "report.md")
    print(f"Wrote {len(rows)} interaction rows to {args.output_dir}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
