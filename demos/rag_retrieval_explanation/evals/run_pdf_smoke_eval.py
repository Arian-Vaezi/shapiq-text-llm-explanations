"""Run generic PDF retrieval smoke checks from a JSON case file.

Usage:
    uv run python demos/rag_retrieval_explanation/evals/run_pdf_smoke_eval.py \
        demos/rag_retrieval_explanation/evals/pdf_cases.example.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _case_args(case: dict[str, Any]) -> list[str]:
    """Build regression_check.py arguments for one JSON case."""
    demo_dir = Path(__file__).resolve().parents[1]
    args = [
        sys.executable,
        str(demo_dir / "regression_check.py"),
        str(case["pdf_path"]),
        "--question",
        str(case["question"]),
        "--top-k",
        str(case.get("top_k", 4)),
        "--method",
        str(case.get("method", "TF-IDF")),
    ]
    for term in case.get("expected_evidence_terms", []):
        args.extend(["--expected-evidence-term", str(term)])
    for term in case.get("expected_answer_terms", []):
        args.extend(["--expected-answer-term", str(term)])
    for term in case.get("forbidden_answer_terms", []):
        args.extend(["--forbidden-answer-term", str(term)])
    if case.get("answer_file"):
        args.extend(["--answer-file", str(case["answer_file"])])
    return args


def main() -> int:
    """Run all configured PDF smoke cases."""
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    args = parser.parse_args()

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    failures = []
    for case in cases:
        case_id = str(case.get("id", case.get("question", "case")))
        print(f"\n=== {case_id} ===")  # noqa: T201
        result = subprocess.run(_case_args(case), check=False)  # noqa: S603
        if result.returncode != 0:
            failures.append(case_id)

    if failures:
        print("\nFailed PDF smoke cases: " + ", ".join(failures))  # noqa: T201
        return 1
    print("\nAll PDF smoke cases passed.")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
