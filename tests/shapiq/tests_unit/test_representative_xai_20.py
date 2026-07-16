"""Model-free tests for the structured 20-case representative runner."""

from __future__ import annotations

import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

runner = importlib.import_module("run_representative_xai")
runner_20 = importlib.import_module("run_representative_xai_20")


def test_selection_has_twenty_unique_cases_and_five_per_category() -> None:
    cases = runner_20.SELECTED_CASES
    assert len(cases) == 20
    assert len({case.case_id for case in cases}) == 20
    assert Counter(case.category for case in cases) == {
        "weather": 5,
        "calculator": 5,
        "web_search": 5,
        "no_tool": 5,
    }


def test_selection_matches_the_existing_holdout() -> None:
    holdout = {
        row["id"]: row for row in json.loads(runner_20.HOLDOUT_PATH.read_text(encoding="utf-8"))
    }
    category_by_tool = runner_20.CATEGORY_BY_TOOL
    for case in runner_20.SELECTED_CASES:
        source = holdout[case.case_id]
        assert case.request == source["request"]
        assert case.expected_tool == source["ground_truth"]
        assert case.category == category_by_tool[case.expected_tool]


def test_selected_cases_are_passed_to_existing_runner(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        runner_20,
        "build_real_dependencies",
        lambda configuration: SimpleNamespace(configuration=configuration),
    )

    def fake_run_experiment(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"failed_case_count": 0}

    monkeypatch.setattr(runner_20, "run_experiment", fake_run_experiment)
    exit_code = runner_20.main(["--output-dir", str(tmp_path)])

    assert exit_code == 0
    assert captured["cases"] is runner_20.SELECTED_CASES
    assert captured["case_selection_description"] == runner_20.CASE_SELECTION_DESCRIPTION
    assert captured["output_dir"] == tmp_path


def test_case_filter_preserves_requested_order(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        runner_20,
        "build_real_dependencies",
        lambda configuration: SimpleNamespace(configuration=configuration),
    )

    def fake_run_experiment(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"failed_case_count": 0}

    monkeypatch.setattr(runner_20, "run_experiment", fake_run_experiment)
    exit_code = runner_20.main(
        [
            "--output-dir",
            str(tmp_path),
            "--case-ids",
            "w04",
            "c08",
            "s03",
            "n08",
            "s09",
        ]
    )

    assert exit_code == 0
    assert [case.case_id for case in captured["cases"]] == [
        "w04",
        "c08",
        "s03",
        "n08",
        "s09",
    ]
    assert "Filtered case IDs: w04, c08, s03, n08, s09" in captured["case_selection_description"]


def test_case_filter_rejects_unknown_and_duplicate_ids() -> None:
    with pytest.raises(SystemExit):
        runner_20.parse_args(["--case-ids", "missing"])
    with pytest.raises(SystemExit):
        runner_20.parse_args(["--case-ids", "w04", "w04"])


def test_default_output_does_not_overwrite_four_case_outputs() -> None:
    args = runner_20.parse_args([])
    assert args.output_dir == Path("outputs/representative_xai_20")
    assert runner.output_paths(args.output_dir) != runner.output_paths(Path("outputs"))
