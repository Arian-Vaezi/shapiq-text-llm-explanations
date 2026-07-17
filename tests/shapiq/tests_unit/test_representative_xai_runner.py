"""Model-free tests for the representative Agent + XAI batch runner."""

from __future__ import annotations

import csv
import datetime
import importlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

runner = importlib.import_module("run_representative_xai")


def _fixed_now() -> datetime.datetime:
    return datetime.datetime(2026, 7, 14, 12, 0, tzinfo=datetime.UTC)


def _monotonic() -> Callable[[], float]:
    values = iter(float(index) for index in range(20))
    return lambda: next(values)


def _fake_xai(request: str, target_tool: str, agent_result: object) -> dict[str, object]:
    del request, target_tool, agent_result
    return {
        "linguistic_players": [
            {"player_index": 0, "text": "first span", "label": "U1", "type": "NOUN_CHUNK"},
            {"player_index": 1, "text": "second span", "label": "U2", "type": "DATE_TIME"},
            {"player_index": 2, "text": "third span", "label": "U3", "type": "STRAY_MERGE"},
        ],
        "first_order_attributions": [
            {"player_index": 0, "label": "U1", "text": "first span", "value": 0.2},
            {"player_index": 1, "label": "U2", "text": "second span", "value": -0.7},
            {"player_index": 2, "label": "U3", "text": "third span", "value": 0.9},
        ],
        "pairwise_interactions": [
            {"player_indices": [0, 1], "pair": ["first span", "second span"], "value": -0.1},
            {"player_indices": [0, 2], "pair": ["first span", "third span"], "value": 0.8},
            {"player_indices": [1, 2], "pair": ["second span", "third span"], "value": 0.05},
        ],
        "empty_coalition_value": -1.25,
        "full_coalition_value": -0.1,
        "sv_algorithm": "fake SV",
        "ksii_algorithm": "fake k-SII",
    }


def _configuration() -> object:
    return runner.BatchConfiguration()


def _metadata() -> dict[str, str]:
    return {"git_branch": "test-branch", "git_commit": "deadbeef"}


def test_fixed_cases_and_cli_defaults_match_the_3b_holdout() -> None:
    assert [case.case_id for case in runner.REPRESENTATIVE_CASES] == ["w01", "c05", "s03", "n07"]
    assert [case.request for case in runner.REPRESENTATIVE_CASES] == [
        "What will the temperature be in Oslo on Friday evening?",
        "A rectangle is 13.2 cm by 8.5 cm; compute its area.",
        "What major news was announced by the European Central Bank today?",
        (
            "I'm studying percentages, but I only need a conceptual explanation "
            "of why multiplying by 0.5 is the same as taking half."
        ),
    ]
    assert [case.expected_tool for case in runner.REPRESENTATIVE_CASES] == [
        "weather_tool",
        "calculator_tool",
        "web_search_tool",
        "no_tool",
    ]
    args = runner.parse_args([])
    assert vars(args) == {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "device": "cuda",
        "dtype": "auto",
        "quantization": "none",
        "max_new_tokens": 512,
        "max_pairs_per_batch": 1,
        "output_dir": Path("outputs"),
    }


def test_actual_tool_is_target_and_all_xai_values_are_retained(tmp_path: Path) -> None:
    requests: list[str] = []
    targets: list[str] = []
    persisted_lengths: list[int] = []
    actual_tools = iter(["no_tool", "calculator_tool", "web_search_tool", "no_tool"])

    def run_agent(request: str) -> object:
        requests.append(request)
        return SimpleNamespace(
            selected_tool=next(actual_tools),
            raw_response=f"raw response for {request}",
            tool_arguments={},
        )

    def run_xai(request: str, target_tool: str, agent_result: object) -> dict[str, object]:
        targets.append(target_tool)
        return _fake_xai(request, target_tool, agent_result)

    experiment = runner.run_experiment(
        dependencies=runner.RunnerDependencies(run_agent=run_agent, run_xai=run_xai),
        configuration=_configuration(),
        output_dir=tmp_path,
        metadata=_metadata(),
        now=_fixed_now,
        monotonic=_monotonic(),
        after_persist=lambda payload: persisted_lengths.append(len(payload["runs"])),
    )

    assert requests == [case.request for case in runner.REPRESENTATIVE_CASES]
    assert targets == ["no_tool", "calculator_tool", "web_search_tool", "no_tool"]
    assert persisted_lengths == [1, 2, 3, 4]
    first = experiment["runs"][0]
    assert first["expected_tool"] == "weather_tool"
    assert first["actual_selected_tool"] == "no_tool"
    assert first["xai_target_tool"] == "no_tool"
    assert first["selection_correct"] is False
    assert [row["value"] for row in first["first_order_attributions"]] == [0.2, -0.7, 0.9]
    assert first["most_supportive_span"] == "third span"
    assert first["supportive_value"] == 0.9
    assert first["most_opposing_span"] == "second span"
    assert first["opposing_value"] == -0.7
    assert [row["value"] for row in first["pairwise_interactions"]] == [-0.1, 0.8, 0.05]
    assert first["strongest_positive_interaction"] == {
        "pair_text": "first span + third span",
        "value": 0.8,
    }
    assert first["strongest_negative_interaction"] == {
        "pair_text": "first span + second span",
        "value": -0.1,
    }


def test_json_and_csv_contain_required_fields(tmp_path: Path) -> None:
    dependencies = runner.RunnerDependencies(
        run_agent=lambda request: SimpleNamespace(
            selected_tool="weather_tool",
            raw_response=request,
            tool_arguments={"location": "Oslo"},
        ),
        run_xai=_fake_xai,
    )
    runner.run_experiment(
        dependencies=dependencies,
        configuration=_configuration(),
        output_dir=tmp_path,
        cases=runner.REPRESENTATIVE_CASES[:1],
        metadata=_metadata(),
        now=_fixed_now,
        monotonic=_monotonic(),
    )
    paths = runner.output_paths(tmp_path)
    payload = json.loads(paths.json_path.read_text(encoding="utf-8"))
    required_metadata = {
        "experiment_name",
        "created_at",
        "git_branch",
        "git_commit",
        "model_name",
        "device",
        "configuration",
        "source_set",
        "case_selection_description",
        "completed_case_count",
        "failed_case_count",
        "runs",
    }
    required_run_fields = {
        "case_id",
        "category",
        "request",
        "expected_tool",
        "actual_selected_tool",
        "selection_correct",
        "agent_result",
        "router_raw_scores",
        "router_calibrated_scores",
        "linguistic_players",
        "first_order_attributions",
        "most_supportive_span",
        "supportive_value",
        "most_opposing_span",
        "opposing_value",
        "pairwise_interactions",
        "strongest_positive_interaction",
        "strongest_negative_interaction",
        "empty_coalition_value",
        "full_coalition_value",
        "runtime_seconds",
        "status",
        "error_type",
        "error_message",
    }
    assert required_metadata <= payload.keys()
    assert required_run_fields <= payload["runs"][0].keys()
    assert payload["runs"][0]["agent_result"]["tool_arguments"] == {"location": "Oslo"}
    with paths.csv_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert tuple(rows[0]) == runner.SUMMARY_FIELDS
    assert rows[0]["number_of_players"] == "3"
    assert paths.log_path.read_text(encoding="utf-8").count("case=w01") == 1


def test_failure_is_persisted_and_later_cases_continue(tmp_path: Path) -> None:
    agent_calls: list[str] = []
    snapshots: list[dict[str, object]] = []

    def run_agent(request: str) -> object:
        agent_calls.append(request)
        if "European Central Bank" in request:
            msg = "synthetic third-case failure"
            raise RuntimeError(msg)
        return SimpleNamespace(selected_tool="no_tool", raw_response=request, tool_arguments={})

    experiment = runner.run_experiment(
        dependencies=runner.RunnerDependencies(run_agent=run_agent, run_xai=_fake_xai),
        configuration=_configuration(),
        output_dir=tmp_path,
        metadata=_metadata(),
        now=_fixed_now,
        monotonic=_monotonic(),
        after_persist=lambda payload: snapshots.append(deepcopy(payload)),
    )

    assert len(agent_calls) == 4
    assert [len(snapshot["runs"]) for snapshot in snapshots] == [1, 2, 3, 4]
    third_snapshot = snapshots[2]
    assert [run["status"] for run in third_snapshot["runs"]] == [
        "completed",
        "completed",
        "failed",
    ]
    assert third_snapshot["runs"][2]["error_type"] == "RuntimeError"
    assert third_snapshot["runs"][2]["error_message"] == "synthetic third-case failure"
    assert experiment["runs"][3]["status"] == "completed"
    assert experiment["completed_case_count"] == 3
    assert experiment["failed_case_count"] == 1
    persisted = json.loads(runner.output_paths(tmp_path).json_path.read_text(encoding="utf-8"))
    assert [run["case_id"] for run in persisted["runs"]] == ["w01", "c05", "s03", "n07"]


def test_fake_outputs_are_deterministic(tmp_path: Path) -> None:
    dependencies = runner.RunnerDependencies(
        run_agent=lambda request: SimpleNamespace(
            selected_tool="weather_tool",
            raw_response=request,
            tool_arguments={},
        ),
        run_xai=_fake_xai,
    )
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    for output_dir in (first_dir, second_dir):
        runner.run_experiment(
            dependencies=dependencies,
            configuration=_configuration(),
            output_dir=output_dir,
            cases=runner.REPRESENTATIVE_CASES[:2],
            metadata=_metadata(),
            now=_fixed_now,
            monotonic=_monotonic(),
        )

    first_paths = runner.output_paths(first_dir)
    second_paths = runner.output_paths(second_dir)
    assert first_paths.json_path.read_bytes() == second_paths.json_path.read_bytes()
    assert first_paths.csv_path.read_bytes() == second_paths.csv_path.read_bytes()
    assert first_paths.log_path.read_bytes() == second_paths.log_path.read_bytes()
