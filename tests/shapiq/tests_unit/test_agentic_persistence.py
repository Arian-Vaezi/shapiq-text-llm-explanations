"""Tests for write-only JSON exports in the agentic tool-use demo."""

from __future__ import annotations

import datetime
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

import persistence  # noqa: E402
from hf_router import select_tool_from_scores  # noqa: E402
from scorers import CALIBRATION_USER_REQUESTS  # noqa: E402


def test_config_snapshot_uses_live_routing_defaults(tmp_path: Path) -> None:
    now = datetime.datetime(2026, 7, 12, 10, 11, 12, tzinfo=datetime.UTC)

    path = persistence.write_config_snapshot(
        hf_model_id="mock/model",
        device="cpu",
        export_dir=tmp_path,
        now=now,
    )
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    signature = inspect.signature(select_tool_from_scores)
    assert payload["routing"]["no_tool_boost_delta"] == float(
        signature.parameters["no_tool_boost_delta"].default
    )
    assert payload["routing"]["selection_mode"] == signature.parameters["mode"].default
    assert payload["calibration"]["calibration_user_requests"] == list(CALIBRATION_USER_REQUESTS)
    assert payload["model"] == {"hf_model_id": "mock/model", "device": "cpu"}
    assert path.name == "config_20260712_101112.json"
    assert list(tmp_path.glob("session_*.json")) == []


def _mock_result_kwargs(*, native_mode: bool = False) -> dict[str, object]:
    segments = [
        SimpleNamespace(label="U1", text="weather in Berlin"),
        SimpleNamespace(label="U2", text="tomorrow morning"),
    ]
    pairs = [
        {
            "pair": ["U1", "U2"],
            "text": [segments[0].text, segments[1].text],
            "k_sii": np.float64(0.25),
        }
    ]
    scores = {
        "weather_tool": np.float32(2.0),
        "calculator_tool": np.float64(-1.0),
        "web_search_tool": np.float32(0.5),
        "no_tool": np.float64(-0.25),
    }
    exported_scores = {} if native_mode else scores
    return {
        "hf_model_id": "mock/model",
        "user_request": "Will it rain in Berlin tomorrow morning?",
        "system_prompt": "Use the appropriate tool.",
        "player_segments": segments,
        "raw_scores": exported_scores,
        "calibrated_scores": (
            {} if native_mode else {name: np.float64(value) for name, value in scores.items()}
        ),
        "selected_tool": "weather_tool",
        "raw_argmax": None if native_mode else "weather_tool",
        "calibrated_argmax": None if native_mode else "weather_tool",
        "target_tool": "weather_tool",
        "baseline_h_empty": np.float32(-0.5),
        "full_h_n": np.float64(1.5),
        "pairwise_interactions": pairs,
    }


def test_session_export_accumulates_calibrated_and_native_runs(tmp_path: Path) -> None:
    session_started_at = datetime.datetime(2026, 7, 12, 10, 0, tzinfo=datetime.UTC)
    recorded_times = [
        datetime.datetime(2026, 7, 12, 10, 12, 13, 123000, tzinfo=datetime.UTC),
        datetime.datetime(2026, 7, 12, 10, 13, 14, 456000, tzinfo=datetime.UTC),
        datetime.datetime(2026, 7, 12, 10, 14, 15, 789000, tzinfo=datetime.UTC),
    ]

    paths = []
    for index, recorded_at in enumerate(recorded_times):
        paths.append(
            persistence.write_result_export(
                session_id="12345678-full-session-id",
                session_started_at=session_started_at,
                export_dir=tmp_path,
                now=recorded_at,
                **_mock_result_kwargs(native_mode=index == 1),
            )
        )
    path = paths[-1]
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    assert len(set(paths)) == 1
    assert path.name == "session_20260712_100000_12345678.json"
    assert payload["session_id"] == "12345678-full-session-id"
    assert payload["session_started_at"] == "2026-07-12T10:00:00.000+00:00"
    assert [run["run_index"] for run in payload["runs"]] == [0, 1, 2]
    assert [run["recorded_at"] for run in payload["runs"]] == [
        "2026-07-12T10:12:13.123+00:00",
        "2026-07-12T10:13:14.456+00:00",
        "2026-07-12T10:14:15.789+00:00",
    ]
    calibrated_run, native_run, _ = payload["runs"]
    assert calibrated_run["object_name"] == "AgenticToolUseResult"
    assert set(calibrated_run["routing_decision"]["raw_scores"]) == set(
        persistence.TOOL_SCORE_NAMES
    )
    assert native_run["routing_decision"]["raw_scores"] == {}
    assert native_run["routing_decision"]["calibrated_scores"] == {}
    assert calibrated_run["xai_explanation"]["delta"] == 2.0
    assert calibrated_run["xai_explanation"]["pairwise_interactions"][0]["k_sii"] == 0.25
    assert calibrated_run["request"]["player_segments"] == [
        "weather in Berlin",
        "tomorrow morning",
    ]


def test_different_session_ids_write_separate_files(tmp_path: Path) -> None:
    session_started_at = datetime.datetime(2026, 7, 12, 10, 0, tzinfo=datetime.UTC)
    paths = {
        persistence.write_result_export(
            session_id=session_id,
            session_started_at=session_started_at,
            export_dir=tmp_path,
            **_mock_result_kwargs(),
        )
        for session_id in ("aaaaaaaa-first-session", "bbbbbbbb-second-session")
    }

    assert {path.name for path in paths} == {
        "session_20260712_100000_aaaaaaaa.json",
        "session_20260712_100000_bbbbbbbb.json",
    }
    assert all(len(json.loads(path.read_text(encoding="utf-8"))["runs"]) == 1 for path in paths)


def test_result_write_failure_is_non_fatal(monkeypatch) -> None:
    def fail_write(*args, **kwargs):
        del args, kwargs
        message = "read-only export directory"
        raise PermissionError(message)

    warnings: list[str] = []
    monkeypatch.setattr(persistence, "write_result_export", fail_write)

    result = persistence.write_result_export_safely(
        warning_callback=warnings.append,
        user_request="test",
    )

    assert result is None
    assert warnings == ["Could not write the run export: read-only export directory"]
