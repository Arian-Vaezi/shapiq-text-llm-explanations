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


def test_mock_result_export_round_trips_numpy_values(tmp_path: Path) -> None:
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

    path = persistence.write_result_export(
        export_dir=tmp_path,
        now=datetime.datetime(2026, 7, 12, 10, 12, 13, tzinfo=datetime.UTC),
        hf_model_id="mock/model",
        user_request="Will it rain in Berlin tomorrow morning?",
        system_prompt="Use the appropriate tool.",
        player_segments=segments,
        raw_scores=scores,
        calibrated_scores={name: np.float64(value) for name, value in scores.items()},
        selected_tool="weather_tool",
        raw_argmax="weather_tool",
        calibrated_argmax="weather_tool",
        target_tool="weather_tool",
        baseline_h_empty=np.float32(-0.5),
        full_h_n=np.float64(1.5),
        pairwise_interactions=pairs,
    )
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)

    assert payload["object_name"] == "AgenticToolUseResult"
    assert set(payload["routing_decision"]["raw_scores"]) == set(persistence.TOOL_SCORE_NAMES)
    assert payload["xai_explanation"]["delta"] == 2.0
    assert payload["xai_explanation"]["pairwise_interactions"][0]["k_sii"] == 0.25
    assert payload["request"]["player_segments"] == [
        "weather in Berlin",
        "tomorrow morning",
    ]
    assert path.name == "result_20260712_101213_will_it_rain_in_berlin_tomorro.json"


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
