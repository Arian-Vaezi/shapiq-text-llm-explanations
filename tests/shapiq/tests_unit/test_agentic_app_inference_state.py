"""Tests for stale inference session state in the agentic tool-use demo app."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

from sample_data import SAMPLE_TRACES  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

APP_PATH = str(DEMO_DIR / "app.py")


def _fake_inference_result() -> SimpleNamespace:
    return SimpleNamespace(
        selected_tool="calculator_tool",
        tool_arguments={},
        assistant_answer="demo",
        final_answer="demo",
        raw_trace={},
        error=None,
        available=True,
    )


def _inject_fake_inference_result(at: AppTest) -> None:
    at.session_state["agentic_inferred_tool"] = "calculator_tool"
    at.session_state["agentic_inference_result"] = _fake_inference_result()
    at.session_state["agentic_inference_backend"] = "Groq"
    at.session_state["agentic_inference_model"] = "llama-3.1-8b-instant"


def test_unrelated_rerun_keeps_inferred_tool() -> None:
    """Re-running without changing the request must not drop a real inference result."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    _inject_fake_inference_result(at)
    at.run()

    assert not at.exception
    assert at.session_state["agentic_inferred_tool"] == "calculator_tool"
    assert at.session_state["agentic_inference_result"] is not None


def test_switching_scenario_clears_stale_inferred_tool() -> None:
    """Switching to a different scenario must invalidate a previous inference result."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    scenario_box = next(box for box in at.sidebar.selectbox if box.label == "Scenario")
    current_trace = scenario_box.value
    other_trace = next(name for name in SAMPLE_TRACES if name != current_trace)

    _inject_fake_inference_result(at)
    at.run()
    assert not at.exception
    assert at.session_state["agentic_inferred_tool"] == "calculator_tool"

    scenario_box = next(box for box in at.sidebar.selectbox if box.label == "Scenario")
    scenario_box.set_value(other_trace).run()

    assert not at.exception
    assert at.session_state["agentic_inferred_tool"] is None
    assert at.session_state["agentic_inference_result"] is None
    assert at.session_state["agentic_inference_backend"] is None
    assert at.session_state["agentic_inference_model"] is None


def test_editing_custom_request_clears_stale_inferred_tool() -> None:
    """Editing the custom request text must invalidate a previous inference result."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    mode_radio = next(radio for radio in at.sidebar.radio if radio.label == "Input")
    mode_radio.set_value("Custom request").run()
    assert not at.exception

    _inject_fake_inference_result(at)
    at.run()
    assert not at.exception
    assert at.session_state["agentic_inferred_tool"] == "calculator_tool"

    request_box = next(area for area in at.text_area if area.label == "Request text")
    request_box.set_value("Explain what photosynthesis is in simple terms.").run()

    assert not at.exception
    assert at.session_state["agentic_inferred_tool"] is None
    assert at.session_state["agentic_inference_result"] is None
