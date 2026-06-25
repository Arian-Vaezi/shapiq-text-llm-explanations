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


def test_selecting_a_different_example_clears_stale_inferred_tool() -> None:
    """Selecting a different "Try example" entry must invalidate a previous inference result.

    The current app has no sidebar "Scenario" selectbox; sample scenarios are applied
    through the "Try example" selectbox in the Inference tab instead, which rewrites
    ``agentic_request_text`` and should invalidate any prior inference result tied to
    the old request text.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception
    default_request = at.session_state["agentic_request_text"]

    _inject_fake_inference_result(at)
    at.run()
    assert not at.exception
    assert at.session_state["agentic_inferred_tool"] == "calculator_tool"

    example_box = next(box for box in at.selectbox if box.label == "Try example")
    other_trace = next(
        name
        for name in SAMPLE_TRACES
        if " ".join(SAMPLE_TRACES[name]["user_segments"]) != default_request
    )
    example_box.select(other_trace).run()

    assert not at.exception
    assert at.session_state["agentic_inferred_tool"] is None
    assert at.session_state["agentic_inference_result"] is None
    assert at.session_state["agentic_inference_backend"] is None
    assert at.session_state["agentic_inference_model"] is None


def test_editing_custom_request_clears_stale_inferred_tool() -> None:
    """Editing the user request text must invalidate a previous inference result.

    The current app's only input mode is the "User request" text area in the
    Inference tab (there is no separate "Custom request" radio/mode anymore).
    """
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    _inject_fake_inference_result(at)
    at.run()
    assert not at.exception
    assert at.session_state["agentic_inferred_tool"] == "calculator_tool"

    request_box = next(area for area in at.text_area if area.label == "User request")
    request_box.set_value("Explain what photosynthesis is in simple terms.").run()

    assert not at.exception
    assert at.session_state["agentic_inferred_tool"] is None
    assert at.session_state["agentic_inference_result"] is None
