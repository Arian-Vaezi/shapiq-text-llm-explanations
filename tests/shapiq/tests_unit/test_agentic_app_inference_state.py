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


def test_linguistic_segmenter_is_selected_by_default() -> None:
    """The "Segmenter" control itself is Developer-mode-only; enable it to find the widget."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    at.toggle(key="agentic_developer_mode").set_value(True)
    at.run()

    segmenter_box = next(box for box in at.selectbox if box.label == "Segmenter")

    assert not at.exception
    assert segmenter_box.value == "Linguistic (spaCy chunking)"


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


def test_switching_hf_model_clears_stale_inferred_tool_and_explanation() -> None:
    """Switching the HF Local model must invalidate the previous agent + XAI result.

    Regression test: switching HF Local models (e.g. 1.5B -> 3B) with Developer mode
    OFF used to leave a stale XAI explanation computed for the *old* model's target
    tool, because ``logprob_model_id`` silently stayed on ``DEFAULT_LOGPROB_MODEL_ID``
    unless Developer mode was on, and neither the inference signature nor the
    explanation result signature accounted for the selected HF model at all. That let
    Agent Result move on to a new tool (e.g. after a real model switch) while XAI
    Explanation kept rendering the previous tool's cached result.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    hf_model_box = next(box for box in at.selectbox if box.label == "HF model")
    assert hf_model_box.value == "Qwen/Qwen2.5-1.5B-Instruct"

    # Simulate a completed HF-local run with the 1.5B model: an agent result plus a
    # cached XAI explanation result, both pointing at `calculator_tool`. The XAI
    # `result` is deliberately a minimal stand-in (not a full explanation payload) --
    # switching the model must invalidate it before anything tries to render it.
    at.session_state["agentic_inferred_tool"] = "calculator_tool"
    at.session_state["agentic_inference_result"] = SimpleNamespace(
        selected_tool="calculator_tool",
        tool_arguments={},
        agent_response="demo",
        raw_trace={},
        error=None,
        available=True,
        backend="HF local",
        model="Qwen/Qwen2.5-1.5B-Instruct",
    )
    at.session_state["agentic_inference_backend"] = "HF local"
    at.session_state["agentic_inference_model"] = "Qwen/Qwen2.5-1.5B-Instruct"
    at.session_state["has_run"] = True
    at.session_state["result"] = {"target_tool": "calculator_tool"}
    at.session_state["result_signature"] = "fake-signature-for-1.5b-calculator_tool"

    hf_model_box.select("Qwen/Qwen2.5-3B-Instruct").run()

    assert not at.exception
    assert at.session_state["agentic_inferred_tool"] is None
    assert at.session_state["agentic_inference_result"] is None
    assert at.session_state["agentic_inference_backend"] is None
    assert at.session_state["agentic_inference_model"] is None
    assert at.session_state["has_run"] is False
    assert at.session_state["result"] is None
    # `result_signature` is eagerly re-synced to the current (pending) settings as soon
    # as the XAI tab re-renders in the same run -- it does not stay `None` -- but it must
    # no longer be the stale signature from the old (1.5B, calculator_tool) result.
    assert at.session_state["result_signature"] != "fake-signature-for-1.5b-calculator_tool"


def _fake_no_tool_inference_result() -> SimpleNamespace:
    return SimpleNamespace(
        selected_tool="no_tool",
        tool_arguments={},
        agent_response="Photosynthesis converts light energy into chemical energy.",
        cleaned_direct_answer="Photosynthesis converts light energy into chemical energy.",
        direct_answer="Photosynthesis converts light energy into chemical energy.",
        parse_error=None,
        raw_trace={},
        error=None,
        available=True,
        backend="HF local",
        model="Qwen/Qwen2.5-1.5B-Instruct",
    )


def test_switching_hf_model_invalidates_frozen_direct_answer_target() -> None:
    """A no_tool Agent Result's frozen direct-answer target is sourced from
    ``agentic_inference_result`` -- switching the HF model must clear that
    result (the same generic invalidation covered above for executable-tool
    results) so a stale direct-answer target can never survive the switch.
    """
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    hf_model_box = next(box for box in at.selectbox if box.label == "HF model")
    at.session_state["agentic_inference_result"] = _fake_no_tool_inference_result()
    at.session_state["agentic_inference_backend"] = "HF local"
    at.session_state["agentic_inference_model"] = "Qwen/Qwen2.5-1.5B-Instruct"

    hf_model_box.select("Qwen/Qwen2.5-3B-Instruct").run()

    assert not at.exception
    assert at.session_state["agentic_inference_result"] is None


def test_editing_request_invalidates_frozen_direct_answer_target() -> None:
    """Editing the user request must clear a no_tool Agent Result the same way."""
    at = AppTest.from_file(APP_PATH, default_timeout=120)
    at.run()
    assert not at.exception

    at.session_state["agentic_inference_result"] = _fake_no_tool_inference_result()
    at.session_state["agentic_inference_backend"] = "HF local"
    at.session_state["agentic_inference_model"] = "Qwen/Qwen2.5-1.5B-Instruct"
    at.run()
    assert at.session_state["agentic_inference_result"] is not None

    request_box = next(area for area in at.text_area if area.label == "User request")
    request_box.set_value("Explain what entropy is in simple terms.").run()

    assert not at.exception
    assert at.session_state["agentic_inference_result"] is None
