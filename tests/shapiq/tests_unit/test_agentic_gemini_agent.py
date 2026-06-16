"""Tests for optional Gemini inference helper."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

import gemini_agent  # noqa: E402
from gemini_agent import classify_gemini_error, run_gemini_tool_inference  # noqa: E402
from tool_schemas import EXECUTABLE_TOOL_SCHEMAS  # noqa: E402

FIXED_SYSTEM_PROMPT = (
    "- Use weather_tool for weather, rain, temperature, forecast, or city-date questions."
)
FIXED_TOOL_CONTEXT = (
    "- weather_tool: Get current weather conditions or a forecast.\n"
    "- calculator_tool: Evaluate exact arithmetic and numeric expressions."
)


def fake_client_factory(tool_name: str, args: dict[str, object]):
    class FakeModels:
        def generate_content(self, *, model, contents, config=None):
            assert model == "gemini-test"
            assert "Will it rain" in contents
            assert config is not None
            return SimpleNamespace(
                text="tool call",
                function_calls=[
                    SimpleNamespace(
                        name=tool_name,
                        args=args,
                    )
                ],
            )

    class FakeClient:
        models = FakeModels()

    return lambda api_key: FakeClient()


def test_gemini_inference_missing_api_key_returns_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    result = run_gemini_tool_inference(
        "Will it rain in Berlin tomorrow?",
        EXECUTABLE_TOOL_SCHEMAS,
        "gemini-2.5-flash",
        system_prompt=FIXED_SYSTEM_PROMPT,
        tool_context=FIXED_TOOL_CONTEXT,
    )

    assert result.available is False
    assert result.selected_tool is None
    assert "GEMINI_API_KEY" in result.error


def test_gemini_inference_fake_client_returns_tool_and_answer(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    result = run_gemini_tool_inference(
        "Will it rain in Berlin tomorrow?",
        EXECUTABLE_TOOL_SCHEMAS,
        "gemini-test",
        system_prompt=FIXED_SYSTEM_PROMPT,
        tool_context=FIXED_TOOL_CONTEXT,
        client_factory=fake_client_factory(
            "weather_tool",
            {"location": "Berlin", "date": "tomorrow"},
        ),
    )

    assert result.available is True
    assert result.selected_tool == "weather_tool"
    assert result.tool_arguments == {"location": "Berlin", "date": "tomorrow"}
    assert "would use the weather tool" in result.assistant_answer
    assert "would use the weather tool" in result.final_answer
    assert "Demo weather" in result.raw_trace["tool_output"]


def test_gemini_weather_tool_accepts_date_or_time(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    result = run_gemini_tool_inference(
        "Will it rain in Berlin tomorrow morning?",
        EXECUTABLE_TOOL_SCHEMAS,
        "gemini-test",
        system_prompt=FIXED_SYSTEM_PROMPT,
        tool_context=FIXED_TOOL_CONTEXT,
        client_factory=fake_client_factory(
            "weather_tool",
            {"location": "Berlin", "date_or_time": "tomorrow morning"},
        ),
    )

    assert result.error is None
    assert result.tool_arguments == {"location": "Berlin", "date_or_time": "tomorrow morning"}
    assert "would use the weather tool" in result.assistant_answer
    assert "tomorrow morning" in result.final_answer
    assert "tomorrow morning" in result.raw_trace["tool_output"]


def test_gemini_unknown_tool_returns_error_without_demo_answer(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    result = run_gemini_tool_inference(
        "Will it rain in Berlin tomorrow?",
        EXECUTABLE_TOOL_SCHEMAS,
        "gemini-test",
        system_prompt=FIXED_SYSTEM_PROMPT,
        tool_context=FIXED_TOOL_CONTEXT,
        client_factory=fake_client_factory(
            "calendar_tool",
            {"location": "Berlin", "date": "tomorrow"},
        ),
    )

    assert result.selected_tool == "calendar_tool"
    assert result.tool_arguments == {"location": "Berlin", "date": "tomorrow"}
    assert result.final_answer == ""
    assert result.error == "Gemini selected unknown tool: calendar_tool"
    assert "tool_output" not in result.raw_trace


def test_gemini_error_classification_returns_friendly_hints() -> None:
    assert classify_gemini_error("429 RESOURCE_EXHAUSTED") == (
        "Gemini quota exhausted for this project/model. "
        "Try another model, wait, or use the preview fallback."
    )
    assert classify_gemini_error("503 UNAVAILABLE") == (
        "Gemini model is temporarily overloaded. Retry later or choose another model."
    )
    assert classify_gemini_error("404 NOT_FOUND") == (
        "The model name is not available for this API version. Check available models."
    )


def test_list_available_gemini_models_filters_generate_content(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    class FakeModels:
        def list(self):
            return [
                SimpleNamespace(
                    name="models/gemini-2.5-flash",
                    supported_generation_methods=["generateContent"],
                ),
                SimpleNamespace(
                    name="models/text-embedding",
                    supported_generation_methods=["embedContent"],
                ),
                {"name": "models/gemini-test", "supported_actions": ["generateContent"]},
            ]

    class FakeClient:
        models = FakeModels()

    monkeypatch.setattr(gemini_agent, "_default_client_factory", lambda api_key: FakeClient())

    assert gemini_agent.list_available_gemini_models() == [
        "gemini-2.5-flash",
        "gemini-test",
    ]


def test_list_available_gemini_models_missing_key_returns_empty(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert gemini_agent.list_available_gemini_models() == []


def test_gemini_inference_prompt_includes_shared_system_prompt_and_tool_context(
    monkeypatch,
) -> None:
    """The real inference prompt must use the same fixed context as the explanation."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    captured: dict[str, object] = {}

    class CapturingModels:
        def generate_content(self, *, model, contents, config=None):
            del model, config
            captured["contents"] = contents
            return SimpleNamespace(
                text="tool call",
                function_calls=[
                    SimpleNamespace(name="weather_tool", args={"location": "Berlin"}),
                ],
            )

    class CapturingClient:
        models = CapturingModels()

    run_gemini_tool_inference(
        "Will it rain in Berlin tomorrow?",
        EXECUTABLE_TOOL_SCHEMAS,
        "gemini-test",
        system_prompt=FIXED_SYSTEM_PROMPT,
        tool_context=FIXED_TOOL_CONTEXT,
        client_factory=lambda api_key: CapturingClient(),
    )

    prompt_text = captured["contents"]
    assert FIXED_SYSTEM_PROMPT in prompt_text
    assert FIXED_TOOL_CONTEXT in prompt_text
