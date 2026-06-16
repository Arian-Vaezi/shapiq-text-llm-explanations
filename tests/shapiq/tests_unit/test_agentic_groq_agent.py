"""Tests for optional Groq inference helper."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

from groq_agent import classify_groq_error, run_groq_tool_inference  # noqa: E402
from tool_schemas import EXECUTABLE_TOOL_SCHEMAS  # noqa: E402

FIXED_SYSTEM_PROMPT = (
    "- Use weather_tool for weather, rain, temperature, forecast, or city-date questions."
)
FIXED_TOOL_CONTEXT = (
    "- weather_tool: Get current weather conditions or a forecast.\n"
    "- calculator_tool: Evaluate exact arithmetic and numeric expressions."
)


def fake_client_factory(content: str | None = None, error: Exception | None = None):
    class FakeCompletions:
        def create(self, *, model, messages, temperature, response_format):
            assert model == "groq-test"
            assert messages
            assert temperature == 0
            assert response_format == {"type": "json_object"}
            if error is not None:
                raise error
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content=content)),
                ],
            )

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    return lambda api_key: FakeClient()


def test_groq_inference_missing_api_key_returns_unavailable(monkeypatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    result = run_groq_tool_inference(
        "Calculate 2 + 3",
        EXECUTABLE_TOOL_SCHEMAS,
        "groq-test",
        system_prompt=FIXED_SYSTEM_PROMPT,
        tool_context=FIXED_TOOL_CONTEXT,
    )

    assert result.available is False
    assert result.selected_tool is None
    assert "GROQ_API_KEY" in result.error


def test_groq_fake_client_returns_calculator_tool(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    result = run_groq_tool_inference(
        "Calculate 2 + 3",
        EXECUTABLE_TOOL_SCHEMAS,
        "groq-test",
        system_prompt=FIXED_SYSTEM_PROMPT,
        tool_context=FIXED_TOOL_CONTEXT,
        client_factory=fake_client_factory(
            '{"selected_tool":"calculator_tool","tool_arguments":{"expression":"2 + 3"},'
            '"assistant_answer":"2 + 3 is 5."}',
        ),
    )

    assert result.error is None
    assert result.selected_tool == "calculator_tool"
    assert result.tool_arguments == {"expression": "2 + 3"}
    assert result.assistant_answer == "2 + 3 is 5."
    assert result.final_answer == "2 + 3 is 5."


def test_groq_unknown_tool_returns_error(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    result = run_groq_tool_inference(
        "Schedule this",
        EXECUTABLE_TOOL_SCHEMAS,
        "groq-test",
        system_prompt=FIXED_SYSTEM_PROMPT,
        tool_context=FIXED_TOOL_CONTEXT,
        client_factory=fake_client_factory(
            '{"selected_tool":"calendar_tool","tool_arguments":{"date":"tomorrow"}}',
        ),
    )

    assert result.selected_tool == "calendar_tool"
    assert result.tool_arguments == {"date": "tomorrow"}
    assert result.final_answer == ""
    assert result.error == "Groq selected unknown tool: calendar_tool"
    assert "tool_output" not in result.raw_trace


def test_groq_malformed_json_returns_clean_error(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    result = run_groq_tool_inference(
        "Calculate 2 + 3",
        EXECUTABLE_TOOL_SCHEMAS,
        "groq-test",
        system_prompt=FIXED_SYSTEM_PROMPT,
        tool_context=FIXED_TOOL_CONTEXT,
        client_factory=fake_client_factory("not json"),
    )

    assert result.selected_tool is None
    assert result.final_answer == ""
    assert result.error == "Groq returned malformed JSON."


def test_groq_rate_limit_error_is_classified(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    result = run_groq_tool_inference(
        "Calculate 2 + 3",
        EXECUTABLE_TOOL_SCHEMAS,
        "groq-test",
        system_prompt=FIXED_SYSTEM_PROMPT,
        tool_context=FIXED_TOOL_CONTEXT,
        client_factory=fake_client_factory(error=RuntimeError("429 rate limit exceeded")),
    )

    assert result.available is False
    assert result.error == (
        "Groq inference failed: Groq rate limit reached. "
        "Try again later or use the preview fallback."
    )
    assert classify_groq_error("401 invalid API key") == (
        "Groq authentication failed. Check GROQ_API_KEY."
    )
    assert classify_groq_error("404 model not found") == (
        "Groq model is unavailable or not found. Try another model."
    )


def test_groq_weather_tool_accepts_date_or_time(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    result = run_groq_tool_inference(
        "Will it rain in Berlin tomorrow morning?",
        EXECUTABLE_TOOL_SCHEMAS,
        "groq-test",
        system_prompt=FIXED_SYSTEM_PROMPT,
        tool_context=FIXED_TOOL_CONTEXT,
        client_factory=fake_client_factory(
            '{"selected_tool":"weather_tool",'
            '"tool_arguments":{"location":"Berlin","date_or_time":"tomorrow morning"}}',
        ),
    )

    assert result.error is None
    assert result.tool_arguments == {"location": "Berlin", "date_or_time": "tomorrow morning"}
    assert "would use the weather tool" in result.assistant_answer
    assert "tomorrow morning" in result.final_answer
    assert "tomorrow morning" in result.raw_trace["tool_output"]


def test_groq_inference_prompt_includes_shared_system_prompt_and_tool_context(
    monkeypatch,
) -> None:
    """The real inference prompt must use the same fixed context as the explanation."""
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    captured: dict[str, object] = {}

    class CapturingCompletions:
        def create(self, *, model, messages, temperature, response_format):
            del model, temperature, response_format
            captured["messages"] = messages
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"selected_tool":"weather_tool",'
                            '"tool_arguments":{"location":"Berlin"}}',
                        ),
                    ),
                ],
            )

    class CapturingChat:
        completions = CapturingCompletions()

    class CapturingClient:
        chat = CapturingChat()

    run_groq_tool_inference(
        "Will it rain in Berlin tomorrow?",
        EXECUTABLE_TOOL_SCHEMAS,
        "groq-test",
        system_prompt=FIXED_SYSTEM_PROMPT,
        tool_context=FIXED_TOOL_CONTEXT,
        client_factory=lambda api_key: CapturingClient(),
    )

    prompt_text = " ".join(message["content"] for message in captured["messages"])
    assert FIXED_SYSTEM_PROMPT in prompt_text
    assert FIXED_TOOL_CONTEXT in prompt_text
