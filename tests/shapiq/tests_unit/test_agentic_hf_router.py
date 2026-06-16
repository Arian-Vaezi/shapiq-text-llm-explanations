"""Tests for the local HuggingFace router helper."""

from __future__ import annotations

import sys
from pathlib import Path

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

from hf_router import LocalHFRouter  # noqa: E402
from tool_schemas import TOOL_DESCRIPTIONS  # noqa: E402


def test_parse_valid_json_web_search_with_query_argument() -> None:
    result = LocalHFRouter.parse_response(
        """
        Here is the decision:
        {"agent_response":"I would search for current F1 results.",
         "selected_tool":"web_search_tool",
         "tool_arguments":{"query":"latest Formula 1 race winner this weekend"}}
        """
    )

    assert result.selected_tool == "web_search_tool"
    assert result.tool_arguments == {
        "query": "latest Formula 1 race winner this weekend",
    }
    assert result.agent_response == "I would search for current F1 results."


def test_parse_valid_json_calculator_with_expression_argument() -> None:
    result = LocalHFRouter.parse_response(
        '{"agent_response":"I would calculate it.",'
        '"selected_tool":"calculator_tool",'
        '"tool_arguments":{"expression":"2 + 3 * 4"}}'
    )

    assert result.selected_tool == "calculator_tool"
    assert result.tool_arguments == {"expression": "2 + 3 * 4"}
    assert result.agent_response == "I would calculate it."


def test_fallback_parser_detects_web_search_from_non_json_output() -> None:
    result = LocalHFRouter.parse_response("This needs web search because it asks latest news.")

    assert result.selected_tool == "web_search_tool"
    assert result.tool_arguments == {}
    assert result.agent_response == "I would use web_search_tool for this request."


def test_fallback_parser_detects_calculator_from_non_json_output() -> None:
    result = LocalHFRouter.parse_response("Use the calculator for the arithmetic.")

    assert result.selected_tool == "calculator_tool"
    assert result.tool_arguments == {}


def test_invalid_selected_tool_falls_back_to_no_tool() -> None:
    result = LocalHFRouter.parse_response(
        '{"selected_tool":"calendar_tool","tool_arguments":{"date":"tomorrow"}}'
    )

    assert result.selected_tool == "no_tool"
    assert result.tool_arguments == {"date": "tomorrow"}
    assert result.agent_response == "I would use no_tool for this request."


def test_choose_tool_can_use_monkeypatched_generation_without_loading_model() -> None:
    class FakeRouter(LocalHFRouter):
        def __init__(self) -> None:
            self.seen_prompt = ""

        def _generate_raw_response(self, router_prompt: str) -> tuple[str, str]:
            self.seen_prompt = router_prompt
            return (
                f"formatted::{router_prompt}",
                '{"agent_response":"I would use weather.",'
                '"selected_tool":"weather_tool",'
                '"tool_arguments":{"location":"Berlin","date":"tomorrow"}}',
            )

    router = FakeRouter()
    result = router.choose_tool("Will it rain in Berlin tomorrow?", TOOL_DESCRIPTIONS)

    assert result.selected_tool == "weather_tool"
    assert result.tool_arguments == {"location": "Berlin", "date": "tomorrow"}
    assert result.debug_prompt.startswith("formatted::")
    assert "weather_tool" in router.seen_prompt
    assert "Return only valid JSON" in router.seen_prompt
