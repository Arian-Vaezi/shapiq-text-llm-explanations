"""Sample traces for the agentic tool-use explanation demo."""

from __future__ import annotations

TOOLS = {
    "weather_tool": "Fetch current weather or forecasts for a place and date.",
    "calculator_tool": "Compute exact arithmetic and numeric expressions.",
    "web_search_tool": "Search the web for current, recent, or external facts.",
    "no_tool": "Answer directly without calling an external tool.",
}


SAMPLE_TRACES = {
    "Weather forecast call": {
        "target_tool": "weather_tool",
        "system_segments": [
            "Use weather_tool for weather, rain, temperature, forecast, or city-date questions.",
            "Use calculator_tool only for exact arithmetic or unit conversions.",
            "Use web_search_tool only when the user asks for latest or current external facts.",
            "If the request is general knowledge and no external data is needed, choose no_tool.",
        ],
        "user_segments": [
            "Will it rain",
            "in Berlin",
            "tomorrow morning?",
        ],
        "takeaway": (
            "Weather language, location, date, and the system weather rule should push the "
            "agent toward weather_tool."
        ),
    },
    "Calculator call": {
        "target_tool": "calculator_tool",
        "system_segments": [
            "Use calculator_tool for exact arithmetic, totals, percentages, and numeric expressions.",
            "Use weather_tool only for forecast or weather conditions.",
            "Use web_search_tool for fresh facts that may have changed recently.",
            "Use no_tool for explanations that can be answered from general knowledge.",
        ],
        "user_segments": [
            "Calculate",
            "238 times 47",
            "and give only the final number.",
        ],
        "takeaway": (
            "The arithmetic verb, numeric expression, and exact-answer instruction should make "
            "calculator_tool the intended action."
        ),
    },
    "Web search call": {
        "target_tool": "web_search_tool",
        "system_segments": [
            "Use web_search_tool when the answer depends on current, latest, recent, or live information.",
            "Use calculator_tool for exact arithmetic.",
            "Use weather_tool for weather forecasts.",
            "Use no_tool when the answer is stable and does not require external lookup.",
        ],
        "user_segments": [
            "Who won",
            "the latest Formula 1 race",
            "this weekend?",
        ],
        "takeaway": (
            "Words such as latest and this weekend should interact with the system recency rule "
            "to push web_search_tool."
        ),
    },
    "No tool needed": {
        "target_tool": "no_tool",
        "system_segments": [
            "Use no_tool for stable conceptual explanations that do not require external data.",
            "Use web_search_tool only for current or recent facts.",
            "Use calculator_tool only for exact calculations.",
            "Use weather_tool only for weather questions.",
        ],
        "user_segments": [
            "Explain",
            "what photosynthesis is",
            "in simple terms.",
        ],
        "takeaway": (
            "This scenario explains a non-tool decision: the user asks for stable conceptual "
            "knowledge, so no_tool should be supported."
        ),
    },
    "Ambiguous Apple request": {
        "target_tool": "web_search_tool",
        "system_segments": [
            "Use web_search_tool only if the user asks for latest, newest, current, or recently changed information.",
            "Do not browse for timeless background explanations.",
            "Use calculator_tool for exact arithmetic.",
            "Use no_tool if the request can be answered without external data.",
        ],
        "user_segments": [
            "Tell me about",
            "Apple's newest product",
            "and why people care about it.",
        ],
        "takeaway": (
            "The interesting interaction is between the system recency rule and the word newest "
            "in the user request."
        ),
    },
}
