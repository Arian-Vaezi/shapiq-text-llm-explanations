"""Canonical structured tool schemas for the agentic tool-use demo."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence

NO_TOOL_NAME = "no_tool"
NO_TOOL_DESCRIPTION = "Answer directly without calling an external tool."

# no_tool is a decision candidate, not an executable function. It stays out of
# EXECUTABLE_TOOL_SCHEMAS so model-native tool calling sees only callable tools.
EXECUTABLE_TOOL_SCHEMAS: tuple[dict[str, object], ...] = (
    {
        "type": "function",
        "function": {
            "name": "weather_tool",
            "description": "Get current weather conditions or a forecast.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City, region, or location.",
                    },
                    "date": {
                        "type": "string",
                        "description": "Requested date or time period.",
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator_tool",
            "description": "Evaluate exact arithmetic and numeric expressions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Arithmetic or numeric expression to evaluate.",
                    },
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search_tool",
            "description": "Find current, recent, changing, or external facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for the needed external facts.",
                    },
                },
                "required": ["query"],
            },
        },
    },
)


def _function_mapping(schema: Mapping[str, object]) -> Mapping[str, object]:
    function = schema.get("function")
    if not isinstance(function, Mapping):
        msg = "Executable tool schema must contain a function mapping."
        raise TypeError(msg)
    return function


def _function_name(schema: Mapping[str, object]) -> str:
    function = _function_mapping(schema)
    name = function.get("name")
    if not isinstance(name, str) or not name.strip():
        msg = "Executable tool schema function.name must be a non-empty string."
        raise ValueError(msg)
    return name


def _function_description(schema: Mapping[str, object]) -> str:
    function = _function_mapping(schema)
    description = function.get("description")
    if not isinstance(description, str) or not description.strip():
        msg = "Executable tool schema function.description must be a non-empty string."
        raise ValueError(msg)
    return description


def executable_tool_names(
    tool_schemas: Sequence[Mapping[str, object]] = EXECUTABLE_TOOL_SCHEMAS,
) -> tuple[str, ...]:
    """Return executable function names from structured tool schemas."""
    return tuple(_function_name(schema) for schema in tool_schemas)


DECISION_NAMES: tuple[str, ...] = (*executable_tool_names(), NO_TOOL_NAME)


def derive_tool_descriptions(
    tool_schemas: Sequence[Mapping[str, object]] = EXECUTABLE_TOOL_SCHEMAS,
    *,
    no_tool_name: str = NO_TOOL_NAME,
    no_tool_description: str = NO_TOOL_DESCRIPTION,
) -> dict[str, str]:
    """Derive the legacy tool-description view from canonical schemas."""
    descriptions = {
        _function_name(schema): _function_description(schema) for schema in tool_schemas
    }
    descriptions[no_tool_name] = no_tool_description
    return descriptions


TOOL_DESCRIPTIONS: dict[str, str] = derive_tool_descriptions()

# Canonical argument keys follow EXECUTABLE_TOOL_SCHEMAS above. Different real
# agent backends (Groq, local HF routers) have been observed to use slightly
# different argument names for the same value -- e.g. a model guessing "city"
# instead of the schema's "location". This is the single shared alias table:
# both router_scorers.TrajectoryArgumentMatchScorer (comparing two trajectories
# for a coalition value function) and groq_agent's demo tool-output templates
# (rendering a user-visible "Tool output" string) key-normalize through it, so
# neither one silently drifts from the other.
DEFAULT_ARGUMENT_ALIASES: dict[str, dict[str, str]] = {
    "weather_tool": {
        "date_or_time": "date",
        "time": "date",
        "when": "date",
        "place": "location",
        "city": "location",
    },
    "calculator_tool": {
        "expr": "expression",
        "equation": "expression",
        "calculation": "expression",
    },
    "web_search_tool": {
        "search_query": "query",
        "q": "query",
        "search": "query",
    },
    "no_tool": {},
}


def canonicalize_tool_arguments(
    tool_name: str,
    arguments: Mapping[str, object],
    alias_map: Mapping[str, str],
) -> dict[str, object]:
    """Map alias argument keys to canonical keys, raising ValueError on conflicting aliases.

    A canonical key already present in ``arguments`` is never overwritten by an
    alias mapping to the same canonical key that appears later; a genuine
    conflict (two different keys mapping to the same canonical key with
    different values) raises rather than silently picking one, since that
    means the source data is ambiguous.
    """
    canonical: dict[str, object] = {}
    for key, value in arguments.items():
        ckey = alias_map.get(key, key)
        if ckey in canonical:
            if canonical[ckey] != value:
                msg = (
                    f"Conflicting values for canonical argument {ckey!r} in {tool_name!r}: "
                    f"{canonical[ckey]!r} vs {value!r}"
                )
                raise ValueError(msg)
        else:
            canonical[ckey] = value
    return canonical


def get_executable_tool_schemas() -> tuple[dict[str, object], ...]:
    """Return deep copies of executable tool schemas for model input."""
    return tuple(copy.deepcopy(schema) for schema in EXECUTABLE_TOOL_SCHEMAS)


def render_tool_schemas_text(tool_schemas: Sequence[Mapping[str, object]]) -> str:
    """Render structured tool schemas deterministically for fallback prompts."""
    rendered = []
    for schema in tool_schemas:
        function = _function_mapping(schema)
        name = _function_name(schema)
        description = _function_description(schema)
        parameters = function["parameters"]
        rendered.append(
            "\n".join(
                [
                    f"- {name}: {description}",
                    "  parameters:",
                    _indent_json(parameters, spaces=4),
                ]
            )
        )
    return "\n".join(rendered)


def validate_tool_configuration(
    tool_schemas: Sequence[Mapping[str, object]] = EXECUTABLE_TOOL_SCHEMAS,
    *,
    no_tool_name: str = NO_TOOL_NAME,
    decision_names: Sequence[str] = DECISION_NAMES,
    descriptions: Mapping[str, str] = TOOL_DESCRIPTIONS,
) -> None:
    """Validate the local structured tool configuration."""
    names = []
    for schema in tool_schemas:
        if schema.get("type") != "function":
            msg = "Executable tool schema must have type == 'function'."
            raise ValueError(msg)
        function = _function_mapping(schema)
        name = _function_name(schema)
        description = _function_description(schema)
        parameters = function.get("parameters")
        if name == no_tool_name:
            msg = "no_tool must not be an executable function schema."
            raise ValueError(msg)
        if not isinstance(parameters, Mapping):
            msg = f"Tool schema {name!r} must define a parameters mapping."
            raise TypeError(msg)
        if not description:
            msg = f"Tool schema {name!r} must define a non-empty description."
            raise ValueError(msg)
        names.append(name)

    if len(names) != len(set(names)):
        msg = "Executable tool names must be unique."
        raise ValueError(msg)
    if list(decision_names).count(no_tool_name) != 1:
        msg = "no_tool must appear exactly once in decision candidate names."
        raise ValueError(msg)
    missing_descriptions = [name for name in decision_names if name not in descriptions]
    if missing_descriptions:
        msg = f"Missing descriptions for decision candidates: {missing_descriptions!r}."
        raise ValueError(msg)


def _indent_json(value: object, *, spaces: int) -> str:
    text = json.dumps(value, indent=2, sort_keys=True)
    prefix = " " * spaces
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


validate_tool_configuration()
