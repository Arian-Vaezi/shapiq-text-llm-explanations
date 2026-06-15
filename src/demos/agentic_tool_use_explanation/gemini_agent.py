"""Gemini-backed inference helper for the agentic tool-use demo."""

from __future__ import annotations

import ast
import json
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field


@dataclass
class GeminiInferenceResult:
    """Structured result returned by the optional Gemini inference path."""

    selected_tool: str | None = None
    tool_arguments: dict[str, object] = field(default_factory=dict)
    final_answer: str = ""
    raw_trace: dict[str, object] = field(default_factory=dict)
    error: str | None = None
    available: bool = True


def run_gemini_tool_inference(
    user_request: str,
    tool_schemas: Sequence[Mapping[str, object]],
    model_name: str,
    *,
    client_factory: Callable[[str], object] | None = None,
) -> GeminiInferenceResult:
    """Ask Gemini to select a demo tool, then run a deterministic local tool output."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return GeminiInferenceResult(
            available=False,
            error="GEMINI_API_KEY is not set. Add it to the environment to run Gemini inference.",
        )

    try:
        if client_factory is None:
            client_factory = _default_client_factory
        client = client_factory(api_key)
    except Exception as error:  # noqa: BLE001
        return GeminiInferenceResult(
            available=False,
            error=f"Gemini client is unavailable: {error}",
        )

    prompt = _build_inference_prompt(user_request, tool_schemas)
    try:
        response = _generate_content(client, model_name, prompt, tool_schemas)
    except Exception as error:  # noqa: BLE001
        friendly_error = classify_gemini_error(str(error))
        return GeminiInferenceResult(
            available=False,
            raw_trace={"prompt": prompt},
            error=f"Gemini inference failed: {friendly_error}",
        )

    selected_tool, tool_arguments = _extract_tool_call(response)
    raw_text = str(getattr(response, "text", "") or "")
    if selected_tool is None:
        selected_tool, tool_arguments = _parse_text_selection(raw_text)

    if selected_tool is None:
        return GeminiInferenceResult(
            selected_tool=None,
            final_answer=raw_text,
            raw_trace={"prompt": prompt, "response_text": raw_text},
            error="Gemini did not return a recognizable tool selection.",
        )

    known_tool_names = _tool_names(tool_schemas)
    if selected_tool not in known_tool_names:
        return GeminiInferenceResult(
            selected_tool=selected_tool,
            tool_arguments=tool_arguments,
            raw_trace={"prompt": prompt, "response_text": raw_text},
            error=f"Gemini selected unknown tool: {selected_tool}",
        )

    tool_output = _run_demo_tool(selected_tool, tool_arguments)
    final_answer = _format_final_answer(selected_tool, tool_output)
    return GeminiInferenceResult(
        selected_tool=selected_tool,
        tool_arguments=tool_arguments,
        final_answer=final_answer,
        raw_trace={
            "prompt": prompt,
            "response_text": raw_text,
            "tool_output": tool_output,
        },
    )


def list_available_gemini_models(model_filter: str = "generateContent") -> list[str]:
    """Return Gemini model IDs available to the configured API key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return []
    try:
        client = _default_client_factory(api_key)
        models = client.models.list()
    except Exception:  # noqa: BLE001
        return []

    model_names = []
    for model in models:
        if model_filter and not _model_supports_method(model, model_filter):
            continue
        model_name = _model_id(model)
        if model_name:
            model_names.append(model_name)
    return model_names


def classify_gemini_error(error_text: str) -> str:
    """Return a short user-facing hint for common Gemini API failures."""
    normalized = error_text.upper()
    if "429" in normalized or "RESOURCE_EXHAUSTED" in normalized:
        return (
            "Gemini quota exhausted for this project/model. "
            "Try another model, wait, or use the preview fallback."
        )
    if "503" in normalized or "UNAVAILABLE" in normalized:
        return "Gemini model is temporarily overloaded. Retry later or choose another model."
    if "404" in normalized or "NOT_FOUND" in normalized:
        return "The model name is not available for this API version. Check available models."
    return error_text


def _default_client_factory(api_key: str) -> object:
    from google import genai

    return genai.Client(api_key=api_key)


def _build_inference_prompt(
    user_request: str,
    tool_schemas: Sequence[Mapping[str, object]],
) -> str:
    tool_names = []
    for schema in tool_schemas:
        function = schema.get("function", {})
        if isinstance(function, Mapping):
            name = function.get("name")
            description = function.get("description", "")
            if isinstance(name, str):
                tool_names.append(f"- {name}: {description}")
    return (
        "Select the single best tool for the user request. "
        "Return a tool call if tool calling is available. Otherwise return JSON with "
        "selected_tool and tool_arguments.\n\n"
        f"Available tools:\n{chr(10).join(tool_names)}\n\n"
        f"User request:\n{user_request}"
    )


def _model_supports_method(model: object, model_filter: str) -> bool:
    supported_methods = getattr(model, "supported_actions", None)
    if supported_methods is None:
        supported_methods = getattr(model, "supported_generation_methods", None)
    if supported_methods is None and isinstance(model, Mapping):
        supported_methods = model.get("supported_actions")
        if supported_methods is None:
            supported_methods = model.get("supported_generation_methods")
    if supported_methods is None:
        return True
    return model_filter in supported_methods


def _model_id(model: object) -> str:
    name = getattr(model, "name", None)
    if name is None and isinstance(model, Mapping):
        name = model.get("name")
    if not isinstance(name, str):
        return ""
    return name.removeprefix("models/")


def _generate_content(
    client: object,
    model_name: str,
    prompt: str,
    tool_schemas: Sequence[Mapping[str, object]],
) -> object:
    config = {"tools": [{"function_declarations": [_to_function_declaration(s) for s in tool_schemas]}]}
    try:
        return client.models.generate_content(model=model_name, contents=prompt, config=config)
    except TypeError:
        return client.models.generate_content(model=model_name, contents=prompt)


def _to_function_declaration(schema: Mapping[str, object]) -> dict[str, object]:
    function = schema.get("function", {})
    if not isinstance(function, Mapping):
        return {}
    return {
        "name": function.get("name"),
        "description": function.get("description"),
        "parameters": function.get("parameters", {}),
    }


def _tool_names(tool_schemas: Sequence[Mapping[str, object]]) -> set[str]:
    names = set()
    for schema in tool_schemas:
        function = schema.get("function", {})
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names


def _extract_tool_call(response: object) -> tuple[str | None, dict[str, object]]:
    function_calls = getattr(response, "function_calls", None)
    if function_calls:
        return _coerce_function_call(function_calls[0])

    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            function_call = getattr(part, "function_call", None)
            if function_call is not None:
                return _coerce_function_call(function_call)
    return None, {}


def _coerce_function_call(function_call: object) -> tuple[str | None, dict[str, object]]:
    name = getattr(function_call, "name", None)
    args = getattr(function_call, "args", None) or {}
    if isinstance(function_call, Mapping):
        name = function_call.get("name", name)
        args = function_call.get("args", args)
    if not isinstance(name, str):
        return None, {}
    if isinstance(args, Mapping):
        return name, dict(args)
    return name, {}


def _parse_text_selection(text: str) -> tuple[str | None, dict[str, object]]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match is None:
        return None, {}
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, {}
    selected_tool = payload.get("selected_tool")
    tool_arguments = payload.get("tool_arguments", {})
    if not isinstance(selected_tool, str):
        return None, {}
    if not isinstance(tool_arguments, Mapping):
        tool_arguments = {}
    return selected_tool, dict(tool_arguments)


def _run_demo_tool(tool_name: str, arguments: Mapping[str, object]) -> str:
    if tool_name == "weather_tool":
        location = str(arguments.get("location", "the requested location"))
        date_or_time = arguments.get("date_or_time", arguments.get("date", "the requested time"))
        return f"Demo weather for {location} at {date_or_time}: mild, partly cloudy, low rain risk."
    if tool_name == "calculator_tool":
        expression = str(arguments.get("expression", ""))
        return f"Demo calculator result: {_safe_eval_arithmetic(expression)}"
    if tool_name == "web_search_tool":
        query = str(arguments.get("query", "the request"))
        return f"Demo search result for {query!r}: representative current-looking result."
    return "Demo direct answer: no external tool was called."


def _safe_eval_arithmetic(expression: str) -> str:
    try:
        node = ast.parse(expression, mode="eval")
        return str(_eval_numeric_node(node.body))
    except Exception:  # noqa: BLE001
        return "unable to evaluate demo expression"


def _eval_numeric_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval_numeric_node(node.operand)
    if isinstance(node, ast.BinOp):
        left = _eval_numeric_node(node.left)
        right = _eval_numeric_node(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
    msg = "unsupported expression"
    raise ValueError(msg)


def _format_final_answer(tool_name: str, tool_output: str) -> str:
    return f"Gemini selected `{tool_name}`. {tool_output}"
