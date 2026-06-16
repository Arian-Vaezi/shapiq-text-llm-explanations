"""Groq-backed inference helper for the agentic tool-use demo."""

from __future__ import annotations

import ast
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

try:
    from demos.agentic_tool_use_explanation.tool_schemas import NO_TOOL_NAME
except ModuleNotFoundError:
    from tool_schemas import NO_TOOL_NAME


@dataclass
class GroqInferenceResult:
    """Structured result returned by the optional Groq inference path."""

    selected_tool: str | None = None
    tool_arguments: dict[str, object] = field(default_factory=dict)
    assistant_answer: str = ""
    final_answer: str = ""
    raw_trace: dict[str, object] = field(default_factory=dict)
    error: str | None = None
    available: bool = True


def run_groq_tool_inference(
    user_request: str,
    tool_schemas: Sequence[Mapping[str, object]],
    model_name: str,
    *,
    system_prompt: str,
    tool_context: str,
    client_factory: Callable[[str], object] | None = None,
) -> GroqInferenceResult:
    """Ask Groq to select a demo tool using JSON output, then run a local demo tool.

    ``system_prompt`` and ``tool_context`` must be the same fixed context the
    explanation uses, so the real inference decision and the Shapley
    explanation are computed over identical context.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return GroqInferenceResult(
            available=False,
            error="GROQ_API_KEY is not set. Add it to the environment to run Groq inference.",
        )

    try:
        if client_factory is None:
            client_factory = _default_client_factory
        client = client_factory(api_key)
    except Exception as error:  # noqa: BLE001
        return GroqInferenceResult(
            available=False,
            error=f"Groq client is unavailable: {classify_groq_error(str(error))}",
        )

    prompt = _build_inference_prompt(
        user_request,
        tool_schemas,
        system_prompt=system_prompt,
        tool_context=tool_context,
    )
    try:
        response_text = _generate_json_selection(client, model_name, prompt)
    except Exception as error:  # noqa: BLE001
        return GroqInferenceResult(
            available=False,
            raw_trace={"prompt": prompt},
            error=f"Groq inference failed: {classify_groq_error(str(error))}",
        )

    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError:
        return GroqInferenceResult(
            raw_trace={"prompt": prompt, "response_text": response_text},
            error="Groq returned malformed JSON.",
        )

    selected_tool = payload.get("selected_tool")
    tool_arguments = payload.get("tool_arguments", {})
    assistant_answer = payload.get("assistant_answer", "")
    if not isinstance(selected_tool, str):
        return GroqInferenceResult(
            raw_trace={"prompt": prompt, "response_text": response_text},
            error="Groq JSON did not contain a valid selected_tool.",
        )
    if not isinstance(tool_arguments, Mapping):
        tool_arguments = {}
    tool_arguments = dict(tool_arguments)
    if selected_tool in {"weather_tool", "web_search_tool"}:
        assistant_answer = _build_assistant_answer(user_request, selected_tool, tool_arguments)
    elif not isinstance(assistant_answer, str) or not assistant_answer.strip():
        assistant_answer = _build_assistant_answer(user_request, selected_tool, tool_arguments)

    if selected_tool not in _valid_decision_names(tool_schemas):
        return GroqInferenceResult(
            selected_tool=selected_tool,
            tool_arguments=tool_arguments,
            assistant_answer=assistant_answer,
            raw_trace={"prompt": prompt, "response_text": response_text},
            error=f"Groq selected unknown tool: {selected_tool}",
        )

    tool_output = _run_demo_tool(selected_tool, tool_arguments)
    return GroqInferenceResult(
        selected_tool=selected_tool,
        tool_arguments=tool_arguments,
        assistant_answer=assistant_answer,
        final_answer=assistant_answer,
        raw_trace={
            "prompt": prompt,
            "response_text": response_text,
            "tool_output": tool_output,
        },
    )


def classify_groq_error(error_text: str) -> str:
    """Return a short user-facing hint for common Groq API failures."""
    normalized = error_text.upper()
    if "429" in normalized or "RATE" in normalized:
        return "Groq rate limit reached. Try again later or use the preview fallback."
    if "401" in normalized or "403" in normalized or "AUTH" in normalized or "API KEY" in normalized:
        return "Groq authentication failed. Check GROQ_API_KEY."
    if "404" in normalized or "NOT_FOUND" in normalized or "MODEL" in normalized:
        return "Groq model is unavailable or not found. Try another model."
    return f"Groq API failure: {error_text}"


def _default_client_factory(api_key: str) -> object:
    from groq import Groq

    return Groq(api_key=api_key)


def _build_inference_prompt(
    user_request: str,
    tool_schemas: Sequence[Mapping[str, object]],
    *,
    system_prompt: str,
    tool_context: str,
) -> str:
    return (
        f"{system_prompt.strip()}\n\n"
        f"Available tools:\n{tool_context.strip()}\n\n"
        "Select the single best tool for the user request. Return JSON only with this shape:\n"
        '{"selected_tool":"...","tool_arguments":{...},"assistant_answer":"..."}\n\n'
        "assistant_answer must be a concise natural-language response to the user. "
        "For calculator_tool, include the computed result if clear. For weather_tool or "
        "web_search_tool, do not invent live facts; say the agent would use that tool to "
        "retrieve the needed information. If no external data is needed, answer directly.\n\n"
        f"Valid selected_tool values: {', '.join(sorted(_valid_decision_names(tool_schemas)))}\n\n"
        f"User request:\n{user_request}"
    )


def _generate_json_selection(client: object, model_name: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a tool router. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    return str(response.choices[0].message.content or "")


def _tool_names(tool_schemas: Sequence[Mapping[str, object]]) -> set[str]:
    names = set()
    for schema in tool_schemas:
        function = schema.get("function", {})
        if isinstance(function, Mapping) and isinstance(function.get("name"), str):
            names.add(function["name"])
    return names


def _valid_decision_names(tool_schemas: Sequence[Mapping[str, object]]) -> set[str]:
    """Return valid selected_tool values: executable tools plus the no_tool decision.

    ``no_tool`` is a decision candidate, not an executable function, so it is
    never part of ``tool_schemas`` itself but must still be a selectable answer
    when no external tool is needed.
    """
    return _tool_names(tool_schemas) | {NO_TOOL_NAME}


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


def _build_assistant_answer(
    user_request: str,
    tool_name: str,
    arguments: Mapping[str, object],
) -> str:
    if tool_name == "calculator_tool":
        expression = str(arguments.get("expression", ""))
        result = _safe_eval_arithmetic(expression)
        if result != "unable to evaluate demo expression":
            return f"The result is {result}."
        return "I would use the calculator tool to evaluate that expression."
    if tool_name == "weather_tool":
        location = str(arguments.get("location", "the requested location"))
        date_or_time = arguments.get("date_or_time", arguments.get("date", "the requested time"))
        return (
            f"I would use the weather tool to retrieve the forecast for {location} "
            f"at {date_or_time}."
        )
    if tool_name == "web_search_tool":
        query = str(arguments.get("query", user_request))
        return f"I would use web search to retrieve current information for: {query}."
    if tool_name == "no_tool":
        return f"I can answer directly: {user_request}"
    return user_request


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
