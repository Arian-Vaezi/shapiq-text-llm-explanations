"""Local HuggingFace structured router for the agentic tool-use demo."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

try:
    from demos.agentic_tool_use_explanation.tool_schemas import NO_TOOL_NAME
except ModuleNotFoundError:
    from tool_schemas import NO_TOOL_NAME

DEFAULT_LOCAL_HF_ROUTER_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

LOGGER = logging.getLogger(__name__)

ALLOWED_TOOLS: tuple[str, ...] = (
    "calculator_tool",
    "weather_tool",
    "web_search_tool",
    NO_TOOL_NAME,
)

ROUTER_TOOL_DESCRIPTIONS: dict[str, str] = {
    "calculator_tool": "exact arithmetic or mathematical calculations",
    "weather_tool": "weather forecasts or current weather",
    "web_search_tool": "current, latest, recent, live, or external information",
    NO_TOOL_NAME: "stable information that does not require an external lookup",
}


@dataclass
class RouterDecision:
    """Structured local-router decision."""

    agent_response: str
    selected_tool: str
    tool_arguments: dict[str, Any]
    raw_response: str
    debug_prompt: str | None = None


class LocalHFRouter:
    """Route tool-use requests with a local HuggingFace causal language model."""

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_HF_ROUTER_MODEL_ID,
        max_new_tokens: int = 256,
        *,
        trust_remote_code: bool = False,
        generation_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self.generation_kwargs = dict(generation_kwargs or {})

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            msg = (
                "Local HuggingFace routing requires the optional dependencies "
                "`torch` and `transformers`. Install them to use the HF local backend."
            )
            raise RuntimeError(msg) from error

        self._torch = torch
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
            )
        except Exception as error:
            msg = (
                f"Could not load local HuggingFace router model {model_name!r}. "
                "Check that the model is available locally or that your environment can "
                f"download it. Details: {error}"
            )
            raise RuntimeError(msg) from error

        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def choose_tool(
        self,
        user_request: str,
        tool_descriptions: Mapping[str, str],
    ) -> RouterDecision:
        """Choose the best demo tool for one user request."""
        router_prompt = self.build_router_prompt(user_request, tool_descriptions)
        debug_prompt, raw_response = self._generate_raw_response(router_prompt)
        return self.parse_response(raw_response, debug_prompt=debug_prompt)

    @classmethod
    def build_router_prompt(
        cls,
        user_request: str,
        tool_descriptions: Mapping[str, str],
    ) -> str:
        """Build the strict JSON router prompt used before chat-template formatting."""
        tool_lines = []
        for tool_name in ALLOWED_TOOLS:
            description = ROUTER_TOOL_DESCRIPTIONS[tool_name]
            extra_description = tool_descriptions.get(tool_name)
            if extra_description and extra_description != description:
                description = f"{description}. Demo description: {extra_description}"
            tool_lines.append(f"- {tool_name}: {description}")

        return (
            "Choose exactly one tool for the user request.\n\n"
            "Available tools:\n"
            f"{chr(10).join(tool_lines)}\n\n"
            "Return only valid JSON with this exact shape:\n"
            "{\n"
            '  "agent_response": "...",\n'
            '  "selected_tool": "...",\n'
            '  "tool_arguments": {...}\n'
            "}\n\n"
            f"Valid selected_tool values: {', '.join(ALLOWED_TOOLS)}.\n"
            "For weather_tool, include location/date fields when available.\n"
            "For calculator_tool, include an expression field when available.\n"
            "For web_search_tool, include a query field when available.\n"
            "For no_tool, use an empty tool_arguments object.\n"
            "Do not execute any tool and do not invent live facts.\n\n"
            f"User request:\n{user_request.strip()}"
        )

    @classmethod
    def parse_response(
        cls,
        raw_response: str,
        *,
        debug_prompt: str | None = None,
    ) -> RouterDecision:
        """Parse model output into a RouterDecision with keyword fallback."""
        payload = cls._parse_json_payload(raw_response)
        if payload is None:
            selected_tool = cls._infer_tool_from_text(raw_response)
            tool_arguments: dict[str, Any] = {}
            agent_response = f"I would use {selected_tool} for this request."
        else:
            selected_tool = payload.get("selected_tool")
            if not isinstance(selected_tool, str) or selected_tool not in ALLOWED_TOOLS:
                selected_tool = NO_TOOL_NAME

            raw_arguments = payload.get("tool_arguments", {})
            tool_arguments = dict(raw_arguments) if isinstance(raw_arguments, Mapping) else {}

            agent_response = payload.get("agent_response")
            if not isinstance(agent_response, str) or not agent_response.strip():
                agent_response = f"I would use {selected_tool} for this request."

        return RouterDecision(
            agent_response=agent_response,
            selected_tool=selected_tool,
            tool_arguments=tool_arguments,
            raw_response=raw_response,
            debug_prompt=debug_prompt,
        )

    @classmethod
    def _parse_json_payload(cls, raw_response: str) -> dict[str, Any] | None:
        json_text = cls._extract_outermost_json_object(raw_response)
        if json_text is None:
            return None
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _extract_outermost_json_object(text: str) -> str | None:
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            character = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue

            if character == '"':
                in_string = True
            elif character == "{":
                depth += 1
            elif character == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        return None

    @staticmethod
    def _infer_tool_from_text(text: str) -> str:
        normalized = text.lower()
        if "web_search_tool" in normalized or "web search" in normalized:
            return "web_search_tool"
        if "calculator_tool" in normalized or "calculator" in normalized:
            return "calculator_tool"
        if "weather_tool" in normalized or "weather" in normalized:
            return "weather_tool"
        return NO_TOOL_NAME

    def _generate_raw_response(self, router_prompt: str) -> tuple[str, str]:
        model_prompt = self._format_model_prompt(router_prompt)
        input_device = self._input_device()
        inputs = self.tokenizer(model_prompt, return_tensors="pt")
        inputs = {key: value.to(input_device) for key, value in inputs.items()}
        prompt_token_count = int(inputs["input_ids"].shape[-1])

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": False,
        }
        generation_kwargs.update(self.generation_kwargs)
        if "pad_token_id" not in generation_kwargs and self.tokenizer.eos_token_id is not None:
            generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        with self._torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        new_token_ids = output_ids[0, prompt_token_count:]
        raw_response = self.tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
        return model_prompt, raw_response

    def _format_model_prompt(self, router_prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a strict tool router. Return JSON only."},
            {"role": "user", "content": router_prompt},
        ]
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                return str(
                    apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                )
            except Exception:  # noqa: BLE001
                LOGGER.debug("Falling back after chat-template formatting failed.", exc_info=True)
        return (
            "System:\nYou are a strict tool router. Return JSON only.\n\n"
            f"User:\n{router_prompt}\n\n"
            "Assistant:"
        )

    def _input_device(self) -> object:
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
