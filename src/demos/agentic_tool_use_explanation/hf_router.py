"""Local HuggingFace structured router for the agentic tool-use demo."""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

try:
    from demos.agentic_tool_use_explanation.scorers import (
        CalibratedToolLogOddsScorer,
        ToolChoice,
        build_native_tool_call_prompt,
    )
except ModuleNotFoundError:
    from scorers import CalibratedToolLogOddsScorer, ToolChoice, build_native_tool_call_prompt

try:
    from demos.agentic_tool_use_explanation.tool_schemas import (
        NO_TOOL_NAME,
        get_executable_tool_schemas,
    )
except ModuleNotFoundError:
    from tool_schemas import NO_TOOL_NAME, get_executable_tool_schemas

DEFAULT_LOCAL_HF_ROUTER_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"

LOGGER = logging.getLogger(__name__)
DEFAULT_NATIVE_HF_MAX_NEW_TOKENS = 512
_SENTINEL_PREFIX_RE = re.compile(
    r"^\s*(?:no_tool|notool|no tool)\s*(?:\r?\n)+",
    flags=re.IGNORECASE,
)
_SENTINEL_ONLY_RE = re.compile(
    r"^\s*(?:no_tool|notool|no tool)\s*$",
    flags=re.IGNORECASE,
)

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
    selected_tool: str | None
    tool_arguments: dict[str, Any]
    raw_response: str
    debug_prompt: str | None = None
    parse_error: str | None = None
    direct_answer: str | None = None
    raw_response_original: str | None = None
    normalized_response_used_for_parsing: str | None = None
    extracted_tool_call_json: str | None = None
    cleaned_direct_answer: str | None = None
    removed_direct_answer_sentinel: bool = False
    generation_parameters: dict[str, Any] | None = None


class NativeToolCallParseError(ValueError):
    """Raised when native output contains malformed tool-call structure."""

    def __init__(self, message: str, *, extracted_tool_call_json: str | None = None) -> None:
        super().__init__(message)
        self.extracted_tool_call_json = extracted_tool_call_json


@dataclass(frozen=True)
class ParsedNativeToolCall:
    """Validated native tool-call payload."""

    name: str
    arguments: dict[str, Any]
    extracted_json: str


def normalize_native_tool_output(raw_response: str) -> str:
    """Remove only known trailing generation artifacts before native parsing."""
    normalized = raw_response.strip()
    while normalized.endswith("<|im_end|>"):
        normalized = normalized[: -len("<|im_end|>")].rstrip()
    return normalized


def clean_direct_answer(text: str) -> tuple[str, bool]:
    """Remove a leading standalone internal sentinel from direct answers only."""
    cleaned = _SENTINEL_PREFIX_RE.sub("", text, count=1).strip()
    return cleaned, cleaned != text.strip()


def is_internal_sentinel_only(text: str) -> bool:
    """Return True when the model emitted only an internal direct-answer sentinel."""
    return bool(_SENTINEL_ONLY_RE.fullmatch(text))


class LocalHFRouter:
    """Route tool-use requests with a local HuggingFace causal language model."""

    def __init__(
        self,
        model_name: str = DEFAULT_LOCAL_HF_ROUTER_MODEL_ID,
        max_new_tokens: int = DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
        *,
        trust_remote_code: bool = False,
        quantization_mode: str = "none",
        device: str | None = "auto",
        dtype: str = "auto",
        generation_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.trust_remote_code = trust_remote_code
        self.requested_quantization_mode = quantization_mode
        self.actual_quantization_mode = "none"
        self.requested_device = device or "auto"
        self.requested_dtype = dtype
        self.generation_kwargs = dict(generation_kwargs or {})
        self.tokenizer_id = model_name
        self.tokenizer_lock = threading.RLock()

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
        if quantization_mode != "none":
            msg = (
                f"Local HF native routing does not currently support quantization mode "
                f"{quantization_mode!r}. Use 'none' so routing and XAI scoring remain "
                "on the same supported model configuration."
            )
            raise RuntimeError(msg)
        # Resolve a single concrete device up front -- the same pattern used by
        # CalibratedToolLogOddsScorer -- instead of accelerate's device_map="auto"
        # dispatch. device_map="auto" is CUDA/multi-GPU oriented; on an
        # MPS-only Mac it can plan a fragile multi-device (or CPU-offloaded)
        # placement instead of the simple, verified-working single-device
        # `.to("mps")` move, and is not needed here since this router is a
        # single dense causal LM sized to fit on one device.
        if self.requested_device not in {"auto", "cuda", "mps", "cpu"}:
            msg = f"Unsupported local HF device setting: {self.requested_device!r}."
            raise RuntimeError(msg)
        if self.requested_device == "auto":
            if torch.cuda.is_available():
                resolved_device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        else:
            resolved_device = self.requested_device
        if dtype != "auto":
            model_dtype = getattr(torch, dtype)
        elif resolved_device == "cuda":
            model_dtype = torch.bfloat16
        elif resolved_device == "mps":
            model_dtype = torch.float16
        else:
            model_dtype = torch.float32
        self.device = resolved_device
        self.dtype = str(model_dtype).removeprefix("torch.")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=model_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=trust_remote_code,
            )
            self.model.to(resolved_device)
        except Exception as error:
            msg = (
                f"Could not load local HuggingFace router model {model_name!r}. "
                "Check that the model is available locally or that your environment can "
                f"download it. Details: {error}"
            )
            raise RuntimeError(msg) from error

        with self.tokenizer_lock:
            if getattr(self.tokenizer, "pad_token", None) is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()

    def choose_tool(
        self,
        user_request: str,
        tool_descriptions: Mapping[str, str],
        *,
        system_prompt: str = "",
    ) -> RouterDecision:
        """Choose the best demo tool with the model's native tool-call format."""
        del tool_descriptions
        debug_prompt, raw_response, generation_parameters = self._generate_native_tool_response(
            user_request,
            system_prompt=system_prompt,
        )
        decision = self.parse_native_response(raw_response, debug_prompt=debug_prompt)
        decision.generation_parameters = generation_parameters
        return decision

    def _tokenizer_access_lock(self) -> threading.RLock:
        """Return this cached router's shared tokenizer/model lock."""
        lock = getattr(self, "tokenizer_lock", None)
        if lock is None:  # Compatibility with lightweight subclasses in tests.
            lock = threading.RLock()
            self.tokenizer_lock = lock
        return lock

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
    def parse_native_response(
        cls,
        raw_response: str,
        *,
        debug_prompt: str | None = None,
    ) -> RouterDecision:
        """Parse a model-native tool-call response into the demo decision shape."""
        normalized_response = normalize_native_tool_output(raw_response)
        try:
            parsed_tool_call = cls._parse_native_tool_call(normalized_response)
        except NativeToolCallParseError as error:
            return RouterDecision(
                agent_response="",
                selected_tool=None,
                tool_arguments={},
                raw_response=raw_response,
                debug_prompt=debug_prompt,
                parse_error=str(error),
                direct_answer=None,
                raw_response_original=raw_response,
                normalized_response_used_for_parsing=normalized_response,
                extracted_tool_call_json=getattr(error, "extracted_tool_call_json", None),
            )

        if parsed_tool_call is None:
            if is_internal_sentinel_only(normalized_response):
                return RouterDecision(
                    agent_response="",
                    selected_tool=None,
                    tool_arguments={},
                    raw_response=raw_response,
                    debug_prompt=debug_prompt,
                    parse_error=(
                        "The model emitted only an internal routing label instead of a "
                        "natural-language answer."
                    ),
                    direct_answer=None,
                    raw_response_original=raw_response,
                    normalized_response_used_for_parsing=normalized_response,
                    extracted_tool_call_json=None,
                    cleaned_direct_answer=None,
                    removed_direct_answer_sentinel=False,
                )
            direct_answer, removed_sentinel = clean_direct_answer(normalized_response)
            if not direct_answer:
                return RouterDecision(
                    agent_response="",
                    selected_tool=None,
                    tool_arguments={},
                    raw_response=raw_response,
                    debug_prompt=debug_prompt,
                    parse_error="The model did not emit a usable natural-language answer.",
                    direct_answer=None,
                    raw_response_original=raw_response,
                    normalized_response_used_for_parsing=normalized_response,
                    extracted_tool_call_json=None,
                    cleaned_direct_answer=None,
                    removed_direct_answer_sentinel=removed_sentinel,
                )
            return RouterDecision(
                agent_response=direct_answer,
                selected_tool=NO_TOOL_NAME,
                tool_arguments={},
                raw_response=raw_response,
                debug_prompt=debug_prompt,
                parse_error=None,
                direct_answer=direct_answer,
                raw_response_original=raw_response,
                normalized_response_used_for_parsing=normalized_response,
                extracted_tool_call_json=None,
                cleaned_direct_answer=direct_answer,
                removed_direct_answer_sentinel=removed_sentinel,
            )

        return RouterDecision(
            agent_response=f"I would use {parsed_tool_call.name} for this request.",
            selected_tool=parsed_tool_call.name,
            tool_arguments=parsed_tool_call.arguments,
            raw_response=raw_response,
            debug_prompt=debug_prompt,
            parse_error=None,
            direct_answer=None,
            raw_response_original=raw_response,
            normalized_response_used_for_parsing=normalized_response,
            extracted_tool_call_json=parsed_tool_call.extracted_json,
        )

    @classmethod
    def _parse_native_tool_call(cls, raw_response: str) -> ParsedNativeToolCall | None:
        text = raw_response.strip()
        if not text:
            return None

        opening_count = len(re.findall(r"<tool_call\s*>", text))
        closing_count = text.count("</tool_call>")
        if opening_count == 0 and closing_count == 0:
            return None
        if opening_count != closing_count:
            msg = "Model output contained an incomplete <tool_call> block."
            raise NativeToolCallParseError(msg)
        if opening_count > 1:
            msg = "Model output contained multiple <tool_call> blocks; only one is supported."
            raise NativeToolCallParseError(msg)

        tool_call_match = re.search(
            r"<tool_call\s*>\s*(.*?)\s*</tool_call>",
            text,
            re.DOTALL,
        )
        if tool_call_match is None:
            msg = "Model output contained tool-call markers but no parseable block."
            raise NativeToolCallParseError(msg)

        json_text = tool_call_match.group(1).strip()
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as error:
            msg = f"Native tool-call JSON could not be parsed: {error.msg}."
            raise NativeToolCallParseError(msg, extracted_tool_call_json=json_text) from error
        if not isinstance(payload, Mapping):
            msg = "Native tool-call payload must be a JSON object."
            raise NativeToolCallParseError(msg, extracted_tool_call_json=json_text)

        name = payload.get("name")
        if not isinstance(name, str):
            msg = "Native tool-call payload must contain a string 'name'."
            raise NativeToolCallParseError(msg, extracted_tool_call_json=json_text)
        executable_tools = {
            str(schema["function"]["name"]) for schema in get_executable_tool_schemas()
        }
        if name not in executable_tools:
            msg = f"Native tool-call payload referenced unknown tool {name!r}."
            raise NativeToolCallParseError(msg, extracted_tool_call_json=json_text)

        arguments = payload.get("arguments", {})
        if not isinstance(arguments, Mapping):
            msg = "Native tool-call 'arguments' must be a JSON object when present."
            raise NativeToolCallParseError(msg, extracted_tool_call_json=json_text)
        return ParsedNativeToolCall(name=name, arguments=dict(arguments), extracted_json=json_text)

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
        with self._tokenizer_access_lock():
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

    def _generate_native_tool_response(
        self,
        user_request: str,
        *,
        system_prompt: str,
    ) -> tuple[str, str, dict[str, Any]]:
        with self._tokenizer_access_lock():
            model_prompt = self._format_native_tool_prompt(
                user_request, system_prompt=system_prompt
            )
            input_device = self._input_device()
            inputs = self.tokenizer(model_prompt, return_tensors="pt")
            inputs = {key: value.to(input_device) for key, value in inputs.items()}
            prompt_token_count = int(inputs["input_ids"].shape[-1])

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            }
            generation_kwargs.update(self.generation_kwargs)
            if generation_kwargs.get("do_sample") is False:
                for sampling_key in ("temperature", "top_p", "top_k"):
                    generation_kwargs.pop(sampling_key, None)
            if "pad_token_id" not in generation_kwargs and self.tokenizer.eos_token_id is not None:
                generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

            with self._torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generation_kwargs)
            new_token_ids = output_ids[0, prompt_token_count:]
            raw_response = self.tokenizer.decode(new_token_ids, skip_special_tokens=False).strip()
            return model_prompt, raw_response, dict(generation_kwargs)

    def _format_native_tool_prompt(self, user_request: str, *, system_prompt: str) -> str:
        with self._tokenizer_access_lock():
            return build_native_tool_call_prompt(
                self.tokenizer,
                system_prompt=system_prompt,
                user_request=user_request,
                tool_schemas=get_executable_tool_schemas(),
            )

    def _format_model_prompt(self, router_prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a strict tool router. Return JSON only."},
            {"role": "user", "content": router_prompt},
        ]
        with self._tokenizer_access_lock():
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
                    LOGGER.debug(
                        "Falling back after chat-template formatting failed.", exc_info=True
                    )
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


HF_SELECTION_MODES = (
    "calibrated",
    "raw_margin_gate",
    "scaled_calibration",
    "no_tool_boost",
)


def select_tool_from_scores(
    raw_scores: Mapping[str, float],
    calibrated_scores: Mapping[str, float],
    *,
    mode: str = "no_tool_boost",
    raw_margin_threshold: float = 1.0,
    calibration_strength: float = 1.0,
    no_tool_boost_delta: float = 0.75,
) -> tuple[str, dict[str, float]]:
    """Select a tool with an isolated, optional confidence-aware adjustment.

    ``no_tool_boost`` is the default production behavior. ``calibrated``
    preserves the unadjusted calibrated evidence, while ``raw_margin_gate``
    trusts the raw argmax when its top-vs-runner-up margin reaches the supplied
    threshold. ``scaled_calibration`` applies only a fraction of the inferred
    baseline correction: ``raw - strength * (raw - calibrated)``.
    ``no_tool_boost`` adds a fixed delta to only the calibrated ``no_tool``
    score before recomputing the argmax.
    """
    if mode not in HF_SELECTION_MODES:
        msg = f"Unknown HF selection mode {mode!r}; expected one of {HF_SELECTION_MODES!r}."
        raise ValueError(msg)
    if raw_scores.keys() != calibrated_scores.keys():
        msg = "Raw and calibrated scores must contain identical candidates."
        raise ValueError(msg)
    if not raw_scores:
        msg = "At least one routing candidate is required."
        raise ValueError(msg)
    if raw_margin_threshold < 0:
        msg = "raw_margin_threshold must be non-negative."
        raise ValueError(msg)
    if not 0.0 <= calibration_strength <= 1.0:
        msg = "calibration_strength must be between 0 and 1."
        raise ValueError(msg)
    if mode == "no_tool_boost" and no_tool_boost_delta < 0:
        msg = "no_tool_boost_delta must be non-negative."
        raise ValueError(msg)

    raw = dict(raw_scores)
    calibrated = dict(calibrated_scores)
    if mode == "calibrated":
        decision_scores = calibrated
    elif mode == "scaled_calibration":
        if calibration_strength == 0.0:
            decision_scores = raw
        elif calibration_strength == 1.0:
            decision_scores = calibrated
        else:
            decision_scores = {
                tool_name: raw[tool_name]
                - calibration_strength * (raw[tool_name] - calibrated[tool_name])
                for tool_name in raw
            }
    elif mode == "raw_margin_gate":
        ranked_raw = sorted(raw.values(), reverse=True)
        raw_margin = ranked_raw[0] - ranked_raw[1] if len(ranked_raw) > 1 else float("inf")
        decision_scores = raw if raw_margin >= raw_margin_threshold else calibrated
    else:
        if "no_tool" not in calibrated:
            msg = "no_tool_boost mode requires a 'no_tool' candidate."
            raise ValueError(msg)
        decision_scores = dict(calibrated)
        decision_scores["no_tool"] += no_tool_boost_delta
    return max(decision_scores, key=decision_scores.get), decision_scores


class LocalHFClassificationRouter:
    """Route tool-use requests with the exact A/B/C/D protocol used by the scorer.

    Unlike ``LocalHFRouter`` (structured JSON tool-calling, used for real
    execution and argument extraction), this router selects a tool by scoring
    every routing-label continuation under ``build_routing_classification_prompt``
    -- the identical canonical protocol used by
    ``scorers.CalibratedToolLogOddsScorer`` -- and taking the argmax of the
    *calibrated* evidence ``r_t(N) = g_t(N) - b_t`` (the same per-label
    calibration baseline subtracted for every Shapley coalition), not the raw
    label likelihood ``g_t(N)``. It is the only target-tool selector that may
    be presented as "the decision explained by" the calibrated multiclass tool
    log-odds (HF local) scorer, since the scorer's own coalition values are
    computed from that same calibrated evidence.

    This router does not execute tools and does not extract call arguments; it
    wraps an already-loaded ``CalibratedToolLogOddsScorer`` so no separate model
    is loaded for routing versus scoring.
    """

    def __init__(
        self,
        scorer: CalibratedToolLogOddsScorer,
        *,
        selection_mode: str = "no_tool_boost",
        raw_margin_threshold: float = 1.0,
        calibration_strength: float = 1.0,
        no_tool_boost_delta: float = 0.75,
    ) -> None:
        self.scorer = scorer
        self.selection_mode = selection_mode
        self.raw_margin_threshold = raw_margin_threshold
        self.calibration_strength = calibration_strength
        self.no_tool_boost_delta = no_tool_boost_delta

    def choose_tool(
        self,
        user_request: str,
        *,
        system_prompt: str,
        tool_descriptions: Mapping[str, str],
    ) -> ToolChoice:
        """Return the calibrated-evidence argmax decision for one full user request."""
        raw_score_method = getattr(self.scorer, "score_full_request_labels", None)
        raw_scores = (
            raw_score_method(
                user_request,
                system_prompt=system_prompt,
                tool_descriptions=tool_descriptions,
            )
            if callable(raw_score_method)
            else None
        )
        calibrated_scores = self.scorer.score_full_request_calibrated_labels(
            user_request,
            system_prompt=system_prompt,
            tool_descriptions=tool_descriptions,
        )
        if raw_scores is None:  # Compatibility with lightweight test doubles.
            raw_scores = calibrated_scores
        raw_argmax = max(raw_scores, key=raw_scores.get)
        calibrated_argmax = max(calibrated_scores, key=calibrated_scores.get)
        selected_tool, decision_scores = select_tool_from_scores(
            raw_scores,
            calibrated_scores,
            mode=self.selection_mode,
            raw_margin_threshold=self.raw_margin_threshold,
            calibration_strength=self.calibration_strength,
            no_tool_boost_delta=self.no_tool_boost_delta,
        )
        calibration_debug = {
            tool_name: raw_scores[tool_name] - calibrated_scores[tool_name]
            for tool_name in calibrated_scores
        }
        LOGGER.debug(
            "route request=%r raw_scores=%s calibrated_scores=%s "
            "calibration_baseline_b_t=%s raw_argmax=%s calibrated_argmax=%s "
            "selection_mode=%r selected_tool=%s",
            user_request[:60],
            raw_scores,
            calibrated_scores,
            calibration_debug,
            raw_argmax,
            calibrated_argmax,
            self.selection_mode,
            selected_tool,
        )
        label = self.scorer.routing_labels.get(selected_tool, "?")
        return ToolChoice(
            tool=selected_tool,
            score=decision_scores[selected_tool],
            reason=(
                f"Highest routing evidence under the {self.selection_mode!r} "
                f"A/B/C/D selection policy (label {label!r})."
            ),
            scores=dict(decision_scores),
        )


class HFArgumentExtractor:
    """Second-stage structured-JSON tool-argument extraction.

    Reuses an already-loaded model/tokenizer/device -- e.g. the ones owned by
    a ``CalibratedToolLogOddsScorer`` -- instead of loading an independent
    model. This is deliberately separate from the A/B/C/D routing
    classification protocol: routing decides *which* tool via
    ``build_routing_classification_prompt`` and the scorer's calibrated
    log-odds; this extractor only runs once routing is already decided, to
    fill in call arguments (e.g. ``weather_tool``'s location/date) via a
    structured JSON generation prompt. Its output is a second-stage,
    display-only convenience and must never enter the Shapley/k-SII payoff.
    """

    def __init__(
        self,
        *,
        model: object,
        tokenizer: object,
        device: str,
        tokenizer_lock: threading.RLock | None = None,
        max_new_tokens: int = 128,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.tokenizer_lock = tokenizer_lock or threading.RLock()
        self.max_new_tokens = max_new_tokens

    def extract_arguments(
        self,
        *,
        user_request: str,
        selected_tool: str,
        tool_descriptions: Mapping[str, str],
    ) -> dict[str, object]:
        """Return structured JSON call arguments for one already-selected tool.

        Returns an empty dict for ``no_tool`` (no arguments to extract) and
        whenever generation does not yield parseable JSON, rather than raising
        -- this is a best-effort display convenience, not a routing decision.
        """
        if selected_tool == NO_TOOL_NAME:
            return {}
        prompt = self._build_argument_prompt(user_request, selected_tool, tool_descriptions)
        raw_response = self._generate(prompt)
        payload = LocalHFRouter._parse_json_payload(raw_response)  # noqa: SLF001
        if payload is None:
            return {}
        raw_arguments = payload.get("tool_arguments", {})
        return dict(raw_arguments) if isinstance(raw_arguments, Mapping) else {}

    @staticmethod
    def _build_argument_prompt(
        user_request: str,
        selected_tool: str,
        tool_descriptions: Mapping[str, str],
    ) -> str:
        """Build the argument-extraction prompt, kept separate from the routing prompt."""
        description = tool_descriptions.get(selected_tool, "")
        return (
            "Extract call arguments for exactly one already-selected tool. The tool "
            "choice is final; do not reconsider or return a different tool.\n\n"
            f"Selected tool: {selected_tool}\n"
            f"Tool description: {description}\n\n"
            "Return only valid JSON with this exact shape:\n"
            '{"tool_arguments": {...}}\n\n'
            "For weather_tool, include location/date fields when available.\n"
            "For calculator_tool, include an expression field when available.\n"
            "For web_search_tool, include a query field when available.\n"
            "Do not include any other keys or text.\n\n"
            f"User request:\n{user_request.strip()}"
        )

    def _generate(self, prompt: str) -> str:
        with self.tokenizer_lock:
            model_prompt = self._format_model_prompt(prompt)
            inputs = self.tokenizer(model_prompt, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            prompt_token_count = int(inputs["input_ids"].shape[-1])

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
            }
            if getattr(self.tokenizer, "eos_token_id", None) is not None:
                generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

            import torch

            with torch.inference_mode():
                output_ids = self.model.generate(**inputs, **generation_kwargs)
            new_token_ids = output_ids[0, prompt_token_count:]
            return str(self.tokenizer.decode(new_token_ids, skip_special_tokens=True)).strip()

    def _format_model_prompt(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "You are a strict tool-argument extractor. Return JSON only.",
            },
            {"role": "user", "content": prompt},
        ]
        with self.tokenizer_lock:
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
                    LOGGER.debug(
                        "Falling back after chat-template formatting failed.", exc_info=True
                    )
        return (
            "System:\nYou are a strict tool-argument extractor. Return JSON only.\n\n"
            f"User:\n{prompt}\n\n"
            "Assistant:"
        )
