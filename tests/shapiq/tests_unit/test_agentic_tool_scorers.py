"""Tests for agentic tool-use demo scorers."""

from __future__ import annotations

import dataclasses
import inspect
import math
import sys
import types
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

import app as app_module  # noqa: E402
from hf_router import (  # noqa: E402
    DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
    LocalHFRouter,
    clean_direct_answer,
    is_internal_sentinel_only,
    normalize_native_tool_output,
)
from scorers import (  # noqa: E402
    CALIBRATION_USER_REQUESTS,
    ROUTING_LABELS,
    CalibratedToolLogOddsScorer,
    FrozenContinuationTarget,
    LexicalToolRouter,
    LexicalToolScorer,
    LLMToolScorer,
    MockLLM,
    NativeDirectAnswerScorer,
    NativeToolCallScorer,
    RoutingLabelTokenization,
    build_canonical_direct_answer_target,
    build_coalition_prompt,
    build_native_direct_answer_prompt,
    build_native_tool_call_continuation,
    build_native_tool_call_prompt,
    build_routing_classification_prompt,
    build_tool_calling_prompt,
    split_coalition_prompt,
    stable_logsumexp,
    validate_routing_label_tokens,
)
from tool_game import ToolUseGame, ToolUseSegment  # noqa: E402
from tool_schemas import (  # noqa: E402
    DECISION_NAMES,
    EXECUTABLE_TOOL_SCHEMAS,
    NO_TOOL_NAME,
    TOOL_DESCRIPTIONS,
    get_executable_tool_schemas,
    validate_tool_configuration,
)


def make_fake_calibrated_scorer(
    coalition_scores: dict[str, float],
    calibration_scores: dict[str, float],
    empty_scores: dict[str, float],
) -> CalibratedToolLogOddsScorer:
    """Build a CalibratedToolLogOddsScorer with a stubbed model-scoring layer.

    Real coalition prompts route to ``coalition_scores``; the calibration probe
    prompts (``CALIBRATION_USER_REQUESTS``) route to ``calibration_scores``; and
    the true empty-coalition prompt (empty user request) routes to
    ``empty_scores``. This lets tests exercise the real calibration/target-vs-all/
    empty-coalition-normalization logic without loading a model.
    """
    scorer = CalibratedToolLogOddsScorer.__new__(CalibratedToolLogOddsScorer)
    scorer.model_id = "fake-model-id"
    scorer.routing_labels = dict(ROUTING_LABELS)
    scorer.routing_label_separator = " "
    scorer.calibration_user_requests = CALIBRATION_USER_REQUESTS
    scorer.last_debug_outputs = []
    scorer._calibration_cache = {}
    scorer._empty_log_odds_cache = {}
    scorer._raw_score_row_cache = {}

    calibration_markers = tuple(
        f"User request:\n{request}\n\n" for request in CALIBRATION_USER_REQUESTS
    )
    empty_marker = "User request:\n\nDecision:"

    def fake_compute_label_log_scores_batched(
        prompts: list[str],
        candidate_tools: list[str],
    ) -> list[dict[str, float]]:
        del candidate_tools
        rows = []
        for prompt in prompts:
            if any(marker in prompt for marker in calibration_markers):
                rows.append(dict(calibration_scores))
            elif prompt.endswith(empty_marker):
                rows.append(dict(empty_scores))
            else:
                rows.append(dict(coalition_scores))
        return rows

    scorer._compute_label_log_scores_batched = fake_compute_label_log_scores_batched
    return scorer


class FakeGenerator:
    """Small deterministic generator for scorer tests."""

    def __init__(self, response: str) -> None:
        self.response = response

    def generate(self, prompt: str) -> str:
        del prompt
        return self.response


class RecordingTokenizer:
    """Fake tokenizer that records chat-template calls."""

    def __init__(self) -> None:
        self.calls = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tools: tuple[dict[str, object], ...],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        return "native chat prompt"


class RejectingToolsTokenizer:
    """Fake tokenizer without tools= support."""

    def apply_chat_template(self, *args, **kwargs) -> str:
        del args, kwargs
        msg = "apply_chat_template() got an unexpected keyword argument 'tools'"
        raise TypeError(msg)


class BrokenTokenizer:
    """Fake tokenizer that raises an unrelated error."""

    def apply_chat_template(self, *args, **kwargs) -> str:
        del args, kwargs
        msg = "tokenizer exploded"
        raise RuntimeError(msg)


class NativeTemplateTokenizer:
    """Small tokenizer that renders native prompts and assistant tool calls."""

    pad_token = "<pad>"
    eos_token = "</s>"
    eos_token_id = 0
    padding_side = "right"

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tools: tuple[dict[str, object], ...],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            }
        )
        rendered = "".join(
            f"<{message['role']}>{message.get('content', '')}" for message in messages
        )
        last_message = messages[-1]
        if "tool_calls" in last_message:
            tool_call = last_message["tool_calls"][0]
            rendered += f"<tool_call>{tool_call['function']['name']}"
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered


class CharacterTensorTokenizer(NativeTemplateTokenizer):
    """Tokenizer that maps characters to tensor ids for teacher-forcing tests."""

    def __call__(
        self,
        texts: list[str],
        *,
        return_tensors: str,
        add_special_tokens: bool,
        padding: bool,
    ) -> dict[str, object]:
        del return_tensors, add_special_tokens, padding
        import torch

        rows = [[ord(character) for character in text] for text in texts]
        max_len = max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        for row_index, row in enumerate(rows):
            input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[row_index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class ControlledLogitModel:
    """Fake causal LM that assigns known log-probabilities by sequence position."""

    def __init__(self, desired_logprobs_by_position: dict[int, float]) -> None:
        self.desired_logprobs_by_position = desired_logprobs_by_position

    def eval(self) -> None:
        return None

    def __call__(self, *, input_ids, attention_mask):
        del attention_mask
        import torch

        vocab_size = 256
        logits = torch.full(
            (input_ids.shape[0], input_ids.shape[1], vocab_size),
            -1000.0,
            dtype=torch.float32,
        )
        for row_index in range(input_ids.shape[0]):
            for position in range(input_ids.shape[1] - 1):
                target_token = int(input_ids[row_index, position + 1])
                desired = self.desired_logprobs_by_position.get(position, -0.1)
                competitor_probability = math.log1p(-math.exp(desired))
                logits[row_index, position, target_token] = desired
                logits[row_index, position, 1] = competitor_probability
        return types.SimpleNamespace(logits=logits)


class TensorLike:
    """Tiny object with the tensor methods used by LocalHFRouter generation."""

    shape = (1, 3)

    def to(self, device):
        del device
        return self

    def __getitem__(self, item):
        del item
        return [1, 2, 3]


class GenerationTokenizer(NativeTemplateTokenizer):
    """Tokenizer fake for LocalHFRouter._generate_native_tool_response."""

    eos_token_id = 0

    def __call__(self, prompt: str, *, return_tensors: str) -> dict[str, TensorLike]:
        del prompt, return_tensors
        return {"input_ids": TensorLike()}

    def decode(self, token_ids, *, skip_special_tokens: bool) -> str:
        del token_ids, skip_special_tokens
        return "A complete direct answer."


class RecordingGenerateModel:
    """Model fake that records generation kwargs."""

    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def parameters(self):
        return iter([types.SimpleNamespace(device="cpu")])

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return TensorLike()


def test_canonical_tool_schema_contents() -> None:
    validate_tool_configuration()
    schema_by_name = {schema["function"]["name"]: schema for schema in EXECUTABLE_TOOL_SCHEMAS}

    assert set(schema_by_name) == {
        "weather_tool",
        "calculator_tool",
        "web_search_tool",
    }
    assert NO_TOOL_NAME not in schema_by_name
    assert NO_TOOL_NAME in DECISION_NAMES
    assert len(DECISION_NAMES) == len(set(DECISION_NAMES))
    assert schema_by_name["weather_tool"]["function"]["parameters"]["required"] == ["location"]
    assert schema_by_name["calculator_tool"]["function"]["parameters"]["required"] == ["expression"]
    assert schema_by_name["web_search_tool"]["function"]["parameters"]["required"] == ["query"]


def test_derived_tool_descriptions_include_no_tool() -> None:
    descriptions_by_name = {
        schema["function"]["name"]: schema["function"]["description"]
        for schema in EXECUTABLE_TOOL_SCHEMAS
    }

    for tool_name, description in descriptions_by_name.items():
        assert TOOL_DESCRIPTIONS[tool_name] == description
    assert TOOL_DESCRIPTIONS[NO_TOOL_NAME] == "Answer directly without calling an external tool."


def test_build_tool_calling_prompt_uses_native_chat_template_tools() -> None:
    tokenizer = RecordingTokenizer()

    prompt = build_tool_calling_prompt(
        tokenizer,
        system_prompt="You are a tool router.",
        user_request="Will it rain in Berlin tomorrow?",
        tool_schemas=EXECUTABLE_TOOL_SCHEMAS,
    )

    assert prompt == "native chat prompt"
    assert len(tokenizer.calls) == 1
    call = tokenizer.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "You are a tool router."},
        {"role": "user", "content": "Will it rain in Berlin tomorrow?"},
    ]
    assert call["tools"] == get_executable_tool_schemas()
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True


def test_build_tool_calling_prompt_keeps_empty_user_message() -> None:
    tokenizer = RecordingTokenizer()

    build_tool_calling_prompt(
        tokenizer,
        system_prompt="You are a tool router.",
        user_request="",
        tool_schemas=EXECUTABLE_TOOL_SCHEMAS,
    )

    call = tokenizer.calls[0]
    assert call["messages"][0] == {"role": "system", "content": "You are a tool router."}
    assert call["messages"][1] == {"role": "user", "content": ""}
    assert call["tools"] == get_executable_tool_schemas()


def test_build_native_tool_call_prompt_uses_structured_schemas_and_instruction() -> None:
    tokenizer = NativeTemplateTokenizer()

    prompt = build_native_tool_call_prompt(
        tokenizer,
        system_prompt="Route carefully.",
        user_request="Who won the latest Formula 1 race?",
    )

    assert prompt.endswith("<assistant>")
    call = tokenizer.calls[0]
    assert call["tools"] == get_executable_tool_schemas()
    assert call["tokenize"] is False
    assert call["add_generation_prompt"] is True
    system_message = call["messages"][0]
    assert "If an external tool is required" in system_message["content"]
    assert "without emitting a <tool_call> block" in system_message["content"]


def test_native_prompt_does_not_tell_model_to_use_no_tool() -> None:
    tokenizer = NativeTemplateTokenizer()

    build_native_tool_call_prompt(
        tokenizer,
        system_prompt="Answer stable conceptual explanations directly without calling a tool.",
        user_request="Explain precision and recall.",
    )

    system_content = str(tokenizer.calls[0]["messages"][0]["content"])
    assert "Use no_tool" not in system_content
    assert "stable conceptual questions" in system_content
    assert "answer the user directly in natural language" in system_content
    assert "Never output the literal strings no_tool" in system_content
    assert "NoTool" in system_content


def test_no_tool_is_not_registered_as_executable_schema() -> None:
    executable_names = {schema["function"]["name"] for schema in get_executable_tool_schemas()}

    assert "no_tool" not in executable_names


def test_build_native_tool_call_continuation_uses_chat_template_through_tool_name() -> None:
    tokenizer = NativeTemplateTokenizer()

    continuation = build_native_tool_call_continuation(
        tokenizer,
        system_prompt="Route carefully.",
        user_request="Will it rain in Berlin tomorrow?",
        target_tool="weather_tool",
    )

    assert continuation == "<tool_call>weather_tool"
    assert tokenizer.calls[-1]["add_generation_prompt"] is False
    assert tokenizer.calls[-1]["tools"] == get_executable_tool_schemas()


def test_native_hf_router_parses_valid_tool_call_and_arguments() -> None:
    raw_response = (
        '<tool_call>{"name": "weather_tool", "arguments": {"location": "Berlin"}}</tool_call>'
    )

    decision = LocalHFRouter.parse_native_response(raw_response)

    assert decision.selected_tool == "weather_tool"
    assert decision.tool_arguments == {"location": "Berlin"}
    assert decision.parse_error is None


def test_native_hf_router_parses_multiline_tool_call_with_trailing_im_end() -> None:
    raw_response = (
        "<tool_call>\n"
        '{"name": "web_search_tool", "arguments": {"query": "role of CEO"}}\n'
        "</tool_call><|im_end|>"
    )

    decision = LocalHFRouter.parse_native_response(raw_response)

    assert decision.selected_tool == "web_search_tool"
    assert decision.tool_arguments == {"query": "role of CEO"}
    assert decision.parse_error is None
    assert decision.raw_response_original == raw_response
    assert decision.normalized_response_used_for_parsing == raw_response.removesuffix("<|im_end|>")
    assert decision.extracted_tool_call_json == (
        '{"name": "web_search_tool", "arguments": {"query": "role of CEO"}}'
    )
    assert decision.extracted_tool_call_json.endswith("}}")


def test_native_hf_router_parses_multiple_argument_fields() -> None:
    raw_response = (
        "<tool_call>\n"
        "{\n"
        '  "name": "weather_tool",\n'
        '  "arguments": {\n'
        '    "location": "Berlin",\n'
        '    "date": "tomorrow"\n'
        "  }\n"
        "}\n"
        "</tool_call>"
    )

    decision = LocalHFRouter.parse_native_response(raw_response)

    assert decision.selected_tool == "weather_tool"
    assert decision.tool_arguments == {
        "location": "Berlin",
        "date": "tomorrow",
    }
    assert decision.parse_error is None


def test_native_hf_router_parses_nested_argument_value_without_truncation() -> None:
    raw_response = (
        "<tool_call>\n"
        '{"name": "web_search_tool", "arguments": {"query": "CEO role", '
        '"filters": {"region": "global", "types": ["overview", "definition"]}}}\n'
        "</tool_call>"
    )

    decision = LocalHFRouter.parse_native_response(raw_response)

    assert decision.selected_tool == "web_search_tool"
    assert decision.tool_arguments == {
        "query": "CEO role",
        "filters": {
            "region": "global",
            "types": ["overview", "definition"],
        },
    }
    assert decision.parse_error is None
    assert decision.extracted_tool_call_json is not None
    assert '"filters": {"region": "global", "types": ["overview", "definition"]}' in (
        decision.extracted_tool_call_json
    )


def test_native_hf_router_parses_tool_call_with_whitespace_around_json() -> None:
    raw_response = (
        '  <tool_call> \n  {"name": "calculator_tool", '
        '"arguments": {"expression": "238 * 47"}} \n </tool_call>  '
    )

    decision = LocalHFRouter.parse_native_response(raw_response)

    assert decision.selected_tool == "calculator_tool"
    assert decision.tool_arguments == {"expression": "238 * 47"}


def test_normalize_native_tool_output_only_strips_known_trailing_artifacts() -> None:
    raw_response = "  answer <|not_special|>  <|im_end|>  "

    assert normalize_native_tool_output(raw_response) == "answer <|not_special|>"


def test_native_hf_router_maps_direct_answer_to_no_tool_and_preserves_text() -> None:
    raw_response = "Precision measures exactness; recall measures coverage."

    decision = LocalHFRouter.parse_native_response(raw_response)

    assert decision.selected_tool == "no_tool"
    assert decision.tool_arguments == {}
    assert decision.agent_response == raw_response
    assert decision.parse_error is None
    assert decision.direct_answer == raw_response


@pytest.mark.parametrize(
    "raw_response",
    [
        "No_tool\n\nPrecision and recall are different metrics.",
        "no_tool\nPhotosynthesis converts light into chemical energy.",
        "no tool\r\nA CEO sets strategy and leads executives.",
        "NoTool\nA model is a simplified representation.",
    ],
)
def test_native_hf_router_cleans_leading_direct_answer_sentinel(raw_response: str) -> None:
    decision = LocalHFRouter.parse_native_response(raw_response)

    assert decision.selected_tool == "no_tool"
    assert decision.parse_error is None
    assert decision.direct_answer is not None
    assert not decision.direct_answer.lower().startswith(("no_tool", "notool", "no tool"))
    assert decision.removed_direct_answer_sentinel is True


def test_direct_answer_cleanup_preserves_middle_no_tool_text() -> None:
    raw_response = "The literal string no_tool is an internal label in this demo."

    cleaned, removed = clean_direct_answer(raw_response)

    assert cleaned == raw_response
    assert removed is False


@pytest.mark.parametrize("raw_response", ["No_tool", "no_tool", "NoTool", "no tool"])
def test_native_hf_router_rejects_sentinel_only_output(raw_response: str) -> None:
    decision = LocalHFRouter.parse_native_response(raw_response)

    assert is_internal_sentinel_only(raw_response) is True
    assert decision.selected_tool is None
    assert decision.direct_answer is None
    assert decision.parse_error is not None
    assert "internal routing label" in decision.parse_error


def test_native_hf_router_does_not_keyword_fallback_when_native_parse_fails() -> None:
    raw_response = "I might need weather_tool, but I am answering directly."

    decision = LocalHFRouter.parse_native_response(raw_response)

    assert decision.selected_tool == "no_tool"
    assert decision.tool_arguments == {}


@pytest.mark.parametrize(
    ("raw_response", "message"),
    [
        ('<tool_call>{"name": "web_search_tool" }', "incomplete"),
        ('<tool_call>{"name": "web_search_tool", bad json}</tool_call>', "JSON"),
        ('<tool_call>{"name": "calendar_tool", "arguments": {}}</tool_call>', "unknown"),
        (
            '<tool_call>{"name": "web_search_tool", "arguments": {"query": "role of CEO"}'
            "</tool_call>",
            "JSON",
        ),
        (
            '<tool_call>{"name": "web_search_tool", "arguments": "role of CEO"}</tool_call>',
            "arguments",
        ),
        (
            '<tool_call>{"name": "weather_tool", "arguments": {}}</tool_call>'
            '<tool_call>{"name": "web_search_tool", "arguments": {}}</tool_call>',
            "multiple",
        ),
    ],
)
def test_native_hf_router_tool_call_parse_failures_are_not_no_tool(
    raw_response: str,
    message: str,
) -> None:
    decision = LocalHFRouter.parse_native_response(raw_response)

    assert decision.selected_tool is None
    assert decision.tool_arguments == {}
    assert decision.direct_answer is None
    assert decision.parse_error is not None
    assert message.lower() in decision.parse_error.lower()
    if message in {"JSON", "unknown", "arguments"}:
        assert decision.extracted_tool_call_json is not None


def test_app_parse_failure_blocks_xai_target() -> None:
    inference_result = types.SimpleNamespace(
        selected_tool=None,
        parse_error="Native tool-call JSON could not be parsed.",
    )

    assert app_module.has_untrusted_native_parse_result(inference_result) is True


def test_sentinel_only_output_blocks_xai_target() -> None:
    decision = LocalHFRouter.parse_native_response("No_tool")

    assert app_module.has_untrusted_native_parse_result(decision) is True


def test_native_hf_generation_is_deterministic_and_uses_larger_budget() -> None:
    import torch

    router = LocalHFRouter.__new__(LocalHFRouter)
    router.tokenizer = GenerationTokenizer()
    router.model = RecordingGenerateModel()
    router.device = "cpu"
    router.max_new_tokens = DEFAULT_NATIVE_HF_MAX_NEW_TOKENS
    router.generation_kwargs = {"temperature": 0.7, "top_p": 0.9, "top_k": 20}
    router._torch = torch

    _debug_prompt, _raw_response, generation_parameters = router._generate_native_tool_response(
        "Explain precision and recall.",
        system_prompt="Answer stable conceptual explanations directly.",
    )

    assert generation_parameters["do_sample"] is False
    assert generation_parameters["max_new_tokens"] == 512
    assert "temperature" not in generation_parameters
    assert "top_p" not in generation_parameters
    assert "top_k" not in generation_parameters


def test_primary_hf_routing_does_not_instantiate_classification_router(monkeypatch) -> None:
    class FakeRouterDecision:
        def __init__(self) -> None:
            self.selected_tool = "web_search_tool"
            self.tool_arguments = {"query": "latest Formula 1 race winner"}
            self.agent_response = "I would use web_search_tool."
            self.raw_response = "<tool_call>web_search_tool"
            self.debug_prompt = "native prompt"
            self.parse_error = None

    class FakeRouter:
        def choose_tool(self, user_request, tool_descriptions, *, system_prompt=""):
            del user_request, tool_descriptions, system_prompt
            return FakeRouterDecision()

    def fake_load_local_hf_router(
        model_name,
        max_new_tokens,
        *,
        trust_remote_code,
        quantization_mode="none",
        device="auto",
        dtype="auto",
    ):
        del model_name, max_new_tokens, trust_remote_code, quantization_mode, device, dtype
        return FakeRouter()

    def exploding_classification_router(*args, **kwargs):
        del args, kwargs
        msg = "Primary HF routing must not instantiate LocalHFClassificationRouter."
        raise AssertionError(msg)

    monkeypatch.setattr(app_module, "load_local_hf_router", fake_load_local_hf_router)
    monkeypatch.setattr(app_module, "LocalHFClassificationRouter", exploding_classification_router)

    agent_callable = app_module.build_complete_agent_callable(
        inference_backend="HF local",
        inference_model_name="fake-native-model",
        system_prompt="Route to the correct tool.",
        tool_context="- web_search_tool: current facts",
    )

    result = agent_callable("Who won the latest Formula 1 race?")

    assert result.selected_tool == "web_search_tool"


def test_split_coalition_prompt_preserves_empty_user_request() -> None:
    system_prompt, user_request = split_coalition_prompt(
        "You are a tool router.\n\n"
        "Available tools:\n- weather_tool: Forecasts\n\n"
        "User request:\n\n"
        "Assistant:"
    )

    assert system_prompt == "You are a tool router."
    assert user_request == ""


def test_build_tool_calling_prompt_fallback_renders_canonical_schemas() -> None:
    prompt = build_tool_calling_prompt(
        RejectingToolsTokenizer(),
        system_prompt="You are a tool router.",
        user_request="Will it rain?",
        tool_schemas=EXECUTABLE_TOOL_SCHEMAS,
    )

    assert "System:\nYou are a tool router." in prompt
    assert "User:\nWill it rain?" in prompt
    for schema in EXECUTABLE_TOOL_SCHEMAS:
        function = schema["function"]
        assert function["name"] in prompt
        assert function["description"] in prompt
    assert '"required": [' in prompt
    assert '"location"' in prompt
    assert '"expression"' in prompt
    assert '"query"' in prompt
    assert NO_TOOL_NAME not in prompt


def test_build_tool_calling_prompt_does_not_swallow_unexpected_errors() -> None:
    with pytest.raises(RuntimeError, match="tokenizer exploded"):
        build_tool_calling_prompt(
            BrokenTokenizer(),
            system_prompt="You are a tool router.",
            user_request="Will it rain?",
            tool_schemas=EXECUTABLE_TOOL_SCHEMAS,
        )


def test_build_tool_calling_prompt_does_not_mutate_shared_schemas() -> None:
    before = deepcopy(EXECUTABLE_TOOL_SCHEMAS)

    build_tool_calling_prompt(
        RecordingTokenizer(),
        system_prompt="You are a tool router.",
        user_request="Will it rain?",
        tool_schemas=EXECUTABLE_TOOL_SCHEMAS,
    )

    assert before == EXECUTABLE_TOOL_SCHEMAS


def test_build_routing_classification_prompt_uses_fixed_label_legend() -> None:
    prompt = build_routing_classification_prompt(
        system_prompt="Route to the correct tool.",
        user_request="Will it rain tomorrow?",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert "You are a tool-routing classifier." in prompt
    assert "Route to the correct tool." in prompt
    for tool_name, label in ROUTING_LABELS.items():
        assert f"{label} = {tool_name}" in prompt
        assert f"Description: {TOOL_DESCRIPTIONS[tool_name]}" in prompt
    assert "Return exactly one decision code and nothing else." in prompt
    assert "User request:\nWill it rain tomorrow?" in prompt
    assert prompt.endswith("Decision:")


def test_build_routing_classification_prompt_changes_with_tool_description() -> None:
    """Blocker 1: changing a tool description changes the generated classifier prompt."""
    descriptions_a = dict(TOOL_DESCRIPTIONS)
    descriptions_b = dict(TOOL_DESCRIPTIONS)
    descriptions_b["weather_tool"] = "A completely different weather description."

    prompt_a = build_routing_classification_prompt(
        system_prompt="Route.", user_request="Will it rain?", tool_descriptions=descriptions_a
    )
    prompt_b = build_routing_classification_prompt(
        system_prompt="Route.", user_request="Will it rain?", tool_descriptions=descriptions_b
    )

    assert prompt_a != prompt_b
    assert descriptions_b["weather_tool"] in prompt_b
    assert descriptions_b["weather_tool"] not in prompt_a


def test_build_routing_classification_prompt_supports_empty_user_request() -> None:
    prompt = build_routing_classification_prompt(
        system_prompt="Route.", user_request="", tool_descriptions=TOOL_DESCRIPTIONS
    )

    assert prompt.endswith("User request:\n\nDecision:")


def test_calibrated_scorer_build_scoring_prompt_uses_routing_protocol() -> None:
    scorer = CalibratedToolLogOddsScorer.__new__(CalibratedToolLogOddsScorer)
    scorer.routing_labels = dict(ROUTING_LABELS)
    full_prompt = build_coalition_prompt(
        "Calculate 238 times 47",
        system_prompt="Use calculator_tool for arithmetic.",
        tool_context="- calculator_tool: Calculator",
    )

    preview = scorer.build_scoring_prompt(
        full_prompt,
        target_tool="calculator_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert preview == build_routing_classification_prompt(
        system_prompt="Use calculator_tool for arithmetic.",
        user_request="Calculate 238 times 47",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )
    assert "The correct tool is" not in preview
    assert TOOL_DESCRIPTIONS["calculator_tool"] in preview


def test_calibrated_scorer_sequence_logprobs_batched_chunks_and_preserves_order() -> None:
    scorer = CalibratedToolLogOddsScorer.__new__(CalibratedToolLogOddsScorer)
    scorer.max_pairs_per_batch = 2
    calls = []
    cache_releases = []

    def fake_sequence_logprobs_batch(
        prompts: list[str],
        continuations: list[str],
    ) -> list[float]:
        calls.append((prompts.copy(), continuations.copy()))
        return [float(prompt.removeprefix("prompt-")) for prompt in prompts]

    scorer._sequence_logprobs_batch = fake_sequence_logprobs_batch
    scorer._release_device_cache = lambda: cache_releases.append(True)

    scores = scorer._sequence_logprobs_batched(
        ["prompt-0", "prompt-1", "prompt-2", "prompt-3", "prompt-4"],
        ["cont-0", "cont-1", "cont-2", "cont-3", "cont-4"],
    )

    assert scores == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert calls == [
        (["prompt-0", "prompt-1"], ["cont-0", "cont-1"]),
        (["prompt-2", "prompt-3"], ["cont-2", "cont-3"]),
        (["prompt-4"], ["cont-4"]),
    ]
    assert len(cache_releases) == len(calls)


def test_llm_tool_scorer_build_scoring_prompt_preview() -> None:
    scorer = LLMToolScorer(llm=MockLLM())

    preview = scorer.build_scoring_prompt(
        "System\n\nUser request:\nCalculate 238 times 47\n\nAssistant:",
        target_tool="calculator_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert "Target tool:" in preview
    assert "calculator_tool" in preview
    assert "Prompt:" in preview


def test_calibrated_scorer_next_token_log_probs_match_full_log_softmax_gather() -> None:
    torch = pytest.importorskip("torch")
    logits = torch.tensor(
        [
            [
                [0.2, -0.1, 1.4, 0.0, -0.7],
                [1.1, 0.3, -0.5, 0.8, -1.2],
                [-0.4, 1.7, 0.2, -0.9, 0.5],
                [0.6, -1.1, 0.4, 1.3, -0.2],
            ],
            [
                [-0.3, 0.9, 0.1, -0.6, 1.2],
                [0.7, -0.8, 1.5, 0.0, -0.4],
                [1.4, 0.2, -1.0, 0.5, -0.2],
                [-0.5, 0.4, 0.8, -0.1, 1.0],
            ],
        ],
        dtype=torch.float32,
    )
    token_ids = torch.tensor(
        [
            [0, 2, 3, 1],
            [4, 0, 2, 3],
        ],
        dtype=torch.long,
    )

    expected = (
        torch.log_softmax(logits[:, :-1, :], dim=-1)
        .gather(
            dim=-1,
            index=token_ids[:, 1:].unsqueeze(-1),
        )
        .squeeze(-1)
    )
    actual = CalibratedToolLogOddsScorer._next_token_log_probs(logits, token_ids)

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_calibrated_scorer_loads_with_dtype_kwarg_not_deprecated_torch_dtype(monkeypatch) -> None:
    """Regression test for the macOS/MPS crash investigation.

    ``from_pretrained`` must be called with the current ``dtype=`` keyword
    (matching the verified-working standalone load script), never the
    deprecated ``torch_dtype=`` alias, and with ``low_cpu_mem_usage=True`` to
    avoid materializing a full fp32 copy before casting/moving to MPS. Half
    precision must be requested on MPS, not just CUDA.
    """
    transformers = pytest.importorskip("transformers")

    captured: dict[str, object] = {}

    class FakeTokenizer:
        pad_token = "eos"
        eos_token = "eos"

        def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
            del add_special_tokens
            return {"input_ids": [ord(character) for character in text]}

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeTokenizer:
            del model_id, kwargs
            return cls()

    class FakeModel:
        def to(self, device: str) -> FakeModel:
            captured["to_device"] = device
            return self

        def eval(self) -> FakeModel:
            return self

        @classmethod
        def from_pretrained(cls, model_id: str, **kwargs: object) -> FakeModel:
            del model_id
            captured["kwargs"] = dict(kwargs)
            return cls()

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeModel)

    CalibratedToolLogOddsScorer(model_id="fake-model", device="mps")

    torch = pytest.importorskip("torch")
    kwargs = captured["kwargs"]
    assert "dtype" in kwargs
    assert "torch_dtype" not in kwargs
    assert kwargs["low_cpu_mem_usage"] is True
    assert kwargs["dtype"] is torch.float16
    assert captured["to_device"] == "mps"


def test_llm_tool_scorer_parse_score_accepts_plain_number() -> None:
    scorer = LLMToolScorer(llm=FakeGenerator("0.7"))

    assert scorer.parse_score("0.7") == 0.7


def test_llm_tool_scorer_parse_score_accepts_labeled_number() -> None:
    scorer = LLMToolScorer(llm=FakeGenerator("Score: 0.7"))

    assert scorer.parse_score("Score: 0.7") == 0.7
    assert scorer.parse_score("The score is 0.7\n") == 0.7


def test_llm_tool_scorer_parse_score_rejects_out_of_range_number() -> None:
    scorer = LLMToolScorer(llm=FakeGenerator("1.5"))

    with pytest.raises(ValueError):
        scorer.parse_score("1.5")


def test_llm_tool_scorer_parse_score_rejects_tool_index() -> None:
    scorer = LLMToolScorer(llm=FakeGenerator("tool 3"))

    with pytest.raises(ValueError):
        scorer.parse_score("tool 3")


def test_llm_tool_scorer_parse_score_rejects_text_without_number() -> None:
    scorer = LLMToolScorer(llm=FakeGenerator("not a number"))

    with pytest.raises(ValueError):
        scorer.parse_score("not a number")


def test_llm_tool_scorer_parses_numeric_output() -> None:
    scorer = LLMToolScorer(llm=MockLLM("0.75"))

    scores = scorer.score_batch(
        ["User asks whether it will rain in Berlin tomorrow."],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [0.75]


def test_llm_tool_scorer_falls_back_on_out_of_range_numeric_output() -> None:
    fallback = LexicalToolScorer()
    scorer = LLMToolScorer(llm=MockLLM("1.7"), fallback_scorer=fallback)
    prompt = "User asks whether it will rain in Berlin tomorrow."

    scores = scorer.score_batch(
        [prompt],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == fallback.score_batch(
        [prompt],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )
    assert scorer.last_debug_outputs[0]["used_fallback"] is True


def test_llm_tool_scorer_score_batch_uses_fake_generator() -> None:
    scorer = LLMToolScorer(llm=FakeGenerator("0.8"))

    scores = scorer.score_batch(
        [
            "Use weather_tool for forecasts.",
            "Will it rain in Berlin tomorrow?",
        ],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == [0.8, 0.8]
    assert scorer.last_debug_outputs == [
        {
            "target_tool": "weather_tool",
            "raw_output": "0.8",
            "parsed_score": 0.8,
            "used_fallback": False,
            "fallback_score": None,
            "final_score": 0.8,
        },
        {
            "target_tool": "weather_tool",
            "raw_output": "0.8",
            "parsed_score": 0.8,
            "used_fallback": False,
            "fallback_score": None,
            "final_score": 0.8,
        },
    ]


def test_llm_tool_scorer_falls_back_on_invalid_output() -> None:
    fallback = LexicalToolScorer()
    scorer = LLMToolScorer(llm=FakeGenerator("not a score"), fallback_scorer=fallback)
    prompt = "Use weather_tool for forecast questions. Will it rain in Berlin tomorrow?"

    scores = scorer.score_batch(
        [prompt],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == fallback.score_batch(
        [prompt],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )
    assert len(scorer.last_debug_outputs) == 1
    debug_output = scorer.last_debug_outputs[0]
    assert debug_output["raw_output"] == "not a score"
    assert debug_output["parsed_score"] is None
    assert debug_output["used_fallback"] is True
    assert debug_output["fallback_score"] == scores[0]
    assert debug_output["final_score"] == scores[0]


SAMPLE_SYSTEM_PROMPT = "Route to the correct tool."
SAMPLE_TOOL_CONTEXT = "- weather_tool: forecasts\n- calculator_tool: arithmetic"


def _sample_coalition_prompt(user_request: str) -> str:
    return build_coalition_prompt(
        user_request,
        system_prompt=SAMPLE_SYSTEM_PROMPT,
        tool_context=SAMPLE_TOOL_CONTEXT,
    )


def test_calibrated_scorer_returns_one_float_per_prompt() -> None:
    coalition_scores = {
        "weather_tool": 2.0,
        "calculator_tool": 0.0,
        "web_search_tool": -1.0,
        "no_tool": -2.0,
    }
    scorer = make_fake_calibrated_scorer(coalition_scores, coalition_scores, coalition_scores)

    scores = scorer.score_batch(
        [_sample_coalition_prompt("Prompt one"), _sample_coalition_prompt("Prompt two")],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert isinstance(scores, list)
    assert len(scores) == 2
    assert all(isinstance(score, float) for score in scores)
    assert len(scorer.last_debug_outputs) == 2


def test_calibrated_scorer_treats_no_tool_as_one_alternative_not_reference() -> None:
    """Test 1: target-vs-all correctness (no target-vs-no_tool contrast)."""
    coalition_scores = {
        "weather_tool": -1.0,
        "calculator_tool": -0.1,
        "web_search_tool": -3.0,
        "no_tool": -4.0,
    }
    zero_scores = dict.fromkeys(coalition_scores, 0.0)
    scorer = make_fake_calibrated_scorer(coalition_scores, zero_scores, zero_scores)
    prompt = _sample_coalition_prompt("Will it rain in Berlin tomorrow?")

    weather_value = scorer.score_batch(
        [prompt], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS
    )[0]
    calculator_value = scorer.score_batch(
        [prompt], target_tool="calculator_tool", tool_descriptions=TOOL_DESCRIPTIONS
    )[0]

    # weather_tool outscores no_tool (-1.0 > -4.0), but calculator_tool (-0.1) is
    # actually the strongest candidate. A target-vs-no_tool contrast would
    # incorrectly report strong support for weather_tool; target-vs-all must not:
    # calculator_tool's value must be clearly higher than weather_tool's.
    assert calculator_value > weather_value
    assert weather_value < calculator_value - 1.0


def test_calibrated_scorer_empty_coalition_normalizes_to_zero() -> None:
    """Test 2: V(empty_coalition; target_tool) == 0.0 for every target tool."""
    coalition_scores = {
        "weather_tool": -1.0,
        "calculator_tool": -0.1,
        "web_search_tool": -3.0,
        "no_tool": -4.0,
    }
    calibration_scores = {
        "weather_tool": -1.2,
        "calculator_tool": -0.3,
        "web_search_tool": -3.1,
        "no_tool": -4.2,
    }
    empty_scores = {
        "weather_tool": -2.0,
        "calculator_tool": -2.5,
        "web_search_tool": -2.2,
        "no_tool": -1.9,
    }
    scorer = make_fake_calibrated_scorer(coalition_scores, calibration_scores, empty_scores)
    empty_coalition_prompt = _sample_coalition_prompt("")

    for target_tool in TOOL_DESCRIPTIONS:
        value = scorer.score_batch(
            [empty_coalition_prompt],
            target_tool=target_tool,
            tool_descriptions=TOOL_DESCRIPTIONS,
        )[0]
        assert math.isclose(value, 0.0, abs_tol=1e-8)


def test_calibrated_scorer_calibration_removes_fixed_label_prior() -> None:
    """Test 3: subtracting calibration scores removes a shared fixed label prior."""

    def build(prior: float) -> CalibratedToolLogOddsScorer:
        coalition_scores = {
            "weather_tool": -1.0,
            "calculator_tool": prior - 0.5,
            "web_search_tool": -2.0,
            "no_tool": -3.0,
        }
        calibration_scores = {
            "weather_tool": -1.2,
            "calculator_tool": prior - 1.0,
            "web_search_tool": -2.1,
            "no_tool": -3.05,
        }
        empty_scores = {
            "weather_tool": -1.1,
            "calculator_tool": prior - 0.8,
            "web_search_tool": -2.05,
            "no_tool": -3.02,
        }
        return make_fake_calibrated_scorer(coalition_scores, calibration_scores, empty_scores)

    prompt = _sample_coalition_prompt("Calculate 12 times 4")
    small_prior_scorer = build(5.0)
    large_prior_scorer = build(50.0)

    small_value = small_prior_scorer.score_batch(
        [prompt], target_tool="calculator_tool", tool_descriptions=TOOL_DESCRIPTIONS
    )[0]
    large_value = large_prior_scorer.score_batch(
        [prompt], target_tool="calculator_tool", tool_descriptions=TOOL_DESCRIPTIONS
    )[0]

    # The final game value must not depend on the magnitude of the shared fixed
    # prior baked into both the coalition and calibration raw scores.
    assert small_value == pytest.approx(large_value, rel=1e-9)
    calibrated_scores = small_prior_scorer.last_debug_outputs[0]["calibrated_scores"]
    assert calibrated_scores["calculator_tool"] == pytest.approx(0.5, abs=1e-9)


def test_calibrated_scorer_calibration_cache_key_varies_with_tool_description() -> None:
    """Blocker 1 / required test 2: a changed tool description is a calibration cache miss."""
    coalition_scores = dict.fromkeys(TOOL_DESCRIPTIONS, -1.0)
    scorer = make_fake_calibrated_scorer(coalition_scores, coalition_scores, coalition_scores)
    prompt = _sample_coalition_prompt("Will it rain tomorrow?")

    descriptions_a = dict(TOOL_DESCRIPTIONS)
    descriptions_b = dict(TOOL_DESCRIPTIONS)
    descriptions_b["weather_tool"] = "A completely different weather description."

    scorer.score_batch([prompt], target_tool="weather_tool", tool_descriptions=descriptions_a)
    scorer.score_batch([prompt], target_tool="weather_tool", tool_descriptions=descriptions_b)

    assert len(scorer._calibration_cache) == 2


def test_calibrated_scorer_calibration_cache_key_varies_with_system_prompt() -> None:
    """Required test 3: calibration scores are not reused across different system prompts."""
    coalition_scores = dict.fromkeys(TOOL_DESCRIPTIONS, -1.0)
    scorer = make_fake_calibrated_scorer(coalition_scores, coalition_scores, coalition_scores)
    prompt_a = build_coalition_prompt(
        "Will it rain tomorrow?", system_prompt="Route A.", tool_context=SAMPLE_TOOL_CONTEXT
    )
    prompt_b = build_coalition_prompt(
        "Will it rain tomorrow?", system_prompt="Route B.", tool_context=SAMPLE_TOOL_CONTEXT
    )

    scorer.score_batch([prompt_a], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS)
    scorer.score_batch([prompt_b], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS)

    assert len(scorer._calibration_cache) == 2


def test_calibrated_scorer_calibration_cache_key_varies_with_routing_label_order() -> None:
    """Required test 4: calibration scores are not reused across different label orders."""
    coalition_scores = dict.fromkeys(TOOL_DESCRIPTIONS, -1.0)
    scorer = make_fake_calibrated_scorer(coalition_scores, coalition_scores, coalition_scores)
    prompt = _sample_coalition_prompt("Will it rain tomorrow?")

    scorer.routing_labels = dict(ROUTING_LABELS)
    scorer.score_batch([prompt], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS)

    scorer.routing_labels = dict(reversed(list(ROUTING_LABELS.items())))
    scorer.score_batch([prompt], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS)

    assert len(scorer._calibration_cache) == 2


def test_calibrated_scorer_calibration_cache_key_varies_with_model_id() -> None:
    coalition_scores = dict.fromkeys(TOOL_DESCRIPTIONS, -1.0)
    scorer = make_fake_calibrated_scorer(coalition_scores, coalition_scores, coalition_scores)
    prompt = _sample_coalition_prompt("Will it rain tomorrow?")

    scorer.model_id = "model-a"
    scorer.score_batch([prompt], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS)

    scorer.model_id = "model-b"
    scorer.score_batch([prompt], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS)

    assert len(scorer._calibration_cache) == 2


def test_calibrated_scorer_reuses_raw_scores_for_repeated_empty_coalition_prompt() -> None:
    """Required test: the empty coalition is not re-evaluated redundantly by the model."""
    coalition_scores = dict.fromkeys(TOOL_DESCRIPTIONS, -1.0)
    scorer = make_fake_calibrated_scorer(coalition_scores, coalition_scores, coalition_scores)
    call_counter = {"prompts_computed": 0}
    original_compute = scorer._compute_label_log_scores_batched

    def counting_compute(prompts: list[str], candidate_tools: list[str]) -> list[dict[str, float]]:
        call_counter["prompts_computed"] += len(prompts)
        return original_compute(prompts, candidate_tools)

    scorer._compute_label_log_scores_batched = counting_compute
    empty_prompt = _sample_coalition_prompt("")

    # Score the literal empty coalition as an ordinary coalition prompt. This
    # populates the raw-score-row cache for that exact prompt text (both from
    # the main-loop evaluation and the internal empty-coalition baseline, which
    # share the identical prompt text and therefore the identical cache entry).
    scorer.score_batch(
        [empty_prompt], target_tool="weather_tool", tool_descriptions=TOOL_DESCRIPTIONS
    )
    prompts_computed_after_first = call_counter["prompts_computed"]
    # One real compute call for the empty-coalition prompt (main-loop scoring)
    # and one for each distinct calibration probe prompt -- the internal
    # empty-coalition-normalization helper reuses the first call's cached row
    # instead of triggering another compute call for the identical text.
    assert prompts_computed_after_first == 1 + len(CALIBRATION_USER_REQUESTS)

    # Ask again with a different target tool: the empty-coalition raw scores
    # for this exact prompt text must be reused from cache, not recomputed --
    # zero additional real model calls for the identical prompt text.
    scorer.score_batch(
        [empty_prompt], target_tool="calculator_tool", tool_descriptions=TOOL_DESCRIPTIONS
    )

    assert call_counter["prompts_computed"] == prompts_computed_after_first


def test_calibrated_scorer_probabilities_are_stable_and_sum_to_one() -> None:
    """Test 4: diagnostic calibrated_probabilities form a stable probability simplex."""
    coalition_scores = {
        "weather_tool": -1.0,
        "calculator_tool": -0.1,
        "web_search_tool": -3.0,
        "no_tool": -4.0,
    }
    calibration_scores = {
        "weather_tool": -1.2,
        "calculator_tool": -0.3,
        "web_search_tool": -3.1,
        "no_tool": -4.2,
    }
    scorer = make_fake_calibrated_scorer(coalition_scores, calibration_scores, calibration_scores)

    scorer.score_batch(
        [_sample_coalition_prompt("Will it rain tomorrow?")],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    probabilities = scorer.last_debug_outputs[0]["calibrated_probabilities"]
    assert set(probabilities) == set(TOOL_DESCRIPTIONS)
    assert all(math.isfinite(value) for value in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_calibrated_scorer_sequence_logprobs_sum_not_average_unequal_lengths() -> None:
    """Test 5: unequal-length continuations are scored by summed, not mean, logprob."""
    torch = pytest.importorskip("torch")

    class FakeBatchTokenizer:
        """Looks up exact-string token ids and right-pads like a real HF tokenizer."""

        def __init__(self, id_map: dict[str, list[int]]) -> None:
            self.id_map = id_map

        def __call__(self, texts, *, return_tensors="pt", add_special_tokens=False, padding=True):
            del return_tensors, add_special_tokens, padding
            sequences = [self.id_map[text] for text in texts]
            max_len = max(len(sequence) for sequence in sequences)
            input_ids = []
            attention_mask = []
            for sequence in sequences:
                pad_len = max_len - len(sequence)
                input_ids.append(sequence + [0] * pad_len)
                attention_mask.append([1] * len(sequence) + [0] * pad_len)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

    class FakeZeroLogitsModel:
        """Returns uniform (all-zero) logits, so every next-token logprob is -log(2)."""

        def __call__(self, input_ids, attention_mask):
            del attention_mask
            batch, seq_len = input_ids.shape
            return type("Output", (), {"logits": torch.zeros((batch, seq_len, 2))})()

        def eval(self) -> None:
            return None

    id_map = {
        "PROMPT": [0, 0, 0],
        "PROMPT A": [0, 0, 0, 1],
        "PROMPT BB": [0, 0, 0, 1, 1],
    }
    scorer = CalibratedToolLogOddsScorer.__new__(CalibratedToolLogOddsScorer)
    scorer._torch = torch
    scorer.tokenizer = FakeBatchTokenizer(id_map)
    scorer.model = FakeZeroLogitsModel()
    scorer.device = "cpu"

    scores = scorer._sequence_logprobs_batch(["PROMPT", "PROMPT"], [" A", " BB"])

    one_token_score, two_token_score = scores
    expected_one_token = -math.log(2)
    expected_two_token_sum = -2 * math.log(2)
    assert one_token_score == pytest.approx(expected_one_token, abs=1e-6)
    # The two-token continuation must be scored by its summed log-likelihood
    # (double the one-token score), not the per-token mean (which would
    # incorrectly equal the one-token score).
    assert two_token_score == pytest.approx(expected_two_token_sum, abs=1e-6)
    assert two_token_score != pytest.approx(expected_one_token, abs=1e-3)


def test_validate_routing_label_tokens_raises_on_non_prefix_stable_tokenizer() -> None:
    """Test 6: a tokenizer that breaks prompt/label prefix stability raises ValueError."""
    prompt_prefix = "...Decision:"

    class BrokenPrefixTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
            del add_special_tokens
            if text == prompt_prefix:
                return {"input_ids": [1, 2, 3]}
            # Any text built from prompt_prefix + label retokenizes completely
            # differently, so it can never keep [1, 2, 3] as a strict prefix.
            return {"input_ids": [9, 9, 9, 9]}

    with pytest.raises(ValueError, match="prefix-stable"):
        validate_routing_label_tokens(
            BrokenPrefixTokenizer(),
            prompt_prefix,
            routing_labels={"weather_tool": "A"},
            separator=" ",
        )


def test_validate_routing_label_tokens_records_token_lengths_for_stable_tokenizer() -> None:
    prompt_prefix = "Decision:"

    class StablePerCharTokenizer:
        def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
            del add_special_tokens
            return {"input_ids": [ord(character) for character in text]}

    diagnostics = validate_routing_label_tokens(
        StablePerCharTokenizer(),
        prompt_prefix,
        routing_labels={"weather_tool": "A", "calculator_tool": "BB"},
        separator="",
    )

    assert diagnostics["weather_tool"] == RoutingLabelTokenization(
        tool_name="weather_tool", label="A", token_length=1
    )
    assert diagnostics["calculator_tool"] == RoutingLabelTokenization(
        tool_name="calculator_tool", label="BB", token_length=2
    )


def test_stable_logsumexp_matches_naive_computation_for_small_inputs() -> None:
    values = [-1.0, -0.1, -3.0, -4.0]

    expected = math.log(sum(math.exp(value) for value in values))

    assert stable_logsumexp(values) == pytest.approx(expected)


def test_calibrated_scorer_rejects_too_few_candidate_tools() -> None:
    scorer = make_fake_calibrated_scorer({"weather_tool": 1.0}, {}, {})

    with pytest.raises(ValueError, match="at least two"):
        scorer.score_batch(
            [_sample_coalition_prompt("Will it rain tomorrow?")],
            target_tool="weather_tool",
            tool_descriptions={"weather_tool": TOOL_DESCRIPTIONS["weather_tool"]},
        )


def test_calibrated_scorer_rejects_unknown_target_tool() -> None:
    coalition_scores = {"weather_tool": 1.0, "no_tool": 0.0}
    scorer = make_fake_calibrated_scorer(coalition_scores, coalition_scores, coalition_scores)

    with pytest.raises(ValueError, match="not available"):
        scorer.score_batch(
            [_sample_coalition_prompt("Will it rain tomorrow?")],
            target_tool="calendar_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )


def test_calibrated_scorer_rejects_candidate_tool_without_routing_label() -> None:
    coalition_scores = {"weather_tool": 1.0, "mystery_tool": 0.0}
    scorer = make_fake_calibrated_scorer(coalition_scores, coalition_scores, coalition_scores)

    with pytest.raises(ValueError, match="mystery_tool"):
        scorer.score_batch(
            [_sample_coalition_prompt("Will it rain tomorrow?")],
            target_tool="weather_tool",
            tool_descriptions={
                "weather_tool": TOOL_DESCRIPTIONS["weather_tool"],
                "mystery_tool": "An undefined decision candidate.",
            },
        )


def test_tool_use_game_normalization_is_a_noop_for_calibrated_scorer() -> None:
    """Required test 7: V(empty_coalition) == 0 and Game(normalize=True) does not change it.

    Ownership: the scorer already owns contextual calibration, target-vs-all
    log-odds, and empty-coalition payoff normalization -- V(empty_coalition) is
    exactly 0.0 by construction. The Game layer only owns Shapley/k-SII
    computation; its own ``normalize=True`` subtraction of
    ``normalization_value`` must therefore be a mathematical no-op for this
    scorer (normalization_value == 0.0), not a second, redundant
    normalization that changes the returned payoff.
    """
    coalition_scores = {
        "weather_tool": -1.0,
        "calculator_tool": -0.1,
        "web_search_tool": -3.0,
        "no_tool": -4.0,
    }
    calibration_scores = {
        "weather_tool": -1.2,
        "calculator_tool": -0.3,
        "web_search_tool": -3.1,
        "no_tool": -4.2,
    }
    scorer = make_fake_calibrated_scorer(coalition_scores, calibration_scores, calibration_scores)
    game = ToolUseGame(
        target_tool="weather_tool",
        user_segments=["Will it rain tomorrow?"],
        system_prompt=SAMPLE_SYSTEM_PROMPT,
        tool_context=SAMPLE_TOOL_CONTEXT,
        scorer=scorer,
        tool_descriptions=TOOL_DESCRIPTIONS,
        normalize=True,
    )

    # The scorer's own empty-coalition normalization already makes this 0.0 --
    # Game.__init__ computing it independently must agree, not introduce a
    # second, different baseline.
    assert game.normalization_value == pytest.approx(0.0, abs=1e-9)

    empty_coalition = np.zeros((1, game.n_players), dtype=bool)
    full_coalition = np.ones((1, game.n_players), dtype=bool)

    raw_empty = float(game.value_function(empty_coalition)[0])
    raw_full = float(game.value_function(full_coalition)[0])
    normalized_empty = float(game(empty_coalition)[0])
    normalized_full = float(game(full_coalition)[0])

    assert raw_empty == pytest.approx(0.0, abs=1e-9)
    assert normalized_empty == pytest.approx(0.0, abs=1e-9)
    # Game-level normalize=True must not alter the scorer's own payoff: the
    # normalized value equals the raw value because normalization_value == 0.
    assert normalized_full == pytest.approx(raw_full, abs=1e-9)


def test_app_load_logprob_scorer_has_no_post_cache_mutation() -> None:
    """Required test 6: no post-cache mutation of max_pairs_per_batch occurs.

    ``load_logprob_scorer`` must take ``max_pairs_per_batch`` as part of its
    (cached) call signature -- so a different UI-selected batch size loads a
    distinct cached scorer instead of reaching into a shared cached instance
    and mutating its configuration -- and app.py must not contain the old
    ``<scorer>.max_pairs_per_batch = ...`` mutation pattern anywhere.
    """
    signature = inspect.signature(app_module.load_logprob_scorer.__wrapped__)
    assert "max_pairs_per_batch" in signature.parameters
    assert signature.parameters["max_pairs_per_batch"].default is inspect.Parameter.empty

    source = inspect.getsource(app_module)
    assert ".max_pairs_per_batch = " not in source


def test_qwen3_is_selectable_and_builds_single_hf_config() -> None:
    config = app_module.selected_hf_model_config("Qwen/Qwen3-4B-Instruct-2507")

    assert "Qwen/Qwen3-4B-Instruct-2507" in app_module.HF_LOCAL_MODEL_OPTIONS
    assert config.model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.tokenizer_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.model_family == "qwen3"
    assert config.quantization_mode == "none"
    assert config.device == "auto"
    assert config.dtype == "auto"
    assert config.supports_native_tools is True


def test_hf_config_cache_key_varies_by_model_and_quantization() -> None:
    qwen25 = app_module.selected_hf_model_config("Qwen/Qwen2.5-3B-Instruct")
    qwen3 = app_module.selected_hf_model_config("Qwen/Qwen3-4B-Instruct-2507")
    qwen3_quantized = app_module.SelectedHFModelConfig(
        model_id=qwen3.model_id,
        model_family=qwen3.model_family,
        quantization_mode="4bit",
        device=qwen3.device,
        dtype=qwen3.dtype,
        supports_native_tools=qwen3.supports_native_tools,
    )

    assert qwen25.cache_key(scorer_mode="native") != qwen3.cache_key(scorer_mode="native")
    assert qwen3.cache_key(scorer_mode="native") != qwen3_quantized.cache_key(scorer_mode="native")


def test_cached_hf_loaders_include_model_runtime_identity() -> None:
    router_signature = inspect.signature(app_module.load_local_hf_router.__wrapped__)
    scorer_signature = inspect.signature(app_module.load_logprob_scorer.__wrapped__)

    for parameter_name in ("quantization_mode", "device", "dtype"):
        assert parameter_name in router_signature.parameters
        assert parameter_name in scorer_signature.parameters
    assert "tokenizer_id" in scorer_signature.parameters


def test_native_scorer_from_router_preserves_selected_qwen3_identity() -> None:
    class FakeModel:
        def eval(self):
            return None

    class FakeTokenizer:
        eos_token = "<eos>"
        pad_token = None

    router = types.SimpleNamespace(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        tokenizer_id="Qwen/Qwen3-4B-Instruct-2507",
        requested_quantization_mode="none",
        actual_quantization_mode="none",
        device="cpu",
        dtype="float32",
        model=FakeModel(),
        tokenizer=FakeTokenizer(),
    )
    config = app_module.selected_hf_model_config("Qwen/Qwen3-4B-Instruct-2507")

    scorer = app_module.build_native_hf_scorer_from_router(
        router,
        max_pairs_per_batch=1,
        selected_config=config,
    )
    diagnostics = app_module.validate_hf_inference_xai_consistency(
        selected_config=config,
        router=router,
        scorer=scorer,
    )

    assert scorer.model is router.model
    assert scorer.tokenizer is router.tokenizer
    assert scorer.model_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert scorer.tokenizer_id == "Qwen/Qwen3-4B-Instruct-2507"
    assert diagnostics["match"] is True
    assert diagnostics["model_family"] == "qwen3"


@pytest.mark.parametrize(
    ("scorer_override", "message"),
    [
        ({"model_id": "Qwen/Qwen2.5-3B-Instruct"}, "same model"),
        ({"tokenizer_id": "Qwen/Qwen2.5-3B-Instruct"}, "same tokenizer"),
        ({"actual_quantization_mode": "4bit"}, "same quantization"),
        ({"device": "mps"}, "same device"),
    ],
)
def test_hf_consistency_guard_blocks_runtime_mismatch(
    scorer_override: dict[str, str],
    message: str,
) -> None:
    config = app_module.selected_hf_model_config("Qwen/Qwen3-4B-Instruct-2507")
    router = types.SimpleNamespace(
        model_name=config.model_id,
        tokenizer_id=config.tokenizer_id,
        requested_quantization_mode=config.quantization_mode,
        actual_quantization_mode=config.quantization_mode,
        device="cpu",
        dtype="float32",
    )
    scorer_values = {
        "model_id": config.model_id,
        "tokenizer_id": config.tokenizer_id,
        "requested_quantization_mode": config.quantization_mode,
        "actual_quantization_mode": config.quantization_mode,
        "device": "cpu",
        "dtype": "float32",
    }
    scorer_values.update(scorer_override)
    scorer = types.SimpleNamespace(**scorer_values)

    with pytest.raises(RuntimeError, match=message):
        app_module.validate_hf_inference_xai_consistency(
            selected_config=config,
            router=router,
            scorer=scorer,
        )


def test_app_logprob_scorer_branch_never_loads_structured_json_router() -> None:
    """Invariant for the macOS/MPS crash investigation: exactly one Qwen model.

    ``CalibratedToolLogOddsScorer`` owns the model for the calibrated
    multiclass tool log-odds (HF local) explanation path.
    ``LocalHFClassificationRouter`` must be the only target-tool selector used
    there, as a thin wrapper reusing that same scorer -- the structured JSON
    ``LocalHFRouter``/``load_local_hf_router`` must never be instantiated
    inside that branch, or two independent Qwen models would be constructed
    (and held resident simultaneously) for what should be a single-model path.
    """
    source = inspect.getsource(app_module)
    branch_start = source.index("elif scorer_backend == LOGPROB_SCORER_LABEL:")
    next_branch = source.index("elif scorer_backend ==", branch_start + 1)
    branch_source = source[branch_start:next_branch]

    assert "LocalHFClassificationRouter" in branch_source
    assert "load_local_hf_router" not in branch_source
    assert "LocalHFRouter(" not in branch_source


def test_app_combined_hf_local_calibrated_mode_skips_structured_router(monkeypatch) -> None:
    """Required tests 1/2/5: HF local + calibrated log-odds share one model instance.

    Exercises the real ``build_complete_agent_callable(calibrated_hf_mode=True)``
    control flow end to end (the same function the Inference tab's "Run
    inference" button calls): only the two cache_resource loaders and
    ``HFArgumentExtractor`` are faked (to avoid loading a real model), while
    ``LocalHFClassificationRouter`` runs for real against a fake scorer. If
    this code path ever called ``load_local_hf_router``/``LocalHFRouter``, the
    faked loader below would raise instead of returning a result.
    """

    class FakeScorer:
        def __init__(self) -> None:
            self.model = object()
            self.tokenizer = object()
            self.device = "cpu"
            self.routing_labels = dict(ROUTING_LABELS)

        def score_full_request_calibrated_labels(
            self, user_request, *, system_prompt, tool_descriptions
        ):
            del user_request, system_prompt, tool_descriptions
            return {
                "weather_tool": 1.0,
                "calculator_tool": 0.0,
                "web_search_tool": -1.0,
                "no_tool": -2.0,
            }

    fake_scorer = FakeScorer()
    captured_extractor_kwargs: dict[str, object] = {}

    class FakeArgumentExtractor:
        def __init__(self, *, model, tokenizer, device) -> None:
            captured_extractor_kwargs["model"] = model
            captured_extractor_kwargs["tokenizer"] = tokenizer
            captured_extractor_kwargs["device"] = device

        def extract_arguments(self, *, user_request, selected_tool, tool_descriptions):
            del user_request, tool_descriptions
            return {"location": "Berlin"} if selected_tool == "weather_tool" else {}

    def exploding_load_local_hf_router(*args, **kwargs):
        del args, kwargs
        msg = "load_local_hf_router must not be called in calibrated HF-local mode"
        raise AssertionError(msg)

    def fake_load_logprob_scorer(
        model_id,
        *,
        max_pairs_per_batch,
        device=None,
        dtype="auto",
        quantization_mode="none",
        tokenizer_id=None,
    ):
        del device, dtype, quantization_mode, tokenizer_id
        captured_load["model_id"] = model_id
        captured_load["max_pairs_per_batch"] = max_pairs_per_batch
        return fake_scorer

    captured_load: dict[str, object] = {}

    monkeypatch.setattr(app_module, "load_local_hf_router", exploding_load_local_hf_router)
    monkeypatch.setattr(app_module, "load_logprob_scorer", fake_load_logprob_scorer)
    monkeypatch.setattr(app_module, "HFArgumentExtractor", FakeArgumentExtractor)

    agent_callable = app_module.build_complete_agent_callable(
        inference_backend="HF local",
        inference_model_name="Qwen/Qwen2.5-1.5B-Instruct",
        system_prompt="Route to the correct tool.",
        tool_context="- weather_tool: forecasts",
        calibrated_hf_mode=True,
        logprob_model_id="Qwen/Qwen2.5-1.5B-Instruct",
        logprob_max_pairs_per_batch=1,
    )

    result = agent_callable("Will it rain in Berlin tomorrow?")

    assert result.error is None
    assert result.available is True
    assert result.selected_tool == "weather_tool"
    assert result.tool_arguments == {"location": "Berlin"}
    assert result.calibrated_hf_mode is True
    assert captured_load["model_id"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert captured_extractor_kwargs["model"] is fake_scorer.model
    assert captured_extractor_kwargs["tokenizer"] is fake_scorer.tokenizer
    assert captured_extractor_kwargs["device"] == "cpu"


def test_app_hf_local_legacy_mode_still_uses_structured_router(monkeypatch) -> None:
    """Required test 6: the old LocalHFRouter path still works when not in calibrated mode.

    Same ``build_complete_agent_callable`` entry point, but with
    ``calibrated_hf_mode=False`` (the default for plain "HF local" inference
    outside the calibrated scorer), must still call ``load_local_hf_router``.
    """
    calls: list[str] = []

    class FakeRouterDecision:
        def __init__(self) -> None:
            self.selected_tool = "weather_tool"
            self.tool_arguments = {"location": "Berlin"}
            self.agent_response = "I would use weather_tool."
            self.raw_response = "{}"
            self.debug_prompt = None

    class FakeRouter:
        def choose_tool(self, user_request, tool_descriptions):
            del user_request, tool_descriptions
            return FakeRouterDecision()

    def fake_load_local_hf_router(
        model_name,
        max_new_tokens,
        *,
        trust_remote_code,
        quantization_mode="none",
        device="auto",
        dtype="auto",
    ):
        calls.append(model_name)
        del max_new_tokens, trust_remote_code, quantization_mode, device, dtype
        return FakeRouter()

    monkeypatch.setattr(app_module, "load_local_hf_router", fake_load_local_hf_router)

    agent_callable = app_module.build_complete_agent_callable(
        inference_backend="HF local",
        inference_model_name="Qwen/Qwen3-4B-Instruct-2507",
        system_prompt="Route to the correct tool.",
        tool_context="- weather_tool: forecasts",
    )

    result = agent_callable("Will it rain in Berlin tomorrow?")

    assert calls == ["Qwen/Qwen3-4B-Instruct-2507"]
    assert result.selected_tool == "weather_tool"
    assert result.tool_arguments == {"location": "Berlin"}


def make_fake_native_scorer(
    *,
    tokenizer: object | None = None,
    desired_logprobs_by_position: dict[int, float] | None = None,
    max_pairs_per_batch: int = 4,
) -> NativeToolCallScorer:
    import torch

    scorer = NativeToolCallScorer.__new__(NativeToolCallScorer)
    scorer.model_id = "fake-native-model"
    scorer.tokenizer_id = "fake-native-model"
    scorer.requested_quantization_mode = "none"
    scorer.actual_quantization_mode = "none"
    scorer.tokenizer = tokenizer or CharacterTensorTokenizer()
    scorer.model = ControlledLogitModel(desired_logprobs_by_position or {})
    scorer.device = "cpu"
    scorer.dtype = "auto"
    scorer.tool_schemas = get_executable_tool_schemas()
    scorer.last_debug_outputs = []
    scorer._score_cache = {}
    scorer._torch = torch
    scorer.max_pairs_per_batch = max_pairs_per_batch
    return scorer


def test_native_tool_call_scorer_rejects_no_tool() -> None:
    scorer = make_fake_native_scorer(tokenizer=NativeTemplateTokenizer())

    with pytest.raises(ValueError, match="not no_tool"):
        scorer.score_batch(
            ["System\n\nAvailable tools:\n- x\n\nUser request:\nExplain precision.\n\nAssistant:"],
            target_tool="no_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )


def test_native_tool_call_scorer_has_no_routing_label_dependency() -> None:
    source = inspect.getsource(NativeToolCallScorer)

    assert "ROUTING_LABELS" not in source
    assert "routing_labels" not in source


def test_native_tool_call_scorer_uses_teacher_forced_mean_logprob() -> None:
    scorer = make_fake_native_scorer(
        desired_logprobs_by_position={
            1: -0.2,
            2: -0.4,
        }
    )

    scores = scorer._sequence_mean_logprobs_batch(["ab"], ["cd"])

    assert scores == pytest.approx([(-0.2 + -0.4) / 2], abs=1e-6)


def test_native_tool_call_scorer_batches_and_returns_finite_scores(monkeypatch) -> None:
    scorer = make_fake_native_scorer(tokenizer=NativeTemplateTokenizer(), max_pairs_per_batch=2)
    batch_sizes = []

    def fake_batch(prompts: list[str], continuations: list[str]) -> list[float]:
        batch_sizes.append(len(prompts))
        assert len(prompts) == len(continuations)
        return [-0.5 for _ in prompts]

    monkeypatch.setattr(scorer, "_sequence_mean_logprobs_batch", fake_batch)
    prompts = [
        build_coalition_prompt(
            f"Request {index}",
            system_prompt="Route.",
            tool_context="- weather_tool: weather",
        )
        for index in range(5)
    ]

    scores = scorer.score_batch(
        prompts,
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert batch_sizes == [2, 2, 1]
    assert scores == [-0.5] * 5
    assert all(math.isfinite(score) for score in scores)


def test_native_tool_call_scorer_empty_subtraction_is_game_normalization(monkeypatch) -> None:
    scorer = make_fake_native_scorer(tokenizer=NativeTemplateTokenizer())

    def fake_score_batch(prompts, *, target_tool, tool_descriptions):
        del target_tool, tool_descriptions
        return [0.0 if "User request:\n\nAssistant:" in prompt else -0.25 for prompt in prompts]

    monkeypatch.setattr(scorer, "score_batch", fake_score_batch)
    game = ToolUseGame(
        target_tool="weather_tool",
        user_segments=["Will it rain", "in Berlin tomorrow?"],
        system_prompt="Route.",
        scorer=scorer,
        tool_descriptions=TOOL_DESCRIPTIONS,
        normalize=True,
    )

    empty = np.zeros((1, game.n_players), dtype=bool)
    full = np.ones((1, game.n_players), dtype=bool)

    assert game(empty)[0] == pytest.approx(0.0)
    assert game(full)[0] == pytest.approx(-0.25)


def test_no_tool_legacy_surrogate_remains_a_developer_ablation() -> None:
    """The legacy A/B/C/D no-tool probe is retained but no longer auto-selected."""
    assert "surrogate" in app_module.NO_TOOL_SURROGATE_SCORER_LABEL.lower()
    assert "surrogate" in app_module.NO_TOOL_SURROGATE_HELP.lower()
    assert "not executable" in app_module.NO_TOOL_SURROGATE_HELP
    assert "not native" in app_module.NO_TOOL_SURROGATE_HELP
    assert app_module.value_function_metadata(app_module.NO_TOOL_SURROGATE_SCORER_LABEL) == (
        "legacy_abcd_no_tool_probe",
        "surrogate_ablation",
    )


def test_no_tool_selects_native_direct_answer_branch_and_metadata() -> None:
    """A no_tool Agent Result now defaults to the native direct-answer scorer, not ABCD."""
    assert "direct-answer" in app_module.NATIVE_DIRECT_ANSWER_SCORER_LABEL.lower()
    assert "no-tool probability" not in app_module.NATIVE_DIRECT_ANSWER_SCORER_HELP.lower()
    assert "no-tool decision likelihood" not in app_module.NATIVE_DIRECT_ANSWER_SCORER_HELP.lower()
    assert "not a probability of choosing no tool" in app_module.NATIVE_DIRECT_ANSWER_SCORER_HELP
    assert (
        app_module.hf_value_function_label_for_target(
            "no_tool",
            app_module.NATIVE_HF_SCORER_LABEL,
        )
        == app_module.NATIVE_DIRECT_ANSWER_SCORER_LABEL
    )
    assert app_module.value_function_metadata(app_module.NATIVE_DIRECT_ANSWER_SCORER_LABEL) == (
        "native_direct_answer_continuation_likelihood",
        "native_direct_answer",
    )


def test_executable_tool_keeps_native_branch_and_metadata() -> None:
    assert "tool-identity" in app_module.NATIVE_HF_SCORER_LABEL
    assert "actual generated continuation" not in app_module.NATIVE_HF_SCORER_HELP
    assert "canonical native-format continuation" in app_module.NATIVE_HF_SCORER_HELP
    assert "excludes free-form argument tokens" in app_module.NATIVE_HF_SCORER_HELP
    assert (
        app_module.hf_value_function_label_for_target(
            "web_search_tool",
            app_module.NATIVE_HF_SCORER_LABEL,
        )
        == app_module.NATIVE_HF_SCORER_LABEL
    )
    assert app_module.value_function_metadata(app_module.NATIVE_HF_SCORER_LABEL) == (
        "native_target_tool_continuation_likelihood",
        "native_tool_call",
    )


def test_native_scorer_debug_metadata_describes_canonical_tool_identity() -> None:
    scorer = make_fake_native_scorer()

    scores = scorer.score_batch(
        [
            build_coalition_prompt(
                "Will it rain in Berlin tomorrow?",
                system_prompt="Route.",
                tool_context="- weather_tool: weather",
            )
        ],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )
    debug = scorer.last_debug_outputs[0]

    assert len(scores) == 1
    assert debug["target_source"] == "full-context native inference"
    assert debug["continuation_type"] == "canonical native template"
    assert debug["continuation_scope"] == "tool identity only"
    assert debug["arguments_included"] == "no"
    assert "tool-identity" in str(debug["score_description"])
    assert "free-form arguments are excluded" in str(debug["score_description"])


def test_documentation_wording_matches_canonical_continuation_design() -> None:
    readme_text = (DEMO_DIR / "README.md").read_text()
    normalized_readme = " ".join(readme_text.split())

    forbidden_phrases = [
        "actual generated continuation",
        "real ChatML tokens produced by the full-context inference",
        "exact full generated tool call",
        "exact generated tool call",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in readme_text
    assert "template-derived canonical native-format continuation" in normalized_readme
    assert "truncated at the tool identity" in normalized_readme
    assert "excludes free-form argument tokens" in normalized_readme
    assert "not argument generation or raw response reproduction" in normalized_readme
    assert "same model, tokenizer, chat template, device, and runtime" in normalized_readme


def test_agent_result_web_search_card_is_agent_action_without_probabilities() -> None:
    result = types.SimpleNamespace(
        selected_tool="web_search_tool",
        tool_arguments={"query": "latest Formula 1 race winner and team"},
    )

    html = app_module.render_agent_result_card(
        result,
        backend="HF local",
        model="Qwen/Qwen3-4B-Instruct-2507",
    )

    assert "🌐 Agent selected <code>web_search_tool</code>" in html
    assert "current or externally retrieved information" in html
    assert "query" in html
    assert "latest Formula 1 race winner and team" in html
    assert "Tool call prepared" in html
    assert "Qwen/Qwen3-4B-Instruct-2507 · HF Local" in html
    assert "Model routed to" not in html
    assert "Native local HF tool-call decision" not in html
    assert "No calibrated probability distribution is shown" not in html
    assert "probability-row" not in html


def test_agent_result_weather_card_renders_icon_and_arguments() -> None:
    result = types.SimpleNamespace(
        selected_tool="weather_tool",
        tool_arguments={"location": "Berlin", "date": "tomorrow"},
    )

    html = app_module.render_agent_result_card(result, backend="HF local", model="Qwen")

    assert "🌦️ Agent selected <code>weather_tool</code>" in html
    assert "weather information for a specific place or time" in html
    assert "location" in html
    assert "Berlin" in html
    assert "date" in html
    assert "tomorrow" in html


def test_agent_result_calculator_card_renders_expression_argument() -> None:
    result = types.SimpleNamespace(
        selected_tool="calculator_tool",
        tool_arguments={"expression": "238 * 47"},
    )

    html = app_module.render_agent_result_card(result, backend="HF local", model="Qwen")

    assert "🧮 Agent selected <code>calculator_tool</code>" in html
    assert "exact numerical calculation" in html
    assert "expression" in html
    assert "238 * 47" in html


def test_agent_result_direct_answer_hides_internal_no_tool_label() -> None:
    result = types.SimpleNamespace(
        selected_tool="no_tool",
        cleaned_direct_answer="Photosynthesis converts light energy into chemical energy.",
        tool_arguments={},
    )

    html = app_module.render_agent_result_card(result, backend="HF local", model="Qwen")

    assert "💬 Agent answered directly" in html
    assert "No external tool was called." in html
    assert "Photosynthesis converts light energy" in html
    assert "no_tool" not in html
    assert "NoTool" not in html
    assert "Arguments" not in html
    assert "Tool call prepared" not in html


def test_agent_result_parse_failure_is_not_direct_answer() -> None:
    result = types.SimpleNamespace(
        selected_tool=None,
        parse_error="Native tool-call JSON could not be parsed.",
        raw_response="<tool_call>{bad json}</tool_call>",
    )

    html = app_module.render_agent_result_card(result, backend="HF local", model="Qwen")

    assert "⚠️ Native tool-call parsing failed" in html
    assert "could not be parsed safely" in html
    assert "not treated as a direct answer" in html
    assert "Agent answered directly" not in html
    assert "no_tool" not in html


def test_agent_result_execution_status_reflects_available_tool_output() -> None:
    prepared = types.SimpleNamespace(selected_tool="weather_tool", tool_arguments={})
    executed = types.SimpleNamespace(
        selected_tool="weather_tool",
        tool_arguments={},
        raw_trace={"tool_output": "Weather lookup completed."},
    )

    prepared_html = app_module.render_agent_result_card(
        prepared,
        backend="HF local",
        model="Qwen",
    )
    executed_html = app_module.render_agent_result_card(
        executed,
        backend="Groq",
        model="llama-3.1-8b-instant",
    )

    assert "Tool call prepared" in prepared_html
    assert "Tool executed successfully" in executed_html
    assert "Weather lookup completed." in executed_html
    assert "llama-3.1-8b-instant · Groq" in executed_html


def test_lexical_tool_scorer_returns_one_score_per_prompt() -> None:
    scorer = LexicalToolScorer()

    scores = scorer.score_batch(
        [
            "Will it rain in Berlin tomorrow?",
            "Explain photosynthesis in simple terms.",
        ],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert len(scores) == 2
    assert all(0.0 <= score <= 1.0 for score in scores)


def test_lexical_tool_router_returns_mock_llm_choice() -> None:
    router = LexicalToolRouter()

    choice = router.choose_tool(
        "Will it rain in Berlin tomorrow morning?",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert choice.tool == "weather_tool"
    assert 0.0 <= choice.score <= 1.0
    assert set(choice.scores) == set(TOOL_DESCRIPTIONS)
    assert "weather" in choice.reason or "rain" in choice.reason


def test_tool_use_game_accepts_llm_scorer() -> None:
    game = ToolUseGame(
        target_tool="weather_tool",
        user_segments=["Will it rain tomorrow?"],
        system_prompt="Use weather_tool for forecasts.",
        scorer=LLMToolScorer(llm=MockLLM("0.75")),
        tool_descriptions=TOOL_DESCRIPTIONS,
        normalize=False,
    )
    coalitions = np.array([[False], [True]])

    scores = game.value_function(coalitions)

    assert scores.tolist() == [0.75, 0.75]


def test_tool_use_game_builds_coalition_prompt() -> None:
    game = ToolUseGame(
        target_tool="weather_tool",
        user_segments=[ToolUseSegment("user", "U1", "Will it rain tomorrow?")],
        system_prompt="Use weather_tool for forecasts.",
        tool_descriptions=TOOL_DESCRIPTIONS,
        normalize=False,
    )

    prompt = game.build_prompt([])

    assert "Use weather_tool for forecasts." in prompt
    assert "Available tools:" in prompt
    assert "Will it rain tomorrow?" not in prompt
    assert "User request:\n\nAssistant:" in prompt


# ===========================================================================
# Native direct-answer continuation: target extraction + scorer
# ===========================================================================


class WordTokenizer:
    """Deterministic, reversible whitespace-word tokenizer for target-extraction tests.

    Splits on literal single spaces so ``decode(encode(text)) == text`` for any
    input using single-space word separation -- no real HF tokenizer is loaded.
    """

    def __init__(self) -> None:
        self._id_to_word: list[str] = []
        self._word_to_id: dict[str, int] = {}

    def _id_for(self, word: str) -> int:
        if word not in self._word_to_id:
            self._word_to_id[word] = len(self._id_to_word)
            self._id_to_word.append(word)
        return self._word_to_id[word]

    def __call__(self, text: str, *, add_special_tokens: bool = False) -> dict[str, list[int]]:
        del add_special_tokens
        return {"input_ids": [self._id_for(word) for word in text.split(" ")]}

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(self._id_to_word[token_id] for token_id in token_ids)


def test_direct_answer_target_single_sentence() -> None:
    answer = "Photosynthesis converts light energy into chemical energy for plants today."
    target = build_canonical_direct_answer_target(answer, tokenizer=WordTokenizer())

    assert target == answer


def test_direct_answer_target_uses_first_qualifying_sentence() -> None:
    answer = (
        "Photosynthesis converts light energy into chemical energy for plants. "
        "Chlorophyll absorbs the light needed for this reaction to occur reliably."
    )

    target = build_canonical_direct_answer_target(answer, tokenizer=WordTokenizer())

    assert target == "Photosynthesis converts light energy into chemical energy for plants."


def test_direct_answer_target_removes_generic_preamble() -> None:
    answer = "Sure, photosynthesis converts light energy into chemical energy for plants."

    target = build_canonical_direct_answer_target(answer, tokenizer=WordTokenizer())

    assert not target.startswith("Sure,")
    assert target.startswith("photosynthesis")


def test_direct_answer_target_without_punctuation_uses_whole_text() -> None:
    answer = "Photosynthesis converts light energy into chemical energy for plants"

    target = build_canonical_direct_answer_target(answer, tokenizer=WordTokenizer())

    assert target == answer


def test_direct_answer_target_extends_past_short_first_sentence() -> None:
    answer = "Yes. It converts light energy into chemical energy for plants reliably."

    target = build_canonical_direct_answer_target(
        answer, tokenizer=WordTokenizer(), min_target_tokens=4
    )

    assert target.startswith("Yes. It converts")
    assert len(target.split(" ")) >= 4


def test_direct_answer_target_truncates_to_max_tokens() -> None:
    words = [f"word{index}" for index in range(40)]
    answer = " ".join(words) + "."

    target = build_canonical_direct_answer_target(
        answer, tokenizer=WordTokenizer(), max_target_tokens=10, min_target_tokens=4
    )

    assert target.split(" ") == words[:10]


def test_direct_answer_target_handles_unicode_text() -> None:
    answer = "Café résumé naïve façade coöperate — a quick summary of these terms."

    target = build_canonical_direct_answer_target(answer, tokenizer=WordTokenizer())

    assert target == answer


def test_direct_answer_target_preserves_decimal_numbers() -> None:
    answer = "Pi is approximately 3.14 which is useful for circle calculations today."

    target = build_canonical_direct_answer_target(answer, tokenizer=WordTokenizer())

    assert target == answer
    assert "3.14" in target


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        (
            "Dr. Smith explained the theory in simple terms for the class. It helped.",
            "Dr. Smith explained the theory in simple terms for the class.",
        ),
        (
            "This is common knowledge, e.g. water boils at 100 degrees. It is well known.",
            "This is common knowledge, e.g. water boils at 100 degrees.",
        ),
    ],
)
def test_direct_answer_target_preserves_abbreviations(answer: str, expected: str) -> None:
    target = build_canonical_direct_answer_target(answer, tokenizer=WordTokenizer())

    assert target == expected


def test_direct_answer_target_empty_answer_raises() -> None:
    with pytest.raises(ValueError, match="No usable"):
        build_canonical_direct_answer_target("   ", tokenizer=WordTokenizer())


def test_direct_answer_target_sentinel_only_raises() -> None:
    with pytest.raises(ValueError, match="sentinel"):
        build_canonical_direct_answer_target("no_tool", tokenizer=WordTokenizer())


def test_direct_answer_target_strips_termination_artifacts() -> None:
    answer = "Photosynthesis converts light energy into chemical energy. <|im_end|>"

    target = build_canonical_direct_answer_target(answer, tokenizer=WordTokenizer())

    assert "<|im_end|>" not in target
    assert target == "Photosynthesis converts light energy into chemical energy."


def test_direct_answer_target_decode_after_truncation_produces_nonempty_target() -> None:
    words = [f"word{index}" for index in range(40)]
    answer = " ".join(words) + "."

    target = build_canonical_direct_answer_target(
        answer, tokenizer=WordTokenizer(), max_target_tokens=5, min_target_tokens=4
    )

    assert target.strip() != ""


def test_direct_answer_target_extraction_is_deterministic() -> None:
    answer = (
        "Photosynthesis converts light energy into chemical energy for plants. "
        "More text follows this sentence for good measure."
    )

    first = build_canonical_direct_answer_target(answer, tokenizer=WordTokenizer())
    second = build_canonical_direct_answer_target(answer, tokenizer=WordTokenizer())

    assert first == second


def test_frozen_continuation_target_is_immutable() -> None:
    target = FrozenContinuationTarget(
        kind="direct_answer", text="abc", token_ids=(1, 2, 3), source="test"
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        target.text = "changed"  # type: ignore[misc]


def test_build_native_direct_answer_prompt_matches_tool_call_prompt() -> None:
    tokenizer = NativeTemplateTokenizer()

    direct_prompt = build_native_direct_answer_prompt(
        tokenizer, system_prompt="Route.", user_request="Explain photosynthesis."
    )
    tool_prompt = build_native_tool_call_prompt(
        tokenizer, system_prompt="Route.", user_request="Explain photosynthesis."
    )

    assert direct_prompt == tool_prompt


class DirectAnswerFakeTokenizer(NativeTemplateTokenizer):
    """Fake tokenizer supporting both single-text id lookup and batched tensor calls.

    ``NativeDirectAnswerScorer`` needs both: a plain ``tokenizer(text,
    add_special_tokens=False)`` call (to freeze the target's token identity at
    construction) and the batched ``tokenizer(list_of_texts, return_tensors=...,
    padding=...)`` call used by the shared teacher-forced scoring machinery.
    """

    def __call__(
        self,
        texts: object,
        *,
        add_special_tokens: bool = False,
        return_tensors: str | None = None,
        padding: bool | None = None,
    ) -> dict[str, object]:
        del add_special_tokens
        if isinstance(texts, str):
            return {"input_ids": [ord(character) for character in texts]}
        del return_tensors, padding
        import torch

        rows = [[ord(character) for character in text] for text in texts]
        max_len = max(len(row) for row in rows)
        input_ids = torch.zeros((len(rows), max_len), dtype=torch.long)
        attention_mask = torch.zeros((len(rows), max_len), dtype=torch.long)
        for row_index, row in enumerate(rows):
            input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long)
            attention_mask[row_index, : len(row)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def make_direct_answer_scorer(
    *,
    direct_answer_target: str = "ok",
    model: object | None = None,
    tokenizer: object | None = None,
    max_pairs_per_batch: int = 4,
) -> NativeDirectAnswerScorer:
    return NativeDirectAnswerScorer(
        model=model or ControlledLogitModel({}),
        tokenizer=tokenizer or DirectAnswerFakeTokenizer(),
        device="cpu",
        model_id="fake-direct-answer-model",
        max_pairs_per_batch=max_pairs_per_batch,
        direct_answer_target=direct_answer_target,
    )


def test_direct_answer_scorer_rejects_non_no_tool_target() -> None:
    scorer = make_direct_answer_scorer()

    with pytest.raises(ValueError, match="no_tool"):
        scorer.score_batch(
            [build_coalition_prompt("Explain X", system_prompt="Route.", tool_context="- x: y")],
            target_tool="weather_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )


def test_direct_answer_scorer_rejects_empty_target() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        make_direct_answer_scorer(direct_answer_target="   ")


def test_direct_answer_scorer_uses_same_frozen_target_for_every_coalition() -> None:
    scorer = make_direct_answer_scorer(direct_answer_target="Photosynthesis converts light.")
    prompts = [
        build_coalition_prompt(text, system_prompt="Route.", tool_context="- x: y")
        for text in ("Explain X", "Explain Y", "")
    ]

    previews = [
        scorer.build_scoring_prompt(
            prompt, target_tool=NO_TOOL_NAME, tool_descriptions=TOOL_DESCRIPTIONS
        )
        for prompt in prompts
    ]

    assert all(preview.endswith("Photosynthesis converts light.") for preview in previews)
    assert scorer.direct_answer_target == "Photosynthesis converts light."


def test_direct_answer_scorer_never_calls_generate() -> None:
    class ExplodingGenerateModel(ControlledLogitModel):
        def generate(self, *args: object, **kwargs: object) -> object:
            msg = "generate() must not be called during coalition scoring."
            raise AssertionError(msg)

    scorer = make_direct_answer_scorer(model=ExplodingGenerateModel({}))
    prompt = build_coalition_prompt("Explain X", system_prompt="Route.", tool_context="- x: y")

    scores = scorer.score_batch(
        [prompt], target_tool=NO_TOOL_NAME, tool_descriptions=TOOL_DESCRIPTIONS
    )

    assert len(scores) == 1


def test_direct_answer_scorer_one_batched_call_scores_all_coalitions_in_order(
    monkeypatch,
) -> None:
    scorer = make_direct_answer_scorer(direct_answer_target="ok")
    seen_prompts: list[list[str]] = []

    def fake_batched(prompts: list[str], continuations: list[str]) -> list[float]:
        seen_prompts.append(list(prompts))
        assert continuations == [scorer.frozen_target.text] * len(prompts)
        return [-0.1 * index for index in range(len(prompts))]

    monkeypatch.setattr(scorer, "_sequence_mean_logprobs_batched", fake_batched)
    prompts = [
        build_coalition_prompt(f"Request {index}", system_prompt="Route.", tool_context="- x: y")
        for index in range(4)
    ]

    scores = scorer.score_batch(
        prompts, target_tool=NO_TOOL_NAME, tool_descriptions=TOOL_DESCRIPTIONS
    )

    expected_model_prompts = [
        build_native_direct_answer_prompt(
            scorer.tokenizer,
            system_prompt=split_coalition_prompt(prompt)[0],
            user_request=split_coalition_prompt(prompt)[1],
        )
        for prompt in prompts
    ]
    assert len(seen_prompts) == 1
    assert seen_prompts[0] == expected_model_prompts
    assert scores == pytest.approx([-0.1 * index for index in range(4)])


def test_direct_answer_scorer_reuses_cache_for_repeated_coalition_prompt(monkeypatch) -> None:
    scorer = make_direct_answer_scorer()
    call_counter = {"n": 0}
    original = scorer._sequence_mean_logprobs_batched

    def counting(prompts: list[str], continuations: list[str]) -> list[float]:
        call_counter["n"] += 1
        return original(prompts, continuations)

    monkeypatch.setattr(scorer, "_sequence_mean_logprobs_batched", counting)
    prompt = build_coalition_prompt("Explain X", system_prompt="Route.", tool_context="- x: y")

    first = scorer.score_batch(
        [prompt], target_tool=NO_TOOL_NAME, tool_descriptions=TOOL_DESCRIPTIONS
    )
    second = scorer.score_batch(
        [prompt, prompt], target_tool=NO_TOOL_NAME, tool_descriptions=TOOL_DESCRIPTIONS
    )

    assert call_counter["n"] == 1
    assert second == [first[0], first[0]]


def test_direct_answer_scorer_cache_key_includes_target_identity() -> None:
    scorer_a = make_direct_answer_scorer(direct_answer_target="alpha")
    scorer_b = make_direct_answer_scorer(direct_answer_target="beta")
    prompt = build_coalition_prompt("Explain X", system_prompt="Route.", tool_context="- x: y")

    scorer_a.score_batch([prompt], target_tool=NO_TOOL_NAME, tool_descriptions=TOOL_DESCRIPTIONS)
    scorer_b.score_batch([prompt], target_tool=NO_TOOL_NAME, tool_descriptions=TOOL_DESCRIPTIONS)

    key_a = next(iter(scorer_a._score_cache))
    key_b = next(iter(scorer_b._score_cache))
    assert key_a[0] == key_b[0] == prompt
    assert key_a[1] != key_b[1]  # different frozen target token identity -- no cache collision


def test_direct_answer_scorer_rejects_non_finite_score() -> None:
    class NaNLogitModel:
        def eval(self) -> None:
            return None

        def __call__(self, *, input_ids: object, attention_mask: object) -> object:
            del attention_mask
            import torch

            vocab_size = 256
            logits = torch.full((input_ids.shape[0], input_ids.shape[1], vocab_size), float("nan"))
            return types.SimpleNamespace(logits=logits)

    scorer = make_direct_answer_scorer(model=NaNLogitModel())
    prompt = build_coalition_prompt("Explain X", system_prompt="Route.", tool_context="- x: y")

    with pytest.raises(ValueError, match="finite"):
        scorer.score_batch([prompt], target_tool=NO_TOOL_NAME, tool_descriptions=TOOL_DESCRIPTIONS)


def test_direct_answer_scorer_empty_subtraction_is_game_normalization(monkeypatch) -> None:
    scorer = make_direct_answer_scorer()

    def fake_score_batch(prompts, *, target_tool, tool_descriptions):
        del target_tool, tool_descriptions
        return [0.0 if "User request:\n\nAssistant:" in prompt else -0.3 for prompt in prompts]

    monkeypatch.setattr(scorer, "score_batch", fake_score_batch)
    game = ToolUseGame(
        target_tool=NO_TOOL_NAME,
        user_segments=["Explain X", "in simple terms"],
        system_prompt="Route.",
        scorer=scorer,
        tool_descriptions=TOOL_DESCRIPTIONS,
        normalize=True,
    )

    empty = np.zeros((1, game.n_players), dtype=bool)
    full = np.ones((1, game.n_players), dtype=bool)

    assert game(empty)[0] == pytest.approx(0.0)
    assert game(full)[0] == pytest.approx(-0.3)


def test_direct_answer_and_tool_call_scorers_share_sequence_scoring_implementation() -> None:
    """Both scorer families must reuse one lower-level teacher-forced scoring implementation."""
    assert (
        NativeDirectAnswerScorer._sequence_mean_logprobs_batch
        is NativeToolCallScorer._sequence_mean_logprobs_batch
    )
    assert (
        NativeDirectAnswerScorer._next_token_log_probs is NativeToolCallScorer._next_token_log_probs
    )
    assert (
        NativeDirectAnswerScorer._sequence_mean_logprobs_batched
        is NativeToolCallScorer._sequence_mean_logprobs_batched
    )


# ---------------------------------------------------------------------------
# app.py integration: direct-answer target sourcing and scorer selection
# ---------------------------------------------------------------------------


def test_extract_direct_answer_text_prefers_cleaned_direct_answer() -> None:
    result = types.SimpleNamespace(
        cleaned_direct_answer="Cleaned answer.",
        direct_answer="Raw direct answer.",
        agent_response="Agent response.",
    )

    assert app_module.extract_direct_answer_text(result) == "Cleaned answer."


def test_extract_direct_answer_text_falls_back_in_priority_order() -> None:
    only_direct_answer = types.SimpleNamespace(
        cleaned_direct_answer=None, direct_answer="Raw direct answer.", agent_response="Agent."
    )
    only_agent_response = types.SimpleNamespace(
        cleaned_direct_answer="", direct_answer=None, agent_response="Agent response."
    )

    assert app_module.extract_direct_answer_text(only_direct_answer) == "Raw direct answer."
    assert app_module.extract_direct_answer_text(only_agent_response) == "Agent response."


def test_extract_direct_answer_text_never_uses_raw_response() -> None:
    result = types.SimpleNamespace(
        cleaned_direct_answer=None,
        direct_answer=None,
        agent_response=None,
        raw_response="no_tool\nSome answer.",
    )

    assert app_module.extract_direct_answer_text(result) is None


def test_require_trusted_direct_answer_target_rejects_parse_failure() -> None:
    result = types.SimpleNamespace(
        parse_error="malformed", selected_tool=None, cleaned_direct_answer=None
    )

    with pytest.raises(RuntimeError, match="trusted current HF-native direct answer"):
        app_module.require_trusted_direct_answer_target(result, inference_backend="HF local")


def test_require_trusted_direct_answer_target_rejects_non_no_tool_selection() -> None:
    result = types.SimpleNamespace(
        parse_error=None, selected_tool="weather_tool", cleaned_direct_answer=None
    )

    with pytest.raises(RuntimeError, match="trusted current HF-native direct answer"):
        app_module.require_trusted_direct_answer_target(result, inference_backend="HF local")


def test_require_trusted_direct_answer_target_rejects_non_hf_local_backend() -> None:
    result = types.SimpleNamespace(
        parse_error=None, selected_tool="no_tool", cleaned_direct_answer="An answer."
    )

    with pytest.raises(RuntimeError, match="trusted current HF-native direct answer"):
        app_module.require_trusted_direct_answer_target(result, inference_backend="Groq")


def test_require_trusted_direct_answer_target_rejects_missing_result() -> None:
    with pytest.raises(RuntimeError, match="trusted current HF-native direct answer"):
        app_module.require_trusted_direct_answer_target(None, inference_backend="HF local")


def test_require_trusted_direct_answer_target_returns_answer_when_trusted() -> None:
    result = types.SimpleNamespace(
        parse_error=None,
        selected_tool="no_tool",
        cleaned_direct_answer="Photosynthesis converts light energy into chemical energy.",
    )

    answer = app_module.require_trusted_direct_answer_target(result, inference_backend="HF local")

    assert answer == "Photosynthesis converts light energy into chemical energy."


def test_hf_direct_answer_ui_wording_does_not_claim_no_tool_probability() -> None:
    """UI/help text for the native direct-answer scorer must not overclaim."""
    forbidden_phrases = [
        "no-tool probability",
        "no-tool decision likelihood",
        "probability of answering directly",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in app_module.NATIVE_DIRECT_ANSWER_SCORER_HELP.lower()
