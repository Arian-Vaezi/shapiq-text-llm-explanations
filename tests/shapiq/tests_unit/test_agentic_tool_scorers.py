"""Tests for agentic tool-use demo scorers."""

from __future__ import annotations

import inspect
import math
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

import app as app_module  # noqa: E402
from scorers import (  # noqa: E402
    CALIBRATION_USER_REQUESTS,
    ROUTING_LABELS,
    CalibratedToolLogOddsScorer,
    LexicalToolRouter,
    LexicalToolScorer,
    LLMToolScorer,
    MockLLM,
    RoutingLabelTokenization,
    build_coalition_prompt,
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

    def fake_load_logprob_scorer(model_id, *, max_pairs_per_batch):
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

    def fake_load_local_hf_router(model_name, max_new_tokens, *, trust_remote_code):
        calls.append(model_name)
        del max_new_tokens, trust_remote_code
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
