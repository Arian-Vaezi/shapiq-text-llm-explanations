"""Tests for agentic tool-use demo scorers."""

from __future__ import annotations

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

from scorers import (  # noqa: E402
    LexicalToolRouter,
    LexicalToolScorer,
    LLMToolScorer,
    LogProbToolScorer,
    MockLLM,
    build_tool_calling_prompt,
    split_coalition_prompt,
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


def make_fake_logprob_scorer(
    log_scores: dict[str, float] | list[dict[str, float]],
    candidate_texts: dict[str, str] | None = None,
) -> LogProbToolScorer:
    scorer = LogProbToolScorer.__new__(LogProbToolScorer)
    scorer.candidate_template = "The correct tool is {tool_name}."
    scorer.candidate_texts = candidate_texts or {}
    scorer.normalize_by_length = True
    scorer.last_debug_outputs = []

    def fake_candidate_log_scores(
        prompts: list[str],
        candidate_tools: list[str],
    ) -> list[dict[str, float]]:
        del candidate_tools
        if isinstance(log_scores, list):
            return log_scores
        return [log_scores.copy() for _ in prompts]

    scorer._candidate_log_scores = fake_candidate_log_scores
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


def test_logprob_tool_scorer_builds_model_prompt_with_structured_schemas() -> None:
    scorer = LogProbToolScorer.__new__(LogProbToolScorer)
    scorer.candidate_template = "The correct tool is {tool_name}."
    scorer.candidate_texts = {}
    scorer.normalize_by_length = True
    scorer.last_debug_outputs = []
    scorer.tool_schemas = get_executable_tool_schemas()
    scorer.tokenizer = RecordingTokenizer()
    captured = {}

    def fake_sequence_logprobs_batched(
        prompts: list[str],
        continuations: list[str],
    ) -> list[float]:
        captured["prompts"] = prompts
        captured["continuations"] = continuations
        return [-1.0 for _ in prompts]

    scorer._sequence_logprobs_batched = fake_sequence_logprobs_batched

    scorer.score_batch(
        [
            "You are a tool router.\n\n"
            "Available tools:\n- weather_tool: old text\n\n"
            "User request:\nWill it rain?\n\n"
            "Assistant:"
        ],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    call = scorer.tokenizer.calls[0]
    assert call["messages"] == [
        {"role": "system", "content": "You are a tool router."},
        {"role": "user", "content": "Will it rain?"},
    ]
    assert call["tools"] == get_executable_tool_schemas()
    assert captured["prompts"] == ["native chat prompt"] * len(TOOL_DESCRIPTIONS)
    assert len(captured["continuations"]) == len(TOOL_DESCRIPTIONS)


def test_logprob_sequence_logprobs_batched_chunks_and_preserves_order() -> None:
    scorer = LogProbToolScorer.__new__(LogProbToolScorer)
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


def test_logprob_tool_scorer_build_scoring_prompt_preview_uses_model_prompt() -> None:
    scorer = LogProbToolScorer.__new__(LogProbToolScorer)
    scorer.tool_schemas = get_executable_tool_schemas()
    scorer.tokenizer = RecordingTokenizer()
    full_prompt = (
        "Use calculator_tool for arithmetic.\n\n"
        "Available tools:\n- calculator_tool: Calculator\n\n"
        "User request:\nCalculate 238 times 47\n\n"
        "Assistant:"
    )

    preview = scorer.build_scoring_prompt(
        full_prompt,
        target_tool="calculator_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert preview == scorer._model_prompt(full_prompt)
    assert preview == "native chat prompt"
    assert scorer.tokenizer.calls[0]["messages"] == [
        {"role": "system", "content": "Use calculator_tool for arithmetic."},
        {"role": "user", "content": "Calculate 238 times 47"},
    ]


def test_logprob_next_token_log_probs_match_full_log_softmax_gather() -> None:
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
    actual = LogProbToolScorer._next_token_log_probs(logits, token_ids)

    assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-6)


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


def test_logprob_tool_scorer_returns_one_float_per_prompt() -> None:
    scorer = make_fake_logprob_scorer(
        {
            "weather_tool": 2.0,
            "calculator_tool": 0.0,
            "web_search_tool": -1.0,
            "no_tool": -2.0,
        }
    )

    scores = scorer.score_batch(
        ["Prompt one", "Prompt two"],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert isinstance(scores, list)
    assert len(scores) == 2
    assert all(isinstance(score, float) for score in scores)
    assert len(scorer.last_debug_outputs) == 2


def test_logprob_tool_scorer_returns_target_vs_no_tool_difference() -> None:
    scorer = make_fake_logprob_scorer(
        {
            "weather_tool": -0.4,
            "calculator_tool": -2.2,
            "web_search_tool": -1.6,
            "no_tool": -1.9,
        }
    )

    scores = scorer.score_batch(
        ["Will it rain tomorrow?"],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == pytest.approx([1.5])


def test_logprob_tool_scorer_returns_negative_contrastive_value() -> None:
    scorer = make_fake_logprob_scorer(
        {
            "weather_tool": -2.5,
            "no_tool": -0.5,
        }
    )
    tool_descriptions = {
        "weather_tool": TOOL_DESCRIPTIONS["weather_tool"],
        "no_tool": TOOL_DESCRIPTIONS["no_tool"],
    }

    scores = scorer.score_batch(
        ["Will it rain tomorrow?"],
        target_tool="weather_tool",
        tool_descriptions=tool_descriptions,
    )

    assert scores == pytest.approx([-2.0])


def test_logprob_tool_scorer_handles_no_tool_as_target() -> None:
    scorer = make_fake_logprob_scorer(
        {
            "no_tool": -0.3,
            "weather_tool": -1.0,
            "calculator_tool": -2.0,
            "web_search_tool": -1.5,
        }
    )

    scores = scorer.score_batch(
        ["Explain photosynthesis."],
        target_tool="no_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == pytest.approx([0.7])


def test_logprob_tool_scorer_preserves_batch_order() -> None:
    scorer = make_fake_logprob_scorer(
        [
            {
                "weather_tool": -0.4,
                "calculator_tool": -2.2,
                "web_search_tool": -1.6,
                "no_tool": -1.9,
            },
            {
                "weather_tool": -2.5,
                "calculator_tool": -1.5,
                "web_search_tool": -1.2,
                "no_tool": -0.5,
            },
        ]
    )

    scores = scorer.score_batch(
        ["Will it rain tomorrow?", "Explain the water cycle."],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert scores == pytest.approx([1.5, -2.0])


def test_logprob_tool_scorer_debug_contains_contrastive_scores() -> None:
    scorer = make_fake_logprob_scorer(
        {
            "weather_tool": 1.5,
            "calculator_tool": 0.5,
            "web_search_tool": -0.5,
            "no_tool": -1.5,
        }
    )

    scorer.score_batch(
        ["Will it rain tomorrow?"],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    debug_output = scorer.last_debug_outputs[0]
    assert debug_output["target_tool"] == "weather_tool"
    assert debug_output["reference_tool"] == "no_tool"
    assert set(debug_output["candidate_tools"]) == set(TOOL_DESCRIPTIONS)
    assert debug_output["candidate_log_scores"] == [1.5, 0.5, -0.5, -1.5]
    assert debug_output["final_score"] == pytest.approx(3.0)


def test_logprob_tool_scorer_uses_candidate_text_override_for_no_tool() -> None:
    no_tool_text = "The assistant should answer directly without using an external tool."
    scorer = make_fake_logprob_scorer(
        {
            "weather_tool": 1.0,
            "calculator_tool": -1.0,
            "web_search_tool": -2.0,
            "no_tool": 0.5,
        },
        candidate_texts={"no_tool": no_tool_text},
    )

    scorer.score_batch(
        ["Explain photosynthesis."],
        target_tool="no_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    debug_output = scorer.last_debug_outputs[0]
    no_tool_index = debug_output["candidate_tools"].index("no_tool")
    assert debug_output["candidate_continuations"][no_tool_index] == no_tool_text


def test_logprob_tool_scorer_falls_back_to_template_for_missing_candidate_text() -> None:
    scorer = make_fake_logprob_scorer(
        {
            "weather_tool": 1.0,
            "calculator_tool": -1.0,
            "web_search_tool": -2.0,
            "no_tool": 0.5,
        },
        candidate_texts={"no_tool": "Answer directly."},
    )

    scorer.score_batch(
        ["Will it rain tomorrow?"],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    debug_output = scorer.last_debug_outputs[0]
    weather_index = debug_output["candidate_tools"].index("weather_tool")
    assert (
        debug_output["candidate_continuations"][weather_index]
        == "The correct tool is weather_tool."
    )


def test_logprob_tool_scorer_debug_contains_candidate_continuations() -> None:
    scorer = make_fake_logprob_scorer(
        {
            "weather_tool": 1.0,
            "calculator_tool": -1.0,
            "web_search_tool": -2.0,
            "no_tool": 0.5,
        }
    )

    scorer.score_batch(
        ["Will it rain tomorrow?"],
        target_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    debug_output = scorer.last_debug_outputs[0]
    assert "candidate_continuations" in debug_output
    assert len(debug_output["candidate_continuations"]) == len(TOOL_DESCRIPTIONS)


def test_logprob_tool_scorer_requires_candidate_tools() -> None:
    scorer = make_fake_logprob_scorer({"weather_tool": 1.0})

    with pytest.raises(ValueError):
        scorer.score_batch(
            ["Will it rain tomorrow?"],
            target_tool="weather_tool",
            tool_descriptions={},
        )


def test_logprob_tool_scorer_requires_no_tool_candidate() -> None:
    scorer = make_fake_logprob_scorer({"weather_tool": 1.0, "calculator_tool": 0.0})

    with pytest.raises(ValueError, match="no_tool"):
        scorer.score_batch(
            ["Will it rain tomorrow?"],
            target_tool="weather_tool",
            tool_descriptions={
                "weather_tool": TOOL_DESCRIPTIONS["weather_tool"],
                "calculator_tool": TOOL_DESCRIPTIONS["calculator_tool"],
            },
        )


def test_logprob_tool_scorer_rejects_unknown_target_tool() -> None:
    scorer = make_fake_logprob_scorer({"weather_tool": 1.0, "no_tool": 0.0})

    with pytest.raises(ValueError, match="not available"):
        scorer.score_batch(
            ["Will it rain tomorrow?"],
            target_tool="calendar_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )


def test_logprob_tool_scorer_rejects_non_finite_candidate_score() -> None:
    scorer = make_fake_logprob_scorer(
        {
            "weather_tool": float("nan"),
            "calculator_tool": -2.2,
            "web_search_tool": -1.6,
            "no_tool": -1.9,
        }
    )

    with pytest.raises(ValueError, match="finite"):
        scorer.score_batch(
            ["Will it rain tomorrow?"],
            target_tool="weather_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )


def test_logprob_tool_scorer_rejects_missing_candidate_score() -> None:
    scorer = make_fake_logprob_scorer(
        {
            "weather_tool": -0.4,
            "calculator_tool": -2.2,
            "no_tool": -1.9,
        }
    )

    with pytest.raises(ValueError, match="web_search_tool"):
        scorer.score_batch(
            ["Will it rain tomorrow?"],
            target_tool="weather_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )


def test_logprob_tool_scorer_rejects_candidate_result_count_mismatch() -> None:
    scorer = make_fake_logprob_scorer(
        [
            {
                "weather_tool": -0.4,
                "calculator_tool": -2.2,
                "web_search_tool": -1.6,
                "no_tool": -1.9,
            }
        ]
    )

    with pytest.raises(ValueError, match="one score dictionary per prompt"):
        scorer.score_batch(
            ["Prompt one", "Prompt two"],
            target_tool="weather_tool",
            tool_descriptions=TOOL_DESCRIPTIONS,
        )


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
