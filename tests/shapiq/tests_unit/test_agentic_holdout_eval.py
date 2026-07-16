"""Tests for the native HF tool-routing hold-out evaluation."""

from __future__ import annotations

import argparse
import json

import pytest

from demos.agentic_tool_use_explanation import run_holdout_eval as holdout
from demos.agentic_tool_use_explanation.hf_router import RouterDecision


def make_sample(
    sample_id: str,
    request: str,
    ground_truth: str,
    *,
    is_boundary: bool = False,
) -> dict[str, object]:
    """Build one valid hold-out sample."""
    return {
        "id": sample_id,
        "request": request,
        "ground_truth": ground_truth,
        "is_boundary": is_boundary,
    }


def make_decision(
    selected_tool: str | None,
    *,
    parse_error: str | None = None,
    raw_response: str = "native response",
) -> RouterDecision:
    """Build a native-router decision without loading a model."""
    return RouterDecision(
        agent_response="Direct answer" if selected_tool == "no_tool" else "",
        selected_tool=selected_tool,
        tool_arguments={"query": "example"} if selected_tool == "web_search_tool" else {},
        raw_response=raw_response,
        parse_error=parse_error,
        direct_answer="Direct answer" if selected_tool == "no_tool" else None,
    )


class FakeNativeRouter(holdout.LocalHFRouter):
    """Return configured native decisions while recording production call arguments."""

    def __init__(self, decisions: dict[str, RouterDecision]) -> None:
        self.decisions = decisions
        self.calls: list[tuple[str, object, str]] = []

    def choose_tool(
        self,
        user_request: str,
        tool_descriptions: object,
        *,
        system_prompt: str,
    ) -> RouterDecision:
        self.calls.append((user_request, tool_descriptions, system_prompt))
        return self.decisions[user_request]


def test_evaluate_sample_supports_all_canonical_native_outcomes() -> None:
    samples = [
        make_sample("w01", "weather request", "weather_tool"),
        make_sample("c01", "calculator request", "calculator_tool"),
        make_sample("s01", "search request", "web_search_tool"),
        make_sample("n01", "direct request", "no_tool", is_boundary=True),
    ]
    router = FakeNativeRouter(
        {
            "weather request": make_decision("weather_tool"),
            "calculator request": make_decision("calculator_tool"),
            "search request": make_decision("web_search_tool"),
            "direct request": make_decision("no_tool", raw_response="A stable direct answer."),
        }
    )

    results = [
        holdout.evaluate_sample(router, sample, system_prompt="system prompt") for sample in samples
    ]

    assert [result["selected_tool"] for result in results] == list(holdout.TOOL_NAMES)
    assert [result["parser_status"] for result in results] == [
        "native_tool_call",
        "native_tool_call",
        "native_tool_call",
        "direct_answer",
    ]
    assert all(result["correct"] is True for result in results)
    assert all(float(result["elapsed_seconds"]) >= 0.0 for result in results)
    assert len(router.calls) == len(samples)
    assert all(call[1] is holdout.TOOLS for call in router.calls)
    assert all(call[2] == "system prompt" for call in router.calls)


def test_parser_failure_is_recorded_without_fabricating_a_tool() -> None:
    sample = make_sample("w01", "malformed request", "weather_tool")
    router = FakeNativeRouter(
        {
            "malformed request": make_decision(
                None,
                parse_error="Native tool-call JSON could not be parsed.",
                raw_response="<tool_call>{bad json}</tool_call>",
            )
        }
    )

    result = holdout.evaluate_sample(router, sample, system_prompt="system prompt")

    assert result["selected_tool"] is None
    assert result["correct"] is False
    assert result["parser_status"] == "parse_failure"
    assert result["parse_error"] == "Native tool-call JSON could not be parsed."
    assert result["raw_response"] == "<tool_call>{bad json}</tool_call>"


@pytest.mark.parametrize(
    ("payload", "expected_error"),
    [
        ({"id": "w01"}, holdout.InvalidTestSetTypeError),
        ([42], holdout.InvalidSampleTypeError),
        ([{}], holdout.InvalidSampleIdError),
        (
            [
                {"id": "duplicate", "request": "first", "ground_truth": "no_tool"},
                {"id": "duplicate", "request": "second", "ground_truth": "no_tool"},
            ],
            holdout.DuplicateSampleIdError,
        ),
        (
            [{"id": "w01", "request": "  ", "ground_truth": "weather_tool"}],
            holdout.InvalidSampleRequestError,
        ),
        (
            [{"id": "w01", "request": "weather", "ground_truth": "calendar_tool"}],
            holdout.InvalidGroundTruthError,
        ),
    ],
)
def test_load_samples_validates_json(
    payload: object, expected_error: type[Exception], tmp_path
) -> None:
    path = tmp_path / "samples.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(expected_error):
        holdout.load_samples(path)


def test_summary_and_json_contain_only_native_routing_results(tmp_path) -> None:
    results = [
        {
            **make_sample("w01", "weather", "weather_tool"),
            "selected_tool": "weather_tool",
            "correct": True,
            "parser_status": "native_tool_call",
            "parse_error": None,
            "tool_arguments": {},
            "raw_response": "<tool_call>weather</tool_call>",
            "elapsed_seconds": 1.25,
        },
        {
            **make_sample("w02", "broken", "weather_tool", is_boundary=True),
            "selected_tool": None,
            "correct": False,
            "parser_status": "parse_failure",
            "parse_error": "broken output",
            "tool_arguments": {},
            "raw_response": "broken",
            "elapsed_seconds": 0.5,
        },
        {
            **make_sample("n01", "stable", "no_tool", is_boundary=True),
            "selected_tool": "no_tool",
            "correct": True,
            "parser_status": "direct_answer",
            "parse_error": None,
            "tool_arguments": {},
            "raw_response": "A direct answer.",
            "elapsed_seconds": 0.75,
        },
    ]

    summary = holdout.summarize(results)
    output = tmp_path / "results.json"
    holdout.write_results(output, results)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert summary["overall_accuracy"] == pytest.approx(2 / 3)
    assert summary["per_category_accuracy"]["weather_tool"] == 0.5
    assert summary["per_category_accuracy"]["no_tool"] == 1.0
    assert summary["boundary_accuracy"] == {"False": 1.0, "True": 0.5}
    assert summary["confusion_matrix"]["weather_tool"]["weather_tool"] == 1
    assert summary["confusion_matrix"]["weather_tool"]["parse_failure"] == 1
    assert payload == summary
    forbidden_fields = {
        "probability",
        "raw_scores",
        "calibrated_scores",
        "raw_argmax",
        "calibrated_argmax",
        "selection_metadata",
        "strategy_sweep",
    }
    assert forbidden_fields.isdisjoint(payload)
    assert all(forbidden_fields.isdisjoint(result) for result in payload["results"])


def test_main_constructs_native_router_and_calls_it_once(monkeypatch, tmp_path) -> None:
    testset = tmp_path / "samples.json"
    output = tmp_path / "results.json"
    testset.write_text(
        json.dumps([make_sample("w01", "weather request", "weather_tool")]),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    class FakeRouter:
        def __init__(self, **kwargs: object) -> None:
            captured["router_kwargs"] = kwargs

        def choose_tool(
            self,
            user_request: str,
            tool_descriptions: object,
            *,
            system_prompt: str,
        ) -> RouterDecision:
            captured["call"] = (user_request, tool_descriptions, system_prompt)
            return make_decision("weather_tool")

    monkeypatch.setattr(holdout, "LocalHFRouter", FakeRouter)
    monkeypatch.setattr(
        holdout,
        "parse_args",
        lambda: argparse.Namespace(
            testset=testset,
            model_id="fake/native-model",
            device="cpu",
            dtype="float32",
            max_new_tokens=64,
            output=output,
        ),
    )

    holdout.main()

    assert captured["router_kwargs"] == {
        "model_name": "fake/native-model",
        "max_new_tokens": 64,
        "device": "cpu",
        "dtype": "float32",
    }
    call = captured["call"]
    assert call[0] == "weather request"
    assert call[1] is holdout.TOOLS
    assert isinstance(call[2], str) and call[2]
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["results"][0]["selected_tool"] == "weather_tool"
    assert saved["results"][0]["correct"] is True
