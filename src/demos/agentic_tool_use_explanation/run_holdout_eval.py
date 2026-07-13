"""Batch evaluation CLI for the native HF tool-calling router."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from demos.agentic_tool_use_explanation.app import (
        MOCK_SYSTEM_SEGMENTS,
        TOOLS,
        build_segments,
        build_system_prompt,
    )
    from demos.agentic_tool_use_explanation.hf_router import (
        DEFAULT_LOCAL_HF_ROUTER_MODEL_ID,
        DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
        LocalHFRouter,
        RouterDecision,
    )
except ModuleNotFoundError:
    from app import MOCK_SYSTEM_SEGMENTS, TOOLS, build_segments, build_system_prompt
    from hf_router import (
        DEFAULT_LOCAL_HF_ROUTER_MODEL_ID,
        DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
        LocalHFRouter,
        RouterDecision,
    )


TOOL_NAMES = ("weather_tool", "calculator_tool", "web_search_tool", "no_tool")
PARSE_FAILURE_LABEL = "parse_failure"
PREDICTION_NAMES = (*TOOL_NAMES, PARSE_FAILURE_LABEL)
DEFAULT_TESTSET = Path(__file__).with_name("holdout_samples.json")
DEFAULT_OUTPUT = Path(__file__).with_name("holdout_eval_results.json")


class InvalidTestSetTypeError(TypeError):
    """Raised when the hold-out test-set root is not a JSON list."""

    def __init__(self) -> None:
        super().__init__("Test set must be a JSON list.")


class InvalidSampleTypeError(TypeError):
    """Raised when a hold-out sample is not a JSON object."""

    def __init__(self, index: int) -> None:
        super().__init__(f"Sample at index {index} must be an object.")


class InvalidSampleIdError(ValueError):
    """Raised when a hold-out sample id is missing or invalid."""

    def __init__(self, index: int) -> None:
        super().__init__(f"Sample at index {index} has an invalid id.")


class DuplicateSampleIdError(ValueError):
    """Raised when a hold-out sample id occurs more than once."""

    def __init__(self, sample_id: str) -> None:
        super().__init__(f"Duplicate sample id: {sample_id!r}.")


class InvalidSampleRequestError(ValueError):
    """Raised when a hold-out sample request is missing or invalid."""

    def __init__(self, sample_id: str) -> None:
        super().__init__(f"Sample {sample_id!r} has an invalid request.")


class InvalidGroundTruthError(ValueError):
    """Raised when a hold-out sample has an unsupported ground-truth tool."""

    def __init__(self, sample_id: str) -> None:
        super().__init__(f"Sample {sample_id!r} has an invalid ground_truth.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--model-id", default=DEFAULT_LOCAL_HF_ROUTER_MODEL_ID)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_samples(path: Path) -> list[dict[str, Any]]:
    """Load and validate the hold-out records."""
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise InvalidTestSetTypeError
    seen_ids: set[str] = set()
    samples: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise InvalidSampleTypeError(index)
        sample_id = item.get("id")
        request = item.get("request")
        ground_truth = item.get("ground_truth")
        if not isinstance(sample_id, str) or not sample_id:
            raise InvalidSampleIdError(index)
        if sample_id in seen_ids:
            raise DuplicateSampleIdError(sample_id)
        if not isinstance(request, str) or not request.strip():
            raise InvalidSampleRequestError(sample_id)
        if ground_truth not in TOOL_NAMES:
            raise InvalidGroundTruthError(sample_id)
        seen_ids.add(sample_id)
        samples.append(
            {
                "id": sample_id,
                "request": request,
                "ground_truth": ground_truth,
                "is_boundary": bool(item.get("is_boundary", False)),
            }
        )
    return samples


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate metrics for complete or partial results."""
    confusion_matrix = {truth: dict.fromkeys(PREDICTION_NAMES, 0) for truth in TOOL_NAMES}
    for result in results:
        selected_tool = result["selected_tool"]
        prediction = selected_tool if selected_tool in TOOL_NAMES else PARSE_FAILURE_LABEL
        confusion_matrix[result["ground_truth"]][prediction] += 1

    total = len(results)
    correct = sum(bool(result["correct"]) for result in results)
    per_category_accuracy: dict[str, float | None] = {}
    for tool_name in TOOL_NAMES:
        category_rows = [result for result in results if result["ground_truth"] == tool_name]
        per_category_accuracy[tool_name] = (
            sum(bool(result["correct"]) for result in category_rows) / len(category_rows)
            if category_rows
            else None
        )

    boundary_accuracy: dict[str, float | None] = {}
    for is_boundary in (False, True):
        split_rows = [result for result in results if result["is_boundary"] is is_boundary]
        boundary_accuracy[str(is_boundary)] = (
            sum(bool(result["correct"]) for result in split_rows) / len(split_rows)
            if split_rows
            else None
        )
    return {
        "results": results,
        "confusion_matrix": confusion_matrix,
        "overall_accuracy": correct / total if total else None,
        "per_category_accuracy": per_category_accuracy,
        "boundary_accuracy": boundary_accuracy,
    }


def parser_status(decision: RouterDecision) -> str:
    """Classify a native router result without inventing a fallback tool."""
    if decision.parse_error or decision.selected_tool not in TOOL_NAMES:
        return PARSE_FAILURE_LABEL
    if decision.selected_tool == "no_tool":
        return "direct_answer"
    return "native_tool_call"


def evaluate_sample(
    router: LocalHFRouter,
    sample: Mapping[str, Any],
    *,
    system_prompt: str,
) -> dict[str, Any]:
    """Run native inference exactly once and build one diagnostic result row."""
    started = time.perf_counter()
    decision = router.choose_tool(
        sample["request"],
        TOOLS,
        system_prompt=system_prompt,
    )
    elapsed_seconds = time.perf_counter() - started
    status = parser_status(decision)
    selected_tool = decision.selected_tool if status != PARSE_FAILURE_LABEL else None
    return {
        **sample,
        "selected_tool": selected_tool,
        "correct": selected_tool == sample["ground_truth"],
        "parser_status": status,
        "parse_error": decision.parse_error,
        "tool_arguments": dict(decision.tool_arguments),
        "raw_response": decision.raw_response,
        "elapsed_seconds": elapsed_seconds,
    }


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    """Atomically overwrite the output with current results and aggregates."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(summarize(results), file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary_path.replace(path)


def format_accuracy(value: float | None) -> str:
    """Format an optional accuracy as a percentage."""
    return "n/a" if value is None else f"{value:.1%}"


def print_summary(summary: dict[str, Any]) -> None:
    """Print the requested readable evaluation summary."""
    results = summary["results"]
    correct = sum(bool(result["correct"]) for result in results)
    print(
        f"Overall accuracy: {correct}/{len(results)} ({format_accuracy(summary['overall_accuracy'])})"
    )
    print("Accuracy by boundary status:")
    for key in ("False", "True"):
        rows = [result for result in results if str(result["is_boundary"]) == key]
        split_correct = sum(bool(result["correct"]) for result in rows)
        print(
            f"  is_boundary={key}: {split_correct}/{len(rows)} "
            f"({format_accuracy(summary['boundary_accuracy'][key])})"
        )

    label_width = max(len(name) for name in TOOL_NAMES)
    cell_width = max(5, label_width)
    print("Confusion matrix (rows=ground_truth, columns=native selected_tool):")
    print(" " * (label_width + 2) + " ".join(f"{name:>{cell_width}}" for name in PREDICTION_NAMES))
    for truth in TOOL_NAMES:
        values = summary["confusion_matrix"][truth]
        print(
            f"{truth:<{label_width}}  "
            + " ".join(f"{values[name]:>{cell_width}}" for name in PREDICTION_NAMES)
        )

    print("Per-category accuracy:")
    for tool_name in TOOL_NAMES:
        rows = [result for result in results if result["ground_truth"] == tool_name]
        category_correct = sum(bool(result["correct"]) for result in rows)
        print(
            f"  {tool_name}: {category_correct}/{len(rows)} "
            f"({format_accuracy(summary['per_category_accuracy'][tool_name])})"
        )


def main() -> None:
    """Run the hold-out evaluation."""
    args = parse_args()
    samples = load_samples(args.testset)
    system_prompt = build_system_prompt(build_segments(MOCK_SYSTEM_SEGMENTS, "system"))
    router = LocalHFRouter(
        model_name=args.model_id,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        dtype=args.dtype,
    )
    results: list[dict[str, Any]] = []

    try:
        for index, sample in enumerate(samples, start=1):
            result = evaluate_sample(
                router,
                sample,
                system_prompt=system_prompt,
            )
            results.append(result)
            verdict = "correct" if result["correct"] else "incorrect"
            print(
                f"[{index:02d}/{len(samples):02d}] {sample['id']} "
                f"{result['elapsed_seconds']:.1f}s {verdict}",
                flush=True,
            )
            if index % 5 == 0:
                write_results(args.output, results)
    except BaseException:
        if results:
            write_results(args.output, results)
        raise

    write_results(args.output, results)
    print_summary(summarize(results))
    print(f"Results JSON: {args.output.resolve()}")


if __name__ == "__main__":
    main()
