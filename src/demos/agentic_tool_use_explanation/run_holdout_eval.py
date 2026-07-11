"""Batch evaluation CLI for the calibrated HF Local routing pipeline."""

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
        HF_SELECTION_MODES,
        LocalHFClassificationRouter,
        select_tool_from_scores,
    )
    from demos.agentic_tool_use_explanation.scorers import CalibratedToolLogOddsScorer
except ModuleNotFoundError:
    from app import MOCK_SYSTEM_SEGMENTS, TOOLS, build_segments, build_system_prompt
    from hf_router import HF_SELECTION_MODES, LocalHFClassificationRouter, select_tool_from_scores
    from scorers import CalibratedToolLogOddsScorer


TOOL_NAMES = ("weather_tool", "calculator_tool", "web_search_tool", "no_tool")
DEFAULT_TESTSET = Path(__file__).with_name("holdout_samples.json")
DEFAULT_OUTPUT = Path(__file__).with_name("holdout_eval_results.json")
RAW_MARGIN_THRESHOLDS = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
CALIBRATION_STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--testset", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-pairs-per-batch", type=int, default=None)
    parser.add_argument("--selection-mode", choices=HF_SELECTION_MODES, default="calibrated")
    parser.add_argument("--raw-margin-threshold", type=float, default=1.0)
    parser.add_argument("--calibration-strength", type=float, default=1.0)
    return parser.parse_args()


def load_samples(path: Path) -> list[dict[str, Any]]:
    """Load and validate the hold-out records."""
    with path.open(encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list):
        raise ValueError("Test set must be a JSON list.")
    seen_ids: set[str] = set()
    samples: list[dict[str, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"Sample at index {index} must be an object.")
        sample_id = item.get("id")
        request = item.get("request")
        ground_truth = item.get("ground_truth")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Sample at index {index} has an invalid id.")
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample id: {sample_id!r}.")
        if not isinstance(request, str) or not request.strip():
            raise ValueError(f"Sample {sample_id!r} has an invalid request.")
        if ground_truth not in TOOL_NAMES:
            raise ValueError(f"Sample {sample_id!r} has an invalid ground_truth.")
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
    confusion_matrix = {
        truth: {prediction: 0 for prediction in TOOL_NAMES} for truth in TOOL_NAMES
    }
    for result in results:
        confusion_matrix[result["ground_truth"]][result["selected_tool"]] += 1

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
        "strategy_sweep": build_strategy_sweep(results),
    }


def strategy_metrics(results: list[dict[str, Any]], predictions: list[str]) -> dict[str, Any]:
    """Return accuracy metrics for one offline selection strategy."""
    correctness = [
        prediction == result["ground_truth"]
        for result, prediction in zip(results, predictions, strict=True)
    ]
    per_category = {}
    for tool_name in TOOL_NAMES:
        indices = [
            index for index, result in enumerate(results) if result["ground_truth"] == tool_name
        ]
        per_category[tool_name] = (
            sum(correctness[index] for index in indices) / len(indices) if indices else None
        )
    key_samples = {}
    for sample_id in ("n08", "n09", "s05", "s09"):
        matching = [
            index for index, result in enumerate(results) if result["id"] == sample_id
        ]
        key_samples[sample_id] = correctness[matching[0]] if matching else None
    return {
        "correct": sum(correctness),
        "total": len(results),
        "accuracy": sum(correctness) / len(results) if results else None,
        "per_category_accuracy": per_category,
        "key_samples_correct": key_samples,
    }


def predictions_for_strategy(
    results: list[dict[str, Any]],
    *,
    mode: str,
    raw_margin_threshold: float = 1.0,
    calibration_strength: float = 1.0,
) -> list[str]:
    """Recompute decisions from saved scores without another model call."""
    return [
        select_tool_from_scores(
            result["raw_scores"],
            result["calibrated_scores"],
            mode=mode,
            raw_margin_threshold=raw_margin_threshold,
            calibration_strength=calibration_strength,
        )[0]
        for result in results
    ]


def build_strategy_sweep(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate candidate gates/scales offline from the single inference pass."""
    if not results or any("raw_scores" not in result for result in results):
        return {}
    baseline = strategy_metrics(
        results, predictions_for_strategy(results, mode="calibrated")
    )
    margin_gate = {
        str(threshold): strategy_metrics(
            results,
            predictions_for_strategy(
                results,
                mode="raw_margin_gate",
                raw_margin_threshold=threshold,
            ),
        )
        for threshold in RAW_MARGIN_THRESHOLDS
    }
    scaled_calibration = {
        str(strength): strategy_metrics(
            results,
            predictions_for_strategy(
                results,
                mode="scaled_calibration",
                calibration_strength=strength,
            ),
        )
        for strength in CALIBRATION_STRENGTHS
    }
    return {
        "calibrated_baseline": baseline,
        "raw_margin_gate": margin_gate,
        "scaled_calibration": scaled_calibration,
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
    print(f"Overall accuracy: {correct}/{len(results)} ({format_accuracy(summary['overall_accuracy'])})")
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
    print("Confusion matrix (rows=ground_truth, columns=calibrated_argmax):")
    print(" " * (label_width + 2) + " ".join(f"{name:>{cell_width}}" for name in TOOL_NAMES))
    for truth in TOOL_NAMES:
        values = summary["confusion_matrix"][truth]
        print(f"{truth:<{label_width}}  " + " ".join(f"{values[name]:>{cell_width}}" for name in TOOL_NAMES))

    print("Per-category accuracy:")
    for tool_name in TOOL_NAMES:
        rows = [result for result in results if result["ground_truth"] == tool_name]
        category_correct = sum(bool(result["correct"]) for result in rows)
        print(
            f"  {tool_name}: {category_correct}/{len(rows)} "
            f"({format_accuracy(summary['per_category_accuracy'][tool_name])})"
        )

    sweep = summary.get("strategy_sweep", {})
    if sweep:
        print("Offline strategy sweep (same saved model scores; no extra inference):")
        baseline = sweep["calibrated_baseline"]
        print(
            f"  calibrated baseline: {baseline['correct']}/{baseline['total']} "
            f"({format_accuracy(baseline['accuracy'])})"
        )
        for threshold, metrics in sweep["raw_margin_gate"].items():
            print(
                f"  raw-margin threshold={threshold}: {metrics['correct']}/{metrics['total']} "
                f"({format_accuracy(metrics['accuracy'])}) keys={metrics['key_samples_correct']}"
            )
        for strength, metrics in sweep["scaled_calibration"].items():
            print(
                f"  calibration strength={strength}: {metrics['correct']}/{metrics['total']} "
                f"({format_accuracy(metrics['accuracy'])}) keys={metrics['key_samples_correct']}"
            )


def main() -> None:
    """Run the hold-out evaluation."""
    args = parse_args()
    samples = load_samples(args.testset)
    system_prompt = build_system_prompt(build_segments(MOCK_SYSTEM_SEGMENTS, "system"))
    scorer = CalibratedToolLogOddsScorer(
        model_id=args.model_id,
        device=args.device,
        max_pairs_per_batch=args.max_pairs_per_batch,
    )
    router = LocalHFClassificationRouter(
        scorer,
        selection_mode=args.selection_mode,
        raw_margin_threshold=args.raw_margin_threshold,
        calibration_strength=args.calibration_strength,
    )
    results: list[dict[str, Any]] = []

    try:
        for index, sample in enumerate(samples, start=1):
            started = time.perf_counter()
            raw_scores = scorer.score_full_request_labels(
                sample["request"],
                system_prompt=system_prompt,
                tool_descriptions=TOOLS,
            )
            raw_argmax = max(raw_scores, key=raw_scores.get)
            calibrated_scores = scorer.score_full_request_calibrated_labels(
                sample["request"],
                system_prompt=system_prompt,
                tool_descriptions=TOOLS,
            )
            full_calibrated_argmax = max(calibrated_scores, key=calibrated_scores.get)
            choice = router.choose_tool(
                sample["request"],
                system_prompt=system_prompt,
                tool_descriptions=TOOLS,
            )
            correct = choice.tool == sample["ground_truth"]
            results.append(
                {
                    **sample,
                    "raw_argmax": raw_argmax,
                    "calibrated_argmax": full_calibrated_argmax,
                    "selected_tool": choice.tool,
                    "raw_scores": dict(raw_scores),
                    "calibrated_scores": dict(calibrated_scores),
                    "correct": correct,
                }
            )
            elapsed = time.perf_counter() - started
            verdict = "correct" if correct else "incorrect"
            print(f"[{index:02d}/{len(samples):02d}] {sample['id']} {elapsed:.1f}s {verdict}", flush=True)
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
