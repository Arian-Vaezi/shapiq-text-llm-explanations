"""Run representative Agent + XAI case studies, not an accuracy benchmark."""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

EXPERIMENT_NAME = "representative_agentic_tool_use_xai"
SOURCE_SET = "40-prompt holdout"
CASE_SELECTION_DESCRIPTION = (
    "Four successful 3B holdout cases selected for interpretable representative "
    "Shapley and interaction case studies; they are not a new accuracy benchmark."
)
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_DEVICE = "cuda"
DEFAULT_DTYPE = "auto"
DEFAULT_QUANTIZATION = "none"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_MAX_PAIRS_PER_BATCH = 1
DEFAULT_OUTPUT_DIR = Path("outputs")


@dataclass(frozen=True)
class RepresentativeCase:
    """One fixed case selected from the existing 40-prompt holdout set."""

    case_id: str
    category: str
    request: str
    expected_tool: str


REPRESENTATIVE_CASES = (
    RepresentativeCase(
        case_id="w01",
        category="weather",
        request="What will the temperature be in Oslo on Friday evening?",
        expected_tool="weather_tool",
    ),
    RepresentativeCase(
        case_id="c05",
        category="calculator",
        request="A rectangle is 13.2 cm by 8.5 cm; compute its area.",
        expected_tool="calculator_tool",
    ),
    RepresentativeCase(
        case_id="s03",
        category="web_search",
        request="What major news was announced by the European Central Bank today?",
        expected_tool="web_search_tool",
    ),
    RepresentativeCase(
        case_id="n07",
        category="no_tool",
        request=(
            "I'm studying percentages, but I only need a conceptual explanation "
            "of why multiplying by 0.5 is the same as taking half."
        ),
        expected_tool="no_tool",
    ),
)


@dataclass(frozen=True)
class BatchConfiguration:
    """Runtime configuration shared by the native agent and XAI scorer."""

    model_name: str = DEFAULT_MODEL_NAME
    device: str = DEFAULT_DEVICE
    dtype: str = DEFAULT_DTYPE
    quantization: str = DEFAULT_QUANTIZATION
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    max_pairs_per_batch: int = DEFAULT_MAX_PAIRS_PER_BATCH


@dataclass(frozen=True)
class RunnerDependencies:
    """Injected inference boundaries that keep unit tests model-free."""

    run_agent: Callable[[str], object]
    run_xai: Callable[[str, str, object], Mapping[str, object]]
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputPaths:
    """The three incrementally maintained experiment outputs."""

    json_path: Path
    csv_path: Path
    log_path: Path


SUMMARY_FIELDS = (
    "case_id",
    "category",
    "expected_tool",
    "actual_selected_tool",
    "selection_correct",
    "most_supportive_span",
    "supportive_value",
    "most_opposing_span",
    "opposing_value",
    "strongest_positive_pair",
    "positive_interaction_value",
    "strongest_negative_pair",
    "negative_interaction_value",
    "number_of_players",
    "runtime_seconds",
    "status",
    "error_message",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the shared batch CLI without initializing model dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument(
        "--max-pairs-per-batch",
        type=int,
        default=DEFAULT_MAX_PAIRS_PER_BATCH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the real batch CLI without initializing model dependencies."""
    return build_parser().parse_args(argv)


def output_paths(output_dir: Path) -> OutputPaths:
    """Return the required output locations under one configurable directory."""
    return OutputPaths(
        json_path=output_dir / "representative_xai_results.json",
        csv_path=output_dir / "representative_xai_summary.csv",
        log_path=output_dir / "representative_xai_run.log",
    )


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.UTC)


def _iso_milliseconds(value: datetime.datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.UTC)
    return value.isoformat(timespec="milliseconds")


def _git_value(*args: str) -> str:
    try:
        return subprocess.run(  # noqa: S603 - fixed git command, controlled arguments
            ["git", *args],  # noqa: S607 - Git is intentionally resolved from PATH
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def git_metadata() -> dict[str, str]:
    """Read the current revision without mutating the repository."""
    return {
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
    }


def _json_native(value: object) -> object:
    """Convert dataclasses, NumPy-like scalars, and containers to JSON values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_native(asdict(value))
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_native(item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, Mapping):
        return {str(key): _json_native(item_value) for key, item_value in value.items()}
    if isinstance(value, list | tuple):
        return [_json_native(item_value) for item_value in value]
    if isinstance(value, Path):
        return str(value)
    return value


def serialize_agent_result(result: object) -> dict[str, object]:
    """Preserve all public fields exposed by the real agent result."""
    if is_dataclass(result) and not isinstance(result, type):
        payload = asdict(result)
    elif hasattr(result, "__dict__"):
        payload = {key: value for key, value in vars(result).items() if not key.startswith("_")}
    elif isinstance(result, Mapping):
        payload = dict(result)
    else:
        payload = {"value": str(result)}
    return _json_native(payload)  # type: ignore[return-value]


def _selected_tool(agent_result: object) -> str | None:
    if isinstance(agent_result, Mapping):
        selected = agent_result.get("selected_tool")
    else:
        selected = getattr(agent_result, "selected_tool", None)
    return selected if isinstance(selected, str) and selected else None


def _score_mapping(agent_result: object, *names: str) -> dict[str, float]:
    for name in names:
        value = (
            agent_result.get(name)
            if isinstance(agent_result, Mapping)
            else getattr(agent_result, name, None)
        )
        if isinstance(value, Mapping):
            return {str(key): float(score) for key, score in value.items()}
    return {}


def _extreme_span(
    attributions: Sequence[Mapping[str, object]],
    *,
    strongest: bool,
) -> tuple[str | None, float | None]:
    candidates = [
        row
        for row in attributions
        if (float(row["value"]) > 0 if strongest else float(row["value"]) < 0)
    ]
    if not candidates:
        return None, None
    selected = (max if strongest else min)(
        candidates,
        key=lambda row: float(row["value"]),
    )
    return str(selected["text"]), float(selected["value"])


def _extreme_interaction(
    interactions: Sequence[Mapping[str, object]],
    *,
    strongest: bool,
) -> tuple[str | None, float | None]:
    candidates = [
        row
        for row in interactions
        if (float(row["value"]) > 0 if strongest else float(row["value"]) < 0)
    ]
    if not candidates:
        return None, None
    selected = (max if strongest else min)(
        candidates,
        key=lambda row: float(row["value"]),
    )
    pair = selected.get("pair")
    pair_text = (
        " + ".join(str(item) for item in pair) if isinstance(pair, list | tuple) else str(pair)
    )
    return pair_text, float(selected["value"])


def _empty_xai_fields() -> dict[str, object]:
    return {
        "linguistic_players": [],
        "first_order_attributions": [],
        "most_supportive_span": None,
        "supportive_value": None,
        "most_opposing_span": None,
        "opposing_value": None,
        "pairwise_interactions": [],
        "strongest_positive_interaction": None,
        "strongest_negative_interaction": None,
        "empty_coalition_value": None,
        "full_coalition_value": None,
    }


def _summary_row(run: Mapping[str, object]) -> dict[str, object]:
    positive = run.get("strongest_positive_interaction")
    negative = run.get("strongest_negative_interaction")
    positive = positive if isinstance(positive, Mapping) else {}
    negative = negative if isinstance(negative, Mapping) else {}
    players = run.get("linguistic_players")
    return {
        "case_id": run["case_id"],
        "category": run["category"],
        "expected_tool": run["expected_tool"],
        "actual_selected_tool": run.get("actual_selected_tool"),
        "selection_correct": run.get("selection_correct"),
        "most_supportive_span": run.get("most_supportive_span"),
        "supportive_value": run.get("supportive_value"),
        "most_opposing_span": run.get("most_opposing_span"),
        "opposing_value": run.get("opposing_value"),
        "strongest_positive_pair": positive.get("pair_text"),
        "positive_interaction_value": positive.get("value"),
        "strongest_negative_pair": negative.get("pair_text"),
        "negative_interaction_value": negative.get("value"),
        "number_of_players": len(players) if isinstance(players, list) else 0,
        "runtime_seconds": run["runtime_seconds"],
        "status": run["status"],
        "error_message": run["error_message"],
    }


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(_json_native(payload), file, indent=2, ensure_ascii=False)
        file.write("\n")
    temporary_path.replace(path)


def _write_csv_atomic(path: Path, runs: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(_summary_row(run) for run in runs)
    temporary_path.replace(path)


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(message.rstrip() + "\n")


def _persist(paths: OutputPaths, experiment: Mapping[str, object]) -> None:
    runs = experiment["runs"]
    if not isinstance(runs, list):
        msg = "Experiment runs must be a list."
        raise TypeError(msg)
    _write_json_atomic(paths.json_path, experiment)
    _write_csv_atomic(paths.csv_path, runs)


def _require_selected_tool(agent_result: object) -> str:
    selected_tool = _selected_tool(agent_result)
    if selected_tool is None:
        msg = "The full-context agent did not return a selected tool."
        raise RuntimeError(msg)
    return selected_tool


def _require_xai_lists(
    xai: Mapping[str, object],
) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]]]:
    attributions = xai.get("first_order_attributions", [])
    interactions = xai.get("pairwise_interactions", [])
    if not isinstance(attributions, list) or not isinstance(interactions, list):
        msg = "XAI output must contain list-valued attributions and interactions."
        raise TypeError(msg)
    return attributions, interactions


def run_experiment(
    *,
    dependencies: RunnerDependencies,
    configuration: BatchConfiguration,
    output_dir: Path,
    cases: Sequence[RepresentativeCase] = REPRESENTATIVE_CASES,
    case_selection_description: str = CASE_SELECTION_DESCRIPTION,
    metadata: Mapping[str, object] | None = None,
    now: Callable[[], datetime.datetime] = _utc_now,
    monotonic: Callable[[], float] = time.perf_counter,
    after_persist: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Process cases in order and persist after every success or failure."""
    paths = output_paths(output_dir)
    paths.log_path.parent.mkdir(parents=True, exist_ok=True)
    paths.log_path.write_text("", encoding="utf-8")
    created_at = _iso_milliseconds(now())
    revision = dict(metadata) if metadata is not None else git_metadata()
    experiment: dict[str, object] = {
        "experiment_name": EXPERIMENT_NAME,
        "created_at": created_at,
        "git_branch": revision.get("git_branch", "unknown"),
        "git_commit": revision.get("git_commit", "unknown"),
        "model_name": configuration.model_name,
        "device": configuration.device,
        "configuration": {**asdict(configuration), **dependencies.metadata},
        "source_set": SOURCE_SET,
        "case_selection_description": case_selection_description,
        "completed_case_count": 0,
        "failed_case_count": 0,
        "runs": [],
    }
    _append_log(paths.log_path, f"{created_at} experiment_started {EXPERIMENT_NAME}")

    runs = experiment["runs"]
    if not isinstance(runs, list):
        msg = "Experiment runs must be a list."
        raise TypeError(msg)
    for case in cases:
        started = monotonic()
        agent_result: object | None = None
        actual_selected_tool: str | None = None
        run: dict[str, object] = {
            "case_id": case.case_id,
            "category": case.category,
            "request": case.request,
            "expected_tool": case.expected_tool,
            "actual_selected_tool": None,
            "selection_correct": False,
            "agent_result": None,
            "router_raw_scores": {},
            "router_calibrated_scores": {},
            "xai_target_tool": None,
            **_empty_xai_fields(),
            "runtime_seconds": 0.0,
            "status": "failed",
            "error_type": None,
            "error_message": None,
        }
        try:
            agent_result = dependencies.run_agent(case.request)
            run["agent_result"] = serialize_agent_result(agent_result)
            actual_selected_tool = _require_selected_tool(agent_result)
            run["actual_selected_tool"] = actual_selected_tool
            run["selection_correct"] = actual_selected_tool == case.expected_tool
            run["router_raw_scores"] = _score_mapping(agent_result, "raw_scores")
            run["router_calibrated_scores"] = _score_mapping(
                agent_result,
                "calibrated_scores",
                "tool_scores",
            )
            # The expected tool is deliberately absent from this call. The real agent's
            # selected tool is the only possible XAI target.
            xai = dict(dependencies.run_xai(case.request, actual_selected_tool, agent_result))
            attributions, interactions = _require_xai_lists(xai)
            supportive_span, supportive_value = _extreme_span(attributions, strongest=True)
            opposing_span, opposing_value = _extreme_span(attributions, strongest=False)
            positive_pair, positive_value = _extreme_interaction(interactions, strongest=True)
            negative_pair, negative_value = _extreme_interaction(interactions, strongest=False)
            run.update(xai)
            run.update(
                {
                    "xai_target_tool": actual_selected_tool,
                    "most_supportive_span": supportive_span,
                    "supportive_value": supportive_value,
                    "most_opposing_span": opposing_span,
                    "opposing_value": opposing_value,
                    "strongest_positive_interaction": (
                        None
                        if positive_pair is None
                        else {"pair_text": positive_pair, "value": positive_value}
                    ),
                    "strongest_negative_interaction": (
                        None
                        if negative_pair is None
                        else {"pair_text": negative_pair, "value": negative_value}
                    ),
                    "status": "completed",
                }
            )
        except Exception as error:  # noqa: BLE001 - each case must be recorded and continue
            run["error_type"] = type(error).__name__
            run["error_message"] = str(error)
        finally:
            run["runtime_seconds"] = float(monotonic() - started)

        runs.append(_json_native(run))
        completed = sum(item["status"] == "completed" for item in runs)
        experiment["completed_case_count"] = completed
        experiment["failed_case_count"] = len(runs) - completed
        _persist(paths, experiment)
        recorded_at = _iso_milliseconds(now())
        _append_log(
            paths.log_path,
            f"{recorded_at} case={case.case_id} status={run['status']} "
            f"actual_tool={actual_selected_tool or 'none'} "
            f"runtime_seconds={run['runtime_seconds']:.6f} "
            f"error={run['error_message'] or ''}",
        )
        if after_persist is not None:
            after_persist(experiment)

    finished_at = _iso_milliseconds(now())
    experiment["finished_at"] = finished_at
    _persist(paths, experiment)
    _append_log(paths.log_path, f"{finished_at} experiment_finished")
    return experiment


def build_real_dependencies(configuration: BatchConfiguration) -> RunnerDependencies:
    """Load the same native Agent + XAI components used by Streamlit."""
    # Lazy import keeps pytest and --help free of Streamlit/HF/spaCy model loading.
    from app import (  # noqa: I001
        MAX_EXACT_DEMO_PLAYERS,
        MOCK_SYSTEM_SEGMENTS,
        TOOLS,
        LinguisticSegmenter,
        LocalHFRouter,
        build_canonical_direct_answer_target,
        build_coalition_prompt,
        build_native_direct_answer_scorer_from_router,
        build_native_hf_scorer_from_router,
        build_segments,
        build_system_prompt,
        budget_for_demo,
        compute_dual_index_explanations,
        format_tool_context,
        pairwise_matrix_from_explanation,
        require_trusted_direct_answer_target,
        selected_hf_model_config,
        validate_hf_inference_xai_consistency,
        values_to_frame,
    )
    from tool_game import ToolUseGame

    selected_config = selected_hf_model_config(configuration.model_name)
    selected_config = type(selected_config)(
        model_id=selected_config.model_id,
        model_family=selected_config.model_family,
        quantization_mode=configuration.quantization,
        device=configuration.device,
        dtype=configuration.dtype,
        supports_native_tools=selected_config.supports_native_tools,
    )
    router = LocalHFRouter(
        model_name=configuration.model_name,
        max_new_tokens=configuration.max_new_tokens,
        trust_remote_code=False,
        quantization_mode=configuration.quantization,
        device=configuration.device,
        dtype=configuration.dtype,
    )
    segmenter = LinguisticSegmenter()
    system_prompt = build_system_prompt(build_segments(MOCK_SYSTEM_SEGMENTS, "system"))
    tool_context = format_tool_context(TOOLS)

    def run_agent(request: str) -> object:
        return router.choose_tool(request, TOOLS, system_prompt=system_prompt)

    def run_xai(request: str, target_tool: str, agent_result: object) -> Mapping[str, object]:
        segment_texts, debug_rows = segmenter.segment_with_debug(request)
        user_segments = build_segments(segment_texts, "user")
        if not user_segments:
            msg = "Linguistic segmentation returned no players."
            raise RuntimeError(msg)
        full_prompt = build_coalition_prompt(
            user_segments,
            system_prompt=system_prompt,
            tool_context=tool_context,
        )
        empty_prompt = build_coalition_prompt(
            [],
            system_prompt=system_prompt,
            tool_context=tool_context,
        )
        direct_answer_target = None
        if target_tool == "no_tool":
            answer = require_trusted_direct_answer_target(
                agent_result,
                inference_backend="HF local",
            )
            direct_answer_target = build_canonical_direct_answer_target(
                answer,
                tokenizer=router.tokenizer,
            )
            scorer = build_native_direct_answer_scorer_from_router(
                router,
                direct_answer_target=direct_answer_target,
                max_pairs_per_batch=configuration.max_pairs_per_batch,
                selected_config=selected_config,
            )
        else:
            scorer = build_native_hf_scorer_from_router(
                router,
                max_pairs_per_batch=configuration.max_pairs_per_batch,
                selected_config=selected_config,
            )
        consistency = validate_hf_inference_xai_consistency(
            selected_config=selected_config,
            router=router,
            scorer=scorer,
        )
        full_score = scorer.score_batch(
            [full_prompt],
            target_tool=target_tool,
            tool_descriptions=TOOLS,
        )[0]
        empty_score = scorer.score_batch(
            [empty_prompt],
            target_tool=target_tool,
            tool_descriptions=TOOLS,
        )[0]
        exact = len(user_segments) <= MAX_EXACT_DEMO_PLAYERS
        budget = None if exact else budget_for_demo(len(user_segments))
        game = ToolUseGame(
            target_tool=target_tool,
            user_segments=user_segments,
            system_prompt=system_prompt,
            tool_context=tool_context,
            scorer=scorer,
            tool_descriptions=TOOLS,
            defer_empty_coalition_evaluation=exact,
        )
        sv, sv_algorithm, ksii, ksii_algorithm = compute_dual_index_explanations(
            game=game,
            budget=budget,
        )
        attribution_frame = values_to_frame(sv.get_n_order(order=1), user_segments)
        matrix = pairwise_matrix_from_explanation(ksii, game.n_players)
        players = []
        for index, segment in enumerate(user_segments):
            debug = debug_rows[index] if index < len(debug_rows) else {}
            players.append(
                {
                    "player_index": index,
                    "text": segment.text,
                    "label": segment.label,
                    "type": debug.get("rule"),
                    "word_start": debug.get("word_start"),
                    "word_end": debug.get("word_end"),
                    "character_start": debug.get("character_start"),
                    "character_end": debug.get("character_end"),
                }
            )
        attributions = [
            {
                "player_index": next(
                    index
                    for index, segment in enumerate(user_segments)
                    if segment.label == row.segment
                ),
                "label": str(row.segment),
                "text": str(row.text),
                "value": float(row.attribution),
            }
            for row in attribution_frame.itertuples(index=False)
        ]
        attributions.sort(key=lambda row: int(row["player_index"]))
        interactions = [
            {
                "player_indices": [left, right],
                "pair": [user_segments[left].text, user_segments[right].text],
                "labels": [user_segments[left].label, user_segments[right].label],
                "value": float(matrix.iloc[left, right]),
            }
            for left in range(game.n_players)
            for right in range(left + 1, game.n_players)
        ]
        return {
            "linguistic_players": players,
            "first_order_attributions": attributions,
            "pairwise_interactions": interactions,
            "empty_coalition_value": float(empty_score),
            "full_coalition_value": float(full_score),
            "sv_algorithm": sv_algorithm,
            "ksii_algorithm": ksii_algorithm,
            "value_function": (
                "native_direct_answer_continuation_likelihood"
                if target_tool == "no_tool"
                else "native_target_tool_continuation_likelihood"
            ),
            "direct_answer_target": direct_answer_target,
            "hf_runtime_consistency": consistency,
        }

    return RunnerDependencies(
        run_agent=run_agent,
        run_xai=run_xai,
        metadata={
            "inference_backend": "HF local native tool calling",
            "system_prompt": system_prompt,
            "tool_descriptions": dict(TOOLS),
            "segmenter": "LinguisticSegmenter(en_core_web_sm)",
            "players": "user-request linguistic segments",
            "fixed_context": "system prompt and tool descriptions",
            "first_order_index": "SV",
            "pairwise_index": "k-SII",
            "pairwise_max_order": 2,
            "max_exact_players": MAX_EXACT_DEMO_PLAYERS,
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the real representative batch and return a process status code."""
    args = parse_args(argv)
    configuration = BatchConfiguration(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
        quantization=args.quantization,
        max_new_tokens=args.max_new_tokens,
        max_pairs_per_batch=args.max_pairs_per_batch,
    )
    dependencies = build_real_dependencies(configuration)
    experiment = run_experiment(
        dependencies=dependencies,
        configuration=configuration,
        output_dir=args.output_dir,
    )
    paths = output_paths(args.output_dir)
    print(f"Results JSON: {paths.json_path.resolve()}")
    print(f"Summary CSV: {paths.csv_path.resolve()}")
    print(f"Run log: {paths.log_path.resolve()}")
    return 1 if experiment["failed_case_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
