"""Run a structured 20-case Agent + XAI analysis from the existing holdout set."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

from run_representative_xai import (
    BatchConfiguration,
    RepresentativeCase,
    build_parser,
    build_real_dependencies,
    output_paths,
    run_experiment,
)

DEMO_DIR = Path(__file__).resolve().parent
HOLDOUT_PATH = DEMO_DIR / "holdout_samples.json"
PRIOR_RESULTS_PATH = DEMO_DIR / "holdout_eval_results_3b.json"
DEFAULT_OUTPUT_DIR = Path("outputs/representative_xai_20")
CASE_SELECTION_DESCRIPTION = (
    "Twenty cases (five per category) selected before XAI from the existing 40-prompt "
    "holdout for interpretability and coverage; this is not a new accuracy benchmark."
)

SELECTION_REASONS = {
    "w01": "Canonical forecast request with location and future time.",
    "w03": "Alternate imperative syntax with hourly, wind, and time spans.",
    "w04": "Indirect umbrella wording tests implied weather intent.",
    "w07": "Boundary case mixing stable cloud context with a dated forecast.",
    "w10": "Concept-versus-live boundary likely to expose a meaningful interaction.",
    "c01": "Canonical single-operation arithmetic request.",
    "c05": "Geometric wording combines dimensions, units, and requested quantity.",
    "c06": "Symbolic expression gives syntactically distinct operator structure.",
    "c07": "Boundary wording combines conceptual context and repeated percentage growth.",
    "c08": "Explanation-plus-computation request should reveal complementary spans.",
    "s03": "Canonical current-news request with institution and recency cue.",
    "s04": "Imperative lookup phrasing combines a venue with present-day information.",
    "s07": "Boundary case mixes stable context with latest company figures.",
    "s08": "Explanation-plus-current-data request should yield a contextual interaction.",
    "s09": "Prior routing miss retained to analyze negation and recency limitations.",
    "n01": "Canonical stable conceptual explanation.",
    "n03": "Procedural knowledge question with distinct interrogative syntax.",
    "n04": "Multi-part direct-answer request asking for a fixed number of tips.",
    "n08": "Weather vocabulary plus explicit no-forecast constraint tests cue resistance.",
    "n10": "Search vocabulary plus explicit no-live-results constraint tests cue resistance.",
}

SELECTED_CASE_IDS = tuple(SELECTION_REASONS)
CATEGORY_BY_TOOL = {
    "weather_tool": "weather",
    "calculator_tool": "calculator",
    "web_search_tool": "web_search",
    "no_tool": "no_tool",
}


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def load_holdout() -> tuple[Mapping[str, object], ...]:
    """Load the source holdout without copying prompts into this runner."""
    payload = _load_json(HOLDOUT_PATH)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        msg = f"Expected a list of objects in {HOLDOUT_PATH}."
        raise TypeError(msg)
    return tuple(payload)


def load_prior_results() -> dict[str, Mapping[str, object]]:
    """Index the committed Qwen 3B routing results by holdout ID."""
    payload = _load_json(PRIOR_RESULTS_PATH)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        msg = f"Expected an object with list-valued results in {PRIOR_RESULTS_PATH}."
        raise TypeError(msg)
    return {
        str(row["id"]): row for row in payload["results"] if isinstance(row, dict) and "id" in row
    }


def build_selected_cases() -> tuple[RepresentativeCase, ...]:
    """Resolve the fixed selection against the canonical 40-prompt holdout."""
    holdout_by_id = {str(row["id"]): row for row in load_holdout()}
    missing = [case_id for case_id in SELECTED_CASE_IDS if case_id not in holdout_by_id]
    if missing:
        msg = f"Selected case IDs missing from holdout: {missing}"
        raise ValueError(msg)
    return tuple(
        RepresentativeCase(
            case_id=case_id,
            category=CATEGORY_BY_TOOL[str(holdout_by_id[case_id]["ground_truth"])],
            request=str(holdout_by_id[case_id]["request"]),
            expected_tool=str(holdout_by_id[case_id]["ground_truth"]),
        )
        for case_id in SELECTED_CASE_IDS
    )


SELECTED_CASES = build_selected_cases()


def parse_args(argv: Sequence[str] | None = None) -> object:
    """Reuse the base CLI while selecting a non-overlapping default output directory."""
    parser = build_parser()
    parser.add_argument(
        "--case-ids",
        nargs="+",
        choices=SELECTED_CASE_IDS,
        help="Run only these fixed representative case IDs, in the provided order.",
    )
    args = parser.parse_args(argv)
    if args.output_dir == Path("outputs"):
        args.output_dir = DEFAULT_OUTPUT_DIR
    if args.case_ids and len(args.case_ids) != len(set(args.case_ids)):
        parser.error("--case-ids must not contain duplicates")
    return args


def select_cases(case_ids: Sequence[str] | None) -> tuple[RepresentativeCase, ...]:
    """Return all fixed cases or the requested ordered subset."""
    if not case_ids:
        return SELECTED_CASES
    by_id = {case.case_id: case for case in SELECTED_CASES}
    return tuple(by_id[case_id] for case_id in case_ids)


def print_selection_table(cases: Sequence[RepresentativeCase] = SELECTED_CASES) -> None:
    """Print the finalized pre-XAI selection and its prior routing evidence."""
    prior = load_prior_results()
    print(
        "| case_id | category | request | expected_tool | prior actual tool | "
        "prior correctness | reason for selection |"
    )
    print("|---|---|---|---|---|---|---|")
    for case in cases:
        previous = prior.get(case.case_id, {})
        request = case.request.replace("|", "\\|")
        reason = SELECTION_REASONS[case.case_id].replace("|", "\\|")
        print(
            f"| {case.case_id} | {case.category} | {request} | {case.expected_tool} | "
            f"{previous.get('selected_tool')} | {previous.get('correct')} | {reason} |"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected cases through the unchanged real dependency pipeline."""
    args = parse_args(argv)
    cases = select_cases(args.case_ids)
    print_selection_table(cases)
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
        cases=cases,
        case_selection_description=(
            CASE_SELECTION_DESCRIPTION
            if cases is SELECTED_CASES
            else f"{CASE_SELECTION_DESCRIPTION} Filtered case IDs: "
            f"{', '.join(case.case_id for case in cases)}."
        ),
    )
    paths = output_paths(args.output_dir)
    print(f"Results JSON: {paths.json_path.resolve()}")
    print(f"Summary CSV: {paths.csv_path.resolve()}")
    print(f"Run log: {paths.log_path.resolve()}")
    return 1 if experiment["failed_case_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
