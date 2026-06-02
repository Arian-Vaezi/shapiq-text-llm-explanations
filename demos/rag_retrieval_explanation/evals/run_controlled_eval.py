"""Run lightweight controlled evals for the RAG Shapley demo.

Usage:
    uv run python demos/rag_retrieval_explanation/evals/run_controlled_eval.py

The script intentionally uses the local lexical value function so it can run
without model downloads or API keys. It records enough artifacts to compare
future changes to the explanation behavior.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import shapiq

DEMO_DIR = Path(__file__).resolve().parents[1]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from evaluation import (  # noqa: E402
    deletion_evaluation,
    insertion_evaluation,
    random_deletion_baseline,
    random_insertion_baseline,
    retrieval_score_deletion_baseline,
    summarize_faithfulness,
)
from rag_game import RAGRetrievalGame, RetrievedChunk  # noqa: E402
from sample_data import SAMPLE_TRACES  # noqa: E402

EXPECTED_BEHAVIOR: dict[str, dict[str, Any]] = {
    "Marie Curie Nobel categories": {
        "expected_top_titles": ["1903 Nobel Prize in Physics", "1911 Nobel Prize in Chemistry"],
        "positive_pairs": [("1903 Nobel Prize in Physics", "1911 Nobel Prize in Chemistry")],
        "scenario_type": "complementary_evidence",
    },
    "2008 Beijing Olympics host city": {
        "expected_top_titles": ["2008 Summer Olympics"],
        "scenario_type": "direct_evidence_with_distractors",
    },
    "Unsupported Eiffel Tower answer": {
        "expected_top_titles": [],
        "scenario_type": "missing_evidence",
    },
    "Australia capital confusion": {
        "expected_top_titles": ["Capital fact"],
        "scenario_type": "conflicting_context",
    },
    "Atlantic Ocean redundancy": {
        "expected_top_titles": [
            "Atlantic Ocean — eastern border",
            "South America — ocean borders",
        ],
        "negative_pairs": [
            ("Atlantic Ocean — eastern border", "South America — ocean borders"),
        ],
        "scenario_type": "redundant_evidence",
    },
}


def trace_chunks(trace: dict[str, object]) -> list[RetrievedChunk]:
    """Convert a controlled trace fixture to game chunks."""
    chunks = []
    for raw in trace["chunks"]:
        chunk = dict(raw)
        chunks.append(
            RetrievedChunk(
                title=str(chunk["title"]),
                text=str(chunk["text"]),
                retrieval_score=float(chunk.get("retrieval_score", 0.0)),
                section_title=str(chunk.get("section_title", "")),
            )
        )
    return chunks


def make_approximator(index: str, n_players: int, max_order: int) -> object:
    """Create the same family of shapiq approximators used in the app."""
    if index == "SV":
        return shapiq.KernelSHAP(n=n_players, random_state=42)
    if index == "STII":
        return shapiq.PermutationSamplingSTII(
            n=n_players,
            max_order=max_order,
            random_state=42,
        )
    if index == "FSII":
        return shapiq.RegressionFSII(
            n=n_players,
            max_order=max_order,
            random_state=42,
        )
    return shapiq.KernelSHAPIQ(
        n=n_players,
        index=index,
        max_order=max_order,
        random_state=42,
    )


def first_order_frame(values: shapiq.InteractionValues, labels: list[str]) -> pd.DataFrame:
    """Convert first-order interaction values to a table."""
    rows = []
    for interaction, score in values.dict_values.items():
        if len(interaction) == 1:
            idx = interaction[0]
            rows.append(
                {
                    "chunk": labels[idx],
                    "player": idx + 1,
                    "attribution": float(score),
                    "abs_attribution": abs(float(score)),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("abs_attribution", ascending=False).drop(columns=["abs_attribution"])


def interaction_frame(
    values: shapiq.InteractionValues,
    labels: list[str],
) -> pd.DataFrame:
    """Convert pairwise interactions to a table."""
    rows = []
    if values.max_order < 2:
        return pd.DataFrame(columns=["chunk_a", "chunk_b", "interaction"])
    matrix = values.get_n_order_values(2)
    for i, j in combinations(range(len(labels)), 2):
        rows.append(
            {
                "chunk_a": labels[i],
                "chunk_b": labels[j],
                "interaction": float(matrix[i, j]),
                "abs_interaction": abs(float(matrix[i, j])),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("abs_interaction", ascending=False).drop(columns=["abs_interaction"])


def coalition_scores(game: RAGRetrievalGame, labels: list[str]) -> pd.DataFrame:
    """Evaluate all coalitions for audit records."""
    rows = []
    for size in range(game.n_players + 1):
        for combo in combinations(range(game.n_players), size):
            mask = np.zeros((1, game.n_players), dtype=bool)
            mask[0, list(combo)] = True
            rows.append(
                {
                    "coalition": "|".join(labels[idx] for idx in combo) if combo else "EMPTY",
                    "selected_chunks": len(combo),
                    "score": float(game(mask)[0]),
                }
            )
    return pd.DataFrame(rows)


def normalized_auc(values: np.ndarray) -> float:
    """Return area under a step curve normalized to the number of intervals."""
    if len(values) <= 1:
        return float(values[0]) if len(values) == 1 else 0.0
    x = np.arange(len(values), dtype=float)
    return float(np.trapezoid(values, x) / (len(values) - 1))


def write_config(path: Path, config: dict[str, object]) -> None:
    """Write a small YAML-like config without adding a dependency."""
    lines = []
    for key, value in config.items():
        if value is None:
            rendered = "null"
        elif isinstance(value, str):
            rendered = json.dumps(value)
        else:
            rendered = str(value)
        lines.append(f"{key}: {rendered}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_trace(
    *,
    trace_name: str,
    trace: dict[str, object],
    output_dir: Path,
    index: str,
    max_order: int,
    git_commit: str,
) -> dict[str, object]:
    """Run one controlled trace and write run artifacts."""
    chunks = trace_chunks(trace)
    labels = [chunk.title for chunk in chunks]
    game = RAGRetrievalGame(
        question=str(trace["question"]),
        target_answer=str(trace["target_answer"]),
        chunks=chunks,
    )
    budget = 2**game.n_players
    approximator = make_approximator(index, game.n_players, max_order)
    explanation = approximator.approximate(budget=budget, game=game)

    first_order = first_order_frame(explanation.get_n_order(order=1), labels)
    interactions = interaction_frame(explanation, labels)
    deletion = deletion_evaluation(game, first_order)
    insertion = insertion_evaluation(game, first_order)
    random_deletion = random_deletion_baseline(game)
    random_insertion = random_insertion_baseline(game)
    retrieval_scores = list(range(game.n_players, 0, -1))
    retrieval_deletion = retrieval_score_deletion_baseline(game, retrieval_scores)
    faithfulness = summarize_faithfulness(game, first_order)

    expected = EXPECTED_BEHAVIOR.get(trace_name, {})
    expected_top_titles = list(expected.get("expected_top_titles", []))
    ranked_titles = list(first_order["chunk"]) if not first_order.empty else []
    top_expected_rank = None
    if expected_top_titles:
        ranks = [
            ranked_titles.index(title) + 1
            for title in expected_top_titles
            if title in ranked_titles
        ]
        top_expected_rank = min(ranks) if ranks else None

    full_support = float(game(game.grand_coalition)[0])
    missing_false_positive = (
        expected.get("scenario_type") == "missing_evidence" and full_support >= 0.5
    )
    interaction_hit = None
    if "positive_pairs" in expected:
        interaction_hit = any(
            _pair_value(interactions, pair) > 0 for pair in expected["positive_pairs"]
        )
    if "negative_pairs" in expected:
        interaction_hit = any(
            _pair_value(interactions, pair) < 0 for pair in expected["negative_pairs"]
        )

    metrics = {
        "trace_name": trace_name,
        "scenario_type": expected.get("scenario_type", "unknown"),
        "full_context_support": full_support,
        "top_expected_evidence_rank": top_expected_rank,
        "score_without_top_shapley_chunk": faithfulness.score_without_top,
        "top_chunk_deletion_drop": faithfulness.deletion_drop,
        "sufficiency_gap": faithfulness.sufficiency_gap,
        "deletion_auc_shapley": normalized_auc(deletion["score"].to_numpy(dtype=float)),
        "deletion_auc_retrieval": (
            normalized_auc(retrieval_deletion) if retrieval_deletion is not None else None
        ),
        "deletion_auc_random": normalized_auc(random_deletion),
        "insertion_auc_shapley": normalized_auc(insertion["score"].to_numpy(dtype=float)),
        "insertion_auc_random": normalized_auc(random_insertion),
        "interaction_hit": interaction_hit,
        "missing_evidence_false_positive": bool(missing_false_positive),
    }

    trace_dir = output_dir / _slug(trace_name)
    trace_dir.mkdir(parents=True, exist_ok=True)
    write_config(
        trace_dir / "config.yaml",
        {
            "git_commit": git_commit,
            "scenario": trace_name,
            "input_source": "controlled scenario",
            "value_function": "Lexical grounding",
            "interaction_index": index,
            "max_order": max_order,
            "budget": budget,
            "retrieval_method": "fixture order",
            "embedding_model": None,
            "generator_model": None,
        },
    )
    (trace_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    first_order.to_csv(trace_dir / "shapley_values.csv", index=False)
    interactions.to_csv(trace_dir / "interactions.csv", index=False)
    deletion.to_csv(trace_dir / "deletion_curve.csv", index=False)
    insertion.to_csv(trace_dir / "insertion_curve.csv", index=False)
    coalition_scores(game, labels).to_csv(trace_dir / "coalition_scores.csv", index=False)
    (trace_dir / "notes.md").write_text(
        f"# {trace_name}\n\n"
        f"Takeaway: {trace.get('takeaway', '')}\n\n"
        "This run uses the local lexical value function. Treat it as a stable "
        "explanation regression check, not as a final model-backed evaluation.\n",
        encoding="utf-8",
    )
    return metrics


def _pair_value(frame: pd.DataFrame, pair: tuple[str, str]) -> float:
    """Return the interaction value for an unordered title pair."""
    if frame.empty:
        return 0.0
    a, b = pair
    match = frame[
        ((frame["chunk_a"] == a) & (frame["chunk_b"] == b))
        | ((frame["chunk_a"] == b) & (frame["chunk_b"] == a))
    ]
    if match.empty:
        return 0.0
    return float(match.iloc[0]["interaction"])


def _slug(text: str) -> str:
    """Return a filesystem-friendly run name."""
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in slug.split("_") if part)


def main() -> int:
    """Run all controlled evals."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="k-SII", choices=["SV", "k-SII", "STII", "FSII"])
    parser.add_argument("--max-order", type=int, default=2)
    parser.add_argument("--git-commit", default="unknown")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent
        / "runs"
        / datetime.now(tz=UTC).strftime("%Y-%m-%d_%H%M%S_controlled_lexical"),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_trace(
            trace_name=name,
            trace=trace,
            output_dir=args.output_dir,
            index=args.index,
            max_order=args.max_order,
            git_commit=args.git_commit,
        )
        for name, trace in SAMPLE_TRACES.items()
    ]
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote controlled eval run to {args.output_dir}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
