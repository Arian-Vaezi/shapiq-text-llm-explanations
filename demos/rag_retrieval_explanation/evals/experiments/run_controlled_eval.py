"""Run retrieval and Shapley evaluation on the bundled controlled corpus."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
import pandas as pd

import shapiq
from demos.rag_retrieval_explanation.core.evaluation import (
    deletion_evaluation,
    deletion_evaluation_from_order,
    insertion_evaluation,
    leave_one_out_order,
    random_deletion_baseline,
    retrieval_score_deletion_baseline,
    retrieval_score_order,
)
from demos.rag_retrieval_explanation.core.generation import format_rag_context
from demos.rag_retrieval_explanation.core.model_backends import LlamaCppBackend
from demos.rag_retrieval_explanation.core.rag_game import RAGRetrievalGame, ScoreCallable
from demos.rag_retrieval_explanation.core.retrieval import (
    resolved_embedding_device,
    retrieve_relevant_chunks_with_debug,
)
from demos.rag_retrieval_explanation.core.value_functions import (
    ContrastiveLikelihoodValue,
    LexicalGroundingValue,
    TargetLikelihoodValue,
)

from .benchmark import (
    DEFAULT_BENCHMARK_PATH,
    BenchmarkCase,
    ControlledBenchmark,
    benchmark_candidates,
    gold_chunks,
    load_benchmark,
    passage_id_from_title,
)
from .metrics import (
    attribution_metrics,
    classify_knowledge_source,
    exact_match,
    interaction_diagnostics,
    normalized_auc,
    retrieval_metrics,
    token_f1,
)

if TYPE_CHECKING:
    from demos.rag_retrieval_explanation.core.schemas import RetrievedChunk

RUNS_DIR = Path(__file__).parents[1] / "runs"
VALUE_FUNCTION_CHOICES = ("lexical", "target_likelihood", "contrastive_likelihood")


class GenerationBackend(Protocol):
    """Minimal model interface used by the optional full protocol."""

    def generate(self, question: str, context: str) -> object:
        """Generate an answer for one context condition."""

    def target_log_likelihood(self, prompt: str, target_answer: str) -> float:
        """Score target tokens after a prompt."""

    def build_prompt(self, question: str, context: str) -> str:
        """Build the model prompt used for target likelihood."""


def _first_order_frame(
    values: shapiq.InteractionValues, chunks: list[RetrievedChunk]
) -> pd.DataFrame:
    rows = [
        {
            "player": interaction[0] + 1,
            "chunk": chunks[interaction[0]].title,
            "attribution": float(score),
        }
        for interaction, score in values.dict_values.items()
        if len(interaction) == 1
    ]
    if not rows:
        return pd.DataFrame(columns=["player", "chunk", "attribution"])
    frame = pd.DataFrame(rows)
    return (
        frame.assign(abs_attribution=frame["attribution"].abs())
        .sort_values("abs_attribution", ascending=False)
        .drop(columns="abs_attribution")
    )


def _explain(
    *,
    case: BenchmarkCase,
    chunks: list[RetrievedChunk],
    target_answer: str,
    scorer: ScoreCallable,
) -> tuple[pd.DataFrame, np.ndarray, RAGRetrievalGame]:
    game = RAGRetrievalGame(
        question=case.question,
        target_answer=target_answer,
        chunks=chunks,
        scorer=scorer,
    )
    budget = 2**game.n_players
    first_order = shapiq.KernelSHAP(n=game.n_players, random_state=42).approximate(
        budget=budget,
        game=game,
    )
    interactions = shapiq.KernelSHAPIQ(
        n=game.n_players,
        index="k-SII",
        max_order=2,
        random_state=42,
    ).approximate(budget=budget, game=game)
    return (
        _first_order_frame(first_order.get_n_order(order=1), chunks),
        interactions.get_n_order_values(2),
        game,
    )


def _interaction_rows(
    case: BenchmarkCase,
    chunks: list[RetrievedChunk],
    matrix: np.ndarray,
) -> list[dict[str, object]]:
    position = {passage_id_from_title(chunk.title): index for index, chunk in enumerate(chunks)}
    rows: list[dict[str, object]] = []
    for interaction in case.expected_interactions:
        left = interaction.left
        right = interaction.right
        relation = interaction.relation
        observed = None
        sign_correct = None
        status = "evaluable"
        if left in position and right in position:
            observed = float(matrix[position[left], position[right]])
            sign_correct = observed > 0 if interaction.expected_sign == "positive" else observed < 0
        else:
            missing = []
            if left not in position:
                missing.append("left")
            if right not in position:
                missing.append("right")
            status = "undefined_missing_" + "_and_".join(missing)
        rows.append(
            {
                "left": left,
                "right": right,
                "relation": relation,
                "expected_sign": interaction.expected_sign,
                "observed_interaction": observed,
                "interaction_sign_correct": sign_correct,
                "status": status,
            }
        )
    return rows


def _all_pairwise_interaction_rows(
    chunks: list[RetrievedChunk],
    matrix: np.ndarray,
) -> list[dict[str, object]]:
    """Return every retrieved pairwise interaction, not only labelled pairs."""
    rows: list[dict[str, object]] = []
    for left_index, left_chunk in enumerate(chunks):
        for right_index in range(left_index + 1, len(chunks)):
            right_chunk = chunks[right_index]
            value = float(matrix[left_index, right_index])
            rows.append(
                {
                    "left": passage_id_from_title(left_chunk.title),
                    "right": passage_id_from_title(right_chunk.title),
                    "left_rank": left_index + 1,
                    "right_rank": right_index + 1,
                    "interaction": value,
                    "sign": "positive" if value > 0 else "negative" if value < 0 else "zero",
                }
            )
    return rows


def _coalition_scores(game: RAGRetrievalGame) -> list[dict[str, object]]:
    """Expose cached coalition scores for artifact-level reproducibility."""
    rows: list[dict[str, object]] = []
    for size in range(game.n_players + 1):
        for coalition in combinations(range(game.n_players), size):
            mask = np.zeros(game.n_players, dtype=bool)
            mask[list(coalition)] = True
            score = float(game(mask.reshape(1, -1))[0])
            rows.append(
                {
                    "players": [player + 1 for player in coalition],
                    "passage_ids": [
                        passage_id_from_title(game.chunks[player].title) for player in coalition
                    ],
                    "score": score,
                }
            )
    return rows


def _case_metadata(case: BenchmarkCase) -> dict[str, object]:
    """Save the controlled label metadata used by an evaluation artifact."""
    return {
        "case_id": case.case_id,
        "pattern": case.pattern,
        "question": case.question,
        "gold_answer": case.gold_answer,
        "gold_evidence_ids": list(case.gold_evidence_ids),
        "top_k": case.top_k,
        "expected_interactions": [
            {
                "left": interaction.left,
                "right": interaction.right,
                "relation": interaction.relation,
                "expected_sign": interaction.expected_sign,
            }
            for interaction in case.expected_interactions
        ],
    }


def _generated_text(result: object) -> str:
    answer = getattr(result, "answer", result)
    return str(answer).strip()


def _generate_conditions(
    backend: GenerationBackend,
    case: BenchmarkCase,
    retrieved: list[RetrievedChunk],
    labelled_gold: list[RetrievedChunk],
    *,
    f1_threshold: float,
) -> dict[str, object]:
    closed_answer = _generated_text(backend.generate(case.question, format_rag_context([])))
    gold_answer = _generated_text(
        backend.generate(case.question, format_rag_context(labelled_gold))
    )
    retrieved_answer = _generated_text(
        backend.generate(case.question, format_rag_context(retrieved))
    )
    closed_f1 = token_f1(closed_answer, case.gold_answer)
    gold_f1 = token_f1(gold_answer, case.gold_answer)
    retrieved_f1 = token_f1(retrieved_answer, case.gold_answer)
    return {
        "closed_book_answer": closed_answer,
        "gold_context_answer": gold_answer,
        "retrieved_context_answer": retrieved_answer,
        "closed_book_f1": closed_f1,
        "gold_context_f1": gold_f1,
        "retrieved_context_f1": retrieved_f1,
        "closed_book_exact_match": exact_match(closed_answer, case.gold_answer),
        "gold_context_exact_match": exact_match(gold_answer, case.gold_answer),
        "retrieved_context_exact_match": exact_match(retrieved_answer, case.gold_answer),
        "knowledge_source_class": classify_knowledge_source(
            closed_book_f1=closed_f1,
            gold_context_f1=gold_f1,
            retrieved_context_f1=retrieved_f1,
            threshold=f1_threshold,
        ),
    }


def evaluate_case(
    benchmark: ControlledBenchmark,
    case: BenchmarkCase,
    *,
    method: str = "TF-IDF",
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_device: str = "auto",
    value_function: str = "lexical",
    backend: GenerationBackend | None = None,
    f1_threshold: float = 0.65,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate one question from retrieval through optional model interventions."""
    ranked, debug = retrieve_relevant_chunks_with_debug(
        case.question,
        benchmark_candidates(benchmark),
        top_k=case.top_k,
        method=method,
        embedding_model_id=embedding_model_id,
        embedding_device=embedding_device,
    )
    retrieved = [item.chunk for item in ranked]
    retrieved_ids = [passage_id_from_title(chunk.title) for chunk in retrieved]
    retrieval_scores = [float(item.score) for item in ranked]

    if value_function == "lexical":
        scorer = LexicalGroundingValue()
    elif value_function == "target_likelihood" and backend is not None:
        scorer = TargetLikelihoodValue(backend=cast("LlamaCppBackend", backend))
    elif value_function == "contrastive_likelihood" and backend is not None:
        scorer = ContrastiveLikelihoodValue(backend=cast("LlamaCppBackend", backend))
    elif value_function not in VALUE_FUNCTION_CHOICES:
        msg = f"Unknown value function: {value_function}"
        raise ValueError(msg)
    else:
        msg = f"{value_function} requires --model-path."
        raise ValueError(msg)
    gold_frame, gold_interactions, gold_game = _explain(
        case=case,
        chunks=retrieved,
        target_answer=case.gold_answer,
        scorer=scorer,
    )
    deletion = deletion_evaluation(gold_game, gold_frame)["score"].to_numpy(dtype=float)
    insertion = insertion_evaluation(gold_game, gold_frame)["score"].to_numpy(dtype=float)
    retrieval_deletion = retrieval_score_deletion_baseline(gold_game, retrieval_scores)
    random_deletion = random_deletion_baseline(gold_game)
    leave_one_out_deletion = deletion_evaluation_from_order(
        gold_game,
        leave_one_out_order(gold_game),
    )

    row: dict[str, object] = {
        "case_id": case.case_id,
        "pattern": case.pattern,
        "question": case.question,
        "gold_answer": case.gold_answer,
        "retrieved_ids": "|".join(retrieved_ids),
        **retrieval_metrics(retrieved_ids, case.gold_evidence_ids),
        **attribution_metrics(gold_frame, case.gold_evidence_ids),
        "gold_deletion_auc": normalized_auc(deletion),
        "gold_insertion_auc": normalized_auc(insertion),
        "retrieval_order_deletion_auc": (
            normalized_auc(retrieval_deletion) if retrieval_deletion is not None else None
        ),
        "leave_one_out_deletion_auc": normalized_auc(
            leave_one_out_deletion["score"].to_numpy(dtype=float)
        ),
        "random_deletion_auc": normalized_auc(random_deletion),
        "full_gold_support": float(gold_game(gold_game.grand_coalition)[0]),
    }
    interaction_rows = _interaction_rows(case, retrieved, gold_interactions)
    row.update(interaction_diagnostics(interaction_rows))

    generated_frame = pd.DataFrame()
    generated_interactions_rows: list[dict[str, object]] = []
    generated_pairwise_interactions: list[dict[str, object]] = []
    generated_coalition_scores: list[dict[str, object]] = []
    generation = {}
    if backend is not None:
        generation = _generate_conditions(
            backend,
            case,
            retrieved,
            gold_chunks(benchmark, case),
            f1_threshold=f1_threshold,
        )
        if not case.gold_evidence_ids:
            generation["knowledge_source_class"] = "missing_evidence"
        row.update(generation)
        generated_frame, generated_interactions, generated_game = _explain(
            case=case,
            chunks=retrieved,
            target_answer=str(generation["retrieved_context_answer"]),
            scorer=(
                ContrastiveLikelihoodValue(backend=cast("LlamaCppBackend", backend))
                if value_function == "contrastive_likelihood"
                else TargetLikelihoodValue(backend=cast("LlamaCppBackend", backend))
            ),
        )
        generated_interactions_rows = _interaction_rows(case, retrieved, generated_interactions)
        generated_pairwise_interactions = _all_pairwise_interaction_rows(
            retrieved,
            generated_interactions,
        )
        generated_coalition_scores = _coalition_scores(generated_game)
        row.update(
            {
                f"generated_{key}": value
                for key, value in interaction_diagnostics(generated_interactions_rows).items()
            }
        )
        row.update(
            {
                f"generated_{key}": value
                for key, value in attribution_metrics(
                    generated_frame, case.gold_evidence_ids
                ).items()
            }
        )
        row["full_generated_support"] = float(generated_game(generated_game.grand_coalition)[0])

    artifacts: dict[str, object] = {
        "case": _case_metadata(case),
        "retrieved": [
            {
                "rank": index,
                "passage_id": passage_id_from_title(item.chunk.title),
                "score": float(item.score),
                "title": item.chunk.title,
                "text": item.chunk.text,
            }
            for index, item in enumerate(ranked, start=1)
        ],
        "gold_attributions": gold_frame.to_dict(orient="records"),
        "generated_attributions": generated_frame.to_dict(orient="records"),
        "interactions": interaction_rows,
        "generated_interactions": generated_interactions_rows,
        "gold_pairwise_interactions": _all_pairwise_interaction_rows(
            retrieved,
            gold_interactions,
        ),
        "generated_pairwise_interactions": generated_pairwise_interactions,
        "gold_coalition_scores": _coalition_scores(gold_game),
        "generated_coalition_scores": generated_coalition_scores,
        "faithfulness_curves": {
            "gold_shapley_deletion": deletion_evaluation(gold_game, gold_frame).to_dict(
                orient="records"
            ),
            "gold_shapley_insertion": insertion_evaluation(gold_game, gold_frame).to_dict(
                orient="records"
            ),
            "retrieval_score_deletion": (
                deletion_evaluation_from_order(
                    gold_game,
                    retrieval_score_order(retrieval_scores, gold_game.n_players),
                ).to_dict(orient="records")
                if retrieval_deletion is not None
                else []
            ),
            "leave_one_out_deletion": leave_one_out_deletion.to_dict(orient="records"),
            "random_deletion_mean": [
                {"step": index, "score": float(score)}
                for index, score in enumerate(random_deletion)
            ],
        },
        "retrieval_debug": asdict(debug),
        "generation": generation,
    }
    return row, artifacts


def run_benchmark(
    *,
    output_dir: Path,
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    method: str = "TF-IDF",
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_device: str = "auto",
    value_function: str = "lexical",
    backend: GenerationBackend | None = None,
    model_path: str | None = None,
    n_gpu_layers: int | None = None,
    case_ids: set[str] | None = None,
    limit: int | None = None,
    f1_threshold: float = 0.65,
) -> pd.DataFrame:
    """Run all controlled cases and write reproducible result artifacts."""
    benchmark = load_benchmark(benchmark_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    for stale_artifact in artifacts_dir.glob("*.json"):
        stale_artifact.unlink()
    rows = []
    selected_cases = [
        case for case in benchmark.cases if case_ids is None or case.case_id in case_ids
    ]
    if limit is not None:
        selected_cases = selected_cases[:limit]
    for case in selected_cases:
        row, artifacts = evaluate_case(
            benchmark,
            case,
            method=method,
            embedding_model_id=embedding_model_id,
            embedding_device=embedding_device,
            value_function=value_function,
            backend=backend,
            f1_threshold=f1_threshold,
        )
        rows.append(row)
        (artifacts_dir / f"{case.case_id}.json").write_text(
            json.dumps(artifacts, indent=2),
            encoding="utf-8",
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    metadata = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "benchmark": str(benchmark_path),
        "retrieval_method": method,
        "embedding_model": embedding_model_id if method == "Dense embeddings" else None,
        "embedding_device_requested": (embedding_device if method == "Dense embeddings" else None),
        "embedding_device_resolved": (
            resolved_embedding_device(embedding_model_id, embedding_device)
            if method == "Dense embeddings"
            else None
        ),
        "value_function": value_function,
        "model_protocol": backend is not None,
        "model_path": model_path,
        "n_gpu_layers": n_gpu_layers,
        "f1_threshold": f1_threshold,
        "case_count": len(rows),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    _write_report(summary, output_dir)
    return summary


def _mean(frame: pd.DataFrame, column: str) -> str:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return f"{values.mean():.3f}" if not values.empty else "n/a"


def _mean_boolean(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].dropna()
    if values.empty:
        return "n/a"
    return f"{values.astype(bool).mean():.3f}"


def _write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    interaction_total = pd.to_numeric(
        summary.get("interaction_total_pairs"),
        errors="coerce",
    ).sum()
    interaction_evaluable = pd.to_numeric(
        summary.get("interaction_evaluable_pairs"),
        errors="coerce",
    ).sum()
    interaction_skipped = pd.to_numeric(
        summary.get("interaction_skipped_pairs"),
        errors="coerce",
    ).sum()
    interaction_correct = pd.to_numeric(
        summary.get("interaction_correct_signs"),
        errors="coerce",
    ).sum()
    interaction_recovery = (
        interaction_correct / interaction_evaluable if interaction_evaluable else None
    )
    interaction_recovery_text = (
        f"{interaction_recovery:.3f}" if interaction_recovery is not None else "n/a"
    )
    lines = [
        "# Controlled Retrieval and Attribution Evaluation",
        "",
        f"- Cases: {len(summary)}",
        f"- Mean retrieval Recall@k: {_mean(summary, 'retrieval_recall_at_k')}",
        f"- Mean retrieval MRR: {_mean(summary, 'retrieval_mrr')}",
        f"- Top attribution is gold: {_mean_boolean(summary, 'top_attribution_is_gold')}",
        f"- Mean gold attribution mass: {_mean(summary, 'gold_attribution_mass')}",
        f"- Mean gold deletion AUC: {_mean(summary, 'gold_deletion_auc')}",
        f"- Mean leave-one-out deletion AUC: {_mean(summary, 'leave_one_out_deletion_auc')}",
        f"- Mean random deletion AUC: {_mean(summary, 'random_deletion_auc')}",
        f"- Interaction sign recovery: {interaction_recovery_text}",
        f"- Interaction pairs: {int(interaction_evaluable)} evaluable / "
        f"{int(interaction_total)} labelled; {int(interaction_skipped)} skipped",
        "",
        "The controlled corpus is searched by the same retrieval code as the PDF path.",
        "Gold-answer attribution and generated-answer attribution are stored separately.",
        "Missing-evidence cases have no gold evidence and are excluded from evidence recall.",
        "Interaction accuracy excludes labelled pairs where either passage was not retrieved.",
        "",
        "See `summary.csv` and `artifacts/` for per-case results.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-path", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument("--method", choices=["TF-IDF", "Dense embeddings"], default="TF-IDF")
    parser.add_argument(
        "--embedding-model",
        default="BAAI/bge-base-en-v1.5",
    )
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument(
        "--value-function",
        choices=VALUE_FUNCTION_CHOICES,
        default="lexical",
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--f1-threshold", type=float, default=0.65)
    args = parser.parse_args()

    backend = None
    if args.model_path is not None:
        backend = LlamaCppBackend(
            model_path=str(args.model_path),
            n_gpu_layers=args.n_gpu_layers,
        )
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or RUNS_DIR / f"{stamp}_controlled"
    summary = run_benchmark(
        output_dir=output_dir,
        benchmark_path=args.benchmark_path,
        method=args.method,
        embedding_model_id=args.embedding_model,
        embedding_device=args.embedding_device,
        value_function=args.value_function,
        backend=backend,
        model_path=str(args.model_path) if args.model_path is not None else None,
        n_gpu_layers=args.n_gpu_layers if args.model_path is not None else None,
        case_ids=set(args.case_ids) if args.case_ids else None,
        limit=args.limit,
        f1_threshold=args.f1_threshold,
    )
    print(f"Wrote {len(summary)} cases to {output_dir}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
