"""Run external retrieval and Shapley evaluation on frozen QASPER cases."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
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
)

import shapiq

from .benchmark import passage_id_from_title
from .metrics import (
    attribution_metrics,
    classify_knowledge_source,
    exact_match,
    normalized_auc,
    retrieval_metrics,
    token_f1,
)
from .qasper import (
    DEFAULT_CACHE_PATH,
    DEFAULT_SELECTION_PATH,
    QasperCase,
    case_metadata,
    download_qasper_validation,
    load_frozen_qasper_cases,
    qasper_candidates,
    qasper_gold_chunks,
    select_qasper_cases,
    write_selection_manifest,
)
from .run_controlled_eval import _first_order_frame

if TYPE_CHECKING:
    import numpy as np
    from demos.rag_retrieval_explanation.core.schemas import RetrievedChunk

RUNS_DIR = Path(__file__).parent / "runs"
DEFAULT_MODEL_PATH = Path("models/llm/gemma-4-E4B-it-Q4_K_M.gguf")
DEFAULT_EMBEDDING_PATH = Path("models/embedding/bge-base-en-v1.5")


def _generated_text(result: object) -> str:
    return str(getattr(result, "answer", result)).strip()


def _explain(
    *,
    question: str,
    chunks: list[RetrievedChunk],
    target_answer: str,
    scorer: ScoreCallable,
) -> tuple[pd.DataFrame, np.ndarray, RAGRetrievalGame]:
    game = RAGRetrievalGame(
        question=question,
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
    chunks: list[RetrievedChunk],
    matrix: np.ndarray,
) -> list[dict[str, object]]:
    return [
        {
            "left": passage_id_from_title(chunks[left].title),
            "right": passage_id_from_title(chunks[right].title),
            "interaction": float(matrix[left, right]),
        }
        for left in range(len(chunks))
        for right in range(left + 1, len(chunks))
    ]


def _generation_conditions(
    backend: LlamaCppBackend,
    case: QasperCase,
    retrieved: list[RetrievedChunk],
    *,
    f1_threshold: float,
) -> dict[str, object]:
    closed_answer = _generated_text(backend.generate(case.question, format_rag_context([])))
    gold_answer = _generated_text(
        backend.generate(case.question, format_rag_context(qasper_gold_chunks(case)))
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


def _qasper_failure_category(
    *,
    retrieval_recall: object,
    generation: dict[str, object],
    generated_top_is_gold: object,
    f1_threshold: float,
) -> str:
    """Assign an interpretable external-validation failure bucket."""
    recall = float(retrieval_recall or 0.0)
    gold_f1 = float(generation["gold_context_f1"])
    retrieved_f1 = float(generation["retrieved_context_f1"])
    if recall == 0.0:
        return "retrieval_miss"
    if gold_f1 < f1_threshold:
        return "generation_failure_with_gold_context"
    if retrieved_f1 < f1_threshold:
        return "retrieved_context_failure"
    if generated_top_is_gold is False:
        return "generated_attribution_mismatch"
    return "externally_supported"


def evaluate_qasper_case(
    case: QasperCase,
    *,
    backend: LlamaCppBackend,
    embedding_model_id: str = "BAAI/bge-base-en-v1.5",
    embedding_device: str = "auto",
    f1_threshold: float = 0.65,
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate one frozen QASPER case end to end."""
    ranked, debug = retrieve_relevant_chunks_with_debug(
        case.question,
        qasper_candidates(case),
        top_k=case.top_k,
        method="Dense embeddings",
        embedding_model_id=embedding_model_id,
        embedding_device=embedding_device,
    )
    retrieved = [item.chunk for item in ranked]
    retrieved_ids = [passage_id_from_title(chunk.title) for chunk in retrieved]
    retrieval_scores = [float(item.score) for item in ranked]
    scorer = ContrastiveLikelihoodValue(backend=backend)

    gold_frame, gold_interactions, gold_game = _explain(
        question=case.question,
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
    generation = _generation_conditions(
        backend,
        case,
        retrieved,
        f1_threshold=f1_threshold,
    )

    generated_frame, generated_interactions, generated_game = _explain(
        question=case.question,
        chunks=retrieved,
        target_answer=str(generation["retrieved_context_answer"]),
        scorer=ContrastiveLikelihoodValue(backend=backend),
    )
    retrieval_metric_values = retrieval_metrics(retrieved_ids, case.gold_evidence_ids)
    generated_attribution_metric_values = attribution_metrics(
        generated_frame,
        case.gold_evidence_ids,
    )
    row: dict[str, object] = {
        **case_metadata(case),
        "passage_count": len(case.passages),
        "retrieved_ids": "|".join(retrieved_ids),
        **retrieval_metric_values,
        **attribution_metrics(gold_frame, case.gold_evidence_ids),
        **generation,
        **{f"generated_{key}": value for key, value in generated_attribution_metric_values.items()},
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
        "full_generated_support": float(generated_game(generated_game.grand_coalition)[0]),
    }
    row["qasper_failure_category"] = _qasper_failure_category(
        retrieval_recall=row["retrieval_recall_at_k"],
        generation=generation,
        generated_top_is_gold=row["generated_top_attribution_is_gold"],
        f1_threshold=f1_threshold,
    )
    artifacts: dict[str, object] = {
        "case": case_metadata(case),
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
        "gold_interactions": _interaction_rows(retrieved, gold_interactions),
        "generated_interactions": _interaction_rows(retrieved, generated_interactions),
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
        "failure_category": row["qasper_failure_category"],
    }
    return row, artifacts


def _mean(frame: pd.DataFrame, column: str) -> str:
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return f"{values.mean():.3f}" if not values.empty else "n/a"


def _mean_boolean(frame: pd.DataFrame, column: str) -> str:
    values = frame[column].dropna()
    return f"{values.astype(bool).mean():.3f}" if not values.empty else "n/a"


def _write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    diagnoses = summary["knowledge_source_class"].value_counts().to_dict()
    failure_categories = summary["qasper_failure_category"].value_counts().to_dict()
    lines = [
        "# QASPER External Evaluation",
        "",
        f"- Cases: {len(summary)}",
        f"- Mean retrieval Recall@6: {_mean(summary, 'retrieval_recall_at_k')}",
        f"- Mean retrieval MRR: {_mean(summary, 'retrieval_mrr')}",
        f"- Gold-answer top attribution is evidence: "
        f"{_mean_boolean(summary, 'top_attribution_is_gold')}",
        f"- Gold-answer attribution mass on evidence: "
        f"{_mean(summary, 'gold_attribution_mass')}",
        f"- Generated-answer top attribution is evidence: "
        f"{_mean_boolean(summary, 'generated_top_attribution_is_gold')}",
        f"- Generated-answer attribution mass on evidence: "
        f"{_mean(summary, 'generated_gold_attribution_mass')}",
        f"- Gold-answer deletion AUC: {_mean(summary, 'gold_deletion_auc')}",
        f"- Random deletion AUC: {_mean(summary, 'random_deletion_auc')}",
        f"- Closed-book token F1: {_mean(summary, 'closed_book_f1')}",
        f"- Gold-context token F1: {_mean(summary, 'gold_context_f1')}",
        f"- Retrieved-context token F1: {_mean(summary, 'retrieved_context_f1')}",
        f"- Knowledge-source diagnoses: {json.dumps(diagnoses, sort_keys=True)}",
        f"- QASPER failure categories: {json.dumps(failure_categories, sort_keys=True)}",
        "",
        "Configuration: QASPER validation, paragraph candidates, BGE-base top-6,",
        "Gemma E4B, and local contrastive likelihood with exact 64-coalition games.",
        "",
        "See `summary.csv`, `manifest.json`, and `artifacts/` for case-level results.",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_qasper(
    *,
    output_dir: Path,
    backend: LlamaCppBackend,
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    embedding_model_id: str = "BAAI/bge-base-en-v1.5",
    embedding_device: str = "auto",
    limit: int | None = None,
    resume_from: Path | None = None,
    f1_threshold: float = 0.65,
) -> pd.DataFrame:
    """Run the frozen QASPER subset and persist reproducible artifacts."""
    selection, cases = load_frozen_qasper_cases()
    selected_cases = cases[:limit] if limit is not None else cases
    if resume_from is not None and resume_from.resolve() != output_dir.resolve():
        shutil.copytree(resume_from, output_dir, dirs_exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    existing_rows: dict[str, dict[str, object]] = {}
    existing_summary_path = output_dir / "summary.json"
    if existing_summary_path.is_file():
        existing_rows = {
            str(row["case_id"]): row
            for row in json.loads(existing_summary_path.read_text(encoding="utf-8"))
        }
    rows = []
    for index, case in enumerate(selected_cases, start=1):
        if case.case_id in existing_rows:
            print(f"[{index}/{len(selected_cases)}] Reusing {case.case_id}")  # noqa: T201
            rows.append(existing_rows[case.case_id])
            continue
        print(f"[{index}/{len(selected_cases)}] {case.case_id}: {case.question}")  # noqa: T201
        row, artifacts = evaluate_qasper_case(
            case,
            backend=backend,
            embedding_model_id=embedding_model_id,
            embedding_device=embedding_device,
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
    manifest = {
        "created_at": datetime.now(tz=UTC).isoformat(),
        "dataset": selection["dataset"],
        "split": selection["split"],
        "source_revision": selection["source_revision"],
        "selection_manifest": str(DEFAULT_SELECTION_PATH),
        "case_count": len(rows),
        "retrieval_method": "Dense embeddings",
        "embedding_model": embedding_model_id,
        "embedding_device_requested": embedding_device,
        "embedding_device_resolved": resolved_embedding_device(
            embedding_model_id,
            embedding_device,
        ),
        "top_k": 6,
        "value_function": "contrastive_likelihood",
        "model_path": model_path,
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "coalition_budget": 64,
        "random_seed": 42,
        "f1_threshold": f1_threshold,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    _write_report(summary, output_dir)
    return summary


def main() -> int:
    """Prepare the frozen subset or run its external evaluation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-gpu-layers", type=int, default=20)
    parser.add_argument("--embedding-model", default=str(DEFAULT_EMBEDDING_PATH))
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument("--f1-threshold", type=float, default=0.65)
    args = parser.parse_args()

    if args.prepare_only or not DEFAULT_SELECTION_PATH.is_file():
        cache = download_qasper_validation(DEFAULT_CACHE_PATH, force=args.force_download)
        cases = select_qasper_cases(cache)
        write_selection_manifest(cache, cases)
        print(f"Wrote {len(cases)} frozen cases to {DEFAULT_SELECTION_PATH}")  # noqa: T201
        if args.prepare_only:
            return 0

    backend = LlamaCppBackend(
        model_path=str(args.model_path),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
    )
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    suffix = f"qasper_{args.limit}" if args.limit is not None else "qasper_30"
    output_dir = args.output_dir or RUNS_DIR / f"{stamp}_{suffix}"
    summary = run_qasper(
        output_dir=output_dir,
        backend=backend,
        model_path=str(args.model_path),
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        embedding_model_id=args.embedding_model,
        embedding_device=args.embedding_device,
        limit=args.limit,
        resume_from=args.resume_from,
        f1_threshold=args.f1_threshold,
    )
    print(f"Wrote {len(summary)} QASPER cases to {output_dir}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
