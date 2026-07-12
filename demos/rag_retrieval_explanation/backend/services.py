"""Framework-independent orchestration for RAG attribution runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pandas as pd
from demos.rag_retrieval_explanation.core.chunking import chunk_pdf_pages, extract_pdf_pages
from demos.rag_retrieval_explanation.core.evaluation import (
    deletion_evaluation,
    insertion_evaluation,
    random_deletion_baseline,
    random_insertion_baseline,
    retrieval_score_deletion_baseline,
    summarize_faithfulness,
)
from demos.rag_retrieval_explanation.core.generation import generate_answer_from_chunks
from demos.rag_retrieval_explanation.core.rag_game import RAGRetrievalGame
from demos.rag_retrieval_explanation.core.retrieval import retrieve_relevant_chunks_with_debug
from demos.rag_retrieval_explanation.core.sample_data import SAMPLE_TRACES
from demos.rag_retrieval_explanation.core.schemas import RetrievedChunk
from demos.rag_retrieval_explanation.core.value_functions import make_value_function

import shapiq

from .schemas import RuntimeSettings

DEMO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GGUF_PATH = str(DEMO_DIR.parents[1] / "models" / "llm" / "gemma-4-E4B-it-Q4_K_M.gguf")
BGE_BASE_MODEL_ID = "BAAI/bge-base-en-v1.5"


def _resolve_model_path(settings: RuntimeSettings) -> str:
    """Return model path: registry lookup takes precedence over raw model_path."""
    from .model_registry import MODEL_REGISTRY, get_model_path

    if settings.model_id and settings.model_id in MODEL_REGISTRY:
        return get_model_path(settings.model_id)
    return settings.model_path or DEFAULT_GGUF_PATH


VALUE_FUNCTIONS = {
    "Lexical grounding": {
        "description": "Answer-term coverage with a small question-overlap bonus.",
        "formula": "0.88 * answer coverage^1.35 + 0.12 * question overlap - noise penalty",
        "requires_model": False,
    },
    "Local target likelihood": {
        "description": "Likelihood of the target answer under the selected context.",
        "formula": "sigmoid(mean token log-likelihood / T)",
        "requires_model": True,
    },
    "Local contrastive likelihood": {
        "description": "Target likelihood gain over the model's no-context prior.",
        "formula": "sigmoid((LL(context) - LL(no context)) / T)",
        "requires_model": True,
    },
    "Local generated answer overlap": {
        "description": "Token overlap between a generated answer and the target answer.",
        "formula": "|generated tokens ∩ target tokens| / |target tokens|",
        "requires_model": True,
    },
}

ProgressCallback = Callable[[float, str], None]


def default_settings() -> RuntimeSettings:
    """Return dashboard defaults for local inference."""
    return RuntimeSettings(
        model_path=DEFAULT_GGUF_PATH,
        model_id="gemma4-e4b",
        n_gpu_layers=20,
    )


def trace_chunks(trace: dict[str, Any]) -> list[RetrievedChunk]:
    """Convert serialized trace chunks into domain objects."""
    return [
        RetrievedChunk(
            title=str(chunk["title"]),
            text=str(chunk["text"]),
            page_number=chunk.get("page_number"),
            chunk_type=str(chunk.get("chunk_type", "body_text")),
            section_title=str(chunk.get("section_title", "")),
            text_length=int(chunk.get("text_length", 0) or 0),
            retrieval_score=chunk.get("rerank_score"),
            dense_score=chunk.get("dense_score"),
            keyword_score=chunk.get("keyword_score"),
            rerank_score=chunk.get("rerank_score"),
            flags=tuple(chunk.get("flags", [])),
        )
        for chunk in trace["chunks"]
    ]


def make_approximator(index: str, n_players: int, max_order: int) -> Any:  # noqa: ANN401
    """Construct the configured shapiq approximation strategy."""
    if index == "SV":
        return shapiq.KernelSHAP(n=n_players, random_state=42)
    if index == "STII":
        return shapiq.PermutationSamplingSTII(n=n_players, max_order=max_order, random_state=42)
    if index == "FSII":
        return shapiq.RegressionFSII(n=n_players, max_order=max_order, random_state=42)
    kernel_index = cast("Literal['k-SII', 'SII']", index)
    return shapiq.KernelSHAPIQ(
        n=n_players,
        index=kernel_index,
        max_order=max_order,
        random_state=42,
    )


def first_order_rows(values: shapiq.InteractionValues, labels: list[str]) -> list[dict]:
    """Serialize first-order values in descending magnitude."""
    rows = [
        {
            "player": interaction[0] + 1,
            "chunk": labels[interaction[0]],
            "attribution": float(score),
        }
        for interaction, score in values.dict_values.items()
        if len(interaction) == 1
    ]
    return sorted(rows, key=lambda row: abs(row["attribution"]), reverse=True)


def strongest_pair(matrix: np.ndarray, labels: list[str]) -> tuple[str, float]:
    """Return the strongest off-diagonal pairwise interaction."""
    if len(labels) < 2:
        return "No pair", 0.0
    pairs = (
        (abs(float(matrix[i, j])), i, j)
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    )
    _, left, right = max(pairs)
    return f"{left + 1}. {labels[left]} + {right + 1}. {labels[right]}", float(matrix[left, right])


def _curve_rows(frame: pd.DataFrame, label_column: str) -> list[dict]:
    return [
        {
            "step": int(row["step"]),
            "score": float(row["score"]),
            "label": str(row[label_column]),
        }
        for row in frame.to_dict(orient="records")
    ]


def _run_one(
    *,
    value_mode: str,
    question: str,
    answer: str,
    chunks: list[RetrievedChunk],
    retrieval_scores: list[float],
    settings: RuntimeSettings,
) -> dict:
    metadata = VALUE_FUNCTIONS[value_mode]
    scorer = make_value_function(
        value_mode,
        model_path=_resolve_model_path(settings),
        n_ctx=settings.n_ctx,
        n_gpu_layers=settings.n_gpu_layers,
        n_threads=settings.n_threads,
        max_new_tokens=settings.max_new_tokens,
    )
    scorer.warmup()
    game = RAGRetrievalGame(question=question, target_answer=answer, chunks=chunks, scorer=scorer)
    full_score = float(game(game.grand_coalition)[0])
    empty_score = float(game(game.empty_coalition)[0])
    max_order = 1 if settings.interaction_index == "SV" else settings.max_order
    explanation = make_approximator(
        settings.interaction_index, game.n_players, max_order
    ).approximate(budget=settings.budget, game=game)
    rows = first_order_rows(explanation.get_n_order(order=1), [c.title for c in chunks])
    frame = pd.DataFrame(rows)
    matrix = (
        explanation.get_n_order_values(2)
        if explanation.max_order >= 2
        else np.zeros((game.n_players, game.n_players))
    )
    pair_label, pair_value = strongest_pair(matrix, [c.title for c in chunks])
    faithfulness = summarize_faithfulness(game, frame)
    deletion = deletion_evaluation(game, frame)
    insertion = insertion_evaluation(game, frame)
    loo_batch = np.ones((game.n_players, game.n_players), dtype=bool)
    for index in range(game.n_players):
        loo_batch[index, index] = False
    loo = [full_score - float(score) for score in game(loo_batch)]
    for row in rows:
        row["loo"] = loo[int(row["player"]) - 1]

    retrieval_baseline = retrieval_score_deletion_baseline(game, retrieval_scores)
    return {
        "value_name": getattr(scorer, "name", value_mode),
        "value_description": metadata["description"],
        "value_formula": metadata["formula"],
        "full_score": full_score,
        "empty_score": empty_score,
        "top_chunk": f"{rows[0]['player']}. {rows[0]['chunk']}" if rows else "No chunk",
        "top_score": rows[0]["attribution"] if rows else 0.0,
        "attributions": rows,
        "pairwise_matrix": matrix.tolist(),
        "strongest_pair": pair_label,
        "strongest_pair_value": pair_value,
        "faithfulness": asdict(faithfulness),
        "deletion_curve": _curve_rows(deletion, "removed_chunk"),
        "insertion_curve": _curve_rows(insertion, "added_chunk"),
        "random_deletion_baseline": random_deletion_baseline(game).tolist(),
        "random_insertion_baseline": random_insertion_baseline(game).tolist(),
        "retrieval_deletion_baseline": (
            retrieval_baseline.tolist() if retrieval_baseline is not None else None
        ),
    }


def run_trace(
    run_name: str,
    trace: dict[str, Any],
    settings: RuntimeSettings,
    progress: ProgressCallback | None = None,
) -> dict:
    """Run every configured explanation target for a prepared trace."""
    chunks = trace_chunks(trace)
    if not chunks:
        msg = "No retrieved chunks are available."
        raise ValueError(msg)
    invalid = [mode for mode in settings.value_modes if mode not in VALUE_FUNCTIONS]
    if invalid:
        msg = f"Unknown value functions: {', '.join(invalid)}"
        raise ValueError(msg)
    reference_answer = str(trace.get("reference_answer", "")).strip()
    generated_answer = str(trace.get("generated_answer", "")).strip()
    legacy_answer = str(trace.get("target_answer", "")).strip()
    legacy_type = str(trace.get("answer_target_type", "reference_answer"))
    if legacy_answer and not reference_answer and not generated_answer:
        if legacy_type == "generated_answer":
            generated_answer = legacy_answer
        else:
            reference_answer = legacy_answer

    target_specs = []
    if reference_answer:
        target_specs.append(("reference_answer", "Gold / reference answer", reference_answer))
    if generated_answer:
        target_specs.append(("generated_answer", "Generated answer", generated_answer))
    if not target_specs:
        msg = "At least one reference or generated answer is required."
        raise ValueError(msg)

    callback = progress or (lambda _value, _message: None)
    retrieval_scores = [float(x) for x in trace.get("retrieval_scores", [])]
    targets = []
    total_jobs = len(target_specs) * len(settings.value_modes)
    completed_jobs = 0
    for target_type, label, answer in target_specs:
        target_results = []
        for mode in settings.value_modes:
            callback(
                completed_jobs / total_jobs,
                f"Running {label}: {mode}",
            )
            target_results.append(
                _run_one(
                    value_mode=mode,
                    question=str(trace["question"]),
                    answer=answer,
                    chunks=chunks,
                    retrieval_scores=retrieval_scores,
                    settings=settings,
                )
            )
            completed_jobs += 1
        targets.append(
            {
                "target_type": target_type,
                "label": label,
                "answer": answer,
                "results": target_results,
            }
        )
    callback(1.0, "Run complete")
    primary_target = next(
        (target for target in targets if target["target_type"] == "generated_answer"),
        targets[0],
    )
    return {
        "run_name": run_name,
        "source": str(trace.get("source", "sample")),
        "answer_target_type": primary_target["target_type"],
        "retrieval_method": str(trace.get("retrieval_method", "Preselected controlled chunks")),
        "retrieval_top_k": len(chunks),
        "question": str(trace["question"]),
        "answer": primary_target["answer"],
        "takeaway": str(trace.get("takeaway", "")),
        "chunks": [asdict(chunk) | {"flags": list(chunk.flags)} for chunk in chunks],
        "retrieval_debug": trace.get("retrieval_debug"),
        "results": primary_target["results"],
        "targets": targets,
    }


def run_comparison(
    *,
    source: str,
    scenario_name: str | None = None,
    pdf_filename: str | None = None,
    pdf_bytes: bytes | None = None,
    question: str | None = None,
    reference_answer: str = "",
    base_settings: RuntimeSettings,
    model_ids: list[str],
    retrieval_method: str = "TF-IDF",
    retrieval_top_k: int = 4,
    chunk_words: int = 180,
    chunk_overlap: int = 40,
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_device: str = "auto",
) -> dict:
    """Run the same question through multiple models, sharing retrieval across them.

    Retrieval is model-independent and performed once. Each model generates its
    own answer, and attribution is computed separately per model+answer pair.
    Missing or unavailable models are reported with status="missing_model" without
    aborting the run for other models.
    """
    from .model_registry import MODEL_REGISTRY, get_model_status

    # --- Phase 1: shared retrieval ---
    if source == "sample":
        if scenario_name not in SAMPLE_TRACES:
            msg = f"Unknown sample scenario: {scenario_name}"
            raise ValueError(msg)
        trace = SAMPLE_TRACES[scenario_name]
        actual_question = str(trace["question"])
        chunks = trace_chunks(trace)
        raw_scores = cast("list[Any]", trace.get("retrieval_scores", []))
        retrieval_scores = [float(x) for x in raw_scores]
        retrieval_debug = trace.get("retrieval_debug")
        base_run_name = scenario_name
        chunks_data: list[dict] = [asdict(c) | {"flags": list(c.flags)} for c in chunks]
    else:
        if pdf_bytes is None or question is None:
            msg = "PDF comparison requires pdf_bytes and question."
            raise ValueError(msg)
        pages = extract_pdf_pages(pdf_bytes)
        candidates = chunk_pdf_pages(
            pages, words_per_chunk=chunk_words, overlap_words=chunk_overlap
        )
        ranked, debug = retrieve_relevant_chunks_with_debug(
            question,
            candidates,
            top_k=retrieval_top_k,
            method=retrieval_method,
            embedding_model_id=embedding_model_id,
            embedding_device=embedding_device,
        )
        actual_question = question
        chunks = [item.chunk for item in ranked]
        retrieval_scores = [item.score for item in ranked]
        retrieval_debug = asdict(debug)
        base_run_name = f"PDF: {pdf_filename}"
        chunks_data = [
            asdict(item.chunk)
            | {
                "flags": list(item.chunk.flags),
                "dense_score": item.dense_score,
                "keyword_score": item.keyword_score,
                "rerank_score": item.rerank_score,
            }
            for item in ranked
        ]

    # --- Phase 2: per-model generation + attribution ---
    per_model_results: list[dict] = []

    for model_id in model_ids:
        model_info = MODEL_REGISTRY.get(model_id)
        display_name = model_info["display_name"] if model_info else model_id

        if model_id not in MODEL_REGISTRY:
            per_model_results.append(
                {
                    "model_id": model_id,
                    "model_display_name": model_id,
                    "status": "failed",
                    "error": f"Unknown model ID: {model_id}",
                    "expected_path": None,
                    "result": None,
                }
            )
            continue

        status_info = get_model_status(model_id)
        if not status_info["available"]:
            per_model_results.append(
                {
                    "model_id": model_id,
                    "model_display_name": display_name,
                    "status": "missing_model",
                    "error": "Model file not found locally.",
                    "expected_path": status_info["path"],
                    "result": None,
                }
            )
            continue

        try:
            model_settings = RuntimeSettings(
                **{
                    k: v
                    for k, v in base_settings.model_dump().items()
                    if k not in ("model_id", "model_path")
                },
                model_id=model_id,
                model_path=str(status_info["path"]),
            )
            resolved_path = _resolve_model_path(model_settings)

            answer = generate_answer_from_chunks(
                question=actual_question,
                chunks=chunks,
                model_path=resolved_path,
                n_ctx=model_settings.n_ctx,
                n_gpu_layers=model_settings.n_gpu_layers,
                n_threads=model_settings.n_threads,
                max_new_tokens=model_settings.max_new_tokens,
            )

            model_trace: dict[str, Any] = {
                "source": source,
                "question": actual_question,
                "reference_answer": (
                    str(trace["target_answer"]) if source == "sample" else reference_answer.strip()
                ),
                "generated_answer": answer,
                "takeaway": f"Generated by {display_name}.",
                "retrieval_method": (
                    "Preselected controlled chunks" if source == "sample" else retrieval_method
                ),
                "retrieval_scores": retrieval_scores,
                "retrieval_debug": retrieval_debug,
                "chunks": chunks_data,
            }

            run_result = run_trace(
                f"{display_name} / {base_run_name}",
                model_trace,
                model_settings,
            )

            per_model_results.append(
                {
                    "model_id": model_id,
                    "model_display_name": display_name,
                    "status": "completed",
                    "error": None,
                    "expected_path": None,
                    "result": run_result,
                }
            )

        except Exception as exc:  # noqa: BLE001 - isolate failures between compared models
            per_model_results.append(
                {
                    "model_id": model_id,
                    "model_display_name": display_name,
                    "status": "failed",
                    "error": str(exc),
                    "expected_path": None,
                    "result": None,
                }
            )

    return {
        "question": actual_question,
        "source": source,
        "results": per_model_results,
    }


def run_sample(
    scenario_name: str,
    settings: RuntimeSettings,
    progress: ProgressCallback | None = None,
) -> dict:
    """Generate and explain an answer for one bundled scenario."""
    if scenario_name not in SAMPLE_TRACES:
        msg = f"Unknown sample scenario: {scenario_name}"
        raise ValueError(msg)
    trace = SAMPLE_TRACES[scenario_name]
    chunks = trace_chunks(trace)
    generated_answer = generate_answer_from_chunks(
        question=str(trace["question"]),
        chunks=chunks,
        model_path=_resolve_model_path(settings),
        n_ctx=settings.n_ctx,
        n_gpu_layers=settings.n_gpu_layers,
        n_threads=settings.n_threads,
        max_new_tokens=settings.max_new_tokens,
    )
    dual_trace = {
        **trace,
        "reference_answer": str(trace["target_answer"]),
        "generated_answer": generated_answer,
    }
    return run_trace(scenario_name, dual_trace, settings, progress)


def run_pdf(
    *,
    filename: str,
    pdf_bytes: bytes,
    question: str,
    reference_answer: str = "",
    settings: RuntimeSettings,
    retrieval_method: str = "TF-IDF",
    retrieval_top_k: int = 4,
    chunk_words: int = 180,
    chunk_overlap: int = 40,
    embedding_model_id: str = "sentence-transformers/all-MiniLM-L6-v2",
    embedding_device: str = "auto",
    progress: ProgressCallback | None = None,
) -> dict:
    """Retrieve, generate, and explain an answer for one PDF."""
    callback = progress or (lambda _value, _message: None)
    callback(0.05, "Parsing PDF")
    pages = extract_pdf_pages(pdf_bytes)
    callback(0.15, "Chunking pages")
    candidates = chunk_pdf_pages(pages, words_per_chunk=chunk_words, overlap_words=chunk_overlap)
    callback(0.25, "Retrieving evidence")
    ranked, debug = retrieve_relevant_chunks_with_debug(
        question,
        candidates,
        top_k=retrieval_top_k,
        method=retrieval_method,
        embedding_model_id=embedding_model_id,
        embedding_device=embedding_device,
    )
    chunks = [item.chunk for item in ranked]
    callback(0.4, "Generating grounded answer")
    answer = generate_answer_from_chunks(
        question=question,
        chunks=chunks,
        model_path=_resolve_model_path(settings),
        n_ctx=settings.n_ctx,
        n_gpu_layers=settings.n_gpu_layers,
        n_threads=settings.n_threads,
        max_new_tokens=settings.max_new_tokens,
    )
    trace: dict[str, Any] = {
        "source": "pdf",
        "question": question,
        "reference_answer": reference_answer.strip(),
        "generated_answer": answer,
        "takeaway": f"Answer and evidence retrieved from {filename}.",
        "retrieval_method": retrieval_method,
        "retrieval_scores": [item.score for item in ranked],
        "retrieval_debug": asdict(debug),
        "chunks": [
            asdict(item.chunk)
            | {
                "flags": list(item.chunk.flags),
                "dense_score": item.dense_score,
                "keyword_score": item.keyword_score,
                "rerank_score": item.rerank_score,
            }
            for item in ranked
        ],
    }
    return run_trace(f"Uploaded PDF: {filename}", trace, settings, progress)
