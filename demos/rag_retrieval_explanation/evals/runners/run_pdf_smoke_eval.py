"""Run generic PDF retrieval smoke checks and write table/plot reports.

Usage (from project root):
    uv run python demos/rag_retrieval_explanation/evals/runners/run_pdf_smoke_eval.py \
        demos/rag_retrieval_explanation/evals/cases/pdf_cases.example.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import pandas as pd

DEMO_DIR = Path(__file__).resolve().parents[2]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

from core.reporting import (  # noqa: E402
    contains_term,
    markdown_table,
    matplotlib_pyplot,
)
from core.rag_pipeline import (  # noqa: E402
    CandidateChunk,
    PDFPage,
    chunk_pdf_pages,
    extract_pdf_pages,
    retrieve_relevant_chunks_with_debug,
)


def _parse_csv(value: str | None, default: list[str]) -> list[str]:
    """Parse a comma-separated CLI value."""
    if value is None:
        return default
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_chunk_grid(value: str | None) -> list[tuple[int, int]]:
    """Parse chunk grid entries formatted as WORDS:OVERLAP."""
    if not value:
        return [(180, 35)]
    grid = []
    for item in value.split(","):
        words, overlap = item.split(":", maxsplit=1)
        grid.append((int(words), int(overlap)))
    return grid


def _case_configs(
    case: dict[str, Any],
    *,
    methods: list[str],
    embedding_models: list[str],
    chunk_grid: list[tuple[int, int]],
) -> list[dict[str, object]]:
    """Return retrieval configs for one case."""
    case_methods = case.get("methods") or case.get("method")
    selected_methods = (
        [str(case_methods)] if isinstance(case_methods, str) else list(case_methods or methods)
    )
    case_embedding_models = case.get("embedding_models") or case.get("embedding_model")
    selected_embedding_models = (
        [str(case_embedding_models)]
        if isinstance(case_embedding_models, str)
        else list(case_embedding_models or embedding_models)
    )
    case_chunk_grid = case.get("chunk_grid")
    selected_chunk_grid = (
        [(int(item["words"]), int(item["overlap"])) for item in case_chunk_grid]
        if case_chunk_grid
        else chunk_grid
    )

    configs = []
    for method, embedding_model, (words, overlap) in product(
        selected_methods,
        selected_embedding_models,
        selected_chunk_grid,
    ):
        effective_embedding_model = embedding_model if method == "Dense embeddings" else "none"
        configs.append(
            {
                "method": method,
                "embedding_model": effective_embedding_model,
                "words_per_chunk": words,
                "overlap_words": overlap,
                "top_k": int(case.get("top_k", 4)),
            }
        )
    return configs


class EvalCorpus:
    """Cache parsed PDFs and chunked candidate corpora across eval cases."""

    def __init__(self) -> None:
        """Initialize empty in-memory caches."""
        self._pages: dict[Path, list[PDFPage]] = {}
        self._chunks: dict[tuple[Path, int, int], list[CandidateChunk]] = {}

    def pages(self, pdf_path: Path) -> list[PDFPage]:
        """Return extracted pages for one PDF, parsing at most once."""
        resolved = pdf_path.resolve()
        if resolved not in self._pages:
            self._pages[resolved] = extract_pdf_pages(resolved.read_bytes())
        return self._pages[resolved]

    def chunks(
        self,
        pdf_path: Path,
        *,
        words_per_chunk: int,
        overlap_words: int,
    ) -> list[CandidateChunk]:
        """Return chunked candidates for one PDF/chunking configuration."""
        resolved = pdf_path.resolve()
        key = (resolved, words_per_chunk, overlap_words)
        if key not in self._chunks:
            self._chunks[key] = chunk_pdf_pages(
                self.pages(resolved),
                words_per_chunk=words_per_chunk,
                overlap_words=overlap_words,
            )
        return self._chunks[key]


def run_case_config(
    case: dict[str, Any],
    config: dict[str, object],
    corpus: EvalCorpus,
) -> dict[str, object]:
    """Run one PDF smoke case under one retrieval config."""
    case_id = str(case.get("id", case.get("question", "case")))
    pdf_path = Path(str(case["pdf_path"]))
    expected_terms = [str(term) for term in case.get("expected_evidence_terms", [])]
    if not pdf_path.exists():
        return {
            "case_id": case_id,
            **config,
            "passed": False,
            "found_terms": "",
            "missing_terms": ", ".join(expected_terms) if expected_terms else "PDF not found",
            "top_chunk": "",
            "top_page": None,
            "top_score": None,
            "error": f"PDF not found: {pdf_path}",
        }

    candidates = corpus.chunks(
        pdf_path,
        words_per_chunk=int(config["words_per_chunk"]),
        overlap_words=int(config["overlap_words"]),
    )
    ranked, debug = retrieve_relevant_chunks_with_debug(
        str(case["question"]),
        candidates,
        top_k=int(config["top_k"]),
        method=str(config["method"]),
        embedding_model_id=str(config["embedding_model"]),
        embedding_device=str(case.get("embedding_device", "auto")),
    )
    retrieved_text = "\n\n".join(item.chunk.text for item in ranked)
    found_terms = [term for term in expected_terms if contains_term(retrieved_text, term)]
    missing_terms = [term for term in expected_terms if term not in found_terms]
    top = ranked[0] if ranked else None
    return {
        "case_id": case_id,
        **config,
        "passed": not missing_terms,
        "found_terms": ", ".join(found_terms),
        "missing_terms": ", ".join(missing_terms),
        "top_chunk": top.chunk.title if top else "",
        "top_page": top.chunk.page_number if top else None,
        "top_score": top.rerank_score if top else None,
        "covered_query_terms": ", ".join(debug.coverage_summary.get("covered_query_terms", [])),
        "error": "",
    }


def write_summary_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write PDF smoke comparison plots."""
    if summary.empty:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plt = matplotlib_pyplot()

    config_frame = summary.copy()
    config_frame["config"] = config_frame.apply(
        lambda row: (
            f"{row['method']} | {row['words_per_chunk']}/{row['overlap_words']} | "
            f"{row['embedding_model']}"
        ),
        axis=1,
    )
    pass_rates = (
        config_frame.groupby("config", as_index=False)["passed"].mean().sort_values("passed")
    )
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.5 * len(pass_rates))))
    ax.barh(pass_rates["config"], pass_rates["passed"], color="#3572a5")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Pass rate")
    ax.set_title("PDF Smoke Eval Pass Rate by Retrieval Config")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "pass_rate_by_config.png", dpi=180)
    plt.close(fig)

    case_rates = summary.groupby("case_id", as_index=False)["passed"].mean().sort_values("passed")
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.5 * len(case_rates))))
    ax.barh(case_rates["case_id"], case_rates["passed"], color="#4c8f6a")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Pass rate across configs")
    ax.set_title("PDF Smoke Eval Pass Rate by Case")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "pass_rate_by_case.png", dpi=180)
    plt.close(fig)


def write_report(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write a Markdown report for PDF smoke evals."""
    report_columns = [
        "case_id",
        "passed",
        "method",
        "embedding_model",
        "words_per_chunk",
        "overlap_words",
        "top_k",
        "found_terms",
        "missing_terms",
        "top_page",
        "top_score",
        "top_chunk",
        "error",
    ]
    report_frame = summary[[column for column in report_columns if column in summary.columns]]
    lines = [
        "# PDF RAG Smoke Eval Report",
        "",
        "This report checks whether fixed PDF questions retrieve expected evidence terms.",
        "It is a lightweight correctness smoke test, not a public RAG benchmark.",
        "",
        "## Summary Table",
        "",
        markdown_table(report_frame),
        "## Plots",
        "",
        "![Pass rate by retrieval config](plots/pass_rate_by_config.png)",
        "",
        "![Pass rate by case](plots/pass_rate_by_case.png)",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run all configured PDF smoke cases."""
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("--methods", help="Comma-separated methods, e.g. TF-IDF,Dense embeddings")
    parser.add_argument(
        "--embedding-models",
        help="Comma-separated embedding models for dense retrieval comparisons.",
    )
    parser.add_argument(
        "--chunk-grid",
        help="Comma-separated WORDS:OVERLAP entries, e.g. 120:25,180:35,240:50",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "runs"
        / datetime.now(tz=UTC).strftime("%Y-%m-%d_%H%M%S_pdf_smoke"),
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Write report and exit 0 even when some retrieval configurations fail.",
    )
    args = parser.parse_args()

    if not args.cases.exists():
        print(f"Case file not found: {args.cases}")  # noqa: T201
        print(  # noqa: T201
            "Create one from the template, for example:\n"
            "  make pdf-cases-template PDF_CASES=local/pdf_cases.json\n"
            "Then edit local/pdf_cases.json and rerun:\n"
            "  make pdf-compare PDF_CASES=local/pdf_cases.json"
        )
        return 2

    cases = json.loads(args.cases.read_text(encoding="utf-8"))
    methods = _parse_csv(args.methods, ["TF-IDF"])
    embedding_models = _parse_csv(args.embedding_models, ["sentence-transformers/all-MiniLM-L6-v2"])
    chunk_grid = _parse_chunk_grid(args.chunk_grid)

    rows = []
    corpus = EvalCorpus()
    for case in cases:
        rows.extend(
            run_case_config(case, config, corpus)
            for config in _case_configs(
                case,
                methods=methods,
                embedding_models=embedding_models,
                chunk_grid=chunk_grid,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary_plots(summary, args.output_dir)
    write_report(summary, args.output_dir)

    print(f"Wrote PDF smoke eval run to {args.output_dir}")  # noqa: T201
    print(f"Open report: {args.output_dir / 'report.md'}")  # noqa: T201
    if args.allow_failures:
        return 0
    return 0 if bool(summary["passed"].all()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
