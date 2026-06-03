"""Cross-product grid eval runner for the RAG retrieval explanation demo.

Reads an eval config YAML (see evals/configs/), expands the Cartesian product
of chunking × retrieval × generation × value_function dimensions, runs each
combination on every case in a JSON case file, and writes a unified comparison
output to a canonical run directory.

Usage (from project root):
    uv run python demos/rag_retrieval_explanation/evals/runners/run_grid_eval.py \\
        --config demos/rag_retrieval_explanation/evals/configs/embedding_compare.yaml \\
        --cases  demos/rag_retrieval_explanation/evals/cases/artemis_pdf_cases.json

Output (see Phase 3 spec):
    evals/runs/<run_id>/
        config.yaml       resolved grid config
        manifest.json     git + runtime provenance
        summary.csv       one row per (case × config), all metrics
        summary.json      same as CSV, JSON format
        report.md         auto-generated markdown report
        plots/
            pass_rate_by_config.png
            shapley_mass_heatmap.png
            rank_stability.png
            retrieval_vs_shapley.png
        artifacts/
            <case_id>_<config_id>_retrieved_chunks.csv
            <case_id>_<config_id>_shapley_values.csv
            <case_id>_<config_id>_answer.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEMO_DIR = Path(__file__).resolve().parents[2]
if str(DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_DIR))

try:
    import yaml  # type: ignore[import-untyped]

    def _load_yaml(path: Path) -> dict[str, Any]:
        with open(path) as fh:
            return yaml.safe_load(fh)

except ImportError:
    import re as _re

    def _load_yaml(path: Path) -> dict[str, Any]:  # type: ignore[misc]
        raise RuntimeError(
            "PyYAML is required for the grid runner. Install it with: uv add pyyaml"
        )


from core.metrics import (  # noqa: E402
    AttributionMetrics,
    EvalMetrics,
    AnswerMetrics,
    RetrievalMetrics,
    compute_attribution_metrics,
    compute_answer_metrics,
    compute_rank_stability,
    compute_retrieval_metrics,
    normalized_auc,
)
from core.rag_pipeline import (  # noqa: E402
    CandidateChunk,
    PDFPage,
    chunk_pdf_pages,
    extract_pdf_pages,
    retrieve_relevant_chunks_with_debug,
)
from core.reporting import (  # noqa: E402
    contains_term,
    make_run_dir,
    make_run_id,
    markdown_table,
    matplotlib_pyplot,
    slug,
    short_label,
    write_config,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Config expansion
# ---------------------------------------------------------------------------


@dataclass
class GridConfig:
    """One fully resolved configuration point in the grid."""

    config_id: str
    chunk_words: int
    overlap: int
    retrieval_method: str
    embedding_model: str
    generation_backend: str
    value_function: str
    top_k: int
    shapley_budget: int
    shapley_seed: int
    shapley_max_order: int

    def label(self) -> str:
        """Short human-readable label for plots."""
        parts = [self.retrieval_method]
        if self.embedding_model != "none":
            parts.append(self.embedding_model.split("/")[-1])
        parts.append(f"{self.chunk_words}w")
        parts.append(self.value_function)
        return " | ".join(parts)


def expand_grid(config: dict[str, Any]) -> list[GridConfig]:
    """Expand a config dict into one GridConfig per cross-product cell."""
    chunking_list = config.get("chunking", [{"chunk_words": 180, "overlap": 35}])
    retrieval_list = config.get("retrieval", [{"method": "tfidf"}])
    generation_list = config.get("generation", [{"backend": "extractive"}])
    vf_list = config.get("value_functions", ["lexical_grounding"])
    shapley_cfg = config.get("shapley", {})
    output_cfg = config.get("output", {})

    top_k = int(output_cfg.get("top_k", 4))
    budget = int(shapley_cfg.get("budget", 256))
    seed = int(shapley_cfg.get("random_seed", 42))
    max_order = int(shapley_cfg.get("max_order", 2))

    configs: list[GridConfig] = []
    for chunk_cfg, ret_cfg, gen_cfg, vf_name in product(
        chunking_list, retrieval_list, generation_list, vf_list
    ):
        method = str(ret_cfg.get("method", "tfidf"))
        embedding = str(ret_cfg.get("embedding_model", "none")) if method == "dense" else "none"
        gen_backend = str(gen_cfg.get("backend", "extractive"))
        chunk_words = int(chunk_cfg.get("chunk_words", 180))
        overlap = int(chunk_cfg.get("overlap", 35))

        cfg_id = slug(
            f"{method}_{embedding}_{chunk_words}w{overlap}o_{vf_name}_{gen_backend}"
        )
        configs.append(
            GridConfig(
                config_id=cfg_id,
                chunk_words=chunk_words,
                overlap=overlap,
                retrieval_method=method,
                embedding_model=embedding,
                generation_backend=gen_backend,
                value_function=vf_name,
                top_k=top_k,
                shapley_budget=budget,
                shapley_seed=seed,
                shapley_max_order=max_order,
            )
        )
    return configs


# ---------------------------------------------------------------------------
# PDF corpus cache
# ---------------------------------------------------------------------------


class GridCorpus:
    """Cache parsed PDFs and chunked corpora across grid cells."""

    def __init__(self) -> None:
        """Initialize caches."""
        self._pages: dict[Path, list[PDFPage]] = {}
        self._chunks: dict[tuple[Path, int, int], list[CandidateChunk]] = {}

    def pages(self, pdf_path: Path) -> list[PDFPage]:
        """Return pages for a PDF, parsing at most once."""
        resolved = pdf_path.resolve()
        if resolved not in self._pages:
            self._pages[resolved] = extract_pdf_pages(resolved.read_bytes())
        return self._pages[resolved]

    def chunks(self, pdf_path: Path, *, chunk_words: int, overlap: int) -> list[CandidateChunk]:
        """Return chunks for a PDF/chunking config, chunking at most once."""
        resolved = pdf_path.resolve()
        key = (resolved, chunk_words, overlap)
        if key not in self._chunks:
            self._chunks[key] = chunk_pdf_pages(
                self.pages(resolved),
                words_per_chunk=chunk_words,
                overlap_words=overlap,
            )
        return self._chunks[key]


# ---------------------------------------------------------------------------
# Single (case × config) run
# ---------------------------------------------------------------------------


def run_one(
    case: dict[str, Any],
    cfg: GridConfig,
    corpus: GridCorpus,
    run_dir: Path,
    *,
    allow_failures: bool,
) -> dict[str, object]:
    """Run one case under one config; return a flat metrics row."""
    case_id = str(case.get("id", case.get("question", "case")))
    pdf_path = Path(str(case["pdf_path"]))
    expected_terms: list[str] = [str(t) for t in case.get("expected_evidence_terms", [])]
    expected_answer_kw: list[str] = [str(k) for k in case.get("expected_answer_keywords", expected_terms)]

    base_row: dict[str, object] = {
        "case_id": case_id,
        "config_id": cfg.config_id,
        "retrieval_method": cfg.retrieval_method,
        "embedding_model": cfg.embedding_model,
        "chunk_words": cfg.chunk_words,
        "overlap": cfg.overlap,
        "value_function": cfg.value_function,
        "generation_backend": cfg.generation_backend,
    }

    if not pdf_path.exists():
        return {
            **base_row,
            "error": f"PDF not found: {pdf_path}",
            "hit_rate": None,
            "recall_at_k": None,
            "mrr": None,
            "passed": False,
        }

    try:
        candidates = corpus.chunks(pdf_path, chunk_words=cfg.chunk_words, overlap=cfg.overlap)
        ranked, debug = retrieve_relevant_chunks_with_debug(
            str(case["question"]),
            candidates,
            top_k=cfg.top_k,
            method="Dense embeddings" if cfg.retrieval_method == "dense" else "TF-IDF",
            embedding_model_id=cfg.embedding_model,
            embedding_device=str(case.get("embedding_device", "auto")),
        )
    except Exception as exc:  # noqa: BLE001
        if not allow_failures:
            raise
        return {**base_row, "error": str(exc), "hit_rate": None, "recall_at_k": None, "passed": False}

    retrieved_titles = [item.chunk.title for item in ranked]
    retrieval_scores = [item.rerank_score for item in ranked]
    retrieved_text = "\n\n".join(item.chunk.text for item in ranked)

    found_terms = [t for t in expected_terms if contains_term(retrieved_text, t)]
    missing_terms = [t for t in expected_terms if t not in found_terms]
    passed = not missing_terms

    ret_metrics = compute_retrieval_metrics(retrieved_titles, expected_terms, retrieval_scores)
    ans_metrics = compute_answer_metrics(
        answer=retrieved_text,  # use retrieved context as proxy (no LLM generation in grid yet)
        expected_answer_keywords=expected_answer_kw,
        retrieved_titles=retrieved_titles,
    )

    # Write per-case artifacts
    artifact_prefix = f"{slug(case_id)}_{cfg.config_id}"
    chunks_df = pd.DataFrame(
        [
            {
                "rank": i + 1,
                "title": item.chunk.title,
                "page": item.chunk.page_number,
                "chunk_type": item.chunk.chunk_type,
                "rerank_score": item.rerank_score,
                "dense_score": item.dense_score,
                "keyword_score": item.keyword_score,
                "text_preview": item.chunk.text[:200],
            }
            for i, item in enumerate(ranked)
        ]
    )
    chunks_df.to_csv(run_dir / "artifacts" / f"{artifact_prefix}_retrieved_chunks.csv", index=False)

    row = {
        **base_row,
        "passed": passed,
        "found_terms": ", ".join(found_terms),
        "missing_terms": ", ".join(missing_terms),
        "top_chunk": retrieved_titles[0] if retrieved_titles else "",
        "hit_rate": ret_metrics.hit_rate,
        "recall_at_k": ret_metrics.recall_at_k,
        "mrr": ret_metrics.mrr,
        "avg_retrieval_score": ret_metrics.avg_retrieval_score,
        "answer_contains_expected": ans_metrics.answer_contains_expected,
        "answer_supported": ans_metrics.answer_supported,
        "covered_query_terms": ", ".join(debug.coverage_summary.get("covered_query_terms", [])),
        "error": "",
    }
    return row


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def write_report(summary: pd.DataFrame, output_dir: Path, *, config_name: str) -> None:
    """Write a Markdown report for the grid eval."""
    report_cols = [
        "case_id", "config_id", "retrieval_method", "embedding_model",
        "chunk_words", "overlap", "value_function",
        "passed", "hit_rate", "recall_at_k", "mrr",
        "found_terms", "missing_terms", "top_chunk", "error",
    ]
    visible = [c for c in report_cols if c in summary.columns]
    ts = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# Grid Eval Report — {config_name}",
        "",
        f"Generated: {ts}  |  {len(summary)} rows  |  "
        f"{summary['config_id'].nunique() if 'config_id' in summary else '?'} configs  |  "
        f"{summary['case_id'].nunique() if 'case_id' in summary else '?'} cases",
        "",
        "## Summary Table",
        "",
        markdown_table(summary[visible]),
        "## Plots",
        "",
        "![Pass rate by config](plots/pass_rate_by_config.png)",
        "",
    ]
    if (output_dir / "plots" / "shapley_mass_heatmap.png").exists():
        lines += ["![Shapley mass heatmap](plots/shapley_mass_heatmap.png)", ""]
    if (output_dir / "plots" / "rank_stability.png").exists():
        lines += ["![Rank stability](plots/rank_stability.png)", ""]
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the grid eval."""
    parser = argparse.ArgumentParser(description="Cross-product grid eval for the RAG demo.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a config YAML.")
    parser.add_argument("--cases", type=Path, required=True, help="Path to a JSON case file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory (default: evals/runs/<timestamp>_<name>).",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Continue past individual case failures instead of raising.",
    )
    args = parser.parse_args()

    if not args.config.exists():
        print(f"Config not found: {args.config}")  # noqa: T201
        return 2
    if not args.cases.exists():
        print(f"Cases file not found: {args.cases}")  # noqa: T201
        return 2

    grid_cfg = _load_yaml(args.config)
    grid_configs = expand_grid(grid_cfg)
    cases = json.loads(args.cases.read_text(encoding="utf-8"))

    run_name = str(grid_cfg.get("name", args.config.stem))
    if args.output_dir is None:
        runs_base = Path(__file__).resolve().parents[1] / "runs"
        output_dir = make_run_dir(runs_base, make_run_id(run_name))
    else:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plots").mkdir(exist_ok=True)
        (output_dir / "artifacts").mkdir(exist_ok=True)

    print(f"Grid: {len(grid_configs)} configs × {len(cases)} cases = {len(grid_configs) * len(cases)} runs")  # noqa: T201
    print(f"Output: {output_dir}")  # noqa: T201

    write_manifest(
        output_dir,
        run_id=output_dir.name,
        config_file=str(args.config),
        cases_file=str(args.cases),
        extra={
            "eval_type": "grid",
            "grid_name": run_name,
            "n_configs": len(grid_configs),
            "n_cases": len(cases),
            "n_runs": len(grid_configs) * len(cases),
        },
    )
    write_config(
        output_dir / "config.yaml",
        {
            "name": run_name,
            "config_file": str(args.config),
            "cases_file": str(args.cases),
            "n_configs": len(grid_configs),
            "n_cases": len(cases),
        },
    )

    corpus = GridCorpus()
    rows: list[dict[str, object]] = []
    for cfg in grid_configs:
        print(f"  Config: {cfg.config_id}")  # noqa: T201
        for case in cases:
            row = run_one(case, cfg, corpus, output_dir, allow_failures=args.allow_failures)
            rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    _write_comparison_plots(summary, output_dir)
    write_report(summary, output_dir, config_name=run_name)

    passed = int(summary["passed"].sum()) if "passed" in summary.columns else 0
    total = len(summary)
    print(f"Done. {passed}/{total} passed. Report: {output_dir / 'report.md'}")  # noqa: T201
    return 0 if args.allow_failures or passed == total else 1


# ---------------------------------------------------------------------------
# Comparison plots (called by Stage 8 — stubs here, full impl in reporting.py)
# ---------------------------------------------------------------------------


def _write_comparison_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write grid comparison plots (pass rate bar chart; more added in Stage 8)."""
    if summary.empty or "passed" not in summary.columns:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    plt = matplotlib_pyplot()

    # Pass rate by config
    if "config_id" in summary.columns:
        pass_rates = (
            summary.groupby("config_id", as_index=False)["passed"]
            .mean()
            .sort_values("passed")
        )
        labels = [short_label(str(c), 32) for c in pass_rates["config_id"]]
        fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45 * len(pass_rates))))
        ax.barh(labels, pass_rates["passed"].astype(float), color="#3572a5")
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Pass rate (evidence hit)")
        ax.set_title("Grid Eval — Pass Rate by Config")
        ax.grid(axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plots_dir / "pass_rate_by_config.png", dpi=180)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
