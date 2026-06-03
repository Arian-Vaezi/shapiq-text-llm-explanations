"""Shared reporting utilities for all eval runners.

Provides:
- Text/table formatting: markdown_table, slug, short_label, contains_term
- Matplotlib helpers: matplotlib_pyplot (non-interactive backend)
- Run directory management: make_run_dir, write_manifest
- Git helpers: get_git_commit
- Comparison plots: plot_pass_rate_by_config, plot_shapley_mass_heatmap,
    plot_rank_stability, plot_retrieval_vs_shapley_scatter,
    plot_deletion_insertion_curves
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------


def slug(text: str) -> str:
    """Return a filesystem-friendly version of text."""
    s = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in s.split("_") if part)


def short_label(text: str, limit: int = 22) -> str:
    """Truncate text for plot axis labels."""
    return text if len(text) <= limit else text[: limit - 1] + "..."


def contains_term(text: str, term: str) -> bool:
    """Return True when text contains a case-insensitive literal term."""
    return re.search(re.escape(term), text, re.IGNORECASE) is not None


def markdown_table(frame: pd.DataFrame, *, float_digits: int = 3) -> str:
    """Render a small DataFrame as a GitHub-flavored Markdown table."""
    if frame.empty:
        return "_No rows._\n"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.{float_digits}f}"
            )
        else:
            display[column] = display[column].map(
                lambda value: "" if pd.isna(value) else str(value)
            )
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in display.to_numpy()]
    return "\n".join([header, separator, *rows]) + "\n"


def write_config(path: Path, config: dict[str, object]) -> None:
    """Write a small YAML-like config without adding a PyYAML dependency."""
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


# ---------------------------------------------------------------------------
# Matplotlib helper
# ---------------------------------------------------------------------------


def matplotlib_pyplot() -> object:
    """Return pyplot with a non-interactive Agg backend."""
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    return plt


# ---------------------------------------------------------------------------
# Git / environment helpers
# ---------------------------------------------------------------------------


def get_git_commit() -> str:
    """Return the current HEAD git commit hash, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def get_git_branch() -> str:
    """Return the current git branch name, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def get_package_versions(packages: list[str]) -> dict[str, str]:
    """Return installed version strings for the listed packages."""
    from importlib.metadata import PackageNotFoundError, version

    versions: dict[str, str] = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


# ---------------------------------------------------------------------------
# Canonical run directory
# ---------------------------------------------------------------------------


def make_run_id(name: str) -> str:
    """Return a timestamped run ID like 2026-06-03_103000_embedding_compare."""
    ts = datetime.now(tz=UTC).strftime("%Y-%m-%d_%H%M%S")
    return f"{ts}_{slug(name)}"


def make_run_dir(base_dir: Path, run_id: str) -> Path:
    """Create and return the canonical run directory for one eval run."""
    run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    return run_dir


def write_manifest(
    run_dir: Path,
    *,
    run_id: str,
    config_file: str | None,
    cases_file: str | None,
    extra: dict[str, object] | None = None,
) -> None:
    """Write manifest.json recording git state, runtime, and run provenance."""
    manifest: dict[str, object] = {
        "run_id": run_id,
        "git_commit": get_git_commit(),
        "git_branch": get_git_branch(),
        "timestamp_utc": datetime.now(tz=UTC).isoformat(),
        "python_version": sys.version.split()[0],
        "packages": get_package_versions(
            ["shapiq", "transformers", "sentence-transformers", "scikit-learn", "streamlit"]
        ),
        "config_file": config_file,
        "cases_file": cases_file,
    }
    if extra:
        manifest.update(extra)
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Comparison plots (all return the saved path or None)
# ---------------------------------------------------------------------------


def plot_pass_rate_by_config(
    summary: pd.DataFrame,
    plots_dir: Path,
    *,
    config_col: str = "config_id",
    pass_col: str = "passed",
    title: str = "Pass Rate by Config",
    filename: str = "pass_rate_by_config.png",
) -> Path | None:
    """Horizontal bar chart: pass rate per config, sorted ascending.

    Args:
        summary:    One row per (case × config).
        plots_dir:  Directory to write the PNG.
        config_col: Column identifying the config.
        pass_col:   Boolean column indicating pass/fail.
        title:      Plot title.
        filename:   Output filename inside plots_dir.
    """
    if summary.empty or config_col not in summary.columns or pass_col not in summary.columns:
        return None
    plots_dir.mkdir(exist_ok=True)
    plt = matplotlib_pyplot()
    pass_rates = (
        summary.groupby(config_col, as_index=False)[pass_col]
        .mean()
        .sort_values(pass_col)
    )
    labels = [short_label(str(c), 36) for c in pass_rates[config_col]]
    fig, ax = plt.subplots(figsize=(10, max(3.5, 0.45 * len(pass_rates))))
    bars = ax.barh(labels, pass_rates[pass_col].astype(float), color="#3572a5")
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=8)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Pass rate (evidence term hit rate)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    out = plots_dir / filename
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_shapley_mass_heatmap(
    mass_df: pd.DataFrame,
    plots_dir: Path,
    *,
    title: str = "Shapley Mass on Expected Evidence",
    filename: str = "shapley_mass_heatmap.png",
) -> Path | None:
    """Heatmap: rows=cases, cols=configs, colour=Shapley mass on expected chunks.

    Args:
        mass_df:   DataFrame indexed by case_id with one column per config_id,
                   values are the shapley_mass_on_expected float (0-1 or NaN).
        plots_dir: Directory to write the PNG.
    """
    if mass_df.empty:
        return None
    plots_dir.mkdir(exist_ok=True)
    plt = matplotlib_pyplot()
    import matplotlib.colors as mcolors

    data = mass_df.values.astype(float)
    row_labels = [short_label(str(r), 28) for r in mass_df.index]
    col_labels = [short_label(str(c), 22) for c in mass_df.columns]

    fig, ax = plt.subplots(
        figsize=(max(6, 1.4 * len(col_labels)), max(4, 0.55 * len(row_labels)))
    )
    masked = np.ma.masked_invalid(data)
    cmap = plt.cm.RdYlGn  # type: ignore[attr-defined]
    cmap.set_bad(color="#cccccc")
    im = ax.imshow(masked, vmin=0, vmax=1, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)
    plt.colorbar(im, ax=ax, label="Mass on expected chunks")
    ax.set_title(title)
    fig.tight_layout()
    out = plots_dir / filename
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_rank_stability(
    stability_df: pd.DataFrame,
    plots_dir: Path,
    *,
    title: str = "Shapley Rank Stability (Kendall τ)",
    filename: str = "rank_stability.png",
) -> Path | None:
    """Heatmap of pairwise Kendall-τ between config Shapley rankings.

    Args:
        stability_df: Square DataFrame from compute_rank_stability().
        plots_dir:    Directory to write the PNG.
    """
    if stability_df.empty:
        return None
    plots_dir.mkdir(exist_ok=True)
    plt = matplotlib_pyplot()

    labels = [short_label(str(c), 22) for c in stability_df.columns]
    data = stability_df.values.astype(float)
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(labels)), max(4, 1.0 * len(labels))))
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, vmin=-1, vmax=1, aspect="auto", cmap="coolwarm")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    # Annotate cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = data[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else "black")
    plt.colorbar(im, ax=ax, label="Kendall τ")
    ax.set_title(title)
    fig.tight_layout()
    out = plots_dir / filename
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_retrieval_vs_shapley_scatter(
    summary: pd.DataFrame,
    plots_dir: Path,
    *,
    retrieval_col: str = "avg_retrieval_score",
    shapley_col: str = "shapley_mass_on_expected",
    config_col: str = "config_id",
    title: str = "Retrieval Score vs Shapley Mass",
    filename: str = "retrieval_vs_shapley.png",
) -> Path | None:
    """Scatter: x=avg retrieval score of expected chunks, y=Shapley mass on expected.

    Each point is one (case × config). Color encodes config.
    """
    needed = {retrieval_col, shapley_col, config_col}
    if summary.empty or not needed.issubset(summary.columns):
        return None
    valid = summary.dropna(subset=[retrieval_col, shapley_col])
    if valid.empty:
        return None
    plots_dir.mkdir(exist_ok=True)
    plt = matplotlib_pyplot()

    configs = sorted(valid[config_col].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(configs), 1)))  # type: ignore[attr-defined]
    color_map = dict(zip(configs, colors, strict=False))

    fig, ax = plt.subplots(figsize=(7, 5))
    for cfg in configs:
        sub = valid[valid[config_col] == cfg]
        ax.scatter(
            sub[retrieval_col].astype(float),
            sub[shapley_col].astype(float),
            label=short_label(str(cfg), 28),
            color=color_map[cfg],
            alpha=0.75,
            s=55,
        )
    ax.set_xlabel("Avg retrieval score of expected chunks")
    ax.set_ylabel("Shapley mass on expected chunks")
    ax.set_xlim(-0.02, None)
    ax.set_ylim(-0.02, 1.05)
    ax.axline((0, 0), slope=1, color="grey", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = plots_dir / filename
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_deletion_insertion_curves(
    deletion_curves: dict[str, np.ndarray],
    insertion_curves: dict[str, np.ndarray],
    plots_dir: Path,
    *,
    case_id: str = "case",
    filename: str | None = None,
) -> Path | None:
    """Line plot of deletion and insertion curves for one case across configs.

    Args:
        deletion_curves:  {config_id: score_array (step 0..n)}.
        insertion_curves: {config_id: score_array (step 0..n)}.
        plots_dir:        Directory to write the PNG.
        case_id:          Case identifier used in the title.
        filename:         PNG filename; defaults to <slug(case_id)>_del_ins.png.
    """
    if not deletion_curves and not insertion_curves:
        return None
    plots_dir.mkdir(exist_ok=True)
    plt = matplotlib_pyplot()

    fname = filename or f"{slug(case_id)}_deletion_insertion_curves.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax, curves, label in [
        (axes[0], deletion_curves, "Deletion"),
        (axes[1], insertion_curves, "Insertion"),
    ]:
        for cfg_id, scores in curves.items():
            ax.plot(range(len(scores)), scores, marker="o", markersize=4,
                    label=short_label(cfg_id, 24), linewidth=1.3)
        ax.set_xlabel("Step")
        ax.set_ylabel("Support score")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title(f"{label} curves — {short_label(case_id, 30)}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.2)

    fig.tight_layout()
    out = plots_dir / fname
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_comparison_plots(summary: pd.DataFrame, output_dir: Path) -> None:
    """Write all available comparison plots for a grid eval run.

    Calls each plot function with the columns it needs; silently skips any
    plot whose required columns are absent.
    """
    plots_dir = output_dir / "plots"
    plot_pass_rate_by_config(summary, plots_dir)
    # Shapley mass heatmap (requires shapley_mass_on_expected column)
    if "shapley_mass_on_expected" in summary.columns and "config_id" in summary.columns:
        mass_pivot = summary.pivot_table(
            index="case_id", columns="config_id", values="shapley_mass_on_expected"
        )
        plot_shapley_mass_heatmap(mass_pivot, plots_dir)
    # Retrieval vs Shapley scatter
    plot_retrieval_vs_shapley_scatter(summary, plots_dir)
