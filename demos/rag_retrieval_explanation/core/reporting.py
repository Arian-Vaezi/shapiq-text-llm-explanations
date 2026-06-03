"""Shared reporting utilities for all eval runners.

Provides:
- Text/table formatting: markdown_table, slug, short_label, contains_term
- Matplotlib helpers: matplotlib_pyplot (non-interactive backend)
- Run directory management: make_run_dir, write_manifest
- Git helpers: get_git_commit
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

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
