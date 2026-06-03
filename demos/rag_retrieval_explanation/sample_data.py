"""Backward-compatible shim — re-exports from evals.cases.sample_data."""

from __future__ import annotations

from evals.cases.sample_data import SAMPLE_TRACES  # noqa: F401

__all__ = ["SAMPLE_TRACES"]
