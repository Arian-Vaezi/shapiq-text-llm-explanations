"""Backward-compatible shim - re-exports fixtures from core."""

from __future__ import annotations

from core.sample_data import SAMPLE_TRACES

__all__ = ["SAMPLE_TRACES"]
