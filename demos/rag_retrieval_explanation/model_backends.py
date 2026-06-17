"""Backward-compatible shim — re-exports from core.model_backends."""

from __future__ import annotations

from core.model_backends import (  # noqa: F401
    GenerationResult,
    LlamaCppBackend,
    cached_llama_cpp_model,
    sigmoid,
)
