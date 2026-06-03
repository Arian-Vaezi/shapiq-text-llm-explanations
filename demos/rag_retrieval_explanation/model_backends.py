"""Backward-compatible shim — re-exports from core.model_backends."""

from __future__ import annotations

from core.model_backends import (  # noqa: F401
    GenerationResult,
    HuggingFaceCausalLMBackend,
    cached_hf_model,
    cached_hf_tokenizer,
)
