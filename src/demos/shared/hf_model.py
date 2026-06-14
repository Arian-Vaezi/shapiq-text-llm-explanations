"""hf_model.py — backwards-compatible factory.

Importing HFModelWrapper from this module still works exactly as before.
Internally it delegates to the focused wrapper classes.
"""

from __future__ import annotations

from .api_model_wrapper import APIModelWrapper
from .causal_model_wrapper import CausalModelWrapper
from .embedding_model_wrapper import EmbeddingModelWrapper
from .encoder_model_wrapper import EncoderModelWrapper

API_MODEL_NAMES = (
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-safeguard-20b",
    "qwen/qwen3-32b",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)


def is_api_model_name(model_name: str) -> bool:
    """Return whether a model name should be routed to an API provider."""
    name = model_name.lower()
    return name in API_MODEL_NAMES or name.startswith("gemini-")


def HFModelWrapper(
    model_name: str,
    device: str | int = "cuda",
    hf_token: str | None = None,
    temperature: float = 0.0,
) -> CausalModelWrapper | EncoderModelWrapper | APIModelWrapper:
    """Factory that returns the right wrapper for model_name.

    - API Models (groq, openai, gemini) -> APIModelWrapper
    - Causal LMs  (llama, mistral, gemma, …) → CausalModelWrapper
    - Encoder models (bert, roberta, …)        → EncoderModelWrapper

    Raises:
        ValueError: If model_name doesn't match any known model family.
    """
    name = model_name.lower()

    if is_api_model_name(model_name):
        # Route to API wrapper for known API models
        return APIModelWrapper(model_name, device=device, temperature=temperature)

    if any(x in name for x in CausalModelWrapper.CAUSAL_MODELS):
        return CausalModelWrapper(
            model_name, device=device, hf_token=hf_token, temperature=temperature
        )

    if any(x in name for x in EncoderModelWrapper.ENCODER_MODELS):
        return EncoderModelWrapper(model_name, device=device, hf_token=hf_token)

    if any(x in name for x in EmbeddingModelWrapper.EMBEDDING_MODELS):
        return EmbeddingModelWrapper(model_name, device=device)

    msg = f"Unsupported model type for model: {model_name}"
    raise ValueError(msg)


__all__ = [
    "API_MODEL_NAMES",
    "APIModelWrapper",
    "CausalModelWrapper",
    "EmbeddingModelWrapper",
    "EncoderModelWrapper",
    "HFModelWrapper",
    "is_api_model_name",
]
