"""hf_model.py — backwards-compatible factory.

Importing HFModelWrapper from this module still works exactly as before.
Internally it delegates to the focused wrapper classes.
"""
from __future__ import annotations

from .causal_model_wrapper import CausalModelWrapper
from .encoder_model_wrapper import EncoderModelWrapper
from .embedding_model_wrapper import EmbeddingModelWrapper


def HFModelWrapper(
    model_name: str,
    device: str | int = "cuda",
    hf_token: str | None = None,
) -> CausalModelWrapper | EncoderModelWrapper:
    """Factory that returns the right wrapper for model_name.

    - Causal LMs  (llama, mistral, gemma, …) → CausalModelWrapper
    - Encoder models (bert, roberta, …)        → EncoderModelWrapper

    Raises:
        ValueError: If model_name doesn't match any known model family.
    """
    name = model_name.lower()

    if any(x in name for x in CausalModelWrapper.CAUSAL_MODELS):
        return CausalModelWrapper(model_name, device=device, hf_token=hf_token)

    if any(x in name for x in EncoderModelWrapper.ENCODER_MODELS):
        return EncoderModelWrapper(model_name, device=device, hf_token=hf_token)

    msg = f"Unsupported model type for model: {model_name}"
    raise ValueError(msg)


__all__ = [
    "HFModelWrapper",
    "CausalModelWrapper",
    "EncoderModelWrapper",
    "EmbeddingModelWrapper",
]