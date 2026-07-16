"""Central model registry for the RAG Shapley demo.

Qwen3, Qwen2.5, and Gemma 4 families are registered here. Each entry carries
the expected GGUF filename so availability can be checked without loading llama.cpp.
"""

from __future__ import annotations

from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[3] / "models" / "llm"

# fmt: off
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    # ── Qwen2.5 (legacy, kept for the original default path that already exists locally) ──
    "qwen2.5-1.5b": {
        "display_name": "Qwen2.5 1.5B",
        "family": "Qwen2.5",
        "params": "1.5B",
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "description": "Legacy Qwen2.5 baseline — already present locally.",
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "download_url": "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
    },
    # ── Qwen3 ─────────────────────────────────────────────────────────────────────
    # Note: Qwen3 smallest dense model is 1.7B. Official Qwen repo only has Q8_0;
    # Q4_K_M is from unsloth.
    "qwen3-1.7b": {
        "display_name": "Qwen3 1.7B",
        "family": "Qwen3",
        "params": "1.7B",
        "filename": "Qwen3-1.7B-Q4_K_M.gguf",
        "description": "Fastest Qwen3 model, best for quick experiments.",
        "repo_id": "unsloth/Qwen3-1.7B-GGUF",
        "download_url": "https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf",
    },
    "qwen3-4b": {
        "display_name": "Qwen3 4B",
        "family": "Qwen3",
        "params": "4B",
        "filename": "Qwen3-4B-Q4_K_M.gguf",
        "description": "Balanced quality and speed.",
        "repo_id": "Qwen/Qwen3-4B-GGUF",
        "download_url": "https://huggingface.co/Qwen/Qwen3-4B-GGUF/resolve/main/Qwen3-4B-Q4_K_M.gguf",
    },
    "qwen3-8b": {
        "display_name": "Qwen3 8B",
        "family": "Qwen3",
        "params": "8B",
        "filename": "Qwen3-8B-Q4_K_M.gguf",
        "description": "Default — high-quality grounded RAG answers.",
        "repo_id": "Qwen/Qwen3-8B-GGUF",
        "download_url": "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf",
    },
    # ── Gemma 4 ───────────────────────────────────────────────────────────────────
    "gemma4-e2b": {
        "display_name": "Gemma 4 E2B",
        "family": "Gemma 4",
        "params": "E2B",
        "filename": "gemma-4-E2B-it-Q4_K_M.gguf",
        "description": "Google Gemma 4 efficient 2B — very fast inference (~3.1 GB).",
        "repo_id": "unsloth/gemma-4-E2B-it-GGUF",
        "download_url": "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf",
    },
    "gemma4-e4b": {
        "display_name": "Gemma 4 E4B",
        "family": "Gemma 4",
        "params": "E4B",
        "filename": "gemma-4-E4B-it-Q4_K_M.gguf",
        "description": "Google Gemma 4 efficient 4B — good quality-speed balance (~5 GB).",
        "repo_id": "unsloth/gemma-4-E4B-it-GGUF",
        "download_url": "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/main/gemma-4-E4B-it-Q4_K_M.gguf",
    },
    "gemma4-26b-a4b": {
        "display_name": "Gemma 4 26B-A4B",
        "family": "Gemma 4",
        "params": "26B-A4B",
        # unsloth uses UD- (Ultra Derived) prefix for their Q4_K_M quantizations
        "filename": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        "description": "Google Gemma 4 MoE: 26B total / 4B active — high quality (~17 GB).",
        "repo_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "download_url": "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    },
}
# fmt: on

DEFAULT_MODEL_ID = "gemma4-e4b"


# ── Path helpers ──────────────────────────────────────────────────────────────


def get_model_path(model_id: str) -> str:
    """Return the expected local GGUF path for a registered model ID."""
    return str(MODELS_DIR / MODEL_REGISTRY[model_id]["filename"])


def get_model_config(model_id: str) -> dict[str, str]:
    """Return the raw registry entry for *model_id*."""
    return MODEL_REGISTRY[model_id]


def get_model_status(model_id: str) -> dict[str, object]:
    """Return registry entry enriched with local file availability."""
    entry = MODEL_REGISTRY[model_id]
    path = MODELS_DIR / entry["filename"]
    return {
        "id": model_id,
        **entry,
        "path": str(path),
        "available": path.is_file(),
        "download_command": f"uv run python scripts/download_models.py --model {model_id}",
    }


# ── Query helpers ─────────────────────────────────────────────────────────────


def list_models() -> list[dict[str, object]]:
    """Return status dicts for every registered model."""
    return [get_model_status(mid) for mid in MODEL_REGISTRY]


def list_models_by_family(family: str) -> list[dict[str, object]]:
    """Return status dicts for models whose family matches *family* (case-insensitive)."""
    family_lower = family.lower()
    return [
        get_model_status(mid)
        for mid, entry in MODEL_REGISTRY.items()
        if entry["family"].lower() == family_lower
    ]


def get_missing_models(model_ids: list[str]) -> list[dict[str, object]]:
    """Return status dicts for models in *model_ids* whose GGUF file is absent."""
    return [
        get_model_status(mid)
        for mid in model_ids
        if mid in MODEL_REGISTRY and not (MODELS_DIR / MODEL_REGISTRY[mid]["filename"]).is_file()
    ]


def get_available_models() -> list[dict[str, object]]:
    """Return status dicts for all models whose GGUF file exists locally."""
    return [
        get_model_status(mid)
        for mid, entry in MODEL_REGISTRY.items()
        if (MODELS_DIR / entry["filename"]).is_file()
    ]
