from __future__ import annotations

import torch


def resolve_device(device: str | int) -> str:
    """Resolve a device argument to a concrete torch device string.

    Args:
        device: "cuda" (auto-detects best available), an int (cuda index),
                or any explicit string ("cpu", "mps", "cuda:1", …).

    Returns:
        A concrete device string ready for use with .to(device).
    """
    if device == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if isinstance(device, int):
        return f"cuda:{device}"

    return device