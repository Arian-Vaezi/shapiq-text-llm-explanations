"""Semantic segmentation prototype for agentic tool-use prompts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_EMBEDDING_MODEL_ID = "sentence-transformers/all-mpnet-base-v2"


def resolve_embedding_device(device: str | int | None = "auto") -> str:
    """Resolve an embedding device with CUDA, then MPS, then CPU priority."""
    if isinstance(device, int):
        return f"cuda:{device}"
    if device not in {None, "auto", "cuda"}:
        return str(device)

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class SemanticSegmenter:
    """Split text into semantic word-partition segments using MPNet embeddings."""

    model_id: str = DEFAULT_EMBEDDING_MODEL_ID
    device: str | int | None = "auto"
    threshold: float = 0.5
    window: int = 2
    min_segment_words: int = 2

    def __post_init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.device = resolve_embedding_device(self.device)
        self.model = SentenceTransformer(self.model_id, device=self.device)

    def segment(self, text: str) -> list[str]:
        """Return semantic segments that preserve each whitespace token exactly once."""
        words = text.split()
        if len(words) <= 1:
            return words

        half_window = self.window // 2
        windows = [
            " ".join(words[max(0, index - half_window) : index + half_window + 1])
            for index in range(len(words))
        ]

        embeddings = self.model.encode(
            windows,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        similarities = [
            float(np.dot(embeddings[index], embeddings[index + 1]))
            for index in range(len(embeddings) - 1)
        ]

        blocks: list[str] = []
        current = [words[0]]
        for index, similarity in enumerate(similarities):
            if similarity < self.threshold:
                blocks.append(" ".join(current))
                current = [words[index + 1]]
            else:
                current.append(words[index + 1])
        blocks.append(" ".join(current))

        segments = self._merge_short_segments(blocks)
        validate_partition(text, segments)
        return segments

    def _merge_short_segments(self, blocks: list[str]) -> list[str]:
        """Merge undersized blocks while preserving order."""
        merged: list[str] = []
        leading_short_block: str | None = None
        for block in blocks:
            if leading_short_block is not None:
                block = f"{leading_short_block} {block}"
                leading_short_block = None

            if len(block.split()) >= self.min_segment_words:
                merged.append(block)
            elif merged:
                merged[-1] = f"{merged[-1]} {block}"
            else:
                leading_short_block = block

        if leading_short_block is not None:
            if merged:
                merged[-1] = f"{merged[-1]} {leading_short_block}"
            else:
                merged.append(leading_short_block)
        return merged


def validate_partition(text: str, segments: list[str]) -> None:
    """Raise if segments are not an exact whitespace-token partition of text."""
    original_words = text.split()
    segmented_words = " ".join(segments).split()
    if segmented_words != original_words:
        msg = (
            "Semantic segments must preserve every whitespace-tokenized original word "
            "exactly once, in original order, without inserting or rewriting words."
        )
        raise ValueError(msg)
