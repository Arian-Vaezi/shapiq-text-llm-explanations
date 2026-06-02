from __future__ import annotations

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from demos.shared.device_utils import resolve_device


class EmbeddingModelWrapper:
    """Wraps a SentenceTransformer model for semantic similarity / segmentation.

    Kept entirely separate from the causal/encoder wrappers because:
    - It has no tokenizer or HF model in the transformers sense
    - It's loaded on demand (only needed for semantic segmentation)
    - SentenceTransformer handles pooling and normalization internally

    Usage:
        emb = EmbeddingModelWrapper("sentence-transformers/all-mpnet-base-v2")
        vecs = emb.encode(["hello world", "foo bar"])  # shape (2, 768), L2-normalized
    """

    
    EMBEDDING_MODELS: list[str] = ("sentence-transformers", "all-mpnet", "all-minilm", "bge-", "e5-")

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: str | int = "cuda",
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)

        print(f"[EmbeddingModelWrapper] Loading '{model_name}' on {self.device}")
        self._model = SentenceTransformer(model_name, device=self.device)

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into L2-normalized embedding vectors.

        Embeddings are normalized so dot product == cosine similarity.

        Args:
            texts: List of strings to embed.

        Returns:
            np.ndarray of shape (len(texts), hidden_dim).
        """
        return self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )