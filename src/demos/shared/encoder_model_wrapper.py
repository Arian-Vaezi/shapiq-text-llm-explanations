from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from .device_utils import resolve_device


class EncoderModelWrapper:
    """Wraps an encoder-only model (BERT, RoBERTa, DistilBERT, …).

    Responsibilities:
    - Classification scoring (score_classifier) for jailbreak detection
      with encoder-based safety classifiers.
    """

    ENCODER_MODELS: list[str] = [
        "distilbert",
        "roberta",
        "bert",
    ]

    def __init__(
        self,
        model_name: str,
        device: str | int = "cuda",
        hf_token: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.is_causal = False
        self.is_encoder = True

        print(f"[EncoderModelWrapper] Loading '{model_name}' on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModel.from_pretrained(model_name, token=hf_token)
        self.model.to(self.device)
        self.model.eval()

    # =========================================================
    # SCORING
    # =========================================================

    @torch.no_grad()
    def score_classifier(self, texts: list[str]) -> np.ndarray:
        """Run texts through the classifier head and return positive-class probabilities.

        For binary models returns P(positive). For multi-class returns max P.

        Args:
            texts: List of strings to classify.

        Returns:
            np.ndarray of shape (len(texts),).
        """
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

        if probs.shape[-1] == 2:
            return probs[:, 1].cpu().numpy()

        return torch.max(probs, dim=-1).values.cpu().numpy()