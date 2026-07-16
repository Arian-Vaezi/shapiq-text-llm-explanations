from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from shapiq.game import Game


class encoderTextImputer(Game):
    def __init__(
        self,
        model_name: str,
        input_text: str,
        *,
        preloaded_model: AutoModelForSequenceClassification | None = None,
        preloaded_tokenizer: AutoTokenizer | None = None,
        device: str | torch.device | None = None,
        batch_size: int = 32,
        positive_label: str = "POSITIVE",
        negative_label: str = "NEGATIVE",
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self.original_text = input_text
        self.batch_size = batch_size

        # ── Model and tokenizer ───────────────────────────────────────────────
        if preloaded_model is not None and preloaded_tokenizer is not None:
            # Mode 2: reuse already-loaded weights — no loading cost
            self._model = preloaded_model
            self.tokenizer = preloaded_tokenizer
            self.device = next(preloaded_model.parameters()).device
        else:
            # Mode 1: load from HuggingFace (first time or standalone use)
            if device is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model.to(self.device)
            self._model.eval()

        # ── Mask token ────────────────────────────────────────────────────────
        if self.tokenizer.mask_token is None:
            raise ValueError(
                f"Tokenizer for {model_name!r} has no mask token. "
                "This imputer requires a BERT/RoBERTa-style mask token."
            )
        self.mask_token: str = self.tokenizer.mask_token  # [MASK] or <mask>

        # ── Word-level players ────────────────────────────────────────────────
        self._words = np.array(input_text.split(), dtype=object)
        if len(self._words) == 0:
            raise ValueError("input_text must contain at least one word.")
        n_players = len(self._words)

        # ── Label indices ─────────────────────────────────────────────────────
        self._pos_idx = self._find_label_index(positive_label)
        self._neg_idx = self._find_label_index(negative_label)

        # ── Compute v(∅) BEFORE super().__init__ ─────────────────────────────
        # One forward pass on the fully masked sentence.
        # This is the baseline all coalition values are centered around.
        empty_text = " ".join([self.mask_token] * n_players)
        v_empty = float(self._evaluate_texts([empty_text])[0])

        # ── Init Game ─────────────────────────────────────────────────────────
        super().__init__(
            n_players=n_players,
            normalize=True,
            normalization_value=v_empty,
            player_names=list(self._words),
            verbose=verbose,
        )

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def players(self) -> np.ndarray:
        """The word players in order."""
        return self._words.copy()

    # ── Label resolution ──────────────────────────────────────────────────────

    def _find_label_index(self, target_label: str) -> int:
        """Find the class index for a label name (case-insensitive).

        Handles:
            {0: "NEGATIVE", 1: "POSITIVE"}     binary uppercase
            {0: "negative", 1: "positive"}     binary lowercase
            {0: "LABEL_0",  1: "LABEL_1"}      generic → fallback NEG=0 POS=1

        Args:
            target_label: Label string to find.

        Returns:
            Integer class index.

        Raises:
            ValueError: If label not found and no fallback applies.
        """
        target = target_label.upper()
        id2label = self._model.config.id2label

        for idx, label in id2label.items():
            if str(label).upper() == target:
                return int(idx)

        # fallback for generic binary models
        if target == "NEGATIVE":
            return 0
        if target == "POSITIVE":
            return 1

        raise ValueError(f"Could not find label {target_label!r} in model labels: {id2label}")

    # ── Masking ───────────────────────────────────────────────────────────────

    def coalition_to_text(self, coalition: np.ndarray) -> str:
        coalition = np.asarray(coalition, dtype=bool)
        if coalition.shape != (self.n_players,):
            raise ValueError(
                f"Coalition must have shape ({self.n_players},), got {coalition.shape}."
            )
        words = self._words.copy()
        words[~coalition] = self.mask_token
        return " ".join(words)

    def coalitions_to_texts(self, coalitions: np.ndarray) -> list[str]:
        """Convert a batch of coalition vectors into masked sentences.

        Args:
            coalitions: Boolean array of shape (n_coalitions, n_players).

        Returns:
            List of masked sentences, one per coalition.
        """
        coalitions = np.asarray(coalitions, dtype=bool)
        if coalitions.ndim == 1:
            coalitions = coalitions.reshape(1, -1)
        if coalitions.shape[1] != self.n_players:
            raise ValueError(
                f"Coalitions must have shape (n_coalitions, {self.n_players}), "
                f"got {coalitions.shape}."
            )
        return [self.coalition_to_text(c) for c in coalitions]

    # ── Model scoring ─────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _evaluate_texts(self, texts: list[str]) -> np.ndarray:
        """Run encoder on texts and return contrastive scores.

        Processes in batches of self.batch_size to avoid OOM.

        Args:
            texts: List of (masked) sentences.

        Returns:
            np.ndarray of shape (len(texts),) in [-1, +1].
            Each value = P(POSITIVE|text) - P(NEGATIVE|text).
        """
        scores: list[float] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)

            outputs = self._model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1)
            batch_scores = probs[:, self._pos_idx] - probs[:, self._neg_idx]
            scores.extend(batch_scores.cpu().tolist())

        return np.array(scores, dtype=float)

    # ── Value function ────────────────────────────────────────────────────────

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Compute v(S) for a batch of coalitions.

        This is the only method shapiq calls during approximation.

        Args:
            coalitions: Boolean array of shape (n_coalitions, n_players).

        Returns:
            np.ndarray of shape (n_coalitions,) in [-1, +1].
        """
        texts = self.coalitions_to_texts(coalitions)
        return self._evaluate_texts(texts)
