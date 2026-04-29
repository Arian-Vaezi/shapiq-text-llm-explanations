"""Text imputer for LLM / NLP models using Shapley value explanations."""

from __future__ import annotations

import numpy as np

from shapiq.imputer.base import Imputer


class TextImputer(Imputer):
    """Text Imputer for LLM / NLP models.

    This class treats tokens (or words) as players and evaluates coalitions
    by masking/removing parts of the input and calling a model.
    """

    def __init__(
        self,
        model_name: str,
        input_text: str,
        *,
        mask_strategy: str = "mask",
        device: int | str | None = None,
        batch_size: int = 32,
        segmentation: str = "token",  # "token" | "word"
        verbose: bool = False,
    ) -> None:
        """Initialize the TextImputer.

        Args:
            model_name: HuggingFace model identifier.
            input_text: The text to explain.
            mask_strategy: Either ``"mask"`` (replace with [MASK]) or ``"remove"``.
            device: Device for the HuggingFace pipeline (e.g. 0 for GPU, ``"cpu"``).
            batch_size: Number of masked texts to score in one model call.
            segmentation: Either ``"token"`` or ``"word"`` player granularity.
            verbose: Whether to print progress information.
        """
        # moved import inside the class to not break CI
        from transformers import pipeline

        if mask_strategy not in {"mask", "remove"}:
            msg = f"Invalid mask_strategy: {mask_strategy}"
            raise ValueError(msg)

        if segmentation not in {"token", "word"}:
            msg = f"Invalid segmentation: {segmentation}"
            raise ValueError(msg)

        self.mask_strategy = mask_strategy
        self.batch_size = batch_size
        self.segmentation = segmentation

        # ---------------- MODEL ----------------
        # TODO @arian: add HF classifier wrapper (#7)
        self._classifier = pipeline(  # ty: ignore[no-matching-overload]
            "sentiment-analysis",
            model=model_name,
            device=device,
        )
        self._tokenizer = self._classifier.tokenizer

        self.original_text = input_text

        # ---------------- SEGMENTATION ----------------
        # TODO @yuanyuan-yili: implement word-level/token-level segmentation (#8)
        if segmentation == "token":
            tokens = self._tokenizer(input_text)["input_ids"][1:-1]
            self._tokens = np.array(tokens)
            self._players = self._tokens

        elif segmentation == "word":
            # naive word split (can be improved)
            words = input_text.split()
            self._players = np.array(words)

            # map words -> tokens later if needed
            tokens = self._tokenizer(input_text)["input_ids"][1:-1]
            self._tokens = np.array(tokens)

        self._mask_token_id = self._tokenizer.mask_token_id

        # ---------------- SEGMENTATION ----------------
        data = np.arange(len(self._players)).reshape(1, -1)

        super().__init__(
            model=self._classifier,
            data=data,
            x=data,
            sample_size=1,
            verbose=verbose,
        )

        # ---------------- NORMALIZATION ----------------
        # Compute the real empty prediction before value_function inserts empty values.
        empty = np.zeros((1, self.n_features), dtype=bool)
        empty_text = self._decode(self._coalition_to_tokens(empty[0]))
        self.empty_prediction = self._evaluate_texts([empty_text])[0]
        self.normalization_value = self.empty_prediction

    # ------------------- Masking -------------------
    # TODO @yuanyuan-yili: implement [MASK] replacement and token removal strategy (#8)
    def _token_coalition_to_tokens(self, coalition: np.ndarray) -> np.ndarray:
        """Convert a token-level coalition mask into token ids."""
        if self.mask_strategy == "remove":
            return self._tokens[coalition]

        tokens = self._tokens.copy()
        tokens[~coalition] = self._mask_token_id
        return tokens

    def _decode(self, tokens: np.ndarray) -> str:
        return self._tokenizer.decode(tokens)
    
    def _word_coalition_to_text(self, coalition: np.ndarray) -> str:
        """Convert a word-level coalition mask into a masked text string."""
        # coalitions refer to words
        words = self._players.astype(str)

        if self.mask_strategy == "remove":
            return " ".join(words[coalition])

        masked_words = words.copy()
        masked_words[~coalition] = "[MASK]"
        return " ".join(masked_words)
    
    def _coalition_to_text(self, coalition: np.ndarray) -> str:
        """Convert a coalition mask into a text string."""
        # dispatch
        if self.segmentation == "word":
            return self._word_coalition_to_text(coalition)

        return self._decode(self._token_coalition_to_tokens(coalition))

    # ------------------- Value Function -------------------
    def _evaluate_texts(self, texts: list[str]) -> np.ndarray:
        """Evaluate the classifier on a batch of texts."""
        results = []

        import torch

        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                outputs = self._classifier(batch)

                scores = [
                    output["score"] if output["label"] == "POSITIVE" else -output["score"]
                    for output in outputs
                ]
                results.extend(scores)

        return np.array(results, dtype=float)

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Core function.

        coalition → masked text → batched model call → score.
        """
        texts = [self._coalition_to_text(c) for c in coalitions]
        outputs = self._evaluate_texts(texts)
        return self.insert_empty_value(outputs, coalitions)

    # ----------------- Helpers ------------------
    @property
    def tokens(self) -> np.ndarray:
        """Return a copy of the token id array."""
        return self._tokens.copy()

    @property
    def players(self) -> np.ndarray:
        """Return the player array (tokens or words depending on segmentation)."""
        return self._players
