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
    ):
        # moved import inside the class to not break CI
        from transformers import pipeline

        if mask_strategy not in {"mask", "remove"}:
            raise ValueError(f"Invalid mask_strategy: {mask_strategy}")

        if segmentation not in {"token", "word"}:
            raise ValueError(f"Invalid segmentation: {segmentation}")

        self.mask_strategy = mask_strategy
        self.batch_size = batch_size
        self.segmentation = segmentation

        # ---------------- MODEL ----------------
        # TODO: (Arian - #7)
        # add HF classifier wrapper
        self._classifier = pipeline(
            task="sentiment-analysis",
            model=model_name,
            device=device,
        )
        self._tokenizer = self._classifier.tokenizer

        self.original_text = input_text

        # ---------------- SEGMENTATION ----------------
        # TODO: (Yuanyuan/Yili - #8)
        # implement word-level/token-level segmentation
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
        data = self._tokens.reshape(1, -1)

        super().__init__(
            model=self._classifier,
            data=data,
            x=data,
            sample_size=1,
            verbose=verbose,
        )

        # ---------------- NORMALIZATION ----------------
        # compute empty prediction (all masked)
        empty = np.zeros((1, self.n_features), dtype=bool)
        self.empty_prediction = self.value_function(empty)[0]
        self.normalization_value = self.empty_prediction

    # ------------------- Masking -------------------
    # TODO: (Yuanyuan/Yili - #8)
    # Implement [Mask] replacement and token removal strategy
    def _coalition_to_tokens(self, coalition: np.ndarray) -> np.ndarray:
        """Convert a coalition mask into token ids."""
        if self.mask_strategy == "remove":
            return self._tokens[coalition]

        tokens = self._tokens.copy()
        tokens[~coalition] = self._mask_token_id
        return tokens

    def _decode(self, tokens: np.ndarray) -> str:
        return self._tokenizer.decode(tokens)

    # ------------------- Value Function -------------------
    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Core function:
        coalition → masked text → batched model call → score
        """
        # 1. build texts from coalitions
        texts = [self._decode(self._coalition_to_tokens(c)) for c in coalitions]

        results = []

        # 2. (batched) model call to do sentiment classification on each coalition
        with torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                # feeding the batch into the model/classifier
                outputs = self._classifier(batch)

                scores = [o["score"] if o["label"] == "POSITIVE" else -o["score"] for o in outputs]

                results.extend(scores)

        outputs = np.array(results, dtype=float)

        # 3. normalization handling
        return self.insert_empty_value(outputs, coalitions)

    # ----------------- Helpers ------------------
    @property
    def tokens(self) -> np.ndarray:
        return self._tokens.copy()

    @property
    def players(self):
        return self._players
