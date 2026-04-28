from __future__ import annotations

import numpy as np
from transformers import pipeline

from shapiq.imputer.base import Imputer


class TextImputer(Imputer):
    def __init__(
        self,
        model_name: str,
        input_text: str,
        *,
        mask_strategy: str = "mask",
        device: int | str | None = None,
    ):
        # generalized to support other models
        self._classifier = pipeline(
            model=model_name,
            task="sentiment-analysis",
            device=device,
        )
        self._tokenizer = self._classifier.tokenizer

        # tokenize input
        self.original_text = input_text
        self._tokens = np.array(
            self._tokenizer(input_text)["input_ids"][1:-1]
        )

        self.mask_strategy = mask_strategy
        self._mask_token_id = self._tokenizer.mask_token_id

        n_features = len(self._tokens)

        # dummy data (required by base class)
        dummy_data = np.zeros((1, n_features))

        super().__init__(
            model=self._classifier,
            data=dummy_data,
            x=self._tokens,
            sample_size=1,
        )

        # compute empty + full outputs
        self.empty_prediction = self._model_call(
            [self._decode(np.full_like(self._tokens, self._mask_token_id))]
        )[0]

        self.normalization_value = self.empty_prediction

    def _decode(self, tokens):
        return self._tokenizer.decode(tokens)

    def _model_call(self, texts):
        outputs = self._classifier(texts)
        return np.array([
            o["score"] if o["label"] == "POSITIVE" else -o["score"]
            for o in outputs
        ])

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        texts = []

        for coalition in coalitions:
            if self.mask_strategy == "remove":
                tokens = self._tokens[coalition]
            else:
                tokens = self._tokens.copy()
                tokens[~coalition] = self._mask_token_id

            texts.append(self._decode(tokens))

        return self._model_call(texts)