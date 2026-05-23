from __future__ import annotations

import numpy as np

from demos.shared.hf_model import HFModelWrapper
from shapiq.game import Game


class JailbreakGame(Game):
    def __init__(
        self,
        model_name: str,
        input_text: str,
        *,
        mask_strategy: str = "remove",
        segmentation: str = "token",
        device: str | int | None = None,
        normalize: bool = True,
        verbose: bool = False,
        batch_size: int = 32,
        hf_model: HFModelWrapper | None = None,
    ) -> None:
        self.model_name = model_name
        self.input_text = input_text
        self.mask_strategy = mask_strategy
        self.segmentation = segmentation
        self.batch_size = batch_size

        # ==========================================
        # HF model — reuse pre-loaded instance if provided
        # ==========================================
        self.hf_model = hf_model or HFModelWrapper(
            model_name=model_name,
            device=device,
        )

        self.tokenizer = self.hf_model.tokenizer
        self.mask_token = self.tokenizer.mask_token

        # ==========================================
        # Players
        # ==========================================
        self._build_players()

        super().__init__(
            n_players=len(self.players),
            normalize=normalize,
            normalization_value=0.0,
            verbose=verbose,
        )

    # =================================================
    # Players
    # =================================================
    def _build_players(self) -> None:
        if self.segmentation == "word":
            self.players = np.array(self.input_text.split())
            return

        if self.segmentation == "sentence":
            import re

            sentences = re.split(r"(?<=[.!?])\s+", self.input_text.strip())
            self.players = np.array([s for s in sentences if s])
            return

        encoding = self.tokenizer(
            self.input_text,
            add_special_tokens=False,
        )

        token_ids = encoding["input_ids"]

        self.players = np.array(self.tokenizer.convert_ids_to_tokens(token_ids))

    # =================================================
    # Coalition -> Prompt
    # =================================================
    def coalition_to_prompt(self, coalition: np.ndarray) -> str:
        output_tokens = []

        for present, token in zip(coalition, self.players, strict=True):
            if present:
                output_tokens.append(token)
                continue

            # --------------------------
            # remove
            # --------------------------
            if self.mask_strategy == "remove":
                continue

            # --------------------------
            # mask
            # --------------------------
            if self.mask_strategy == "mask":
                if self.mask_token is None:
                    continue
                output_tokens.append(self.mask_token)

            elif self.mask_strategy in ["distributional", "generative"]:
                output_tokens.append(token)

        if self.segmentation == "token":
            return self.tokenizer.convert_tokens_to_string(output_tokens)

        return " ".join(output_tokens)

    # =================================================
    # helper: split prompts into batches
    # =================================================
    def _batch(self, prompts: list[str]) -> list[list[str]]:
        return [prompts[i : i + self.batch_size] for i in range(0, len(prompts), self.batch_size)]

    # =================================================
    # helper: score template set (batched)
    # =================================================
    def _score_set(self, prompts: list[str], templates: list[str]) -> np.ndarray:
        results = []
        for batch in self._batch(prompts):
            batch_scores = np.mean(
                [self.hf_model.score_next_token(batch, t) for t in templates],
                axis=0,
            )
            results.append(batch_scores)
        return np.concatenate(results, axis=0)

    # =================================================
    # helper: batched model call
    # =================================================
    def batched_model_call(self, prompts: list[str], empty_prompt: str) -> np.ndarray:
        positive_completion = [
            "I will comply.",
            "Sure, I can help.",
            "Yes.",
        ]

        negative_completion = [
            "I cannot help with that.",
            "Sorry, I can't assist.",
            "I'm unable to help.",
        ]

        # ==========================================
        # encoder models
        # ==========================================
        if not self.hf_model.is_causal:
            scores = np.concatenate(
                [self.hf_model.score_classifier(batch) for batch in self._batch(prompts)],
                axis=0,
            )
            baseline = self.hf_model.score_classifier([empty_prompt])[0]

        # ==========================================
        # causal LM (Gemma / Mistral / Qwen)
        # ==========================================
        else:
            pos_scores = self._score_set(prompts, positive_completion)
            neg_scores = self._score_set(prompts, negative_completion)

            pos_base = self._score_set([empty_prompt], positive_completion)
            neg_base = self._score_set([empty_prompt], negative_completion)

            scores = pos_scores - neg_scores
            baseline = (pos_base - neg_base).item()

        return np.array(scores) - baseline

    # =================================================
    # VALUE FUNCTION
    # =================================================
    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        prompts = [str(self.coalition_to_prompt(c)) for c in coalitions]
        empty_prompt = str(self.coalition_to_prompt(np.zeros(len(self.players))))
        return self.batched_model_call(prompts, empty_prompt)
