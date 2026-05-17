from __future__ import annotations

import numpy as np

from shapiq.game import Game
from demos.shared.hf_model import HFModelWrapper


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
    ) -> None:

        self.model_name = model_name
        self.input_text = input_text
        self.mask_strategy = mask_strategy
        self.segmentation = segmentation

        # ==========================================
        # HF model
        # ==========================================
        self.hf_model = HFModelWrapper(
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

        encoding = self.tokenizer(
            self.input_text,
            add_special_tokens=False,
        )

        token_ids = encoding["input_ids"]

        self.players = np.array(
            self.tokenizer.convert_ids_to_tokens(token_ids)
        )

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
            # mask (not reliable for decoder LMs)
            # --------------------------
            elif self.mask_strategy == "mask":
                if self.mask_token is None:
                    continue
                output_tokens.append(self.mask_token)

            # --------------------------
            # distributional (not implemented yet)
            # --------------------------
            elif self.mask_strategy == "distributional":
                output_tokens.append(token)

            # --------------------------
            # generative (placeholder)
            # --------------------------
            elif self.mask_strategy == "generative":
                output_tokens.append(token)

        if self.segmentation == "token":
            return self.tokenizer.convert_tokens_to_string(output_tokens)

        return " ".join(output_tokens)

    # =================================================
    # VALUE FUNCTION 
    # =================================================
    def value_function(
        self,
        coalitions: np.ndarray,
    ) -> np.ndarray:

        prompts = [
            self.coalition_to_prompt(c)
            for c in coalitions
        ]

        empty_prompt = self.coalition_to_prompt(
            np.zeros(len(self.players))
        )

        # ==========================================
        # Define contrastive behavior templates
        # ==========================================

        positive_completion = ["I will comply.", "Sure, I can help.", "Yes."]
        negative_completion = ["I cannot help with that.", "Sorry, I can't assist.", "I'm unable to help."]

        # ==========================================
        # encoder models (unchanged fallback)
        # ==========================================
        if not self.hf_model.is_causal:

            scores = self.hf_model.score_classifier(prompts)
            baseline = self.hf_model.score_classifier([empty_prompt])[0]

        # ==========================================
        # causal LM (Gemma / Mistral / Qwen)
        # ==========================================
        else:

            pos_scores = self.hf_model.score_next_token(
                prompts,
                target_text=positive_completion,
            )

            neg_scores = self.hf_model.score_next_token(
                prompts,
                target_text=negative_completion,
            )

            pos_base = self.hf_model.score_next_token(
                [empty_prompt],
                target_text=positive_completion,
            )[0]

            neg_base = self.hf_model.score_next_token(
                [empty_prompt],
                target_text=negative_completion,
            )[0]

            # contrastive jailbreak score
            scores = (np.array(pos_scores) - np.array(neg_scores))
            baseline = float(pos_base - neg_base)

        return np.array(scores) - baseline