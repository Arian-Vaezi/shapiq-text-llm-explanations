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

        self.tokenizer = (
            self.hf_model.tokenizer
        )

        self.mask_token = (
            self.tokenizer.mask_token
        )

        # ==========================================
        # Players
        # ==========================================

        self._build_players()

        super().__init__(
            n_players=len(
                self.players
            ),
            normalize=normalize,
            normalization_value=0.0,
            verbose=verbose,
        )

    # =================================================
    # Players
    # =================================================

    def _build_players(
        self,
    ) -> None:

        # word-level
        if self.segmentation == "word":

            self.players = np.array(
                self.input_text.split()
            )

            return

        # token-level

        encoding = self.tokenizer(
            self.input_text,
            add_special_tokens=False,
        )

        token_ids = encoding[
            "input_ids"
        ]

        self.players = np.array(
            self.tokenizer
            .convert_ids_to_tokens(
                token_ids
            )
        )

    # =================================================
    # Coalition -> Prompt
    # =================================================

    def coalition_to_prompt(
        self,
        coalition: np.ndarray,
    ) -> str:

        output_tokens = []

        for present, token in zip(
            coalition,
            self.players,
            strict=True,
        ):

            # present player

            if present:

                output_tokens.append(
                    token
                )

                continue

            # ======================================
            # remove
            # ======================================

            if (
                self.mask_strategy
                == "remove"
            ):
                continue

            # ======================================
            # mask
            # ======================================

            elif (
                self.mask_strategy
                == "mask"
            ):

                if (
                    self.mask_token
                    is None
                ):

                    raise ValueError(
                        "Model has no mask token."
                    )

                output_tokens.append(
                    self.mask_token
                )

            # ======================================
            # distributional
            # ======================================

            elif (
                self.mask_strategy
                == "distributional"
            ):

                output_tokens.append(
                    token
                )

            # ======================================
            # generative
            # ======================================

            elif (
                self.mask_strategy
                == "generative"
            ):

                output_tokens.append(
                    token
                )

        # token segmentation

        if self.segmentation == "token":

            return (
                self.tokenizer
                .convert_tokens_to_string(
                    output_tokens
                )
            )

        return " ".join(
            output_tokens
        )

    # =================================================
    # Value Function
    # =================================================

    def value_function(
        self,
        coalitions: np.ndarray,
    ) -> np.ndarray:

        prompts = [
            self.coalition_to_prompt(c)
            for c in coalitions
        ]

        # ==========================================
        # encoder classifier
        # ==========================================

        if not self.hf_model.is_causal:

            scores = (
                self.hf_model
                .score_classifier(
                    prompts
                )
            )

        # ==========================================
        # causal LM
        # ==========================================

        else:

            scores = (
                self.hf_model
                .score_next_token(
                    prompts,
                    target_token="Sure",
                )
            )

        return np.array(
            scores
        )