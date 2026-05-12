from __future__ import annotations

import numpy as np

from shapiq.game import Game


class JailbreakAnalysisGame(Game):
    """Game for jailbreak / prompt injection analysis."""

    def __init__(
        self,
        model_name: str,
        input_text: str,
        *,
        mask_strategy: str = "mask",
        segmentation: str = "token",
        device: int | str | None = None,
        batch_size: int = 32,
        verbose: bool = False,
        normalize: bool = True,
    ) -> None:

        from transformers import pipeline

        # =====================================================
        # Validate configuration
        # =====================================================

        if mask_strategy not in {"mask", "remove"}:
            msg = f"Invalid mask_strategy: {mask_strategy}"
            raise ValueError(msg)

        if segmentation not in {"token", "word"}:
            msg = f"Invalid segmentation: {segmentation}"
            raise ValueError(msg)

        self.model_name = model_name
        self.input_text = input_text
        self.mask_strategy = mask_strategy
        self.segmentation = segmentation
        self.batch_size = batch_size

        # =====================================================
        # Model
        # =====================================================

        self._classifier = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=device,
        )

        self._tokenizer = self._classifier.tokenizer

        self._mask_token_id = self._tokenizer.mask_token_id
        self._mask_token = getattr(
            self._tokenizer,
            "mask_token",
            "[MASK]",
        )

        # =====================================================
        # Build Players
        # =====================================================

        self._build_players()

        # =====================================================
        # Initialize Game
        # =====================================================

        super().__init__(
            n_players=len(self._players),
            normalize=normalize,
            normalization_value=0.0,
            verbose=verbose,
        )

    # =====================================================
    # Player Construction
    # =====================================================

    def _build_players(self) -> None:
        """Create players and token representation."""

        if self.segmentation == "token":

            tokens = self._tokenizer(
                self.input_text
            )["input_ids"][1:-1]

            self._tokens = np.array(tokens)

            self._players = self._tokens

        else:

            words = self.input_text.split()

            self._players = np.array(words)

            tokens = self._tokenizer(
                self.input_text
            )["input_ids"][1:-1]

            self._tokens = np.array(tokens)

    # =====================================================
    # Coalition -> Prompt
    # =====================================================

    def coalition_to_prompt(
        self,
        coalition: np.ndarray,
    ) -> str:
        """Convert coalition into a textual prompt.

        TODO:
            Convert coalition mask into prompt text Given different masking strategies.
        """

        pass

    # =====================================================
    # Model Call
    # =====================================================

    def model_call(
        self,
        prompts: list[str],
    ) -> np.ndarray:
        """Call model on prompt batch and compute scores.

        TODO:
        Calling a given model to compute a numeric score.

        """

        pass

    # =====================================================
    # Coalition Validation
    # =====================================================

    def _validate_coalition(
        self,
        coalition: np.ndarray,
    ) -> np.ndarray:

        coalition = np.asarray(
            coalition,
            dtype=bool,
        )

        if coalition.shape != (self.n_players,):

            msg = (
                f"Coalition must have shape "
                f"({self.n_players},), "
                f"got {coalition.shape}."
            )

            raise ValueError(msg)

        return coalition

    # =====================================================
    # Value Function
    # =====================================================

    def value_function(
        self,
        coalitions: np.ndarray,
    ) -> np.ndarray:
        """Evaluate coalitions."""

        prompts = [
            self.coalition_to_prompt(c)
            for c in coalitions
        ]

        outputs = self.model_call(prompts)

        return self.insert_empty_value(
            outputs,
            coalitions,
        )

    # =====================================================
    # Helper Properties
    # =====================================================

    @property
    def players(self) -> np.ndarray:
        return self._players.copy()
