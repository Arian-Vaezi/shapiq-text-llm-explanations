"""SentimentDecoderGame — Sentiment analysis game for decoder-only causal LMs.

This module adapts the JailbreakGame infrastructure for sentiment analysis.
Instead of measuring compliance, we measure sentiment using contrastive
log-odds between positive and negative completion templates.

Key difference from encoder approach (TextImputer + DistilBERT):
    - Encoder: coalition → [MASK] masking → classification score
    - Decoder: coalition → word removal → P(positive) - P(negative)

The value function is:
    v(coalition) = mean(log P(pos_template | masked_text))
                 - mean(log P(neg_template | masked_text))

This gives a scalar score comparable to the encoder approach:
    - Positive values → model predicts positive sentiment
    - Negative values → model predicts negative sentiment
    - Zero → neutral / uncertain

Why contrastive log-odds?
    Decoder models have no classification head. We force sentiment
    by measuring how likely the model is to continue with positive
    vs negative completions. Taking the difference removes the
    baseline language model bias.

Usage:
    from demos.SentimentAnalysis.SentimentDecoderGame import SentimentDecoderGame
    from demos.shared.hf_model import HFModelWrapper

    model = HFModelWrapper('google/gemma-3-1b-it', device='cuda')
    game = SentimentDecoderGame(
        model_name='google/gemma-3-1b-it',
        input_text='This film is not bad at all',
        hf_model=model,
    )
"""

from __future__ import annotations

import re

import numpy as np

from demos.shared.hf_model import HFModelWrapper
from shapiq.game import Game


class SentimentDecoderGame(Game):
    """Sentiment analysis game for decoder-only causal language models.

    Treats each word as a player. The value of a coalition is the
    contrastive log-odds score between positive and negative sentiment
    completion templates, evaluated on the masked input text.

    Attributes:
        model_name: HuggingFace model identifier.
        input_text: The original input sentence to explain.
        mask_strategy: How to handle absent words — 'remove' or 'mask'.
        segmentation: Granularity of players — 'word', 'token', 'sentence'.
        batch_size: Number of prompts per model call.
        hf_model: The loaded HFModelWrapper instance.
        players: Array of player strings (words/tokens/sentences).

    Example:
        >>> game = SentimentDecoderGame(
        ...     model_name='google/gemma-3-1b-it',
        ...     input_text='This film is not bad at all',
        ... )
        >>> game.n_players
        7
    """

    # ── Sentiment completion templates ────────────────────────────────────────
    # These are fed to score_next_token() to measure sentiment direction.
    # The contrastive score = mean(pos) - mean(neg).
    # Chosen to be short, unambiguous, and domain-appropriate for movie reviews.
    POSITIVE_COMPLETIONS = [
        " This is positive.",
        " I loved it.",
        " Excellent!",
        " Great film.",
    ]

    NEGATIVE_COMPLETIONS = [
        " This is negative.",
        " I hated it.",
        " Terrible!",
        " Awful film.",
    ]

    def __init__(
        self,
        model_name: str,
        input_text: str,
        *,
        mask_strategy: str = "remove",   # 'remove' is safer for decoder models
        segmentation: str = "word",       # word-level for linguistic analysis
        device: str | int | None = None,
        normalize: bool = True,
        verbose: bool = False,
        batch_size: int = 8,              # smaller batches — decoder models are larger
        hf_model: HFModelWrapper | None = None,
        hf_token: str | None = None,
    ) -> None:
        """Initialize the SentimentDecoderGame.

        Args:
            model_name: HuggingFace model identifier string.
            input_text: The sentence to explain.
            mask_strategy: 'remove' (drop absent words) or 'mask' (replace
                with mask token if available). Defaults to 'remove' since
                decoder models handle removal more naturally than [MASK].
            segmentation: Player granularity — 'word', 'token', or 'sentence'.
                Defaults to 'word' for linguistic interpretability.
            device: Device for inference. 'cuda' auto-falls back to mps/cpu.
            normalize: Whether to normalize by the empty coalition value.
            verbose: Print progress information.
            batch_size: Coalitions per model call. Keep small for large models.
            hf_model: Pre-loaded HFModelWrapper — reuse to avoid reloading.
            hf_token: HuggingFace token for gated models like Gemma.
        """
        if mask_strategy not in {"remove", "mask"}:
            msg = f"Invalid mask_strategy: {mask_strategy!r}. Use 'remove' or 'mask'."
            raise ValueError(msg)

        if segmentation not in {"word", "token", "sentence"}:
            msg = f"Invalid segmentation: {segmentation!r}."
            raise ValueError(msg)

        self.model_name    = model_name
        self.input_text    = input_text
        self.mask_strategy = mask_strategy
        self.segmentation  = segmentation
        self.batch_size    = batch_size

        # ── Load or reuse HFModelWrapper ──────────────────────────────────────
        # Reusing a pre-loaded model avoids the ~30s reload cost on every call.
        self.hf_model = hf_model or HFModelWrapper(
            model_name=model_name,
            device=device or "cuda",  # auto-falls back to mps/cpu
            hf_token=hf_token,
        )
        self.tokenizer = self.hf_model.tokenizer
        self.mask_token = getattr(self.tokenizer, "mask_token", None)

        # ── Build players from input text ─────────────────────────────────────
        self._build_players()

        # ── Initialize shapiq Game base class ─────────────────────────────────
        super().__init__(
            n_players=len(self.players),
            normalize=normalize,
            normalization_value=0.0,
            verbose=verbose,
        )

    # ── Player construction ───────────────────────────────────────────────────

    def _build_players(self) -> None:
        """Build the players array based on the segmentation strategy.

        Word segmentation: splits on whitespace — each word is one player.
        Token segmentation: uses the tokenizer — each token is one player.
        Sentence segmentation: splits on punctuation — each sentence is one player.
        """
        if self.segmentation == "word":
            # Whitespace split — consistent with TextImputer word segmentation
            self.players = np.array(self.input_text.split())

        elif self.segmentation == "sentence":
            sentences = re.split(r"(?<=[.!?])\s+", self.input_text.strip())
            self.players = np.array([s for s in sentences if s])

        else:  # token
            encoding  = self.tokenizer(self.input_text, add_special_tokens=False)
            token_ids = encoding["input_ids"]
            self.players = np.array(self.tokenizer.convert_ids_to_tokens(token_ids))

    # ── Coalition → masked text ───────────────────────────────────────────────

    def coalition_to_text(self, coalition: np.ndarray) -> str:
        """Convert a binary coalition mask to a masked text string.

        For 'remove' strategy: absent words are dropped entirely.
        For 'mask' strategy: absent words are replaced with [MASK] token.

        Args:
            coalition: Boolean array of shape (n_players,).

        Returns:
            The masked text string to feed to the model.
        """
        output_tokens = []

        for present, token in zip(coalition, self.players, strict=True):
            if present:
                output_tokens.append(token)
            elif self.mask_strategy == "mask" and self.mask_token:
                output_tokens.append(self.mask_token)
            # else: remove — skip the token entirely

        if self.segmentation == "token":
            return self.tokenizer.convert_tokens_to_string(output_tokens)

        return " ".join(output_tokens)

    # ── Batched scoring ───────────────────────────────────────────────────────

    def _score_completions(
        self,
        prompts: list[str],
        templates: list[str],
    ) -> np.ndarray:
        """Score a list of prompts against completion templates.

        For each prompt, computes the mean log-likelihood across all templates.
        Batches the model calls for efficiency.

        Args:
            prompts: List of masked text strings.
            templates: List of completion strings to score against.

        Returns:
            Array of shape (n_prompts,) with mean log-likelihood scores.
        """
        # Average log-likelihood across all templates for each prompt
        all_scores = []
        for template in templates:
            template_scores = []
            # Process in batches
            for i in range(0, len(prompts), self.batch_size):
                batch = prompts[i : i + self.batch_size]
                scores = self.hf_model.score_next_token(batch, template)
                template_scores.extend(scores.tolist())
            all_scores.append(template_scores)

        # Mean across templates — shape (n_prompts,)
        return np.mean(all_scores, axis=0)

    # ── Value function ────────────────────────────────────────────────────────

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Compute the sentiment score for each coalition.

        The score is the contrastive log-odds:
            v(S) = mean_pos_log_prob(S) - mean_neg_log_prob(S)

        Positive values indicate positive sentiment tendency.
        Negative values indicate negative sentiment tendency.

        This is normalized by the empty coalition baseline when normalize=True,
        so the empty coalition maps to 0.

        Args:
            coalitions: Boolean array of shape (n_coalitions, n_players).

        Returns:
            Float array of shape (n_coalitions,) with sentiment scores.
        """
        # Convert each coalition to a masked text prompt
        prompts = [self.coalition_to_text(c) for c in coalitions]

        # Score against positive and negative templates
        pos_scores = self._score_completions(prompts, self.POSITIVE_COMPLETIONS)
        neg_scores = self._score_completions(prompts, self.NEGATIVE_COMPLETIONS)

        # Contrastive log-odds — positive = positive sentiment
        return pos_scores - neg_scores