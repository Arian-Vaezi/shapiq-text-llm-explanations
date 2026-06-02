"""SentimentDecoderGame — Shapley game for decoder-only causal LMs.

VALUE FUNCTION
--------------
Decoder models have no classification head — they cannot directly output
P(POSITIVE) or P(NEGATIVE). Instead we measure sentiment by asking:

    "How likely is the model to CONTINUE the text with positive phrases
     vs negative phrases?"

This is the contrastive log-odds:

    v(S) = mean_{t ∈ T+} log P(t | text(S))
         - mean_{t ∈ T-} log P(t | text(S))

Where:
    text(S)  = sentence with only words in coalition S (absent words removed)
    T+       = positive continuation templates
    T-       = negative continuation templates

Interpretation:
    v(S) > 0  → model finds positive continuations more likely → POSITIVE
    v(S) < 0  → model finds negative continuations more likely → NEGATIVE
    v(S) ≈ 0  → model is uncertain

MASKING STRATEGY
----------------
Absent words are REMOVED entirely (not replaced with [MASK]).
Decoder models were never trained with [MASK] tokens — removal keeps
the remaining text natural and grammatical.

Example:
    Full text:  "This film is not bad at all"
    Coalition:  {not, bad}
    Masked:     "not bad"  (other words removed)

NORMALIZATION
-------------
v(∅) is computed at init time (empty string — all words removed).
shapiq subtracts v(∅) from all coalition values so the baseline is 0.
Shapley values then sum to v(N) - v(∅): the total sentiment shift
from an empty prompt to the full sentence.

TEMPLATES
---------
2 positive + 2 negative domain-agnostic templates.
Domain-agnostic means no movie/product references — works on any text.
Taking the mean across templates reduces sensitivity to specific wording.
"""

from __future__ import annotations

import numpy as np

from shapiq.game import Game


class SentimentDecoderGame(Game):
    """Shapley interaction game for decoder sentiment analysis.

    Each word in the sentence is one player.
    Absent words are removed from the text (decoder-appropriate masking).
    The value function is contrastive log-odds over sentiment templates.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
        Supported: TinyLlama/TinyLlama-1.1B-Chat-v1.0,
                   google/gemma-3-1b-it
    input_text:
        The sentence to explain.
    hf_model:
        Pre-loaded HFModelWrapper. If None, loads from model_name.
        Always pass a pre-loaded model to avoid reloading weights.
    device:
        Inference device. Ignored if hf_model is provided.
    hf_token:
        HuggingFace token for gated models (e.g. Gemma).
    verbose:
        Whether to show shapiq progress bar.

    Examples
    --------
    >>> from demos.SentimentAnalysis.HFModelWrapper import HFModelWrapper
    >>> model = HFModelWrapper("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    >>> game = SentimentDecoderGame(
    ...     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ...     input_text="This film is not bad at all",
    ...     hf_model=model,
    ... )
    >>> game.n_players
    7
    >>> game.players
    array(['This', 'film', 'is', 'not', 'bad', 'at', 'all'], dtype='<U4')
    """

    # ── Sentiment templates ───────────────────────────────────────────────────
    # Domain-agnostic — no movie/product/news references.
    # Works on any input text.
    # Leading space is intentional — matches how causal LMs tokenize
    # a continuation that follows a sentence.
    #
    # Why 2 per class?
    #   - 2 templates averages out wording bias
    #   - Keeps forward passes manageable: 256 coalitions × 4 templates
    #   - Empirically: (not, bad) interaction still clearly detected
    #
    # To change to 1+1 for speed:
    #   POSITIVE_TEMPLATES = [" This is positive."]
    #   NEGATIVE_TEMPLATES = [" This is negative."]

    POSITIVE_TEMPLATES: list[str] = [
        " This is positive.",
        " I agree with this.",
    ]

    NEGATIVE_TEMPLATES: list[str] = [
        " This is negative.",
        " I disagree with this.",
    ]

    def __init__(
        self,
        model_name: str,
        input_text: str,
        *,
        hf_model=None,
        device: str | None = None,
        hf_token: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self.input_text = input_text

        # ── Load or reuse model ───────────────────────────────────────────────
        if hf_model is not None:
            self._model = hf_model
        else:
            from demos.SentimentAnalysis.HFModelWrapper import HFModelWrapper
            self._model = HFModelWrapper(
                model_name=model_name,
                device=device,
                hf_token=hf_token,
            )

        # ── Word-level players ────────────────────────────────────────────────
        self._words = np.array(input_text.split(), dtype=object)
        if len(self._words) == 0:
            raise ValueError("input_text must contain at least one word.")
        n_players = len(self._words)

        # ── Compute v(∅) BEFORE super().__init__ ─────────────────────────────
        # Empty coalition = empty string (all words removed).
        # This is the baseline — shapiq will subtract this from all values.
        v_empty = self._compute_score("")

        # ── Init Game ─────────────────────────────────────────────────────────
        super().__init__(
            n_players=n_players,
            normalize=True,
            normalization_value=v_empty,
            player_names=list(self._words.astype(str)),
            verbose=verbose,
        )

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def players(self) -> np.ndarray:
        """The word players in order."""
        return self._words.copy()

    # ── Masking ───────────────────────────────────────────────────────────────

    def _coalition_to_text(self, coalition: np.ndarray) -> str:
        """Convert a boolean coalition vector to a text string.

        Absent words are removed entirely — no [MASK] replacement.

        Args:
            coalition: Boolean array of shape (n_players,).
                True  = word is present.
                False = word is removed.

        Returns:
            Text string with only present words, joined by spaces.
            Returns empty string "" if no words are present.

        Example:
            Words:     ["This", "film", "is", "not", "bad", "at", "all"]
            Coalition: [F, F, F, T, T, F, F]
            Output:    "not bad"
        """
        present_words = self._words[np.asarray(coalition, dtype=bool)]
        return " ".join(present_words.astype(str))

    def _coalitions_to_texts(self, coalitions: np.ndarray) -> list[str]:
        """Convert a batch of coalition vectors to text strings.

        Args:
            coalitions: Boolean array of shape (n_coalitions, n_players).

        Returns:
            List of text strings, one per coalition.
        """
        return [self._coalition_to_text(c) for c in coalitions]

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _compute_score(self, text: str) -> float:
        """Compute contrastive sentiment score for a single text.

        Used only for computing v(∅) at init time.

        Args:
            text: Input text string (can be empty).

        Returns:
            Scalar contrastive score = mean(log P(T+)) - mean(log P(T-)).
        """
        # Score against all positive templates
        pos_scores = np.mean([
            self._model.score_continuations([text], t)[0]
            for t in self.POSITIVE_TEMPLATES
        ])

        # Score against all negative templates
        neg_scores = np.mean([
            self._model.score_continuations([text], t)[0]
            for t in self.NEGATIVE_TEMPLATES
        ])

        return float(pos_scores - neg_scores)

    def _compute_scores_batch(self, texts: list[str]) -> np.ndarray:
        """Compute contrastive scores for a batch of texts efficiently.

        Uses score_all_continuations() to batch all templates together
        instead of making separate calls per template.

        Steps:
            1. Score all texts against all positive templates in one call
            2. Score all texts against all negative templates in one call
            3. Return mean(pos) - mean(neg) per text

        Args:
            texts: List of text strings (one per coalition).

        Returns:
            np.ndarray of shape (len(texts),) with contrastive scores.
        """
        # Score against all positive templates
        # Shape: (n_pos_templates, n_texts)
        pos_matrix = self._model.score_all_continuations(
            texts,
            self.POSITIVE_TEMPLATES,
        )

        # Score against all negative templates
        # Shape: (n_neg_templates, n_texts)
        neg_matrix = self._model.score_all_continuations(
            texts,
            self.NEGATIVE_TEMPLATES,
        )

        # Mean across templates for each text
        pos_scores = pos_matrix.mean(axis=0)  # (n_texts,)
        neg_scores = neg_matrix.mean(axis=0)  # (n_texts,)

        return pos_scores - neg_scores

    # ── Value function ────────────────────────────────────────────────────────

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Compute v(S) for a batch of coalitions.

        This is the only method shapiq calls during approximation.

        Steps:
            1. Convert each coalition to a text string (remove absent words)
            2. Score all texts against positive and negative templates
            3. Return mean(log P(T+|text)) - mean(log P(T-|text)) per coalition

        Args:
            coalitions: Boolean array of shape (n_coalitions, n_players).

        Returns:
            np.ndarray of shape (n_coalitions,) with contrastive scores.
            Higher = more positive sentiment tendency.
        """
        texts = self._coalitions_to_texts(coalitions)
        return self._compute_scores_batch(texts)