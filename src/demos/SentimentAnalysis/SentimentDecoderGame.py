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

TEMPLATES — LANGUAGE AND REGISTER OPTIONS
-------------------------------------------
Templates MUST match the language of the input sentence. Scoring a German
sentence's continuation against an English template (" This is positive.")
introduces a code-switching confound — you'd be measuring "can the model
switch to English fluently" mixed in with "does it understand sentiment."

Two independent options are available:

    language : "en" | "fr" | "de"        (must match the input sentence)
    register : "formal" | "informal"     (phrasing style)

Defaults are language="en", register="formal" — this preserves the exact
behavior used by the Streamlit app (English-only), so existing app code
is completely unaffected by these additions.

For the multilingual experiments script, pass the sentence's actual
language explicitly:

    SentimentDecoderGame(..., language="de", register="formal")

NORMALIZATION
-------------
v(∅) is computed at init time (empty string — all words removed).
shapiq subtracts v(∅) from all coalition values so the baseline is 0.
"""

from __future__ import annotations

import numpy as np

from shapiq.game import Game


class SentimentDecoderGame(Game):
    """Shapley interaction game for decoder sentiment analysis.

    Each word in the sentence is one player.
    Absent words are removed from the text (decoder-appropriate masking).
    The value function is contrastive log-odds over sentiment templates,
    selectable by language and register.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
    input_text:
        The sentence to explain.
    hf_model:
        Pre-loaded HFModelWrapper. If None, loads from model_name.
    language:
        Template language: "en", "fr", or "de". Defaults to "en".
        MUST match the language of input_text to avoid a code-switching
        confound in the value function.
    register:
        Template phrasing style: "formal" or "informal". Defaults to "formal".
    device:
        Inference device. Ignored if hf_model is provided.
    hf_token:
        HuggingFace token for gated models.
    verbose:
        Whether to show a shapiq progress bar.

    Examples
    --------
    >>> # English (app default — unchanged behavior)
    >>> game = SentimentDecoderGame(
    ...     model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ...     input_text="This film is not bad at all",
    ...     hf_model=model,
    ... )

    >>> # German, formal register (for multilingual experiments)
    >>> game = SentimentDecoderGame(
    ...     model_name="Qwen/Qwen3.5-9B-Instruct",
    ...     input_text="Dieser Film ist überhaupt nicht schlecht.",
    ...     hf_model=model,
    ...     language="de",
    ...     register="formal",
    ... )
    """

    # ── Templates by language and register ────────────────────────────────────
    # Each (language, register) pair gives 2 positive + 2 negative templates.
    # Leading space matches how causal LMs tokenize a continuation following
    # a sentence.
    #
    # formal   : neutral, declarative sentiment statements
    # informal : exclamatory, colloquial phrasing
    #
    # IMPORTANT: language must match the input sentence's language, or the
    # value function conflates sentiment understanding with code-switching.

    TEMPLATES_BY_LANG: dict[str, dict[str, dict[str, list[str]]]] = {
        "en": {
            "formal": {
                "pos": [" This is positive.", " I agree with this."],
                "neg": [" This is negative.", " I disagree with this."],
            },
            "informal": {
                "pos": [" That's great!", " I love it."],
                "neg": [" That's awful!", " I hate it."],
            },
        },
        "fr": {
            "formal": {
                "pos": [" C'est positif.", " Je suis d'accord avec cela."],
                "neg": [" C'est négatif.", " Je ne suis pas d'accord avec cela."],
            },
            "informal": {
                "pos": [" C'est génial !", " J'adore ça."],
                "neg": [" C'est nul !", " Je déteste ça."],
            },
        },
        "de": {
            "formal": {
                "pos": [" Das ist positiv.", " Dem stimme ich zu."],
                "neg": [" Das ist negativ.", " Dem stimme ich nicht zu."],
            },
            "informal": {
                "pos": [" Das ist toll!", " Ich liebe es."],
                "neg": [" Das ist furchtbar!", " Ich hasse es."],
            },
        },
    }

    def __init__(
        self,
        model_name: str,
        input_text: str,
        *,
        hf_model=None,
        language: str = "en",
        register: str = "formal",
        device: str | None = None,
        hf_token: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.model_name = model_name
        self.input_text = input_text

        # ── Resolve templates from language + register ───────────────────────
        if language not in self.TEMPLATES_BY_LANG:
            msg = (
                f"Unsupported language {language!r}. "
                f"Available: {list(self.TEMPLATES_BY_LANG.keys())}"
            )
            raise ValueError(msg)
        if register not in self.TEMPLATES_BY_LANG[language]:
            msg = (
                f"Unsupported register {register!r} for language {language!r}. "
                f"Available: {list(self.TEMPLATES_BY_LANG[language].keys())}"
            )
            raise ValueError(msg)

        self.language = language
        self.register = register
        templates = self.TEMPLATES_BY_LANG[language][register]
        self.POSITIVE_TEMPLATES: list[str] = templates["pos"]
        self.NEGATIVE_TEMPLATES: list[str] = templates["neg"]

        # ── Load or reuse model ───────────────────────────────────────────────
        if hf_model is not None:
            self._model = hf_model
        else:
            from demos.shared.hf_model import HFModelWrapper
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
        v_empty = self._compute_score("")

        # ── Init Game base class ──────────────────────────────────────────────
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

        Returns:
            Text string with only present words, joined by spaces.
        """
        present_words = self._words[np.asarray(coalition, dtype=bool)]
        return " ".join(present_words.astype(str))

    def _coalitions_to_texts(self, coalitions: np.ndarray) -> list[str]:
        """Convert a batch of coalition vectors to text strings."""
        return [self._coalition_to_text(c) for c in coalitions]

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _compute_score(self, text: str) -> float:
        """Compute contrastive sentiment score for a single text.

        Used for computing v(∅) at init time.

        Args:
            text: Input text string (can be empty).

        Returns:
            Scalar contrastive score = mean(log P(T+)) - mean(log P(T-)).
        """
        pos_scores = np.mean([
            self._model.score_continuations([text], t)[0]
            for t in self.POSITIVE_TEMPLATES
        ])
        neg_scores = np.mean([
            self._model.score_continuations([text], t)[0]
            for t in self.NEGATIVE_TEMPLATES
        ])
        return float(pos_scores - neg_scores)

    def _compute_scores_batch(self, texts: list[str]) -> np.ndarray:
        """Compute contrastive scores for a batch of texts efficiently.

        Args:
            texts: List of text strings (one per coalition).

        Returns:
            np.ndarray of shape (len(texts),) with contrastive scores.
        """
        pos_matrix = self._model.score_all_continuations(texts, self.POSITIVE_TEMPLATES)
        neg_matrix = self._model.score_all_continuations(texts, self.NEGATIVE_TEMPLATES)

        pos_scores = pos_matrix.mean(axis=0)
        neg_scores = neg_matrix.mean(axis=0)

        return pos_scores - neg_scores

    # ── Value function ────────────────────────────────────────────────────────

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        """Compute v(S) for a batch of coalitions.

        Args:
            coalitions: Boolean array of shape (n_coalitions, n_players).

        Returns:
            np.ndarray of shape (n_coalitions,) with contrastive scores.
        """
        texts = self._coalitions_to_texts(coalitions)
        return self._compute_scores_batch(texts)