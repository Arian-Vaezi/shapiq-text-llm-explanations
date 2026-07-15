from __future__ import annotations

import numpy as np

from shapiq.game import Game


class sentimentDecoderGame(Game):


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
     
        present_words = self._words[np.asarray(coalition, dtype=bool)]
        return " ".join(present_words.astype(str))

    def _coalitions_to_texts(self, coalitions: np.ndarray) -> list[str]:
        """Convert a batch of coalition vectors to text strings."""
        return [self._coalition_to_text(c) for c in coalitions]

    # ── Scoring ───────────────────────────────────────────────────────────────

    def _compute_score(self, text: str) -> float:
     
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

        pos_matrix = self._model.score_all_continuations(texts, self.POSITIVE_TEMPLATES)
        neg_matrix = self._model.score_all_continuations(texts, self.NEGATIVE_TEMPLATES)

        pos_scores = pos_matrix.mean(axis=0)
        neg_scores = neg_matrix.mean(axis=0)

        return pos_scores - neg_scores

    # ── Value function ────────────────────────────────────────────────────────

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
 
        texts = self._coalitions_to_texts(coalitions)
        return self._compute_scores_batch(texts)