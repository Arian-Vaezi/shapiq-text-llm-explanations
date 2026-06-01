from __future__ import annotations

import numpy as np

from demos.shared.causal_model_wrapper import CausalModelWrapper
from demos.shared.encoder_model_wrapper import EncoderModelWrapper
from demos.shared.embedding_model_wrapper import EmbeddingModelWrapper
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
        # Reuse a pre-loaded causal/encoder model (skips reload)
        hf_model: CausalModelWrapper | EncoderModelWrapper | None = None,
        # Semantic segmentation settings (ignored unless segmentation="semantic")
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        semantic_threshold: float = 0.5,
        semantic_window: int = 6,
        semantic_min_segment_words: int = 3,
    ) -> None:
        self.model_name = model_name
        self.input_text = input_text
        self.mask_strategy = mask_strategy
        self.segmentation = segmentation
        self.batch_size = batch_size
        self.semantic_threshold = semantic_threshold
        self.semantic_window = semantic_window
        self.semantic_min_segment_words = semantic_min_segment_words

        # ==========================================
        # Main scoring model
        # ==========================================
        self.hf_model = hf_model or HFModelWrapper(
            model_name=model_name,
            device=device,
        )

        # ==========================================
        # Embedding model — only loaded when needed
        # ==========================================
        self.embedding_model: EmbeddingModelWrapper | None = None
        if self.segmentation == "semantic":
            self.embedding_model = EmbeddingModelWrapper(
                model_name=embedding_model_name,
                device=device or "cuda",
            )

        self.tokenizer = self.hf_model.tokenizer
        self.mask_token = self.tokenizer.mask_token

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

        if self.segmentation == "semantic":
            self.players = np.array(self._semantic_segments())
            return

        # token-level (default)
        encoding = self.tokenizer(self.input_text, add_special_tokens=False)
        token_ids = encoding["input_ids"]
        self.players = np.array(self.tokenizer.convert_ids_to_tokens(token_ids))

    # =================================================
    # Semantic segmentation
    # =================================================

    def _semantic_segments(self) -> list[str]:
        """Word-granularity segmentation via sliding-window cosine similarity.

        Algorithm:
          1. For each word position i, embed the surrounding window of words
             (window=semantic_window) to get a context-aware representation.
          2. Compute cosine similarity between adjacent window embeddings.
             (dot product of L2-normalized vectors)
          3. Split wherever sim < semantic_threshold.
          4. Merge segments shorter than semantic_min_segment_words into
             the previous segment to avoid isolated function-word segments.

        Tuning:
          - semantic_threshold: lower → fewer splits. Start at 0.5, adjust by 0.05.
          - semantic_window:    larger → smoother similarities, fewer spurious splits.
          - semantic_min_segment_words: safety net for isolated function words.
        """
        assert self.embedding_model is not None  # guaranteed by __init__

        words = self.input_text.split()
        if len(words) <= 1:
            return words

        half = self.semantic_window // 2
        windows = [
            " ".join(words[max(0, i - half) : i + half + 1])
            for i in range(len(words))
        ]

        # shape: (n_words, hidden_dim) — L2-normalized → dot == cosine sim
        embeddings = self.embedding_model.encode(windows)

        similarities = [
            float(np.dot(embeddings[i], embeddings[i + 1]))
            for i in range(len(embeddings) - 1)
        ]

        # Split at similarity drops
        blocks: list[str] = []
        current: list[str] = [words[0]]
        for i, sim in enumerate(similarities):
            if sim < self.semantic_threshold:
                blocks.append(" ".join(current))
                current = [words[i + 1]]
            else:
                current.append(words[i + 1])
        blocks.append(" ".join(current))

        # Merge short segments (isolated function words) into previous
        merged: list[str] = []
        for block in blocks:
            if merged and len(block.split()) < self.semantic_min_segment_words:
                merged[-1] = merged[-1] + " " + block
            else:
                merged.append(block)

        return merged

    # =================================================
    # Coalition -> Prompt
    # =================================================

    def coalition_to_prompt(self, coalition: np.ndarray) -> str:
        output_tokens = []

        for present, token in zip(coalition, self.players, strict=True):
            if present:
                output_tokens.append(token)
                continue

            if self.mask_strategy == "remove":
                continue

            if self.mask_strategy == "mask":
                if self.mask_token is not None:
                    output_tokens.append(self.mask_token)

            elif self.mask_strategy in ["distributional", "generative"]:
                output_tokens.append(token)

        if self.segmentation == "token":
            return self.tokenizer.convert_tokens_to_string(output_tokens)

        return " ".join(output_tokens)

    # =================================================
    # Batching helpers
    # =================================================

    def _batch(self, prompts: list[str]) -> list[list[str]]:
        return [prompts[i : i + self.batch_size] for i in range(0, len(prompts), self.batch_size)]

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
    # Model call
    # =================================================

    def batched_model_call(self, prompts: list[str], empty_prompt: str) -> np.ndarray:
        positive_templates = ["I will comply.", "Sure, I can help.", "Yes."]
        negative_templates = ["I cannot help with that.", "Sorry, I can't assist.", "I'm unable to help."]

        if not self.hf_model.is_causal:
            scores = np.concatenate(
                [self.hf_model.score_classifier(batch) for batch in self._batch(prompts)],
                axis=0,
            )
            baseline = self.hf_model.score_classifier([empty_prompt])[0]
        else:
            pos_scores = self._score_set(prompts, positive_templates)
            neg_scores = self._score_set(prompts, negative_templates)
            pos_base = self._score_set([empty_prompt], positive_templates)
            neg_base = self._score_set([empty_prompt], negative_templates)
            scores = pos_scores - neg_scores
            baseline = (pos_base - neg_base).item()

        return np.array(scores) - baseline

    # =================================================
    # Value function
    # =================================================

    def value_function(self, coalitions: np.ndarray) -> np.ndarray:
        prompts = [str(self.coalition_to_prompt(c)) for c in coalitions]
        empty_prompt = str(self.coalition_to_prompt(np.zeros(len(self.players))))
        return self.batched_model_call(prompts, empty_prompt)