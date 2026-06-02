"""HFModelWrapper — unified wrapper for causal language models.

Supports:
    - TinyLlama/TinyLlama-1.1B-Chat-v1.0
    - google/gemma-3-1b-it

Key design decisions:
    - True batching with left-padding for causal LMs
      (right-padding breaks autoregressive scoring)
    - float16 on CUDA/MPS, float32 on CPU
    - Device auto-detection with graceful fallback
    - score_continuations() is the only method SentimentDecoderGame needs
"""

from __future__ import annotations

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFModelWrapper:
    """Wrapper for causal language models with batched continuation scoring.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier.
        Supported: TinyLlama/TinyLlama-1.1B-Chat-v1.0,
                   google/gemma-3-1b-it
    device:
        Inference device. If None, auto-selects CUDA > MPS > CPU.
    hf_token:
        HuggingFace token for gated models (e.g. Gemma).
    batch_size:
        Number of prompts per forward pass. Defaults to 32.

    Examples
    --------
    >>> model = HFModelWrapper("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    >>> scores = model.score_continuations(
    ...     prompts=["The film is great", "The film is terrible"],
    ...     continuation=" This is positive.",
    ... )
    >>> scores.shape
    (2,)
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        hf_token: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size

        # ── Device selection ──────────────────────────────────────────────────
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[HFModelWrapper] {model_name} → device: {self.device}")

        # ── Dtype selection ───────────────────────────────────────────────────
        # float16 on GPU (faster, less memory)
        # float32 on CPU (float16 not well supported on CPU)
        dtype = (
            torch.float16
            if self.device in ("cuda", "mps")
            else torch.float32
        )

        # ── Tokenizer ─────────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            use_fast=True,
        )

        # Causal LMs need a pad token — use eos if not defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ── Model ─────────────────────────────────────────────────────────────
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=None,
            token=hf_token,
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"[HFModelWrapper] Ready. "
              f"Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB used"
              if self.device == "cuda" else
              f"[HFModelWrapper] Ready on {self.device}.")

    # ── Core scoring method ───────────────────────────────────────────────────

    @torch.inference_mode()
    def score_continuations(
        self,
        prompts: list[str],
        continuation: str,
    ) -> np.ndarray:
        """Compute log P(continuation | prompt) for a batch of prompts.

        Uses true batching with LEFT padding — critical for causal LMs.
        Right-padding would corrupt the autoregressive scores because the
        model would attend to padding tokens before the actual content.

        The score is the sum of log-probabilities of all continuation tokens
        given the prompt tokens:

            score(prompt) = Σ_{t ∈ continuation} log P(token_t | prompt + tokens_before_t)

        Args:
            prompts: List of context strings to score against.
            continuation: The string whose likelihood we measure.
                          Should start with a space e.g. " This is positive."

        Returns:
            np.ndarray of shape (len(prompts),) with log-likelihood scores.
            Higher = model finds the continuation more likely given the prompt.
        """
        if not prompts:
            return np.array([], dtype=float)

        # How many tokens does the continuation have?
        # We need this to know which positions to score.
        continuation_ids = self.tokenizer(
            continuation,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        n_continuation = continuation_ids.shape[1]

        # Build full texts: each prompt concatenated with the continuation
        full_texts = [p + continuation for p in prompts]

        # Tokenize with LEFT padding
        # LEFT padding is required for causal LMs — the model reads left to
        # right, so padding must come before the actual content, not after.
        original_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"

        max_len = min(
            getattr(self.tokenizer, "model_max_length", 2048),
            2048,
        )

        encoded = self.tokenizer(
            full_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(self.device)

        self.tokenizer.padding_side = original_side

        # Single forward pass for the entire batch
        outputs  = self.model(**encoded)
        logits   = outputs.logits                    # (batch, seq, vocab)

        # Shift: logits[i] predicts token[i+1]
        # So to score token at position t, use logits at position t-1
        shift_logits = logits[:, :-1, :]             # (batch, seq-1, vocab)
        shift_ids    = encoded["input_ids"][:, 1:]   # (batch, seq-1)
        shift_mask   = encoded["attention_mask"][:, 1:]  # (batch, seq-1)

        # Log probabilities
        log_probs = torch.log_softmax(shift_logits, dim=-1)  # (batch, seq-1, vocab)

        # Gather the log prob of each actual token
        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_ids.unsqueeze(-1),
        ).squeeze(-1)                                # (batch, seq-1)

        # Zero out padding positions
        token_log_probs = token_log_probs * shift_mask

        # Sum only over the continuation tokens (last n_continuation positions)
        # This is the log P(continuation | prompt)
        scores = token_log_probs[:, -n_continuation:].sum(dim=-1)

        return scores.cpu().numpy()

    # ── Convenience: score against multiple continuations ─────────────────────

    @torch.inference_mode()
    def score_all_continuations(
        self,
        prompts: list[str],
        continuations: list[str],
    ) -> np.ndarray:
        """Score all prompts against all continuations efficiently.

        Instead of calling score_continuations() once per continuation,
        this method interleaves all (prompt, continuation) pairs into
        a single set of batched forward passes.

        Args:
            prompts: List of context strings.
            continuations: List of continuation strings to score against.

        Returns:
            np.ndarray of shape (len(continuations), len(prompts)).
            result[i, j] = log P(continuations[i] | prompts[j])
        """
        results = []
        for continuation in continuations:
            # Process in batches of self.batch_size
            all_scores = []
            for start in range(0, len(prompts), self.batch_size):
                batch  = prompts[start : start + self.batch_size]
                scores = self.score_continuations(batch, continuation)
                all_scores.extend(scores.tolist())
            results.append(all_scores)

        return np.array(results)  # (n_continuations, n_prompts)