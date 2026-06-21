"""HFModelWrapper — unified wrapper for causal language models.

Supports:
    - TinyLlama/TinyLlama-1.1B-Chat-v1.0
    - google/gemma-3-1b-it
    - Qwen/Qwen3.5-9B  (requires load_in_4bit=True on 8GB GPUs)

Key design decisions:
    - True batching with left-padding for causal LMs
      (right-padding breaks autoregressive scoring)
    - float16 on CUDA/MPS, float32 on CPU
    - Device auto-detection with graceful fallback
    - score_continuations() is the only method SentimentDecoderGame needs
    - Optional 4-bit quantization (bitsandbytes) for large models on
      memory-constrained GPUs. Defaults to OFF — does not change app.py
      behavior for TinyLlama / Gemma 3 1B, which already fit in fp16.

QUANTIZATION
------------
Qwen/Qwen3.5-9B needs ~18GB in fp16 — too large for an 8GB GPU.
Loading with load_in_4bit=True via bitsandbytes reduces this to ~5-6GB,
fitting comfortably with headroom for batching.

    model = HFModelWrapper("Qwen/Qwen3.5-9B", load_in_4bit=True)

This requires the bitsandbytes package:
    uv pip install bitsandbytes
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
    device:
        Inference device. If None, auto-selects CUDA > MPS > CPU.
    hf_token:
        HuggingFace token for gated models (e.g. Gemma).
    batch_size:
        Number of prompts per forward pass. Defaults to 32.
    load_in_4bit:
        Whether to load the model in 4-bit quantization via bitsandbytes.
        Defaults to False — required for large models (e.g. Qwen3.5-9B)
        on memory-constrained GPUs (≤8GB). Has no effect on small models
        like TinyLlama or Gemma 3 1B; leave False for those.

    Examples
    --------
    >>> # Small model — app default, unchanged behavior
    >>> model = HFModelWrapper("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    >>> # Large model on 8GB GPU — needs quantization
    >>> model = HFModelWrapper("Qwen/Qwen3.5-9B", load_in_4bit=True)

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
        load_in_4bit: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.load_in_4bit = load_in_4bit

        # ── Device selection ──────────────────────────────────────────────────
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        print(f"[HFModelWrapper] {model_name} → device: {self.device}"
              f"{' (4-bit quantized)' if load_in_4bit else ''}")

        if load_in_4bit and self.device != "cuda":
            raise ValueError(
                "load_in_4bit=True requires a CUDA device. "
                f"Current device: {self.device!r}. "
                "bitsandbytes 4-bit quantization is not supported on CPU/MPS."
            )

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
        if load_in_4bit:
            # 4-bit quantization via bitsandbytes — required for large
            # models (e.g. 9B+) on GPUs with ≤8GB VRAM.
            #
            # device_map="auto" can decide to offload some layers to CPU/disk
            # if it estimates the quantized model won't fully fit — this
            # silently breaks our scoring logic (mixed-device tensors) and
            # is drastically slower. We instead force everything onto the
            # GPU with an explicit max_memory budget, and fail loudly if it
            # truly doesn't fit rather than degrading to CPU offload.
            from transformers import BitsAndBytesConfig

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            gpu_index = 0
            total_gpu_mem = torch.cuda.get_device_properties(gpu_index).total_memory
            # Reserve ~1GB headroom for activations/KV cache during inference
            budget_gb = max(int(total_gpu_mem / 1e9) - 1, 1)
            max_memory = {gpu_index: f"{budget_gb}GiB", "cpu": "0GiB"}

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                max_memory=max_memory,
                token=hf_token,
            )
            # device_map="auto" places the model; no manual .to() needed
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=dtype,
                device_map=None,
                token=hf_token,
            )
            self.model.to(self.device)

        self.model.eval()

        if self.device == "cuda":
            print(f"[HFModelWrapper] Ready. "
                  f"Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB used")
        else:
            print(f"[HFModelWrapper] Ready on {self.device}.")

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
        continuation_ids = self.tokenizer(
            continuation,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
        n_continuation = continuation_ids.shape[1]

        # Build full texts: each prompt concatenated with the continuation
        full_texts = [p + continuation for p in prompts]

        # Tokenize with LEFT padding
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
        ).to(self.model.device)  # use model's device — important for device_map="auto"

        self.tokenizer.padding_side = original_side

        # Single forward pass for the entire batch
        outputs  = self.model(**encoded)
        logits   = outputs.logits                    # (batch, seq, vocab)

        # Shift: logits[i] predicts token[i+1]
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

        # Sum only over the continuation tokens
        scores = token_log_probs[:, -n_continuation:].sum(dim=-1)

        # Cast to float32 before numpy conversion — numpy has no native
        # bfloat16 support, and some models (e.g. Qwen3.5) compute logits
        # in bfloat16 even when bnb_4bit_compute_dtype=float16 is set,
        # because the underlying linear-attention kernels default to bf16.
        return scores.float().cpu().numpy()

    # ── Convenience: score against multiple continuations ─────────────────────

    @torch.inference_mode()
    def score_all_continuations(
        self,
        prompts: list[str],
        continuations: list[str],
    ) -> np.ndarray:
        """Score all prompts against all continuations efficiently.

        Args:
            prompts: List of context strings.
            continuations: List of continuation strings to score against.

        Returns:
            np.ndarray of shape (len(continuations), len(prompts)).
            result[i, j] = log P(continuations[i] | prompts[j])
        """
        results = []
        for continuation in continuations:
            all_scores = []
            for start in range(0, len(prompts), self.batch_size):
                batch  = prompts[start : start + self.batch_size]
                scores = self.score_continuations(batch, continuation)
                all_scores.extend(scores.tolist())
            results.append(all_scores)

        return np.array(results)  # (n_continuations, n_prompts)