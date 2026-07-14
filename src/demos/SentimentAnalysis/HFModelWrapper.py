from __future__ import annotations

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer


class HFModelWrapper:
    
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
     
        results = []
        for continuation in continuations:
            all_scores = []
            for start in range(0, len(prompts), self.batch_size):
                batch  = prompts[start : start + self.batch_size]
                scores = self.score_continuations(batch, continuation)
                all_scores.extend(scores.tolist())
            results.append(all_scores)

        return np.array(results)  # (n_continuations, n_prompts)