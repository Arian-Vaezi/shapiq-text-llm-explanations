from __future__ import annotations

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class HFModelWrapper:
    ENCODER_MODELS = [
        "distilbert",
        "roberta",
        "bert",
    ]

    CAUSAL_MODELS = [
        "mistral",
        "llama",
        "gemma",
        "qwen",
        # lightweight models
        "phi",
        "tinyllama",
        "stablelm",
        "gpt-neo",
        "gemma-2",
    ]

    def __init__(
        self,
        model_name: str,
        device: str | int = "cuda",
        hf_token: str | None = None,
    ):
        self.model_name = model_name

        # -----------------------------
        # device handling
        # -----------------------------
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif isinstance(device, int):
            self.device = f"cuda:{device}"
        else:
            self.device = "cpu"

        # -----------------------------
        # tokenizer (always needed)
        # -----------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            use_fast=True,
        )

        # fix decoder-only padding issue
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # -----------------------------
        # routing (simple + stable)
        # -----------------------------
        self.is_encoder = any(x in model_name.lower() for x in self.ENCODER_MODELS)

        # =====================================================
        # ENCODER MODELS (DistilBERT, RoBERTa)
        # =====================================================
        if self.is_encoder:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                token=hf_token,
            )
            self.model.to(self.device)

        # =====================================================
        # CAUSAL MODELS (Mistral, LLaMA, Gemma, Qwen, etc.)
        # =====================================================
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if "cuda" in self.device else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                token=hf_token,
            )

        self.model.eval()
