from __future__ import annotations

import torch
import numpy as np

from torch.nn.functional import log_softmax
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

        # =====================================================
        # DEVICE
        # =====================================================
        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        elif isinstance(device, int):
            self.device = f"cuda:{device}"
        else:
            self.device = "cpu"

        # =====================================================
        # TOKENIZER
        # =====================================================
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # =====================================================
        # MODEL TYPE ROUTING
        # =====================================================
        self.is_encoder = any(
            x in model_name.lower() for x in self.ENCODER_MODELS
        )
        self.is_causal = not self.is_encoder

        # =====================================================
        # ENCODER MODELS
        # =====================================================
        if self.is_encoder:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                token=hf_token,
            )
            self.model.to(self.device)
        

        # =====================================================
        # CAUSAL MODELS
        # =====================================================
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                token=hf_token,
            )

        self.model.eval()

    # =========================================================
    # ENCODER SCORING (classification models)
    # =========================================================
    @torch.no_grad()
    def score_classifier(self, texts: list[str]) -> np.ndarray:

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)

        # binary classification case
        if probs.shape[-1] == 2:
            return probs[:, 1].cpu().numpy()

        return torch.max(probs, dim=-1).values.cpu().numpy()

    # =========================================================
    # CAUSAL LM SCORING (log P(completion | prompt))
    # =========================================================
    @torch.no_grad()
    def score_next_token(
        self,
        prompts: list[str],
        target_text: str,
    ) -> np.ndarray:

        scores = []

        for prompt in prompts:

            full_text = prompt + target_text

            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
            ).to(self.model.device)

            input_ids = inputs["input_ids"]

            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits  # (B, T, V)

            # shift for next-token prediction
            log_probs = log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = input_ids[:, 1:]

            token_log_probs = log_probs.gather(
                -1,
                target_ids.unsqueeze(-1)
            ).squeeze(-1)

            scores.append(token_log_probs.sum().item())

        return np.array(scores)

    
    # =========================================================
    # TEXT GENERATION
    # =========================================================
    @torch.no_grad()
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:

        if self.is_encoder:
            raise ValueError(
                f"Model '{self.model_name}' is an encoder model "
                "and cannot generate text."
            )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )

        # remove original prompt from output
        response = generated_text[len(prompt):].strip()

        return response