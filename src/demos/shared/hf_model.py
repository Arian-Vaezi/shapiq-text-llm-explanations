from __future__ import annotations

from threading import Thread
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Iterator

import numpy as np
import torch
from torch.nn.functional import log_softmax
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TextIteratorStreamer,
    pipeline,
)
from sentence_transformers import SentenceTransformer


class HFModelWrapper:
    ENCODER_MODELS: ClassVar[list[str]] = [
        "distilbert",
        "roberta",
        "bert",
    ]

    CAUSAL_MODELS: ClassVar[list[str]] = [
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
    ) -> None:
        self.model_name = model_name
        self._st_model: SentenceTransformer | None = None  # loaded lazily via load_embedding_model()

        # =====================================================
        # DEVICE
        # =====================================================

        if device == "cuda":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        elif isinstance(device, int):
            self.device = f"cuda:{device}"
        else:
            self.device = device

        print(f"Using device: {self.device}")

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

        self.is_encoder = any(x in model_name.lower() for x in self.ENCODER_MODELS)
        self.is_causal = any(x in model_name.lower() for x in self.CAUSAL_MODELS)

        # =====================================================
        # ENCODER MODELS
        # =====================================================

        if self.is_encoder:
            self.model = AutoModel.from_pretrained(
                model_name,
                token=hf_token,
            )
            self.model.to(self.device)
            self.pipe = None  # not available for encoder models

        # =====================================================
        # CAUSAL MODELS
        # =====================================================

        elif self.is_causal:
            dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=None,
                low_cpu_mem_usage=False,
                token=hf_token,
            )
            self.model.to(self.device)

            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )

        else:
            msg = f"Unsupported model type for model: {model_name}"
            raise ValueError(msg)

        self.model.eval()

    # =========================================================
    # ENCODER SCORING
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

        if probs.shape[-1] == 2:
            return probs[:, 1].cpu().numpy()

        return torch.max(probs, dim=-1).values.cpu().numpy()

    # =========================================================
    # CAUSAL LM SCORING
    # =========================================================

    @torch.no_grad()
    def score_next_token(self, prompts: list[str], target_text: str) -> np.ndarray:
        scores = []

        # safe upper bound — respect model limit but cap at 2048
        max_length = min(getattr(self.tokenizer, "model_max_length", 2048), 2048)

        for prompt in prompts:
            full_text = prompt + target_text

            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=max_length,  # explicit — avoids fast tokenizer bug
            ).to(self.device)

            input_ids = inputs["input_ids"]
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits

            log_probs = log_softmax(logits[:, :-1, :], dim=-1)
            target_ids = input_ids[:, 1:]

            token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
            scores.append(token_log_probs.sum().item())

        return np.array(scores)

    # =========================================================
    # TEXT GENERATION
    # =========================================================

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        *,
        chat: bool = False,
    ) -> str:
        """Generates text from the model given a prompt.

        If chat=True, formats the prompt using the model's chat template.
        """
        if self.is_encoder:
            msg = f"Model '{self.model_name}' cannot generate text."
            raise ValueError(msg)

        if chat:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        result = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )

        return result[0]["generated_text"].strip()

    # =========================================================
    # TEXT GENERATION STREAMING
    # =========================================================

    def generate_text_stream(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        *,
        chat: bool = False,
    ) -> Iterator[str]:
        """Same as generate_text but yields tokens as they are produced."""
        if self.is_encoder:
            msg = f"Model '{self.model_name}' cannot generate text."
            raise ValueError(msg)

        if chat:
            messages = [{"role": "user", "content": prompt}]
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
        )

        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        yield from streamer

        thread.join()

    # =========================================================
    # SEMANTIC EMBEDDINGS  (SentenceTransformer, separate from main model)
    # =========================================================

    def load_embedding_model(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
    ) -> None:
        """Loads a SentenceTransformer for semantic segmentation.

        Kept separate from the main HF model so causal LMs can still use
        embedding-based segmentation without conflict.

        Args:
            embedding_model_name: Any SentenceTransformer-compatible model name.
                Defaults to all-mpnet-base-v2 (best quality for cosine similarity).
        """
        print(f"Loading embedding model: {embedding_model_name}")
        self._st_model = SentenceTransformer(embedding_model_name, device=self.device)
        print("Embedding model loaded.")

    @torch.no_grad()
    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into normalized embeddings via the SentenceTransformer.

        Embeddings are L2-normalized, so dot product == cosine similarity.

        Args:
            texts: List of strings to embed.

        Returns:
            np.ndarray of shape (len(texts), hidden_dim).

        Raises:
            RuntimeError: If load_embedding_model() has not been called yet.
        """
        if self._st_model is None:
            raise RuntimeError(
                "No embedding model loaded. Call load_embedding_model() first."
            )

        return self._st_model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-norm → dot product == cosine sim
            show_progress_bar=False,
        )