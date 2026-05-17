from __future__ import annotations

import torch

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
)


class HFModelWrapper:

    def __init__(
        self,
        model_name: str,
        device: str | int | None = None,
    ) -> None:

        self.model_name = model_name

        self.is_causal = any(
            x in model_name.lower()
            for x in [
                "gemma",
                "llama",
                "mistral",
                "qwen",
            ]
        )

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        # ==========================================
        # Encoder classifier
        # ==========================================

        if not self.is_causal:

            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=device,
            )

        # ==========================================
        # Causal LM
        # ==========================================

        else:

            self.model = (
                AutoModelForCausalLM
                .from_pretrained(
                    model_name
                )
            )

            self.model.eval()

    # =================================================
    # Encoder scoring
    # =================================================

    def score_classifier(
        self,
        prompts: list[str],
    ) -> list[float]:

        outputs = self.pipeline(
            prompts
        )

        scores = []

        for out in outputs:

            score = out["score"]

            if (
                out["label"]
                == "NEGATIVE"
            ):
                score = -score

            scores.append(score)

        return scores

    # =================================================
    # Generative scoring
    # =================================================

    def score_next_token(
        self,
        prompts: list[str],
        target_token: str,
    ) -> list[float]:

        scores = []

        target_ids = self.tokenizer(
            target_token,
            add_special_tokens=False,
        )["input_ids"]

        target_id = target_ids[0]

        for prompt in prompts:

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
            )

            with torch.no_grad():

                outputs = self.model(
                    **inputs
                )

            logits = outputs.logits[
                0,
                -1,
            ]

            probs = torch.softmax(
                logits,
                dim=-1,
            )

            score = probs[
                target_id
            ].item()

            scores.append(score)

        return scores