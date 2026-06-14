from __future__ import annotations

import os
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator


class APIModelWrapper:
    """Wraps an API-based language model (Groq, Gemini)."""

    def __init__(
        self,
        model_name: str,
        device: str | int = "cpu",
        temperature: float = 0.0,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.is_causal = False
        self.is_encoder = False
        self.temperature = temperature

        # Determine provider
        if "gemini" in model_name.lower():
            self.provider = "gemini"
        else:
            # Fallback to Groq for all other provided models
            self.provider = "groq"

        print(f"[APIModelWrapper] Initializing {self.provider} API model '{model_name}'")

        from dotenv import load_dotenv

        load_dotenv()

        if self.provider == "gemini":
            try:
                from google import genai
            except ImportError:
                raise ImportError("Please install google-genai to use Gemini models.")
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not found. Please add it to your .env file.")
            self.client = genai.Client(api_key=api_key)
        else:
            try:
                from groq import Groq
            except ImportError:
                raise ImportError("Please install groq to use Groq models.")
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not found. Please add it to your .env file.")
            self.client = Groq(api_key=api_key)

    # =========================================================
    # SCORING
    # =========================================================

    def score_next_token(self, prompts: list[str], target_text: str) -> np.ndarray:
        raise NotImplementedError(
            "Logprob scoring is not supported for API models. "
            "Please use 'llm-as-a-judge' value function instead."
        )

    # =========================================================
    # GENERATION
    # =========================================================

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        *,
        chat: bool = False,
        temperature: float | None = None,
    ) -> str:
        temp = temperature if temperature is not None else self.temperature

        if self.provider == "gemini":
            from google.genai import types

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temp,
                    max_output_tokens=max_new_tokens,
                ),
            )
            return response.text or ""
        else:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=max_new_tokens,
            )
            return response.choices[0].message.content or ""

    def generate_text_stream(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        *,
        chat: bool = False,
        temperature: float | None = None,
    ) -> Iterator[str]:
        temp = temperature if temperature is not None else self.temperature

        if self.provider == "gemini":
            from google.genai import types

            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temp,
                    max_output_tokens=max_new_tokens,
                ),
            )
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        else:
            messages = [{"role": "user", "content": prompt}]
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=max_new_tokens,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
