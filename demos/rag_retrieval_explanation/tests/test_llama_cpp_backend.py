"""Tests for the demo's llama.cpp backend without loading a real GGUF model."""

from __future__ import annotations

import numpy as np
import pytest

from demos.rag_retrieval_explanation.core.model_backends import LlamaCppBackend


class FakeLlama:
    """Minimal llama-cpp-python stand-in for deterministic backend tests."""

    def __init__(self) -> None:
        self.scores = np.empty((0, 0))
        self.reset_called = False

    def tokenize(
        self,
        text: bytes,
        *,
        add_bos: bool,
        special: bool,
    ) -> list[int]:
        """Return fixed prompt or target tokens."""
        del special
        if text == b"target":
            return [3, 4]
        return [1, 2] if add_bos else [2]

    def reset(self) -> None:
        """Record context reset."""
        self.reset_called = True

    def eval(self, tokens: list[int]) -> None:
        """Create logits where target tokens 3 and 4 are preferred."""
        assert tokens == [1, 2, 3, 4]
        self.scores = np.zeros((4, 5), dtype=float)
        self.scores[1, 3] = 2.0
        self.scores[2, 4] = 3.0

    def create_completion(self, **kwargs: object) -> dict[str, object]:
        """Return one deterministic completion."""
        assert "Retrieved context:" in str(kwargs["prompt"])
        assert kwargs["temperature"] == 0.0
        return {"choices": [{"text": " grounded answer "}]}


def backend_with_fake_model() -> tuple[LlamaCppBackend, FakeLlama]:
    """Create a backend whose cached model is replaced by a fake."""
    backend = LlamaCppBackend("unused.gguf", n_ctx=32)
    fake = FakeLlama()
    backend.__dict__["model"] = fake
    return backend, fake


def test_generate_uses_grounded_prompt() -> None:
    """Generation should return stripped text from a deterministic completion."""
    backend, _ = backend_with_fake_model()

    result = backend.generate("Question?", "Evidence.")

    assert result.answer == "grounded answer"


def test_qwen3_prompt_disables_thinking_mode() -> None:
    """Qwen3 prompts should avoid expensive reasoning traces for coalition scoring."""
    backend = LlamaCppBackend("Qwen3-8B-Q4_K_M.gguf")

    prompt = backend.build_prompt("Question?", "Evidence.")

    assert prompt.startswith("<|im_start|>system")
    assert "/no_think" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")


def test_backends_for_same_model_share_lock() -> None:
    """Backends sharing a cached model must also serialize access with one lock."""
    first = LlamaCppBackend("shared.gguf", n_ctx=32, n_gpu_layers=2, n_threads=1)
    second = LlamaCppBackend("shared.gguf", n_ctx=32, n_gpu_layers=2, n_threads=1)

    assert first._lock is second._lock


def test_target_log_likelihood_uses_target_token_logits() -> None:
    """Likelihood should average next-token log probabilities for the target."""
    backend, fake = backend_with_fake_model()

    score = backend.target_log_likelihood("prompt", "target")

    expected = np.mean(
        [
            2.0 - np.log(np.exp(2.0) + 4.0),
            3.0 - np.log(np.exp(3.0) + 4.0),
        ]
    )
    assert fake.reset_called
    assert score == pytest.approx(expected)


def test_target_log_likelihood_rejects_context_overflow() -> None:
    """Oversized prompt/target pairs should fail before llama.cpp evaluation."""
    backend, _ = backend_with_fake_model()
    backend.n_ctx = 3

    with pytest.raises(RuntimeError, match="context size"):
        backend.target_log_likelihood("prompt", "target")
