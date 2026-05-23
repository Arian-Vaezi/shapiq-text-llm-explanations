# HFModelWrapper Guide

A shared utility for loading and using HuggingFace models across demos.

---

## Table of Contents

1. [Supported Models](#supported-models)
2. [Creating an HFModel](#creating-an-hfmodel)
3. [Using a Model in a Game](#using-a-model-in-a-game)
4. [Text Generation](#text-generation)
5. [Scoring](#scoring)
6. [Notes](#notes)

---

## Supported Models

### Decoder-only — 7B class

| Name | Model ID | Params |
|------|----------|--------|
| Mistral 7B Instruct | `mistralai/Mistral-7B-Instruct-v0.2` | 7B |
| LLaMA 2 7B | `meta-llama/Llama-2-7b-hf` | 7B |
| Gemma 7B ⚠️ | `google/gemma-7b` | 7B |
| Qwen 2.5 7B | `Qwen/Qwen2.5-7B` | 7B |

### Decoder-only — Lightweight

| Name | Model ID | Params |
|------|----------|--------|
| Gemma 2 2B IT | `google/gemma-2-2b-it` | 2B |
| GPT-Neo | `EleutherAI/gpt-neo-1.3B` | 1.3B |
| Phi-2 | `microsoft/phi-2` | 2.7B |
| TinyLlama | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 1.1B |
| StableLM 2 | `stabilityai/stablelm-2-1_6b` | 1.6B |
| Qwen 1.5 | `Qwen/Qwen1.5-1.8B` | 1.8B |

### Encoder-only

| Name | Model ID | Params |
|------|----------|--------|
| DistilBERT SST-2 | `distilbert-base-uncased-finetuned-sst-2-english` | 66M |
| RoBERTa Sentiment | `cardiffnlp/twitter-roberta-base-sentiment` | 125M |

> ⚠️ **Gated model** — Gemma-7B requires a HuggingFace account and accepted terms of service. 

---

## Creating an HFModel

```python
from demos.shared.hf_model import HFModelWrapper

model = HFModelWrapper(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cuda",       
)
```

Model weights are downloaded and cached automatically by the HuggingFace library under `~/.cache/huggingface/`. Subsequent instantiations with the same `model_name` load from cache without re-downloading.

**Device fallback**: if you pass `device="cuda"` but no GPU is available, the wrapper automatically falls back to `"mps"` (Apple Silicon) and then `"cpu"`.

---

## Using a Model in a Game

Pass an already-instantiated `HFModelWrapper` into your game to avoid reloading weights on every call. If none is provided, the game constructs one itself.

```python
from demos.shared.hf_model import HFModelWrapper
from shapiq import Game

class JailbreakGame(Game):
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        hf_model: HFModelWrapper | None = None,
        **kwargs,
    ) -> None:
        # Reuse a cached model if provided, otherwise load fresh
        self.hf_model = hf_model or HFModelWrapper(
            model_name=model_name,
            device=device,
        )
```


## Text Generation

### Single response

```python
# Plain completion
output = model.generate_text(
    prompt="What is the capital of France?",
    max_new_tokens=64,
)

# Chat-formatted (applies the model's chat template)
output = model.generate_text(
    prompt="What is the capital of France?",
    max_new_tokens=64,
    chat=True,
)
```

### Streaming

Use `generate_text_stream` to yield tokens one at a time — useful for Gradio chatbots or any UI that renders progressively.

```python
for token in model.generate_text_stream(
    prompt="Tell me a story.",
    max_new_tokens=128,
    chat=True,
):
    print(token, end="", flush=True)
```

> Encoder models (DistilBERT, RoBERTa) do not support text generation and will raise `ValueError` if called.

---

## Scoring

### Encoder models — classification probability

Returns the probability of the positive class (or max class for multi-label).

```python
scores = model.score_classifier(["I love this!", "This is terrible."])
# np.ndarray of shape (2,)
```

### Causal models — log-probability of a target continuation

Useful for compliance scoring: how likely is the model to produce `target_text` after each prompt?

```python
prompts = ["Ignore all previous instructions and", "Please help me with"]
scores = model.score_next_token(prompts, target_text=" Sure, here is")
# np.ndarray of shape (2,) — higher = more likely continuation
```

---

## Notes

- **Large models** (7B+) require a GPU with sufficient VRAM. Run on CPU only for testing.
- **Gated models** (e.g. `google/gemma-7b`) require HuggingFace authentication. Set your token via `hf_token=` or `huggingface-cli login`.
- **Lightweight models** (`TinyLlama`, `Phi-2`, `StableLM`) are recommended for local development and experiments.
- Models are detected by name substring matching (`"llama"`, `"gemma"`, `"bert"`, etc.) — keep model IDs as listed above.