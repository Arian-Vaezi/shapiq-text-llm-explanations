
# Model Creation Guide (HFModelWrapper)

---
## How to use HF models in your demo
```python
from demos.shared.hf_model import HFModelWrapper
self.model = HFModelWrapper(
            model_name,
            device=device
        )
```
You can add this code snippet in your game class. Currently added models are listed below.


##  Decoder-only models (7B class)

### Mistral-7B
```python
model = HFModelWrapper(
    "mistralai/Mistral-7B-Instruct-v0.2",
    device="cuda"
)
```

### LLaMA-7B
```python
model = HFModelWrapper(
    "meta-llama/Llama-2-7b-hf",
    device="cuda"
)
```

### Gemma-7B (gated)
```python
model = HFModelWrapper(
    "google/gemma-7b",
    device="cpu"
)
```

### Qwen2.5-7B
```python
model = HFModelWrapper(
    "Qwen/Qwen2.5-7B",
    device="cuda"
)
```

---

##  Lightweight decoder-only models

### Gemma-2B
```
model = HFModelWrapper(
    "google/gemma-2-2b-it",
    device="cuda"
)
```

### Gpt-neo
```
model = HFModelWrapper(
    "EleutherAI/gpt-neo-1.3B",
    device="cuda"
)
```


### Phi-2 (2.7B)
```python
model = HFModelWrapper(
    "microsoft/phi-2",
    device="cuda"
)
```

### TinyLlama (1.1B)
```python
model = HFModelWrapper(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device="cuda"
)
```

### StableLM 2 (1.6B)
```python
model = HFModelWrapper(
    "stabilityai/stablelm-2-1_6b",
    device="cuda"
)
```

### Qwen 1.5 (1.8B)
```python
model = HFModelWrapper(
    "Qwen/Qwen1.5-1.8B",
    device="cuda"
)
```

---

## Encoder models

### DistilBERT
```python
model = HFModelWrapper(
    "distilbert-base-uncased-finetuned-sst-2-english",
    device="cpu"
)
```

### RoBERTa Sentiment
```python
model = HFModelWrapper(
    "cardiffnlp/twitter-roberta-base-sentiment",
    device="cpu"
)
```

---

## HFModelWrapper responsibilities

- Load model
- Load tokenizer
- Provide logits/probabilities （model call）
- Provide generic inference utilities

#  Notes
- Large models may require HuggingFace authentication （such as Gemma-7B）
- Lightweight models are recommended for experiments
