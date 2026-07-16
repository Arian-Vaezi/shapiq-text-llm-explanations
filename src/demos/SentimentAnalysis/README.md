# Sentiment Analysis: Shapley Interactions Demo

Explaining sentiment models with **Shapley Values (SV)** and **pairwise Shapley
Interactions (k-SII)**, built on the [`shapiq`](https://github.com/mmschlk/shapiq)
library. Part of the `shapiq-text-llm-explanations` project (Demo 1).

The central question: sentiment is *compositional*. "not bad" is positive even
though "not" and "bad" look weak or negative individually. First-order Shapley
Values explain **which words** matter; k-SII explains **which word pairs** matter
together — capturing negation, redundancy, and sarcasm cues that SVs alone miss.

---

## What's inside

| File | Purpose |
|---|---|
| `sentiment_analysis.py` | Core logic — model loading, value functions, SV / k-SII computation, plots. No UI code. |
| `SentimentDecoderGame.py` | Decoder value function (contrastive log-odds) as a `shapiq.Game`. |
| `app.py` | Live Streamlit app — type any sentence, pick a model, get SV + k-SII + plots. |
| `results_app.py` | Dashboard for pre-computed multilingual results, with a launcher for the live app. |
| `experiments.py` | Batch runner for the multilingual SLURM experiments. |
| `build_diagrams.py` | Reconstructs figures from `results/summary.json`. |


---

## Two pipelines, one framework

Both pipelines feed the **same** `KernelSHAP` and `KernelSHAPIQ` engine — only the
value function and masking strategy differ.

**Encoder** (classifier models)

```
v(S) = +P(POSITIVE | masked_text)   if predicted POSITIVE
     = -P(NEGATIVE | masked_text)   if predicted NEGATIVE
```

Absent words are replaced with `[MASK]`. Score range `[-1, +1]`. For 3-class
models (RoBERTa) the score is the contrastive `P(pos) - P(neg)`.

**Decoder** (causal LMs)

```
v(S) = mean log P(positive templates | text(S))
     - mean log P(negative templates | text(S))
```

Absent words are removed entirely (decoder models were not trained with `[MASK]`).
Templates are domain-agnostic ("This is positive." / "This is negative.").
Grounded in CELL (ICLR 2025): contrastive scoring functions need no class label.

---

## Models

**Encoders** (swappable live in the app)

- `lvwerra/distilbert-imdb` — movie reviews, binary
- `distilbert-base-uncased-finetuned-sst-2-english` — general English, binary
- `cardiffnlp/twitter-roberta-base-sentiment-latest` — social media, 3-class

**Decoders**

- `google/gemma-3-1b-it` — GPU recommended (~2 GB)
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0` — lighter, CPU-friendly

**Multilingual experiments** use `cardiffnlp/twitter-xlm-roberta-base-sentiment`
(encoder) and `Qwen3.5-4B` (decoder) for genuine EN / FR / DE coverage.

---

## Interaction index

We use **k-SII at order 2**. It gives directly interpretable pairwise scores and
satisfies a clean completeness property:

```
v(N) - v(∅) = Σ φᵢ  +  Σ φᵢⱼ
```

i.e. individual contributions plus pairwise interactions together exactly explain
the gap between the full-sentence score and the all-masked baseline.

`shapiq` also ships `STII` and `FSII` as one-line swaps. STII (Sundararajan &
Dhamdhere, 2020) prioritizes top-order faithfulness and is a natural robustness
check — worth noting since the standard interaction index can produce unstable or
sign-ambiguous estimates in some settings.

---

## Setup

The project uses [`uv`](https://github.com/astral-sh/uv) for environment
management.

```bash
uv sync
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

The live app additionally needs Streamlit and Gradio-free plotting:

```bash
uv pip install streamlit matplotlib pillow
```

For gated/decoder models, authenticate with Hugging Face:

```bash
python -c "from huggingface_hub import login; login(token='YOUR_TOKEN')"
```

> **Note:** running `shapiq` from source on Windows requires the Microsoft C++
> Build Tools (the library has a compiled extension).

---

## Running

**Live app** — type any sentence, choose a model, get an instant explanation:

```bash
streamlit run app.py
# → http://localhost:8501
```

**Results dashboard** — browse the 24 pre-computed multilingual runs:

```bash
streamlit run results_app.py
# → http://localhost:8501
```

The dashboard's sidebar has an **⚡ Instant Analysis** launcher that starts the
live app on port `8502` and opens it in a new tab, so both run side by side.

**Regenerate figures** from saved results:

```bash
python build_diagrams.py
```

---

## The multilingual experiment

24 sentences: **4 phenomena** (negative, positive, negation, sarcasm) ×
**2 topics** (film, service) × **3 languages** (EN, FR, DE), each run through an
encoder and a decoder on the LMU CIP cluster (SLURM). Results — exact and
approximate SV / k-SII, labels, and metadata — are saved to
`results/summary.json`.

Example negation triplet:

```
EN  "This film is not bad at all"
FR  "Ce film n'est pas mauvais du tout"
DE  "Dieser Film ist überhaupt nicht schlecht"
```

---

## Headline findings

- **Negation is a k-SII signal, not an SV one.** For "This film is not bad at
  all", `SV(not) ≈ +0.08` and `SV(bad) ≈ +0.32` are individually misleading, but
  `k-SII(not, bad) ≈ +2.88` dominates — the pair carries the meaning.
- **Model-agnostic.** The same interaction appears in a decoder
  (`k-SII(not, bad) ≈ +4.30` on Gemma) despite a completely different value
  function — evidence the signal is linguistic, not an artifact of one model.
- **Sarcasm is hard.** Both models mostly fail to detect it; k-SII shows diffuse
  interactions with no clean structure. An honest null result — the method can't
  reveal a phenomenon the model never learned.

---

## Reproducibility

- Fixed random seed: `42`
- Exact computation for ≤ 20 players; approximation (budget 512) beyond that
- Pinned model IDs (see above)
- Runs on CPU; GPU only speeds up the decoder models

---

## References

- Muschalik et al. — *shapiq: Shapley Interactions for Machine Learning*, NeurIPS 2024
- Sundararajan & Dhamdhere — *The Shapley Taylor Interaction Index*, 2020
- *Order-sensitive Shapley Values for Evaluating Conceptual Soundness of NLP Models*, 2022
- *CELL your Model: Contrastive Explanations for Large Language Models*, ICLR 2025
