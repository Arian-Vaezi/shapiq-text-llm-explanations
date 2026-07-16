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
| `sentiment_analysis.py` | Core logic — model registries, preloading, SV / k-SII computation, plots. No UI code. |
| `encoderTextImputer.py` | Encoder game — `[MASK]` imputation + contrastive classification score. |
| `sentimentDecoderGame.py` | Decoder game — word removal + contrastive log-odds over language-matched templates. |
| `HFModelWrapper.py` | Causal-LM loading (incl. 4-bit) and batched continuation scoring. |
| `app.py` | Live Streamlit app — type any sentence, pick a model, get SV + k-SII + plots. |
| `results_app.py` | Dashboard for the pre-computed multilingual results, with a launcher for the live app. |
| `Experiments.py` | Batch runner for the multilingual SLURM experiments (exact + approximate). |
| `run_experiments_half1.sbatch` / `run_experiments_half2.sbatch` | SLURM job scripts — sentences 1–12 and 13–24, to fit the 8h window. |
| `build_diagrams.py` | Rebuilds `InteractionValues` from `results/summary.json` and re-renders figures. |

---

## Two pipelines, one framework

Both pipelines are `shapiq.Game` subclasses and feed the **same** `KernelSHAP` /
`KernelSHAPIQ` engine. Only the value function and the masking strategy differ.

**Encoder** — `encoderTextImputer`

```
v(S) = P(positive | masked_text) − P(negative | masked_text)      ∈ [−1, +1]
```

Contrastive for *all* encoder models, binary and 3-class alike. Absent words are
replaced with the tokenizer's mask token (`[MASK]` / `<mask>`). The baseline
`v(∅)` is one forward pass on a fully masked sentence. Label indices are resolved
from `model.config.id2label`, case-insensitively, with a `NEG=0 / POS=1` fallback
for models that only expose generic `LABEL_0` / `LABEL_1` names.

**Decoder** — `sentimentDecoderGame`

```
v(S) = mean log P(T⁺ | text(S)) − mean log P(T⁻ | text(S))        unbounded
```

Absent words are **removed entirely** — causal LMs have no mask token. The
baseline `v(∅)` is the empty string. Grounded in CELL (ICLR 2025): a contrastive
scoring function needs no class label.

Templates are selected by **language × register**, 2 positive + 2 negative each:

| | `formal` | `informal` |
|---|---|---|
| `en` | "This is positive." / "This is negative." | "That's great!" / "That's awful!" |
| `fr` | "C'est positif." / "C'est négatif." | "C'est génial !" / "C'est nul !" |
| `de` | "Das ist positiv." / "Das ist negativ." | "Das ist toll!" / "Das ist furchtbar!" |

> **The `language` argument must match the input sentence's language.** Otherwise
> the value function conflates sentiment understanding with code-switching, and
> the resulting attributions are not interpretable.

---

## Models

**Encoders** — swappable live in the app (`ENCODER_MODELS`)

| Key | Model | Domain |
|---|---|---|
| `imdb` | `lvwerra/distilbert-imdb` | Movie reviews, binary |
| `sst2` | `distilbert-base-uncased-finetuned-sst-2-english` | General English, binary (**app default**) |
| `roberta` | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Social media, 3-class |

**Decoders** (`DECODER_MODELS`)

| Key | Model | Notes |
|---|---|---|
| `tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Lightest, CPU-friendly (**app default**) |
| `gemma3` | `google/gemma-3-1b-it` | ~2 GB, runs fine on a 4 GB GPU |
| `gemma4` | `google/gemma-4-E2B-it` | ⚠️ See known issues — needs ≫ 4 GB VRAM |

**Multilingual experiments** (`Experiments.py`) use a different, genuinely
multilingual pair:

- Encoder: `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- Decoder: `Qwen/Qwen3.5-4B`, loaded **4-bit quantized** (`load_in_4bit=True`),
  `register="formal"`

---

## Interaction index

We use **k-SII at order 2**. It gives directly interpretable pairwise scores and
satisfies a clean completeness property:

```
v(N) − v(∅) = Σ φᵢ  +  Σ φᵢⱼ
```

i.e. individual contributions plus pairwise interactions together exactly explain
the gap between the full-sentence score and the empty baseline.

`shapiq` also ships `STII` and `FSII` as one-line swaps. STII (Sundararajan &
Dhamdhere, 2020) prioritizes top-order faithfulness and is a natural robustness
check — worth noting since the standard interaction index can produce unstable or
sign-ambiguous estimates in some settings.

### Budgets

Budgets differ between the interactive app and the batch experiments, and this is
a deliberate speed/accuracy tradeoff:

| | Encoder | Decoder | Exact ground truth |
|---|---|---|---|
| `sentiment_analysis.py` (live app) | 256 | **10** | never |
| `Experiments.py` (SLURM) | 512 | 512 | encoder: always · decoder: if n ≤ 20 |

The live decoder budget of 10 is very low — each coalition costs 4 forward passes
(2 positive + 2 negative templates), so a realistic budget would make the app
unusably slow. Treat live decoder numbers as indicative; the SLURM runs are the
ones to cite.

---

## Setup

The project uses [`uv`](https://github.com/astral-sh/uv) for environment
management.

```bash
uv sync
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Extra dependencies for the apps:

```bash
uv pip install streamlit matplotlib pillow
```

For gated models (Gemma), authenticate with Hugging Face:

```bash
python -c "from huggingface_hub import login; login(token='YOUR_TOKEN')"
```

> **Windows:** building `shapiq` from source requires the Microsoft C++ Build
> Tools ("Desktop development with C++"), since the library ships a compiled
> extension.

---

## Running

**Live app** — type any sentence, choose a model, get an instant explanation:

```bash
streamlit run app.py
# → http://localhost:8501
```

`app.py` calls `preload_models()` at startup so that each click has zero
model-loading cost. See known issues below.

**Results dashboard** — browse the 24 pre-computed multilingual runs:

```bash
streamlit run results_app.py
# → http://localhost:8501
```

Sections: Story · Accuracy · Negation · Sarcasm · Explorer. The sidebar has an
**⚡ Instant Analysis** launcher that spawns `app.py` on port `8502` and opens it
in a new tab, so both run side by side. This is a local-demo pattern — for a
deployed version, use Streamlit's native multipage layout instead of `subprocess`.

**Regenerate figures** from saved results:

```bash
python build_diagrams.py                              # all runs, approximate values
python build_diagrams.py --index exact                # use exact ground truth
python build_diagrams.py --run_key neg8_film_en_decoder
python build_diagrams.py --phenomenon sarcasm --lang fr
```

Reads `results/summary.json`, writes to `figures/`.

**SLURM experiments** — split across two jobs to fit the 8h window:

```bash
sbatch run_experiments_half1.sbatch    # sentences 1-12
sbatch run_experiments_half2.sbatch    # sentences 13-24
```

`Experiments.py` is resume-safe: completed runs are cached as pickles in
`results/raw/` and skipped on rerun.

---

## The multilingual experiment

24 sentences: **4 phenomena** (negative, positive, negation, sarcasm) ×
**2 topics** (film, service) × **3 languages** (EN, FR, DE), each run through both
an encoder and a decoder.

Example negation triplet:

```
EN  "This film is not bad at all."
FR  "Ce film n'est pas mauvais du tout."
DE  "Dieser Film ist überhaupt nicht schlecht."
```

**Outputs**

```
results/
    raw/{run_key}.pkl     full InteractionValues objects (gitignored)
    summary.json          JSON-safe values, labels, metadata (committed)
    progress.log          human-readable progress
figures/
    {run_key}_sentence.png · _network.png · _heatmap.png
```

`summary.json` stores both approximate and exact SV / k-SII per run, plus
`v_full`, `v_empty`, `budget`, `exact_skipped`, and timings.

---

## Headline findings

**Negation is a k-SII signal, not an SV one.** Gemma 3 1B on
*"This film is not bad at all"* (live app, `formal` / `en`):

| | value |
|---|---|
| SV(`not`) | **−0.7797** |
| SV(`bad`) | +1.4176 |
| sum of the two SVs | +0.6379 |
| **k-SII(`not`, `bad`)** | **+3.7554** |

Read individually, `not` looks like it pushes the sentence *negative*. The pair
carries roughly six times the two SVs combined, in the opposite direction to what
`not`'s own SV suggests. This is exactly the structure first-order attribution
cannot represent.

**Model-agnostic.** The same dominant `(not, bad)` pair appears across the encoder
and decoder pipelines despite completely different value functions, masking
strategies, and score scales — evidence the signal is linguistic, not an artifact
of one model.

**Sarcasm is hard.** Both models mostly fail to detect it; k-SII shows diffuse
interactions with no clean structure. The one correct French case appears to be an
accident of literal negative vocabulary (*ennuyeux*), not irony detection. An
honest null result — the method cannot reveal a phenomenon the model never learned.

---

## Known issues

**`preload_models()` loads every model, including `gemma4`.** `google/gemma-4-E2B-it`
is *not* a 2 GB model — "E2B" means *effective* 2B parameters (Per-Layer
Embeddings + multimodal encoders), and it allocates ~18–19 GB. On a 4 GB GPU the
app OOMs at startup before rendering anything. Until preloading is made lazy or
gemma4 is gated behind a VRAM check, comment it out of `DECODER_MODELS` on small
GPUs. Partially-loaded weights also produce **silent NaN Shapley values** rather
than a clean error — if every SV is `nan`, suspect VRAM before suspecting the math.

**Three different import paths for `HFModelWrapper`.** `sentiment_analysis.py`
imports `demos.SentimentAnalysis.HFModelWrapper`, `sentimentDecoderGame.py`'s
fallback imports `demos.shared.hf_model`, and `Experiments.py` imports it flat as
`HFModelWrapper`. If `demos/shared/hf_model.py` still exists with the older
`score_next_token` API, the fallback path raises
`AttributeError: 'HFModelWrapper' object has no attribute 'score_continuations'`.
Consolidate to one module.

**Class names are `lowerCamelCase`** (`encoderTextImputer`, `sentimentDecoderGame`)
while several imports assume `PascalCase` (`from EncoderTextImputer import
EncoderTextImputer` in `Experiments.py`). This survives on Windows because the
filesystem is case-insensitive, but breaks on macOS/Linux and violates PEP 8.
Renaming to `EncoderTextImputer` / `SentimentDecoderGame` everywhere would fix
both.

**Minor:** `app.py` checks `model_choice == "gemma"` for its decoder label, but the
key is `gemma3` — the label always falls through to "TinyLlama 1.1B".

---

## Reproducibility

- Fixed random seed: `42` everywhere
- Pinned model IDs (see Models)
- Budgets as tabulated above; `Experiments.py` additionally saves exact ground
  truth (`EXACT_MAX_PLAYERS_DECODER = 20`)
- Encoder pipeline runs comfortably on CPU; decoders want a GPU (and `gemma4`
  wants a large one)

---

## References

- Muschalik et al. — *shapiq: Shapley Interactions for Machine Learning*, NeurIPS 2024
- Sundararajan & Dhamdhere — *The Shapley Taylor Interaction Index*, 2020
- *Order-sensitive Shapley Values for Evaluating Conceptual Soundness of NLP Models*, 2022
- *CELL your Model: Contrastive Explanations for Large Language Models*, ICLR 2025
