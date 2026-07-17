# Sentiment Analysis — Shapley Values & Interactions Demo

Explains how sentiment models make decisions using **Shapley Values (SV)**
and **Shapley Interactions (k-SII)**. The core finding driving this project:
individual word contributions (SV) tell part of the story — pairwise word
*interactions* (k-SII) reveal what first-order explanations miss entirely.

> Example: in *"This film is not bad at all"*, `SV(not) ≈ -0.1` looks
> unimportant on its own. But `k-SII(not, bad) = +1.86` — 15× larger than
> any other pair — shows that "not" and "bad" together flip the sentiment
> in a way neither word does alone.

This repository has two parts:

1. **The app** — an interactive Streamlit demo for exploring single sentences
2. **The experiments** — an offline SLURM batch pipeline that ran a
   structured 24-sentence study across English, French, and German

---

## Part 1 — The App

### What it does

You type a sentence, pick a model, and the app shows:

- **Segmentation** — how the sentence is split into word "players"
- **Step 1: Shapley Values** — a colored sentence plot + table showing
  each word's individual contribution to the prediction
- **Step 2: k-SII Interactions** — a network graph + heatmap showing
  which word *pairs* drive the prediction beyond their individual effects
- **Step 3: Interpretation** — automatic synergy/redundancy explanation
  comparing the dominant pair's k-SII to the sum of its individual SVs

### Two pipelines

The app is **English-only**. (Multilingual coverage is explored separately
in Part 2's experiments, which reuse this same infrastructure.)

| | Encoder | Decoder |
|---|---|---|
| **Models** | DistilBERT IMDb, DistilBERT SST-2, RoBERTa Twitter | TinyLlama 1.1B, Gemma 3 1B, Gemma 4 |
| **Masking** | absent words → `[MASK]` token | absent words → **removed** entirely |
| **Value function** | `v(S) = P(POSITIVE\|S) - P(NEGATIVE\|S)` | `v(S) = mean log P(T⁺\|S) - mean log P(T⁻\|S)` |
| **Why this masking?** | Encoders were pretrained with `[MASK]` — natural for them | Decoders never saw `[MASK]` during training — removal keeps text grammatical |
| **Why this value fn?** | Direct classification head output, contrastive and bounded to `[-1, +1]` | No classification head exists — instead we measure how likely the model is to *continue* with positive vs negative phrases |

Both pipelines feed into the same shapiq computation:
**KernelSHAP** (first-order SV) + **KernelSHAPIQ** (pairwise k-SII, order 2).

Larger decoder models (e.g. Gemma 4) need more VRAM than the smaller ones
(TinyLlama, Gemma 3 1B) and may require 4-bit quantization on
consumer-grade GPUs — see `HFModelWrapper`'s `load_in_4bit` parameter.

### Key files

```
src/demos/SentimentAnalysis/
├── app.py                    UI only — layout, styling, event routing
├── sentiment_analysis.py     All computation: pipelines, Shapley calls, plots, model preloading
├── EncoderTextImputer.py     shapiq Game subclass for BERT-family encoders
├── SentimentDecoderGame.py   shapiq Game subclass for causal LMs (contrastive log-odds)
└── HFModelWrapper.py         Batched scoring for decoder models (left-padding, true batching)
```

### Design notes worth knowing

- **`EncoderTextImputer`** subclasses `shapiq.game.Game` directly (not
  `Imputer`) — text doesn't fit the tabular `Imputer` interface cleanly.
  `v(∅)` (the all-masked baseline) is computed *before* `super().__init__()`
  so normalization is correct from the start.
- **`SentimentDecoderGame`** supports `language` and `register` parameters
  for its sentiment templates (`"en"/"fr"/"de"` × `"formal"/"informal"`).
  Defaults to `language="en", register="formal"` — this preserves the
  app's original English-only behavior unchanged. The language-aware
  templates exist specifically for the multilingual experiments (see
  Part 2) where using English templates on French/German sentences would
  introduce a code-switching confound.
- **Models are preloaded once at app startup** (`preload_models()`,
  wrapped in `@st.cache_resource` in `app.py`) so every click after the
  first is instant — no per-request model loading cost.
- **`BUDGET_ENCODER = 256`**, **`BUDGET_DECODER`** is set lower by default
  since decoder inference is far more expensive per coalition (one full
  forward pass per template per coalition, vs. one batched call for the
  encoder). Tune this based on your GPU.

### Running the app

```bash
cd src/demos/SentimentAnalysis
streamlit run app.py
```

Models download from HuggingFace on first run and cache locally
(`HF_HOME`, defaults to `~/.cache/huggingface`). Subsequent runs are fast.

**Hardware notes:**
- Encoder models (DistilBERT, RoBERTa) run comfortably on CPU.
- Decoder models need a GPU for reasonable speed. TinyLlama and Gemma 3 1B
  are lightweight; Gemma 4 requires a stronger GPU and may need 4-bit
  quantization (`bitsandbytes`) to fit on consumer hardware.

---

## Part 2 — The Multilingual Experiments

### Goal

> **Do Shapley interactions reveal the same understanding (and the same
> failure patterns) across English, French, and German — or does a
> model's grasp of negation and sarcasm degrade differently per language?**

The app lets you explore one sentence at a time. The experiments push
further: a structured, reproducible study across **4 sentiment phenomena
× 2 topics × 3 languages = 24 sentences**, run offline on a SLURM GPU
cluster, with both approximate (KernelSHAP/IQ) and **exact** (brute-force)
Shapley/k-SII values computed for validation.

### The 24-sentence dataset

8 base items, each with parallel English/French/German translations,
keeping sentence structure (especially negation/sarcasm marker position)
as parallel as the languages allow:

| Phenomenon | Topic: Film | Topic: Service |
|---|---|---|
| **Negative** | "This film was absolutely terrible." | "The service at this restaurant was really bad." |
| **Positive** | "This film was absolutely wonderful." | "The service at this restaurant was truly amazing." |
| **Negation** | "This film is not bad at all." | "The service at this restaurant is not terrible at all." |
| **Sarcasm** | "Oh great, another boring film, exactly what I was hoping for." | "Oh wonderful, another slow service, just what I needed." |

Negative/positive use *different* intensity words per topic (terrible vs.
bad, wonderful vs. amazing) rather than identical phrasing — this checks
whether the model has learned the *grammatical pattern* of negation/sarcasm
rather than just memorizing one fixed phrase.

### Models

| | Model | Why |
|---|---|---|
| **Encoder** | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | Multilingual (covers EN/FR/DE + 5 more), 3-class (pos/neu/neg), same architecture family as the app's RoBERTa model so existing label-handling logic carries over unchanged |
| **Decoder** | `Qwen/Qwen3.5-4B` (4-bit quantized) | Strong multilingual coverage (201 languages), fits the available GPU with headroom for batching |

### Per-language templates (decoder value function)

The decoder's contrastive log-odds value function needs sentiment
*continuation templates*. Using English templates on French/German
sentences would force the model to code-switch mid-generation, conflating
"does it understand sentiment" with "can it switch languages fluently."
`SentimentDecoderGame` therefore selects templates by the sentence's own
language:

```python
TEMPLATES_BY_LANG = {
    "en": {"formal": {"pos": [" This is positive.", ...], "neg": [...]}},
    "fr": {"formal": {"pos": [" C'est positif.", ...], "neg": [...]}},
    "de": {"formal": {"pos": [" Das ist positiv.", ...], "neg": [...]}},
}
```

An `"informal"` register also exists per language (exclamatory/colloquial
phrasing) as a robustness check — whether findings hold regardless of
template wording style.

### Exact vs. approximate validation

For every encoder run (and decoder runs with ≤10 words), we compute
**both**:

- **Approximate**: `KernelSHAP` / `KernelSHAPIQ` with `budget=256`
- **Exact**: `game.exact_values()` — brute-force over all `2ⁿ` coalitions

This isn't redundant — it validates that the approximation budget is
large enough to trust. Exact values serve as ground truth for sanity
checking on any sentence short enough to compute directly (`n ≤ 10` for
the decoder; always for the encoder).

### Pipeline

```
24 sentences
    → split into two SLURM jobs (12 sentences each, ~8hr time limit each)
    → for each sentence × {encoder, decoder}:
          build the Game (EncoderTextImputer / SentimentDecoderGame)
          compute raw prediction
          compute approximate SV + k-SII
          compute exact SV + k-SII (where feasible)
          save immediately (checkpoint after every sentence)
    → results/summary.json   (JSON-safe, all runs, human-readable)
      results/raw/*.pkl       (full InteractionValues objects, one per run)
```

**Resume-safe**: each sentence/model run is checkpointed individually, so
a killed or timed-out job can be resubmitted and will only redo what's
missing.

### Key files

```
src/demos/SentimentAnalysis/
├── Experiments.py              Standalone runner — no UI, no Streamlit deps
├── build_diagrams.py           Regenerates shapiq plots from summary.json (no GPU needed)
├── run_experiments_half1.sbatch   SLURM job: sentences 1-12 (negative + positive)
└── run_experiments_half2.sbatch   SLURM job: sentences 13-24 (negation + sarcasm)
```

### Running the experiments

```bash
# On the SLURM cluster, with the project's uv environment active:
sbatch run_experiments_half1.sbatch
sbatch run_experiments_half2.sbatch

# Monitor:
squeue -u $USER
tail -f logs/shapiq_sentiment_half1-<jobid>.log
```

Or run a single half manually outside SLURM:

```bash
python Experiments.py --half 1 --model both
python Experiments.py --half 2 --model encoder   # encoder only, e.g. for a quick check
```

### Building diagrams from results

`Experiments.py` deliberately has **no plotting code** — it's a pure
compute/save job meant to run headless on a cluster node. Once
`summary.json` is downloaded locally, regenerate the same sentence plots,
interaction networks, and heatmaps the app produces:

```bash
python build_diagrams.py                                   # everything
python build_diagrams.py --phenomenon sarcasm --index exact  # one phenomenon, exact values
python build_diagrams.py --run_key sar_film_en_decoder        # one specific run
```

Output goes to `figures/{run_key}_sentence.png`, `_network.png`, `_heatmap.png`.

### Hardware

GeForce RTX 2060 Super (8GB VRAM), LMU CIP SLURM cluster (`NvidiaAll`
partition, 8-hour job limit).

---

## Scientific grounding

- **shapiq**: Muschalik et al., *shapiq: Shapley Interactions for Machine
  Learning*, NeurIPS 2024
- **KernelSHAP-IQ**: Fumagalli et al., ICML 2024 — the approximator behind
  `KernelSHAPIQ`
- **Shapley interactions and linguistic structure**: Singhvi et al., 2025
  — shows interaction strength correlates with syntactic proximity and
  multi-word-expression boundaries, motivating k-SII order 2 as the right
  granularity for word-level sentiment analysis
