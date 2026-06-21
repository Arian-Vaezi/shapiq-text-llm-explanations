"""Core logic for the Sentiment Analysis demo.

ARCHITECTURE OVERVIEW
---------------------
This file is the single computation engine for the demo app.

    1. MODEL MANAGEMENT
       All models are preloaded at app startup into _MODEL_CACHE.
       Each cache entry stores the raw model weights + tokenizer.
       When the user clicks Analyse, we build a fresh EncoderTextImputer
       using the already-loaded weights — zero loading cost per click.

    2. SHAPLEY COMPUTATION
       Given a text and a model, computes:
           - First-order Shapley Values  (KernelSHAP)
           - Pairwise k-SII interactions (KernelSHAPIQ)

    3. VISUALIZATION
       Converts results into three PIL images:
           - Sentence plot  : words colored by SV
           - Network plot   : word-pair interaction graph
           - Heatmap        : word × word interaction matrix

TWO PIPELINES
-------------
Pipeline 1 — ENCODER:
    EncoderTextImputer + BERT-family classifier
    Masking  : replace absent words with [MASK] / <mask>
    Value fn : v(S) = P(POSITIVE|S) - P(NEGATIVE|S) ∈ [-1, +1]

Pipeline 2 — DECODER:
    SentimentDecoderGame + causal LM (TinyLlama / Gemma)
    Masking  : remove absent words entirely
    Value fn : contrastive log-odds over sentiment templates

PRELOADING STRATEGY
-------------------
All models are loaded into _MODEL_CACHE at startup via preload_models().
Structure of _MODEL_CACHE:

    _MODEL_CACHE = {
        "imdb": {
            "model":     <AutoModelForSequenceClassification on device>,
            "tokenizer": <AutoTokenizer>,
        },
        "sst2":    { ... },
        "roberta": { ... },
        "tinyllama": { ... },   # HFModelWrapper
        "gemma":     { ... },   # HFModelWrapper
    }

When run_pipeline() is called, _get_encoder_imputer() builds a fresh
EncoderTextImputer passing the preloaded weights — no loading happens,
only v(∅) is computed (one forward pass) before Shapley starts.
"""

from __future__ import annotations

import io
import os

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import shapiq
from shapiq.plot import sentence_interaction_heatmap, sentence_plot

# ── Model registries ──────────────────────────────────────────────────────────

ENCODER_MODELS: dict[str, dict] = {
    "imdb": {
        "model_id":       "lvwerra/distilbert-imdb",
        "label":          "DistilBERT IMDb",
        "domain":         "Movie reviews (IMDb)",
        "positive_label": "POSITIVE",
        "negative_label": "NEGATIVE",
        "description":    (
            "Trained on IMDb movie reviews. Strong movie-domain priors. "
            "Binary: POSITIVE / NEGATIVE."
        ),
    },
    "sst2": {
        "model_id":       "distilbert-base-uncased-finetuned-sst-2-english",
        "label":          "DistilBERT SST-2",
        "domain":         "General English (SST-2)",
        "positive_label": "POSITIVE",
        "negative_label": "NEGATIVE",
        "description":    (
            "Trained on Stanford Sentiment Treebank. "
            "General English sentences. Binary: POSITIVE / NEGATIVE."
        ),
    },
    "roberta": {
        "model_id":       "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "label":          "RoBERTa Twitter",
        "domain":         "Social media (Twitter)",
        "positive_label": "positive",
        "negative_label": "negative",
        "description":    (
            "RoBERTa trained on 124M tweets. "
            "3-class: positive / neutral / negative. "
            "Best for informal language and social media text."
        ),
    },
}

DECODER_MODELS: dict[str, str] = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "gemma3":     "google/gemma-3-1b-it",
    "gemma4":    "google/gemma-4-E2B-it"
    
}

RANDOM_STATE: int = 42
BUDGET_ENCODER: int = 256
BUDGET_DECODER: int = 10

# ── Model cache ───────────────────────────────────────────────────────────────
# Populated by preload_models() at app startup.
# Structure:
#   encoder keys → {"model": ..., "tokenizer": ...}
#   decoder keys → {"wrapper": HFModelWrapper}

_MODEL_CACHE: dict[str, dict] = {}


def preload_models() -> None:
    """Preload all models into memory at app startup.

    Call this ONCE before launching the Gradio app.
    After this returns, all subsequent calls to run_pipeline()
    and run_pipeline_decoder() will be instant (no loading cost).

    Loads:
        Encoders : DistilBERT IMDb, DistilBERT SST-2, RoBERTa Twitter
        Decoders : TinyLlama, Gemma

    Device selection is automatic:
        GPU (CUDA)  → if available
        MPS (Apple) → if on Mac with M-series chip
        CPU         → fallback
    """
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"[preload] Using device: {device}")

    # ── Encoder models ────────────────────────────────────────────────────────
    for key, info in ENCODER_MODELS.items():
        if key in _MODEL_CACHE:
            print(f"[preload] {info['label']} already loaded, skipping.")
            continue

        print(f"[preload] Loading {info['label']}...")
        tokenizer = AutoTokenizer.from_pretrained(info["model_id"])
        model     = AutoModelForSequenceClassification.from_pretrained(info["model_id"])
        model.to(device)
        model.eval()

        _MODEL_CACHE[key] = {
            "model":     model,
            "tokenizer": tokenizer,
        }
        print(f"[preload]   ✓ {info['label']} ready on {device}")

    # ── Decoder models ────────────────────────────────────────────────────────
    for key, model_name in DECODER_MODELS.items():
        if key in _MODEL_CACHE:
            print(f"[preload] {model_name} already loaded, skipping.")
            continue

        print(f"[preload] Loading {model_name}...")
        from demos.SentimentAnalysis.HFModelWrapper import HFModelWrapper
        wrapper = HFModelWrapper(model_name=model_name, device=device)
        _MODEL_CACHE[key] = {"wrapper": wrapper}
        print(f"[preload]   ✓ {model_name} ready")

    print("[preload] All models loaded and ready.")


# ── Imputer factory ───────────────────────────────────────────────────────────

def _get_encoder_imputer(text: str, model_key: str = "sst2"):
    """Build an EncoderTextImputer using preloaded model weights.

    If models have been preloaded via preload_models(), this is instant —
    only v(∅) is computed (one forward pass).

    If preload_models() was not called, falls back to loading on demand.

    Args:
        text: The input sentence to explain.
        model_key: One of 'imdb', 'sst2', 'roberta'.

    Returns:
        Ready-to-use EncoderTextImputer.
    """
    from demos.SentimentAnalysis.EncoderTextImputer import EncoderTextImputer

    model_info = ENCODER_MODELS[model_key]

    if model_key in _MODEL_CACHE:
        # Fast path: reuse preloaded weights
        cached = _MODEL_CACHE[model_key]
        return EncoderTextImputer(
            model_name=model_info["model_id"],
            input_text=text,
            preloaded_model=cached["model"],
            preloaded_tokenizer=cached["tokenizer"],
            positive_label=model_info["positive_label"],
            negative_label=model_info["negative_label"],
        )
    else:
        # Slow path: load on demand (fallback if preload was not called)
        print(f"[warning] {model_info['label']} not preloaded — loading now...")
        return EncoderTextImputer(
            model_name=model_info["model_id"],
            input_text=text,
            positive_label=model_info["positive_label"],
            negative_label=model_info["negative_label"],
        )


def _get_decoder(model_key: str):
    """Return preloaded HFModelWrapper for a decoder model.

    Args:
        model_key: 'tinyllama' or 'gemma'.

    Returns:
        HFModelWrapper instance.
    """
    if model_key in _MODEL_CACHE:
        return _MODEL_CACHE[model_key]["wrapper"]

    # fallback
    model_name = DECODER_MODELS[model_key]
    print(f"[warning] {model_name} not preloaded — loading now...")
    from demos.SentimentAnalysis.HFModelWrapper import HFModelWrapper
    wrapper = HFModelWrapper(model_name=model_name)
    _MODEL_CACHE[model_key] = {"wrapper": wrapper}
    return wrapper


# ── Shapley computation ───────────────────────────────────────────────────────

def compute_shapley_values(game) -> shapiq.InteractionValues:
    """Compute first-order Shapley Values (KernelSHAP).

    φ_i measures average marginal contribution of word i:
        φ_i > 0 → pushes toward POSITIVE
        φ_i < 0 → pushes toward NEGATIVE
    """
    from demos.SentimentAnalysis.SentimentDecoderGame import SentimentDecoderGame
    budget = BUDGET_DECODER if isinstance(game, SentimentDecoderGame) else BUDGET_ENCODER

    return shapiq.KernelSHAP(
        n=game.n_players,
        random_state=RANDOM_STATE,
    ).approximate(budget=budget, game=game)


def compute_interactions(game) -> shapiq.InteractionValues:
    """Compute pairwise k-SII interactions (KernelSHAPIQ).

    k-SII(i,j) measures synergy between word i and word j:
        > 0 → pair contributes MORE together than individually
        < 0 → pair is redundant
        ≈ 0 → words act independently
    """
    from demos.SentimentAnalysis.SentimentDecoderGame import SentimentDecoderGame
    budget = BUDGET_DECODER if isinstance(game, SentimentDecoderGame) else BUDGET_ENCODER

    return shapiq.KernelSHAPIQ(
        n=game.n_players,
        index="k-SII",
        max_order=2,
        random_state=RANDOM_STATE,
    ).approximate(budget=budget, game=game)


def get_top_interactions(
    sii: shapiq.InteractionValues,
    words: list[str],
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """Return top-k pairwise interactions sorted by absolute value."""
    order2 = {k: v for k, v in sii.interaction_lookup.items() if len(k) == 2}
    sorted_pairs = sorted(
        order2.items(),
        key=lambda x: abs(sii.values[x[1]]),
        reverse=True,
    )[:top_k]
    return [
        (words[indices[0]], words[indices[1]], sii.values[idx])
        for indices, idx in sorted_pairs
    ]


# ── Visualization ─────────────────────────────────────────────────────────────

def fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Convert matplotlib Figure to PIL Image and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    return img


def blank_placeholder(message: str = "") -> Image.Image:
    """Return a plain placeholder image with a message."""
    fig, ax = plt.subplots(figsize=(6, 1.5))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    ax.text(0.5, 0.5, message, ha="center", va="center",
            color="#555577", fontsize=11, transform=ax.transAxes)
    ax.axis("off")
    return fig_to_pil(fig)


def make_sentence_plot(sv: shapiq.InteractionValues, words: list[str]) -> Image.Image:
    """Word-colored sentence plot — colors = first-order Shapley Values."""
    try:
        fig, _ = sentence_plot(sv, words, show=False)
        fig.patch.set_facecolor("white")
        fig.set_size_inches(10, 2.5)
        fig.tight_layout()
        return fig_to_pil(fig)
    except Exception as e:  # noqa: BLE001
        return blank_placeholder(f"Sentence plot error: {e}")


def make_network_plot(sii: shapiq.InteractionValues, words: list[str]) -> Image.Image:
    """Interaction network — nodes=words, edges=k-SII values."""
    try:
        result = sii.plot_network(feature_names=words, show=False)
        if result is None:
            return blank_placeholder("Network plot unavailable")
        fig = result[0] if isinstance(result, tuple) else result
        fig.patch.set_facecolor("white")
        fig.set_size_inches(7, 7)
        return fig_to_pil(fig)
    except Exception as e:  # noqa: BLE001
        return blank_placeholder(f"Network error: {e}")


def make_heatmap_plot(sii: shapiq.InteractionValues, words: list[str]) -> Image.Image:
    """Word × word k-SII interaction heatmap."""
    try:
        fig, _ = sentence_interaction_heatmap(sii, words, show=False)
        fig.patch.set_facecolor("white")
        fig.set_size_inches(7, 6)
        fig.tight_layout()
        return fig_to_pil(fig)
    except Exception as e:  # noqa: BLE001
        return blank_placeholder(f"Heatmap error: {e}")


# ── Pipeline 1: Encoder ───────────────────────────────────────────────────────

def run_pipeline(text: str, model_key: str = "sst2") -> dict:
    """Run the encoder sentiment analysis pipeline.

    Called by the app on every user click.
    Model loading cost = 0 if preload_models() was called at startup.

    Steps:
        1. Build EncoderTextImputer with preloaded weights
           (only v(∅) is computed — one forward pass)
        2. Get raw prediction on full sentence
        3. Compute Shapley Values
        4. Compute k-SII interactions
        5. Extract top-5 interactions
        6. Generate three plots

    Args:
        text: Input sentence to explain.
        model_key: 'imdb', 'sst2', or 'roberta'.

    Returns:
        dict with all results and images for the app to render.
    """
    model_info = ENCODER_MODELS[model_key]

    # Step 1 — instant if preloaded, slow only on first call otherwise
    imputer = _get_encoder_imputer(text, model_key)
    words   = imputer.players.tolist()

    # Step 2 — raw prediction on full sentence
    grand  = np.ones((1, imputer.n_players), dtype=bool)
    v_full = float(imputer(grand)[0]) + imputer.normalization_value
    label  = "POSITIVE" if v_full >= 0 else "NEGATIVE"
    score  = abs(v_full)

    # Steps 3 & 4 — Shapley computation (this is where the time is spent)
    sv  = compute_shapley_values(imputer)
    sii = compute_interactions(imputer)

    return {
        "label":             label,
        "score":             score,
        "words":             words,
        "baseline":          imputer.normalization_value,
        "n_players":         imputer.n_players,
        "sv":                sv,
        "sii":               sii,
        "top_interactions":  get_top_interactions(sii, words),
        "img_sentence":      make_sentence_plot(sv, words),
        "img_network":       make_network_plot(sii, words),
        "img_heatmap":       make_heatmap_plot(sii, words),
        "model_type":        "encoder",
        "model_name":        model_info["model_id"],
        "model_key":         model_key,
        "model_domain":      model_info["domain"],
        "model_description": model_info["description"],
    }


# ── Pipeline 2: Decoder ───────────────────────────────────────────────────────

def run_pipeline_decoder(text: str, model_key: str = "tinyllama") -> dict:
    """Run the decoder sentiment pipeline.

    Uses SentimentDecoderGame with a causal LM.
    Absent words are removed (not masked).
    Value function: contrastive log-odds over sentiment templates.

    Args:
        text: Input sentence to explain.
        model_key: 'tinyllama' or 'gemma'.

    Returns:
        Same dict structure as run_pipeline().
    """
    from demos.SentimentAnalysis.SentimentDecoderGame import SentimentDecoderGame

    model_name = DECODER_MODELS[model_key]

    game = SentimentDecoderGame(
    model_name=model_name,
    input_text=text,
    hf_model=_get_decoder(model_key),
    )

    words      = game.players.tolist()
    grand      = np.ones((1, game.n_players), dtype=bool)
    score_full = float(game.value_function(grand)[0])
    label      = "POSITIVE" if score_full > 0 else "NEGATIVE"
    score      = min(abs(score_full) / 10.0, 1.0)

    sv  = compute_shapley_values(game)
    sii = compute_interactions(game)

    return {
        "label":             label,
        "score":             score,
        "words":             words,
        "baseline":          game.normalization_value,
        "n_players":         game.n_players,
        "sv":                sv,
        "sii":               sii,
        "top_interactions":  get_top_interactions(sii, words),
        "img_sentence":      make_sentence_plot(sv, words),
        "img_network":       make_network_plot(sii, words),
        "img_heatmap":       make_heatmap_plot(sii, words),
        "model_type":        "decoder",
        "model_name":        model_name,
        "model_key":         model_key,
        "model_domain":      "General (domain-agnostic templates)",
        "model_description": (
            "Contrastive log-odds with general sentiment templates. "
            "Works on any domain — not restricted to movie reviews."
        ),
    }