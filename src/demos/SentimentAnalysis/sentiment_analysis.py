"""Core logic for the Sentiment Analysis demo.

This module contains ALL computation — model loading, imputer setup,
Shapley value computation, k-SII interaction computation, and plot generation.

Two pipelines are supported:

  1. ENCODER pipeline (default) — uses TextImputer + DistilBERT:
       TextImputer(word, [MASK]) → KernelSHAP → KernelSHAPIQ(k-SII)

  2. DECODER pipeline — uses SentimentDecoderGame + Gemma/TinyLlama:
       SentimentDecoderGame(word, remove) → KernelSHAP → KernelSHAPIQ(k-SII)

Both pipelines produce the same output format so app.py stays unchanged.
The decoder pipeline uses contrastive log-odds as the value function:
    v(S) = mean(log P(positive templates | S)) - mean(log P(negative templates | S))


"""

from __future__ import annotations

import io
import os

# Prevent OpenMP/MKL thread conflicts with PyTorch — must be set before imports
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib as mpl

mpl.use("Agg")  # non-interactive backend — required for server-side rendering
import matplotlib.pyplot as plt
from PIL import Image

import shapiq
from shapiq.imputer import TextImputer
from shapiq.plot import sentence_interaction_heatmap, sentence_plot

# ── Configuration ─────────────────────────────────────────────────────────────

# Encoder model — DistilBERT fine-tuned on IMDb reviews
ENCODER_MODEL = "lvwerra/distilbert-imdb"

# Decoder models — causal LMs using contrastive log-odds value function
DECODER_MODELS = {
    "gemma":     "google/gemma-3-1b-it",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
}

RANDOM_STATE = 42   # fixed seed for reproducibility
BUDGET       = 200  # coalition budget (exact for ≤7 players due to border trick)

# ── Model caches — loaded once, reused on every call ─────────────────────────
# Without caching, every Analyse click reloads the model from disk.
_ENCODER_CACHE: dict = {}
_DECODER_CACHE: dict = {}


def _get_encoder():
    """Return cached HuggingFace encoder pipeline, loading only on first call."""
    if "clf" not in _ENCODER_CACHE:
        from transformers import pipeline as hf_pipeline
        _ENCODER_CACHE["clf"] = hf_pipeline(
            "sentiment-analysis",
            model=ENCODER_MODEL,
            device="cpu",
        )
    return _ENCODER_CACHE["clf"]


def _get_decoder(model_name: str):
    """Return cached HFModelWrapper for decoder model, loading only on first call.

    Args:
        model_name: HuggingFace model identifier string.

    Returns:
        Cached HFModelWrapper instance.
    """
    if model_name not in _DECODER_CACHE:
        from demos.shared.hf_model import HFModelWrapper
        _DECODER_CACHE[model_name] = HFModelWrapper(
            model_name=model_name,
            device="cuda",  # auto-falls back to mps/cpu
        )
    return _DECODER_CACHE[model_name]


# ── Plot helpers ──────────────────────────────────────────────────────────────

def fig_to_pil(fig: plt.Figure) -> Image.Image:
    """Convert a matplotlib Figure to a PIL Image for Gradio rendering.

    Args:
        fig: The matplotlib figure to convert.

    Returns:
        A PIL Image object with the figure content.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)  # free memory
    return img


def blank_placeholder(message: str = "") -> Image.Image:
    """Generate a dark placeholder image with a centered message.

    Args:
        message: Text to display in the center.

    Returns:
        A PIL Image with a dark background and centered message.
    """
    fig, ax = plt.subplots(figsize=(6, 1.5))
    fig.patch.set_facecolor("#0d0d1a")
    ax.set_facecolor("#0d0d1a")
    ax.text(0.5, 0.5, message, ha="center", va="center",
            color="#555577", fontsize=11, transform=ax.transAxes)
    ax.axis("off")
    return fig_to_pil(fig)


# ── Shared computation functions ──────────────────────────────────────────────
# These work with any shapiq Game object — encoder or decoder.

def compute_shapley_values(game) -> shapiq.InteractionValues:
    """Compute first-order Shapley Values using KernelSHAP.

    Works with any shapiq Game — TextImputer or SentimentDecoderGame.
    Each word receives a score for its individual contribution.

    Args:
        game: Any shapiq Game with n_players and value_function.

    Returns:
        InteractionValues containing one SV per word.
    """
    approx = shapiq.KernelSHAP(
        n=game.n_players,
        random_state=RANDOM_STATE,
    )
    return approx.approximate(budget=BUDGET, game=game)


def compute_interactions(game) -> shapiq.InteractionValues:
    """Compute pairwise k-SII interactions using KernelSHAPIQ.

    Works with any shapiq Game — TextImputer or SentimentDecoderGame.
    For every word pair, measures how much they contribute TOGETHER
    beyond their individual SVs.

    Args:
        game: Any shapiq Game with n_players and value_function.

    Returns:
        InteractionValues containing k-SII scores for all word pairs.
    """
    approx = shapiq.KernelSHAPIQ(
        n=game.n_players,
        index="k-SII",
        max_order=2,
        random_state=RANDOM_STATE,
    )
    return approx.approximate(budget=BUDGET, game=game)


def get_top_interactions(
    sii: shapiq.InteractionValues,
    words: list[str],
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """Extract the top-k pairwise interactions by absolute value.

    Args:
        sii: InteractionValues from compute_interactions().
        words: List of word strings (players).
        top_k: Number of top interactions to return.

    Returns:
        List of (word1, word2, interaction_value) tuples sorted by |value|.
    """
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


# ── Visualization functions ───────────────────────────────────────────────────

def make_sentence_plot(sv: shapiq.InteractionValues, words: list[str]) -> Image.Image:
    """Generate sentence-level attribution plot (word colors = SV values).

    Args:
        sv: First-order Shapley Values.
        words: List of word strings.

    Returns:
        PIL Image of the sentence plot.
    """
    fig, _ = sentence_plot(sv, words, show=False)
    fig.patch.set_facecolor("white")
    fig.set_size_inches(10, 2.5)
    fig.tight_layout()
    return fig_to_pil(fig)


def make_network_plot(sii: shapiq.InteractionValues, words: list[str]) -> Image.Image:
    """Generate k-SII interaction network plot.

    Nodes = words, edges = k-SII values.
    Red edges = positive (synergy), blue = negative (redundancy).

    Args:
        sii: k-SII InteractionValues.
        words: List of word strings.

    Returns:
        PIL Image of the network plot.
    """
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
    """Generate word x word k-SII interaction heatmap.

    Args:
        sii: k-SII InteractionValues.
        words: List of word strings.

    Returns:
        PIL Image of the heatmap.
    """
    try:
        fig, _ = sentence_interaction_heatmap(sii, words, show=False)
        fig.patch.set_facecolor("white")
        fig.set_size_inches(7, 6)
        fig.tight_layout()
        return fig_to_pil(fig)
    except Exception as e:  # noqa: BLE001
        return blank_placeholder(f"Heatmap error: {e}")


# ── Pipeline 1: Encoder (DistilBERT + TextImputer) ───────────────────────────

def run_pipeline(text: str) -> dict:
    """Run the encoder sentiment pipeline on a given text.

    Uses TextImputer with DistilBERT (lvwerra/distilbert-imdb).
    Value function = classification score in [-1, +1].

    Args:
        text: The input sentence to explain.

    Returns:
        Result dict with keys: label, score, words, baseline, n_players,
        sv, sii, top_interactions, img_sentence, img_network, img_heatmap,
        model_type ('encoder'), model_name.
    """
    # Build TextImputer — uses cached classifier after first call
    imputer = TextImputer(
        ENCODER_MODEL,
        text,
        segmentation="word",
        mask_strategy="mask",
        device="cpu",
    )
    imputer._classifier = _get_encoder()
    imputer._tokenizer  = imputer._classifier.tokenizer

    words = imputer.players.tolist()

    # Raw prediction on full text
    raw   = imputer._classifier(text)[0]
    label = raw["label"]
    score = raw["score"]

    # Shapley Values and interactions
    sv  = compute_shapley_values(imputer)
    sii = compute_interactions(imputer)

    return {
        "label":            label,
        "score":            score,
        "words":            words,
        "baseline":         imputer.normalization_value,
        "n_players":        imputer.n_features,
        "sv":               sv,
        "sii":              sii,
        "top_interactions": get_top_interactions(sii, words),
        "img_sentence":     make_sentence_plot(sv, words),
        "img_network":      make_network_plot(sii, words),
        "img_heatmap":      make_heatmap_plot(sii, words),
        "model_type":       "encoder",
        "model_name":       ENCODER_MODEL,
    }


# ── Pipeline 2: Decoder (Gemma / TinyLlama + SentimentDecoderGame) ───────────

def run_pipeline_decoder(text: str, model_key: str = "gemma") -> dict:
    """Run the decoder sentiment pipeline on a given text.

    Uses SentimentDecoderGame with a causal LM (Gemma or TinyLlama).
    Value function = contrastive log-odds:
        v(S) = mean(log P(positive templates | S))
             - mean(log P(negative templates | S))

    Positive values = positive sentiment, negative = negative sentiment.

    Args:
        text: The input sentence to explain.
        model_key: Either 'gemma' or 'tinyllama'. Defaults to 'gemma'.

    Returns:
        Result dict with same keys as run_pipeline() plus model_type='decoder'.
    """
    from demos.SentimentAnalysis.SentimentDecoderGame import SentimentDecoderGame

    model_name = DECODER_MODELS.get(model_key, DECODER_MODELS["gemma"])

    # Build game — reuses cached HFModelWrapper
    game = SentimentDecoderGame(
        model_name=model_name,
        input_text=text,
        segmentation="word",
        mask_strategy="remove",  # removal is more natural for decoder models
        batch_size=4,
        hf_model=_get_decoder(model_name),
    )

    words = game.players.tolist()

    # Raw prediction — score full coalition vs empty coalition
    import numpy as np
    full  = np.ones((1, game.n_players), dtype=bool)
    empty = np.zeros((1, game.n_players), dtype=bool)
    score_full  = float(game.value_function(full)[0])
    score_empty = float(game.value_function(empty)[0])

    # Convert contrastive score to label
    label = "POSITIVE" if score_full > score_empty else "NEGATIVE"
    # Normalize score to [0,1] range for display
    score = min(abs(score_full) / 10.0, 1.0)

    # Shapley Values and interactions
    sv  = compute_shapley_values(game)
    sii = compute_interactions(game)

    return {
        "label":            label,
        "score":            score,
        "words":            words,
        "baseline":         score_empty,
        "n_players":        game.n_players,
        "sv":               sv,
        "sii":              sii,
        "top_interactions": get_top_interactions(sii, words),
        "img_sentence":     make_sentence_plot(sv, words),
        "img_network":      make_network_plot(sii, words),
        "img_heatmap":      make_heatmap_plot(sii, words),
        "model_type":       "decoder",
        "model_name":       model_name,
    }