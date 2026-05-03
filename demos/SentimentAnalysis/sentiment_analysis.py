"""
sentiment_analysis.py
======================
Core logic for the Sentiment Analysis demo.

This module contains ALL computation — model loading, TextImputer setup,
Shapley value computation, k-SII interaction computation, and plot generation.

It is intentionally decoupled from the UI (app.py) so that:
  - The same logic can be used in the notebook AND the Gradio app
  - The UI can be swapped (Gradio → Streamlit → CLI) without touching this file
  - Each function can be tested independently

Pipeline (mirrors 01_sentiment_analysis_interactions.ipynb):
    TextImputer(model, text, segmentation="word")
        ↓
    KernelSHAP → InteractionValues (first-order Shapley Values)
        ↓
    KernelSHAPIQ(k-SII, order=2) → InteractionValues (pairwise interactions)
        ↓
    sentence_plot / plot_network / sentence_interaction_heatmap
"""

from __future__ import annotations

import io
import os

# Prevent OpenMP/MKL thread conflicts with PyTorch — must be set before imports
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required for server-side rendering
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import shapiq
from shapiq.imputer import TextImputer
from shapiq.plot import sentence_plot, sentence_interaction_heatmap

# ── Configuration (same as notebook) ─────────────────────────────────────────
MODEL_NAME   = "lvwerra/distilbert-imdb"  # DistilBERT fine-tuned on IMDb reviews
RANDOM_STATE = 42                          # fixed seed for reproducibility
BUDGET       = 200                         # coalition budget (exact for ≤7 players)


# ── Plot helper ───────────────────────────────────────────────────────────────

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
    plt.close(fig)  # free memory — important in long-running server apps
    return img


def blank_placeholder(message: str = "") -> Image.Image:
    """Generate a placeholder image with a centered message.

    Used when a plot is unavailable or computation hasn't started yet.

    Args:
        message: Text to display in the center of the placeholder.

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


# ── Step 1: TextImputer setup ─────────────────────────────────────────────────

def build_imputer(text: str) -> TextImputer:
    """Build and return a TextImputer for the given input text.

    This is Step 1 of the pipeline (mirrors notebook Cell 4).
    Each word in the sentence becomes one player in the cooperative game.
    Absent words are replaced with [MASK] when evaluating coalitions.

    Args:
        text: The input sentence to explain.

    Returns:
        A configured TextImputer instance ready for Shapley computation.
    """
    return TextImputer(
        MODEL_NAME,
        text,
        segmentation="word",   # each word = one player
        mask_strategy="mask",  # absent words → [MASK] token
        device="cpu",          # CPU inference — no GPU required
    )


# ── Step 2: Raw model prediction ─────────────────────────────────────────────

def get_prediction(imputer: TextImputer, text: str) -> tuple[str, float]:
    """Get the raw model prediction for the full input text.

    Calls the classifier directly on the original sentence (no masking).
    This is separate from the Shapley computation which evaluates subsets.

    Args:
        imputer: A configured TextImputer (used to access the classifier).
        text: The original input sentence.

    Returns:
        A tuple of (label, score) where label is "POSITIVE" or "NEGATIVE"
        and score is the model confidence in [0, 1].
    """
    result = imputer._classifier(text)[0]
    return result["label"], result["score"]


# ── Step 3: Shapley Values (first-order) ─────────────────────────────────────

def compute_shapley_values(imputer: TextImputer) -> shapiq.InteractionValues:
    """Compute first-order Shapley Values using KernelSHAP.

    This is Step 3 of the pipeline (mirrors notebook Cell 6).
    Each word receives a score indicating its individual contribution
    to pushing the prediction above the baseline (all-masked) score.

    For ≤7 players and budget=200, the border trick gives exact results.

    Args:
        imputer: A configured TextImputer instance.

    Returns:
        InteractionValues containing one SV per word.
    """
    approx = shapiq.KernelSHAP(
        n=imputer.n_features,
        random_state=RANDOM_STATE,
    )
    return approx.approximate(budget=BUDGET, game=imputer)


# ── Step 4: k-SII Pairwise Interactions ──────────────────────────────────────

def compute_interactions(imputer: TextImputer) -> shapiq.InteractionValues:
    """Compute pairwise Shapley Interactions using KernelSHAPIQ (k-SII, order 2).

    This is Step 4 of the pipeline (mirrors notebook Cell 10).
    For every pair of words, k-SII measures how much that pair contributes
    TOGETHER beyond what each word contributes individually:

      - Positive interaction = synergy (e.g. "not" + "bad" flip sentiment together)
      - Negative interaction = redundancy (e.g. "loved" + "amazing" compete)

    Why k-SII and not STII or FSII?
      - k-SII at order 2 gives directly interpretable pairwise scores
      - Efficient approximation even for longer sentences
      - Clean side-by-side comparison with first-order SVs

    Args:
        imputer: A configured TextImputer instance.

    Returns:
        InteractionValues containing SVs (order 1) and pairwise k-SII (order 2).
    """
    approx = shapiq.KernelSHAPIQ(
        n=imputer.n_features,
        index="k-SII",
        max_order=2,
        random_state=RANDOM_STATE,
    )
    return approx.approximate(budget=BUDGET, game=imputer)


# ── Step 5: Extract top interactions ─────────────────────────────────────────

def get_top_interactions(
    sii: shapiq.InteractionValues,
    words: list[str],
    top_k: int = 5,
) -> list[tuple[str, str, float]]:
    """Extract the top-k pairwise interactions by absolute value.

    Filters the k-SII results to only order-2 interactions and sorts
    them by magnitude so the most impactful word pairs appear first.

    Args:
        sii: InteractionValues from compute_interactions().
        words: List of word strings (players).
        top_k: Number of top interactions to return.

    Returns:
        List of (word1, word2, interaction_value) tuples sorted by |value|.
    """
    # filter to order-2 interactions only
    order2 = {
        k: v
        for k, v in sii.interaction_lookup.items()
        if len(k) == 2
    }

    # sort by absolute value — strongest interactions first
    sorted_pairs = sorted(
        order2.items(),
        key=lambda x: abs(sii.values[x[1]]),
        reverse=True,
    )[:top_k]

    return [
        (words[indices[0]], words[indices[1]], sii.values[idx])
        for indices, idx in sorted_pairs
    ]


# ── Step 6: Visualization ────────────────────────────────────────────────────

def make_sentence_plot(
    sv: shapiq.InteractionValues,
    words: list[str],
) -> Image.Image:
    """Generate the sentence-level attribution plot.

    Colors each word by its first-order Shapley Value.
    Pink/red = positive contribution, blue = negative contribution.

    Mirrors notebook Cell 7 (sentence_plot).

    Args:
        sv: First-order Shapley Values from compute_shapley_values().
        words: List of word strings.

    Returns:
        PIL Image of the sentence plot.
    """
    fig, _ = sentence_plot(sv, words, show=False)
    fig.patch.set_facecolor("white")
    fig.set_size_inches(10, 2.5)
    fig.tight_layout()
    return fig_to_pil(fig)


def make_network_plot(
    sii: shapiq.InteractionValues,
    words: list[str],
) -> Image.Image:
    """Generate the k-SII interaction network plot.

    Nodes = words, edges = pairwise k-SII values.
    Edge thickness and color encode interaction strength and sign:
      - Red/pink edges = positive (synergy)
      - Blue edges = negative (redundancy)

    The headline result "(not, bad) = +2.878" appears as a single
    dominant red edge when analysing "This film is not bad at all".

    Mirrors notebook Cell 11 (plot_network).

    Args:
        sii: k-SII InteractionValues from compute_interactions().
        words: List of word strings.

    Returns:
        PIL Image of the network plot, or a placeholder if unavailable.
    """
    try:
        result = sii.plot_network(feature_names=words, show=False)
        if result is None:
            return blank_placeholder("Network plot unavailable")
        # plot_network returns either a Figure or a (fig, ax) tuple
        fig = result[0] if isinstance(result, tuple) else result
        fig.patch.set_facecolor("white")
        fig.set_size_inches(7, 7)
        return fig_to_pil(fig)
    except Exception as e:
        return blank_placeholder(f"Network error: {e}")


def make_heatmap_plot(
    sii: shapiq.InteractionValues,
    words: list[str],
) -> Image.Image:
    """Generate the pairwise interaction heatmap.

    A word × word matrix where each cell shows the k-SII value
    for that pair. Strong interactions appear as bright red (positive)
    or bright blue (negative) cells.

    When (not, bad) = +2.878, that cell dominates the colorscale —
    visually demonstrating how extreme the negation interaction is.

    Mirrors notebook Cell 11 (sentence_interaction_heatmap).

    Args:
        sii: k-SII InteractionValues from compute_interactions().
        words: List of word strings.

    Returns:
        PIL Image of the heatmap, or a placeholder if unavailable.
    """
    try:
        fig, _ = sentence_interaction_heatmap(sii, words, show=False)
        fig.patch.set_facecolor("white")
        fig.set_size_inches(7, 6)
        fig.tight_layout()
        return fig_to_pil(fig)
    except Exception as e:
        return blank_placeholder(f"Heatmap error: {e}")


# ── Full pipeline (called by app.py) ─────────────────────────────────────────

def run_pipeline(text: str) -> dict:
    """Run the full sentiment explanation pipeline on a given text.

    This is the single entry point called by app.py.
    It runs all steps in order and returns a structured result dict
    that the UI can render without knowing any shapiq internals.

    Steps:
        1. Build TextImputer (word-level players, [MASK] strategy)
        2. Get raw model prediction (label + confidence)
        3. Compute first-order Shapley Values (KernelSHAP)
        4. Compute pairwise k-SII interactions (KernelSHAPIQ)
        5. Extract top interactions
        6. Generate all three plots

    Args:
        text: The input sentence to explain.

    Returns:
        A dict with keys:
            - label (str): "POSITIVE" or "NEGATIVE"
            - score (float): model confidence
            - words (list[str]): word players
            - baseline (float): empty coalition score
            - n_players (int): number of words
            - sv (InteractionValues): first-order Shapley Values
            - sii (InteractionValues): k-SII pairwise interactions
            - top_interactions (list): top-5 (w1, w2, value) tuples
            - img_sentence (PIL.Image): sentence attribution plot
            - img_network (PIL.Image): interaction network plot
            - img_heatmap (PIL.Image): interaction heatmap
    """
    # Step 1 — build imputer
    imputer = build_imputer(text)
    words   = imputer.players.tolist()

    # Step 2 — raw prediction
    label, score = get_prediction(imputer, text)

    # Step 3 — Shapley Values
    sv = compute_shapley_values(imputer)

    # Step 4 — k-SII interactions
    sii = compute_interactions(imputer)

    # Step 5 — top interactions
    top_interactions = get_top_interactions(sii, words, top_k=5)

    # Step 6 — plots
    img_sentence = make_sentence_plot(sv, words)
    img_network  = make_network_plot(sii, words)
    img_heatmap  = make_heatmap_plot(sii, words)

    return {
        "label":            label,
        "score":            score,
        "words":            words,
        "baseline":         imputer.normalization_value,
        "n_players":        imputer.n_features,
        "sv":               sv,
        "sii":              sii,
        "top_interactions": top_interactions,
        "img_sentence":     img_sentence,
        "img_network":      img_network,
        "img_heatmap":      img_heatmap,
    }