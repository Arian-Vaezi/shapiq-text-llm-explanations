"""
app.py
=======
Streamlit web application for the Sentiment Analysis demo.

This file contains ONLY the UI logic — layout, styling, and event routing.
All computation logic lives in sentiment_analysis.py.

Two pipelines are supported:
  - Encoder : BERT-family classifier with [MASK] imputation
  - Decoder : Causal LM with contrastive log-odds value function

To run:
    streamlit run app.py

Related files:
    sentiment_analysis.py    <- core pipeline logic (encoder + decoder)
    EncoderTextImputer.py    <- Game subclass for encoder models
    SentimentDecoderGame.py  <- decoder value function
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sentiment_analysis import (  # noqa: E402
    ENCODER_MODELS,
    preload_models,
    run_pipeline,
    run_pipeline_decoder,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Explainer",
    page_icon="🔍",
    layout="wide",
)

# ── Preload all models once at startup ────────────────────────────────────────
# This runs only on the first page load.
# After this, run_pipeline() has zero model-loading cost per click.
if "models_loaded" not in st.session_state:
    with st.spinner("Loading models… (this happens once at startup)"):
        preload_models()
    st.session_state["models_loaded"] = True

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --accent: #5b6cff;
    --accent-soft: #eef0ff;
    --accent-dark: #4338ca;
    --green: #10b981;
    --red: #ef4444;
    --muted: #6b7280;
    --border: #e5e7eb;
    --surface: #ffffff;
    --bg: #f7f8fc;
}

#MainMenu, footer, header { visibility: hidden; }
.stApp { background: var(--bg); }

.hero {
    background: linear-gradient(135deg, #ffffff 0%, #f1f4ff 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1rem;
}
.hero h1 {
    font-size: 2.2rem;
    font-weight: 800;
    color: #111827;
    letter-spacing: -0.03em;
    margin: 0 0 0.5rem 0;
}
.hero p {
    color: var(--muted);
    font-size: 1rem;
    margin: 0;
    line-height: 1.6;
}

.step-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin: 1.5rem 0 0.4rem 0;
}
.step-number {
    background: var(--accent-soft);
    color: var(--accent-dark);
    border-radius: 50%;
    width: 1.8rem;
    height: 1.8rem;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.8rem;
    flex-shrink: 0;
}
.step-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #111827;
}

.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
}
.metric-row {
    display: flex;
    gap: 1rem;
    margin-top: 0.8rem;
    flex-wrap: wrap;
}
.metric-chip {
    background: #f9fafb;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.5rem 0.8rem;
    font-size: 0.82rem;
}
.metric-chip span {
    display: block;
    color: var(--muted);
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
}
.metric-chip strong {
    display: block;
    color: #111827;
    font-weight: 700;
    margin-top: 0.15rem;
}

.seg-box {
    background: #f9fafb;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0 1rem 0;
    font-size: 0.9rem;
}
.seg-box .seg-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 0.4rem;
}
.seg-token {
    display: inline-block;
    background: var(--accent-soft);
    color: var(--accent-dark);
    border-radius: 6px;
    padding: 0.2rem 0.5rem;
    margin: 0.15rem;
    font-weight: 700;
    font-size: 0.85rem;
    font-family: monospace;
}
.mask-token {
    display: inline-block;
    background: #fef3c7;
    color: #92400e;
    border-radius: 6px;
    padding: 0.2rem 0.5rem;
    margin: 0.15rem;
    font-weight: 700;
    font-size: 0.85rem;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)


# ── Example sentences ─────────────────────────────────────────────────────────
DEMO_EXAMPLES = [
    "This film is not bad at all",
    "I really loved this amazing film",
    "What a magnificent disaster of a film",
    "The acting was superb but the plot was absolutely terrible",
    "I would not say this was a bad experience",
]


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("**Model**")
    model_choice = st.radio(
        "Model",
        options=["encoder", "tinyllama", "gemma3", "gemma4"],
        format_func=lambda x: {
            "encoder":   "DistilBERT (encoder · fast · CPU)",
            "tinyllama": "TinyLlama 1.1B  (decoder · lighter)",
            "gemma3":     "Gemma 3 1B  (decoder · GPU recommended)",
            "gemma4":    "Gemma 4 E2B  (decoder · 16GB VRAM)",
            
        }[x],
        label_visibility="collapsed",
    )

    # Encoder sub-selector (only shown when encoder is selected)
    encoder_key = "imdb"
    if model_choice == "encoder":
        st.markdown("**Encoder variant**")
        encoder_key = st.radio(
            "Encoder variant",
            options=list(ENCODER_MODELS.keys()),
            format_func=lambda k: ENCODER_MODELS[k]["label"],
            label_visibility="collapsed",
        )
        st.caption(ENCODER_MODELS[encoder_key]["description"])

    # Value function formula — updated to match actual implementation
    with st.expander("📐 Value function"):
        if model_choice == "encoder":
            st.markdown("**Contrastive classification score**")
            st.latex(
                r"v(S) = P(\text{POS} \mid \text{masked}(S))"
                r"- P(\text{NEG} \mid \text{masked}(S))"
            )
            st.markdown("""
            | Value | Meaning |
            |-------|---------|
            | `+1`  | strongly positive |
            | ` 0`  | uncertain / balanced |
            | `−1`  | strongly negative |

            **Masking strategy:** absent words → `[MASK]` token  
            **Baseline v(∅):** all words masked → model score on `[MASK] [MASK] ...`  
            **Normalization:** all coalition values centered around v(∅)
            """)
        else:
            model_label = "Gemma 3 1B" if model_choice == "gemma" else "TinyLlama 1.1B"
            st.markdown(f"**{model_label} — contrastive log-odds**")
            st.latex(
                r"v(S) = \frac{1}{|T^+|}\sum_{t \in T^+} \log p(t \mid S)"
                r"- \frac{1}{|T^-|}\sum_{t \in T^-} \log p(t \mid S)"
            )
            st.markdown("""
            **T⁺ (positive templates):**
            > " This is positive." / " I loved it." / " Excellent!"

            **T⁻ (negative templates):**
            > " This is negative." / " I hated it." / " Terrible!"

            **Masking strategy:** absent words → **removed entirely**  
            **Baseline v(∅):** empty string (no words)  
            **Score range:** unbounded (typically ±5 to ±20)
            """)

    st.divider()

    st.markdown("**📓 Examples**")
    st.caption("Click to load.")
    for i, example in enumerate(DEMO_EXAMPLES):
        label = example[:45] + "…" if len(example) > 45 else example
        if st.button(label, key=f"example_{i}", use_container_width=True):
            st.session_state["text_input"] = example

    st.divider()

    st.markdown("**Pipeline**")
    st.markdown("""
    1. **Segmentation** — split sentence into word players
    2. **Masking** — build masked sentence per coalition
    3. **KernelSHAP** — first-order Shapley Values
    4. **KernelSHAPIQ** — pairwise k-SII interactions
    5. **Plots** — sentence · network · heatmap
    """)


# ── Main content ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>🔍 Sentiment Explainer</h1>
    <p>
        Explain how a sentiment model makes its decision using
        <strong>Shapley Values</strong> and <strong>Shapley Interactions (k-SII)</strong>.
        Individual word contributions tell part of the story —
        interactions reveal what first-order explanations miss.
    </p>
</div>
""", unsafe_allow_html=True)

# Text input
col_input, col_btn = st.columns([5, 1])
with col_input:
    text = st.text_input(
        "Input sentence",
        key="text_input",
        placeholder="Example: This film is not bad at all",
        label_visibility="collapsed",
    )
with col_btn:
    st.markdown("<div style='margin-top:0.3rem'></div>", unsafe_allow_html=True)
    run_btn = st.button("Analyse →", type="primary", use_container_width=True)


# ── Analysis ──────────────────────────────────────────────────────────────────

if run_btn and text.strip():

    with st.spinner("Computing Shapley Values and k-SII interactions…"):
        if model_choice == "encoder":
            result = run_pipeline(text.strip(), model_key=encoder_key)
        else:
            result = run_pipeline_decoder(text.strip(), model_key=model_choice)

    # Unpack
    label    = result["label"]
    score    = result["score"]
    words    = result["words"]
    baseline = result["baseline"]
    n        = result["n_players"]
    sv       = result["sv"]
    top_int  = result["top_interactions"]
    mtype    = result["model_type"]
    mname    = result["model_name"].split("/")[-1]
    mask_tok = "[MASK]" if mtype == "encoder" else "removed"

    emoji = "😊" if label == "POSITIVE" else "😠"
    color = "#10b981" if label == "POSITIVE" else "#ef4444"
    badge = "Encoder" if mtype == "encoder" else "Decoder"

    # ── Prediction result card ────────────────────────────────────────────────
    st.markdown(f"""
    <div class="result-card">
        <div style="font-size:1.6rem; font-weight:800; color:{color};">
            {emoji} {label} &nbsp;
            <span style="font-size:1.2rem;">{score:.3f}</span>
            &nbsp;
            <span style="background:{color}22; color:{color}; padding:0.2rem 0.6rem;
                  border-radius:999px; font-size:0.8rem; font-weight:700;">
                {badge} · {mname}
            </span>
        </div>
        <div class="metric-row">
            <div class="metric-chip">
                <span>v(∅) baseline</span>
                <strong>{baseline:+.4f}</strong>
            </div>
            <div class="metric-chip">
                <span>v(N) full sentence</span>
                <strong>{score:+.4f}</strong>
            </div>
            <div class="metric-chip">
                <span>Players</span>
                <strong>{n} words</strong>
            </div>
            <div class="metric-chip">
                <span>Masking</span>
                <strong>{mask_tok}</strong>
            </div>
            <div class="metric-chip">
                <span>Index</span>
                <strong>k-SII order 2</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Segmentation details ──────────────────────────────────────────────────
    st.markdown("""
    <div class="step-header">
        <span class="step-number">0</span>
        <span class="step-title">Segmentation — how the sentence becomes players</span>
    </div>
    """, unsafe_allow_html=True)

    # Build token display
    tokens_html = "".join(
        f'<span class="seg-token">{w}</span>' for w in words
    )

    # Show one masked example — mask all except first two words
    example_coalition = [True, True] + [False] * (n - 2)
    masked_words_html = "".join(
        f'<span class="seg-token">{w}</span>'
        if example_coalition[i]
        else f'<span class="mask-token">{mask_tok}</span>'
        for i, w in enumerate(words)
    )

    st.markdown(f"""
    <div class="seg-box">
        <div class="seg-label">Word players (n = {n})</div>
        {tokens_html}
    </div>
    <div class="seg-box">
        <div class="seg-label">
            Masking strategy — absent words →
            <code>{"[MASK]" if mtype == "encoder" else "removed"}</code>
            &nbsp;|&nbsp; Example coalition S = {{word 0, word 1}}
        </div>
        {masked_words_html}
    </div>
    """, unsafe_allow_html=True)

    if mtype == "encoder":
        st.caption(
            f"The sentence is split into **{n} word players**. "
            f"For each of the ~{2**n:,} possible coalitions S, absent words are replaced "
            f"with `[MASK]`. The model scores each masked sentence and returns "
            f"`P(POSITIVE|S) - P(NEGATIVE|S) ∈ [-1, +1]`. "
            f"KernelSHAP samples {result.get('budget', 256)} coalitions to approximate "
            f"the full Shapley computation."
        )
    else:
        st.caption(
            f"The sentence is split into **{n} word players**. "
            f"For each coalition S, absent words are **removed entirely** "
            f"(decoder models have no [MASK] token). "
            f"The model scores each truncated sentence via contrastive log-odds "
            f"over positive and negative templates."
        )

    # ── Step 1: Shapley Values ────────────────────────────────────────────────
    st.markdown("""
    <div class="step-header">
        <span class="step-number">1</span>
        <span class="step-title">Individual word contributions — Shapley Values</span>
    </div>
    """, unsafe_allow_html=True)

    if mtype == "encoder":
        st.caption(
            "Each bar shows how much a word shifts `v(S)` on average across all coalitions. "
            "The value function is `v(S) = P(POSITIVE|S) - P(NEGATIVE|S)`. "
            "After normalization, v(∅) = 0 — so each SV is a contribution "
            "*relative to the all-masked baseline*. "
            "**Positive SV** → word pushes toward POSITIVE. "
            "**Negative SV** → word pushes toward NEGATIVE. "
            "Words near zero have little individual effect — but may interact strongly (see Step 2)."
        )
    else:
        st.caption(
            "Each bar shows how much a word shifts the contrastive log-odds score on average. "
            "The value function is `v(S) = mean log P(T⁺|S) - mean log P(T⁻|S)`. "
            "After normalization, v(∅) = 0 (empty string baseline). "
            "**Positive SV** → word steers completions toward positive templates. "
            "**Negative SV** → word steers toward negative templates."
        )

    st.image(result["img_sentence"], use_container_width=True)

    # SV table with actual values
    sv_rows = sorted(
        zip(words, sv.values[:n]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    sv_data = {
        "Word":          [r[0] for r in sv_rows],
        "SV":            [f"{r[1]:+.4f}" for r in sv_rows],
        "Direction":     ["↑ positive" if r[1] >= 0 else "↓ negative" for r in sv_rows],
    }
    st.dataframe(sv_data, use_container_width=True, hide_index=True)

    if mtype == "encoder":
        st.caption(
            "⚠️ Individual SVs can be counterintuitive — e.g. 'bad' may score positive "
            "because the model learned from IMDb that 'not bad' is a positive phrase. "
            "The SV of 'bad' alone does not capture this. "
            "That is exactly what k-SII interactions reveal in Step 2."
        )
    else:
        st.caption(
            "⚠️ Decoder SVs reflect how a word changes the model's completion tendency "
            "averaged over all subsets — not how it reads in the full sentence. "
            "Words that seem neutral individually may form strong pairs (Step 2)."
        )

    # ── Step 2: k-SII Interactions ────────────────────────────────────────────
    st.markdown("""
    <div class="step-header">
        <span class="step-number">2</span>
        <span class="step-title">Pairwise word interactions — k-SII order 2</span>
    </div>
    """, unsafe_allow_html=True)

    st.caption(
        "k-SII(i,j) = `v(S∪{i,j}) - v(S∪{i}) - v(S∪{j}) + v(S)` averaged over all S. "
        "It measures how much words i and j contribute **together beyond their individual SVs**. "
        "**Positive k-SII** (synergy) → the pair is more powerful together than the sum of parts. "
        "**Negative k-SII** (redundancy) → the words overlap — having both adds less than expected. "
        "This is what first-order Shapley Values cannot capture."
    )

    col_net, col_heat = st.columns(2)
    with col_net:
        st.markdown("**Interaction network**")
        st.caption(
            "Nodes = words. Edges = k-SII values. "
            "🔴 Red / thick = strong synergy. "
            "🔵 Blue = redundancy. "
            "No edge ≈ k-SII ≈ 0 (independent words)."
        )
        st.image(result["img_network"], use_container_width=True)
    with col_heat:
        st.markdown("**Interaction heatmap**")
        st.caption(
            "Cell (i,j) = k-SII(word_i, word_j). "
            "🟥 Bright red = strong synergy. "
            "🟦 Bright blue = strong redundancy. "
            "Diagonal = first-order SVs. "
            "A dominant off-diagonal cell means that pair drives the prediction."
        )
        st.image(result["img_heatmap"], use_container_width=True)

    # Top interactions table
    st.markdown("**Top pairwise interactions (sorted by |k-SII|)**")
    int_data = {
        "Word 1":  [t[0] for t in top_int],
        "Word 2":  [t[1] for t in top_int],
        "k-SII":   [f"{t[2]:+.4f}" for t in top_int],
        "Type":    ["🟢 synergy" if t[2] > 0 else "🔵 redundancy" for t in top_int],
    }
    st.dataframe(int_data, use_container_width=True, hide_index=True)

    # ── Step 3: Interpretation ────────────────────────────────────────────────
    st.markdown("""
    <div class="step-header">
        <span class="step-number">3</span>
        <span class="step-title">Interpretation</span>
    </div>
    """, unsafe_allow_html=True)

    if top_int:
        w1, w2, val = top_int[0]
        sv_w1 = sv.values[words.index(w1)] if w1 in words else 0.0
        sv_w2 = sv.values[words.index(w2)] if w2 in words else 0.0

        if abs(val) > 0.3:
            if val > 0:
                st.success(f"""
                **Strongest synergy: `{w1}` + `{w2}` → k-SII = {val:+.4f}**

                Individual SVs: `{w1}` = {sv_w1:+.4f}, `{w2}` = {sv_w2:+.4f}  
                Sum of individual SVs: {sv_w1 + sv_w2:+.4f}  
                Extra contribution from the pair together: **{val:+.4f}**

                The pair contributes more to `v(S)` than their individual SVs predict.
                This is the interaction that first-order Shapley Values miss.
                """)
            else:
                st.info(f"""
                **Strongest redundancy: `{w1}` + `{w2}` → k-SII = {val:+.4f}**

                Individual SVs: `{w1}` = {sv_w1:+.4f}, `{w2}` = {sv_w2:+.4f}  
                Sum of individual SVs: {sv_w1 + sv_w2:+.4f}  
                Overlap (diminishing returns): **{val:+.4f}**

                These words carry similar sentiment signals — the model already
                gets the information from one word, adding the other contributes less.
                """)
        else:
            st.info(f"""
            **No dominant interaction detected** (strongest k-SII = {val:+.4f})

            Individual SVs: `{w1}` = {sv_w1:+.4f}, `{w2}` = {sv_w2:+.4f}

            Words act approximately independently in this sentence.
            First-order Shapley Values give a complete picture here.
            """)

elif run_btn and not text.strip():
    st.warning("Please enter a sentence before clicking Analyse.")

else:
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem; color:#9ca3af;">
        <div style="font-size:3rem;">🔍</div>
        <div style="font-size:1.1rem; font-weight:600; margin-top:0.5rem;">
            Select a model and enter a sentence to begin
        </div>
        <div style="font-size:0.9rem; margin-top:0.4rem;">
            Results will show segmentation, Shapley Values, interaction network, and heatmap
        </div>
    </div>
    """, unsafe_allow_html=True)