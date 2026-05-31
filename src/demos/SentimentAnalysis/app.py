"""
app.py
=======
Streamlit web application for the Sentiment Analysis demo.

This file contains ONLY the UI logic layout, styling, and event routing.
All computation logic lives in sentiment_analysis.py.

Two pipelines are supported:
  - Encoder: DistilBERT classifier with [MASK] imputation
  - Decoder: Causal LM with contrastive log-odds value function

To run:
    streamlit run app.py

Related files:
    sentiment_analysis.py    <- core pipeline logic (encoder + decoder)
    SentimentDecoderGame.py  <- decoder value function using HFModelWrapper
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Add src to path so imports from demos.shared work correctly
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sentiment_analysis import (  # noqa: E402
    run_pipeline,
    run_pipeline_decoder,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment Explainer",
    page_icon="🔍",
    layout="wide",
)

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

    # Model selector
    st.markdown("**Model**")
    model_choice = st.radio(
        "Model",
        options=["encoder", "gemma", "tinyllama"],
        format_func=lambda x: {
            "encoder":   "DistilBERT IMDb  (encoder · fast · CPU)",
            "gemma":     "Gemma 3 1B  (decoder · GPU recommended)",
            "tinyllama": "TinyLlama 1.1B  (decoder · lighter)",
        }[x],
        label_visibility="collapsed",
    )

    # Value function explanation changes per model
    with st.expander("📐 Value function formula"):
        if model_choice == "encoder":
            st.markdown("**Classification score**")
            st.latex(
                r"v(S) = \begin{cases} +p & \text{if POSITIVE} \\ -p & \text{if NEGATIVE} \end{cases}"
            )
            st.markdown("""
            - `p` = model output probability
            - Absent words → `[MASK]` token
            - Score range: **[-1, +1]**
            - Baseline: all words masked
            """)
        else:
            model_label = "Gemma 3 1B" if model_choice == "gemma" else "TinyLlama 1.1B"
            st.markdown(f"**{model_label} Contrastive Log-Odds**")
            st.latex(
                r"v(S) = \frac{1}{|T^+|}\sum_{t \in T^+} \log p(t \mid S)"
                r"- \frac{1}{|T^-|}\sum_{t \in T^-} \log p(t \mid S)"
            )
            st.markdown("""
            - T⁺ = positive templates, T⁻ = negative templates
            - Absent words → **removed entirely**
            - Score range: **unbounded** (typically ±5 to ±20)
            - Baseline: empty string (no words)
            """)
            st.markdown("**Positive templates:**")
            st.code('" This is positive."\n" I loved it."\n" Excellent!"\n" Great film."')
            st.markdown("**Negative templates:**")
            st.code('" This is negative."\n" I hated it."\n" Terrible!"\n" Awful film."')

    st.divider()

    # Example sentences
    st.markdown("**📓 Examples**")
    st.caption("Click to load into the input box.")
    for example in DEMO_EXAMPLES:
        label = example[:45] + "…" if len(example) > 45 else example
        if st.button(label, use_container_width=True):
            st.session_state["text_input"] = example

    st.divider()

    # Pipeline steps
    st.markdown("**How It Works**")
    st.markdown("""
    1. **Imputer**  masks or removes words to form coalitions
    2. **KernelSHAP**  estimates individual word contributions
    3. **KernelSHAPIQ** computes pairwise k-SII interactions
    4. **Visualizations** sentence plot, network, heatmap
    """)


# ── Main content ──────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero">
    <h1>🔍 Sentiment Explainer</h1>
    <p>
        Enter any sentence to explain how a sentiment model makes its decision.
        Choose between an encoder classifier and decoder language models.
        Results show individual word contributions and pairwise word interactions.
    </p>
</div>
""", unsafe_allow_html=True)

# App-level introduction changes per model
if model_choice == "encoder":
    st.info("""
    **DistilBERT (Encoder) mode**

    Each word in your sentence becomes a player. To measure a word's importance,
    we hide it by replacing it with `[MASK]` and observe how the sentiment score changes.
    We use **Shapley Values** to fairly distribute credit across all words,
    and **k-SII** to find which word *pairs* jointly drive the prediction.
    """)
else:
    model_label = "Gemma 3 1B" if model_choice == "gemma" else "TinyLlama 1.1B"
    st.info(f"""
    **{model_label} (Decoder) mode**

    Decoder models have no sentiment classifier. Instead, we measure how likely the model
    is to continue your sentence with positive vs negative phrases the **contrastive log-odds**.
    To measure a word's importance, we *remove* it entirely and observe how the score changes.
    We use the same **Shapley Values** and **k-SII** framework as the encoder mode.
    """)

# Text input + submit
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

    with st.spinner("Computing Shapley values and k-SII interactions…"):
        if model_choice == "encoder":
            result = run_pipeline(text.strip())
        else:
            result = run_pipeline_decoder(text.strip(), model_key=model_choice)

    # Unpack result
    label    = result["label"]
    score    = result["score"]
    words    = result["words"]
    baseline = result["baseline"]
    n        = result["n_players"]
    sv       = result["sv"]
    top_int  = result["top_interactions"]
    mtype    = result["model_type"]
    mname    = result["model_name"].split("/")[-1]

    emoji = "😊" if label == "POSITIVE" else "😠"
    color = "#10b981" if label == "POSITIVE" else "#ef4444"
    badge = "Encoder" if mtype == "encoder" else "Decoder"
    vf    = "Classification score" if mtype == "encoder" else "Contrastive log-odds"

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
                <span>Baseline</span>
                <strong>{baseline:.4f}</strong>
            </div>
            <div class="metric-chip">
                <span>Players</span>
                <strong>{n} words</strong>
            </div>
            <div class="metric-chip">
                <span>Value function</span>
                <strong>{vf}</strong>
            </div>
            <div class="metric-chip">
                <span>Index</span>
                <strong>k-SII order 2</strong>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Score explanation changes per model
    if model_choice == "encoder":
        st.caption(f"""
        The **baseline ({baseline:.4f})** is the model score when every word is replaced with [MASK].
        The **prediction score ({score:.3f})** is the model confidence on your full sentence (range: 0 to 1).
        Shapley values explain how each word moved the score from baseline → prediction.
        """)
    else:
        st.caption(f"""
        The **baseline ({baseline:.4f})** is the contrastive log-odds score on an empty input (no words).
        The **prediction score ({score:.3f})** is normalized from the raw log-odds for display.
        Raw decoder scores are unbounded what matters is the direction (positive vs negative)
        and how much each word changes it.
        """)

    # ── Step 1: Shapley Values ────────────────────────────────────────────────
    st.markdown("""
    <div class="step-header">
        <span class="step-number">1</span>
        <span class="step-title">Individual word contributions (Shapley Values)</span>
    </div>
    """, unsafe_allow_html=True)

    # SV explanation changes per model
    if model_choice == "encoder":
        st.caption("""
        A **Shapley Value** measures a word's average contribution across all possible
        subsets of words. Absent words are replaced with **[MASK]**.
        Positive = pushes toward positive sentiment. Negative = pushes toward negative.
        Words near zero contribute little individually but may matter in pairs (see Step 2).
        """)
    else:
        st.caption("""
        A **Shapley Value** measures a word's average contribution across all possible
        subsets of words. Absent words are **removed entirely** from the sentence.
        Positive = makes the model more likely to continue with positive phrases.
        Negative = makes it more likely to continue with negative phrases.
        """)

    # Sentence plot colors each word by its SV
    st.image(result["img_sentence"], use_container_width=True)

    # SV table
    sv_data = {
        "Word": words,
        "Shapley Value": [f"{v:+.4f}" for v in sv.values],
        "Direction": ["↑ positive" if v >= 0 else "↓ negative" for v in sv.values],
    }
    st.dataframe(sv_data, use_container_width=True, hide_index=True)

    # Post-SV note changes per model
    if model_choice == "encoder":
        st.caption("""
        ⚠️ Some results may seem surprising for example, "bad" scoring positive.
        This happens because DistilBERT was trained on IMDb reviews where "not bad"
        frequently appears in positive reviews. Individual SVs reflect learned corpus
        patterns, not literal word meaning his is exactly why interactions matter.
        """)
    else:
        st.caption("""
        ⚠️ Decoder SVs can be harder to interpret individually because the model scores
        completions, not classifications. A word may score near zero individually but
        interact strongly with another word this is what Step 2 reveals.
        """)

    # ── Step 2: k-SII Interactions ────────────────────────────────────────────
    st.markdown("""
    <div class="step-header">
        <span class="step-number">2</span>
        <span class="step-title">Pairwise word interactions (k-SII order 2)</span>
    </div>
    """, unsafe_allow_html=True)

    # Interaction explanation changes per model
    if model_choice == "encoder":
        st.caption("""
        **k-SII** measures how much two words contribute *together* beyond their individual SVs.
        This reveals phenomena that SVs miss like negation: "not" and "bad" individually
        score low, but together they flip the sentiment.
        🟢 **Synergy:** the pair contributes more together than individually.
        🔵 **Redundancy:** the words compete having both adds less than the sum of parts.
        """)
    else:
        st.caption("""
        **k-SII** measures how much two words jointly change the contrastive log-odds score
        beyond what each word changes individually. Since decoder models process full sequences,
        interactions can reveal phrase-level patterns the model has learned.
        🟢 **Synergy:** the pair steers sentiment more strongly together.
        🔵 **Redundancy:** the words overlap in their effect diminishing returns.
        """)

    # Network + heatmap side by side
    col_net, col_heat = st.columns(2)
    with col_net:
        st.markdown("**Interaction network**")
        st.caption("Edge thickness = interaction strength. Red = synergy, blue = redundancy.")
        st.image(result["img_network"], use_container_width=True)
    with col_heat:
        st.markdown("**Interaction heatmap**")
        if model_choice == "encoder":
            st.caption("Word × word k-SII matrix. A dominant bright cell means that pair drives sentiment.")
        else:
            st.caption("Word × word k-SII matrix. Bright cells show pairs that jointly steer the log-odds score.")
        st.image(result["img_heatmap"], use_container_width=True)

    # Top interactions table
    st.markdown("**Top pairwise interactions**")
    int_data = {
        "Word 1": [t[0] for t in top_int],
        "Word 2": [t[1] for t in top_int],
        "k-SII": [f"{t[2]:+.4f}" for t in top_int],
        "Type": ["🟢 synergy" if t[2] > 0 else "🔵 redundancy" for t in top_int],
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
        if abs(val) > 0.5:
            if val > 0:
                if model_choice == "encoder":
                    st.success(f"""
                    **Synergy detected:** `{w1}` + `{w2}` = {val:+.4f}

                    These two words contribute **more together** than their individual SVs predict.
                    The pair is the unit of meaning neither word alone explains the prediction.
                    This is what first-order Shapley values miss.
                    """)
                else:
                    st.success(f"""
                    **Synergy detected:** `{w1}` + `{w2}` = {val:+.4f}

                    These two words jointly steer the model's continuation toward positive sentiment
                    more than either does alone. The decoder model has learned a phrase-level pattern
                    that individual word scores cannot capture.
                    """)
            else:
                if model_choice == "encoder":
                    st.info(f"""
                    **Redundancy detected:** `{w1}` + `{w2}` = {val:+.4f}

                    These two words compete for the same sentiment signal.
                    Having both adds less than the sum of their individual contributions
                    the model has diminishing returns when multiple similar signals are present.
                    """)
                else:
                    st.info(f"""
                    **Redundancy detected:** `{w1}` + `{w2}` = {val:+.4f}

                    These two words overlap in how they influence the model's completions.
                    The decoder already "gets the point" from one word adding the other
                    changes the log-odds score less than expected.
                    """)
        else:
            st.info("""
            No dominant pairwise interaction detected.
            Individual word contributions dominate the words act mostly independently
            in this sentence under the selected model.
            """)

elif run_btn and not text.strip():
    st.warning("Please enter a sentence before clicking Analyse.")

else:
    # Default empty state
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem; color:#9ca3af;">
        <div style="font-size:3rem;">🔍</div>
        <div style="font-size:1.1rem; font-weight:600; margin-top:0.5rem;">
            Select a model and enter a sentence to begin
        </div>
        <div style="font-size:0.9rem; margin-top:0.4rem;">
            Results will show Shapley values, interaction network, and heatmap
        </div>
    </div>
    """, unsafe_allow_html=True)