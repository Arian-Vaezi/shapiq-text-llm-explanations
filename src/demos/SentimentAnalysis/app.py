"""Gradio web application for the Sentiment Analysis demo.

This file contains only the UI logic.
All computation is handled by sentiment_analysis.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr
from sentiment_analysis import blank_placeholder, run_pipeline

if TYPE_CHECKING:
    from collections.abc import Iterator

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #f7f8fc;
    --card: #ffffff;
    --card-soft: #f3f5fb;
    --text: #111827;
    --muted: #6b7280;
    --border: #e5e7eb;
    --accent: #5b6cff;
    --accent-dark: #4338ca;
    --accent-soft: #eef0ff;
    --green: #10b981;
    --red: #ef4444;
    --blue: #3b82f6;
    --shadow: 0 18px 45px rgba(31, 41, 55, 0.08);
    --radius: 22px;
    --font: 'Inter', sans-serif;
    --mono: 'JetBrains Mono', monospace;
}

body, .gradio-container {
    background:
        radial-gradient(circle at top left, rgba(91,108,255,0.14), transparent 34%),
        radial-gradient(circle at top right, rgba(16,185,129,0.10), transparent 30%),
        linear-gradient(180deg, #f8f9ff 0%, #f7f8fc 100%) !important;
    color: var(--text) !important;
    font-family: var(--font) !important;
}

.gradio-container {
    max-width: 1280px !important;
    margin: auto !important;
}

/* Header */
.hero {
    margin: 2rem 0 1.5rem;
    padding: 2.4rem;
    border-radius: 30px;
    background: linear-gradient(135deg, #ffffff 0%, #f1f4ff 100%);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}

.hero-badge {
    display: inline-flex;
    padding: 0.42rem 0.8rem;
    border-radius: 999px;
    background: var(--accent-soft);
    color: var(--accent-dark);
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.hero h1 {
    margin: 0;
    font-size: 2.75rem;
    line-height: 1.05;
    letter-spacing: -0.045em;
    font-weight: 800;
    color: #111827;
}

.hero p {
    max-width: 780px;
    margin: 0.9rem 0 0;
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.7;
}

/* Cards */
.input-card,
.examples-card,
.info-card {
    background: rgba(255,255,255,0.94);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 1.2rem;
}

.section-title {
    font-size: 0.74rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    font-weight: 800;
    margin-bottom: 0.4rem;
}

.section-subtitle {
    font-size: 0.8rem;
    color: var(--muted);
    margin-bottom: 0.85rem;
}

/* Input */
textarea, input[type=text] {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.03) !important;
}

textarea:focus, input[type=text]:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 4px rgba(91,108,255,0.12) !important;
}

/* Button */
button.primary {
    background: linear-gradient(135deg, var(--accent), #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    font-weight: 800 !important;
    padding: 0.85rem 1.3rem !important;
    box-shadow: 0 12px 25px rgba(91,108,255,0.25) !important;
}

button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 16px 30px rgba(91,108,255,0.30) !important;
}

/* Style gr.Examples as chips — using Gradio's real class names */

/* Strip all chrome from the wrapper */
.examples-card .table-wrap {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
    overflow: visible !important;
}

/* Hide the label above the table */
.examples-card .label {
    display: none !important;
}

/* Hide the column header row */
.examples-card .tr-head {
    display: none !important;
}

/* Turn the table into a flex chip row */
.examples-card table {
    display: block !important;
    border: none !important;
    background: transparent !important;
}

.examples-card tbody {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 0.5rem !important;
    background: transparent !important;
    padding: 0.2rem 0 0 !important;
}

/* Each row = one chip */
.examples-card .tr-body {
    display: block !important;
    background: transparent !important;
    border: none !important;
    cursor: pointer !important;
    transition: transform 0.15s !important;
}

/* The td is the visible pill */
.examples-card .tr-body td {
    display: inline-flex !important;
    align-items: center !important;
    max-width: none !important;
    width: auto !important;
    padding: 0.48rem 0.95rem !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 999px !important;
    background: #ffffff !important;
    color: #374151 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    cursor: pointer !important;
    box-shadow: 0 2px 8px rgba(31,41,55,0.06) !important;
    line-height: 1.4 !important;
    white-space: normal !important;
    text-align: left !important;
    transition: border-color 0.15s, background 0.15s, color 0.15s, box-shadow 0.15s !important;
}

.examples-card .tr-body:hover td {
    border-color: #5b6cff !important;
    background: #eef0ff !important;
    color: #4338ca !important;
    box-shadow: 0 5px 16px rgba(91,108,255,0.18) !important;
    transform: translateY(-1px) !important;
}

/* Pipeline */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    padding: 0.62rem 0;
    border-bottom: 1px solid #f0f1f5;
    color: #374151;
    font-size: 0.86rem;
}

.pipeline-step:last-child { border-bottom: none; }

.pipeline-number {
    width: 1.7rem;
    height: 1.7rem;
    border-radius: 50%;
    background: var(--accent-soft);
    color: var(--accent-dark);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 0.75rem;
    flex-shrink: 0;
}

.idea-box {
    margin-top: 1rem;
    padding: 0.95rem;
    background: #f8f9ff;
    border: 1px solid #e0e4ff;
    border-radius: 16px;
    color: #4f46e5;
    font-size: 0.84rem;
    line-height: 1.65;
}

/* Result markdown */
.result-panel {
    background: #ffffff;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.45rem 1.65rem;
    box-shadow: var(--shadow);
}

.result-panel h2  { margin-top: 0; font-size: 1.65rem; letter-spacing: -0.03em; }
.result-panel h3  { margin-top: 1.45rem; font-size: 1rem; color: #111827; }

.result-panel table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--mono);
    font-size: 0.84rem;
    margin-top: 0.65rem;
    border-radius: 14px;
    overflow: hidden;
}

.result-panel th {
    background: #f3f4f6;
    color: #4b5563;
    padding: 0.65rem 0.8rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
}

.result-panel td {
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid #f0f0f0;
}

.result-panel blockquote {
    border-left: 4px solid var(--accent);
    background: #f8f9ff;
    padding: 0.75rem 1rem;
    border-radius: 12px;
    color: var(--muted);
}

/* Metrics */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.7rem;
    margin-top: 1rem;
}

.metric {
    background: #f9fafb;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 0.8rem;
}

.metric-label {
    color: var(--muted);
    font-size: 0.72rem;
    text-transform: uppercase;
    font-weight: 800;
    letter-spacing: 0.08em;
}

.metric-value {
    margin-top: 0.25rem;
    font-family: var(--mono);
    font-weight: 700;
    color: #111827;
}

/* Tabs */
.tab-nav button {
    font-family: var(--font) !important;
    font-weight: 700 !important;
    color: var(--muted) !important;
    background: transparent !important;
    border-radius: 12px !important;
}

.tab-nav button.selected {
    color: var(--accent-dark) !important;
    background: var(--accent-soft) !important;
}

img {
    border-radius: 18px !important;
    border: 1px solid var(--border);
    background: white;
}
"""


DEMO_EXAMPLES = [
    ["This film is not bad at all"],
    ["I really loved this amazing film"],
    ["What a magnificent disaster of a film"],
    ["The acting was superb but the plot was absolutely terrible"],
    ["I would not say this was a bad experience"],
]


def load_example(example: list[str]) -> str:
    """Called by gr.Examples .select() — just returns the chosen sentence."""
    return example[0]


def on_analyse(text: str) -> Iterator[tuple[object, object, object, object, object]]:
    """Run the analysis callback and stream UI updates."""
    text = text.strip()

    if not text:
        yield (
            "### Type a sentence and click *Analyse →*",
            blank_placeholder("Enter a sentence above"),
            blank_placeholder(""),
            blank_placeholder(""),
            gr.update(visible=False),
        )
        return

    yield (
        "### Computing explanation...\n\n"
        "The model is evaluating word contributions and pairwise interactions.",
        blank_placeholder("Computing Shapley values..."),
        blank_placeholder("Computing interactions..."),
        blank_placeholder("Building heatmap..."),
        gr.update(visible=False),
    )

    result = run_pipeline(text)

    label = result["label"]
    score = result["score"]
    words = result["words"]
    baseline = result["baseline"]
    n = result["n_players"]
    sv = result["sv"]
    top_int = result["top_interactions"]

    emoji = "😊" if label == "POSITIVE" else "😠"
    color = "#10b981" if label == "POSITIVE" else "#ef4444"

    sv_rows = ""
    for word, val in zip(words, sv.values, strict=False):
        bar_len = min(int(abs(val) * 15), 10)
        bar_color = "#10b981" if val >= 0 else "#ef4444"
        bar = f'<span style="color:{bar_color}">{"█" * bar_len}</span>'
        sv_rows += f"| {word} | {val:+.4f} | {bar} |\n"

    int_rows = ""
    for w1, w2, val in top_int:
        sign = "synergy" if val > 0 else "redundancy"
        icon = "🟢" if val > 0 else "🔵"
        int_rows += f"| {w1} + {w2} | {val:+.4f} | {icon} {sign} |\n"

    result_md = f"""
## {emoji} Sentiment: *{label}*
<span style="font-size:1.4rem; font-weight:800; color:{color};">{score:.3f}</span>

<div class="metric-grid">
    <div class="metric">
        <div class="metric-label">Baseline</div>
        <div class="metric-value">{baseline:.4f}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Players</div>
        <div class="metric-value">{n} words</div>
    </div>
    <div class="metric">
        <div class="metric-label">Model</div>
        <div class="metric-value">DistilBERT IMDB</div>
    </div>
</div>

---

### Word-level contributions

| Word | Shapley Value | Attribution |
|---|---:|---|
{sv_rows}

---

### Strongest pairwise interactions

| Word Pair | k-SII | Interpretation |
|---|---:|---|
{int_rows}

**Synergy** means the words become more influential together.
**Redundancy** means the words partially overlap or compete in their contribution.
"""

    yield (
        result_md,
        result["img_sentence"],
        result["img_network"],
        result["img_heatmap"],
        gr.update(visible=True),
    )


with gr.Blocks(title="Sentiment Explainer", css=CSS) as demo:
    gr.HTML("""
    <div class="hero">
        <div class="hero-badge">Explainable AI Demo</div>
        <h1>Sentiment Explainer</h1>
        <p>
            Explore how a sentiment model makes decisions by combining
            word-level Shapley values with pairwise interaction analysis.
            This demo turns the original notebook into a clean interactive application.
        </p>
    </div>
    """)

    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=350):
            with gr.Group(elem_classes=["input-card"]):
                gr.HTML('<div class="section-title">Input Sentence</div>')
                text_input = gr.Textbox(
                    placeholder="Example: This film is not bad at all",
                    lines=3,
                    show_label=False,
                )
                analyse_btn = gr.Button("Analyse →", variant="primary")

            with gr.Group(elem_classes=["examples-card"]):
                gr.HTML("""
                <div class="section-title">Examples</div>
                <div class="section-subtitle">Click a chip to load it into the input.</div>
                """)
                examples_widget = gr.Examples(
                    examples=DEMO_EXAMPLES,
                    inputs=[text_input],
                    label="",
                    # Disable the built-in auto-run so clicking only fills the box
                    run_on_click=False,
                    fn=None,
                )

            with gr.Group(elem_classes=["info-card"]):
                gr.HTML("""
                <div class="section-title">Pipeline</div>
                <div class="pipeline-step">
                    <span class="pipeline-number">1</span>
                    <span>TextImputer masks words</span>
                </div>
                <div class="pipeline-step">
                    <span class="pipeline-number">2</span>
                    <span>KernelSHAP estimates word contributions</span>
                </div>
                <div class="pipeline-step">
                    <span class="pipeline-number">3</span>
                    <span>KernelSHAPIQ computes pairwise interactions</span>
                </div>
                <div class="pipeline-step">
                    <span class="pipeline-number">4</span>
                    <span>Plots explain the model decision</span>
                </div>
                <div class="idea-box">
                    <strong>Key idea:</strong><br>
                    Some sentiment effects only appear when words are interpreted together,
                    for example negation pairs such as <code>not</code> + <code>bad</code>.
                </div>
                """)

        with gr.Column(scale=2):
            result_md = gr.Markdown(
                "### Enter a sentence and run the analysis.",
                elem_classes=["result-panel"],
            )

            with gr.Tabs(visible=False) as plot_tabs:
                with gr.Tab("Sentence Contribution"):
                    img_sentence = gr.Image(show_label=False)
                with gr.Tab("Interaction Network"):
                    img_network = gr.Image(show_label=False)
                with gr.Tab("Interaction Heatmap"):
                    img_heatmap = gr.Image(show_label=False)

    analyse_btn.click(
        fn=on_analyse,
        inputs=[text_input],
        outputs=[result_md, img_sentence, img_network, img_heatmap, plot_tabs],
    )

    text_input.submit(
        fn=on_analyse,
        inputs=[text_input],
        outputs=[result_md, img_sentence, img_network, img_heatmap, plot_tabs],
    )


if __name__ == "__main__":
    demo.launch()
