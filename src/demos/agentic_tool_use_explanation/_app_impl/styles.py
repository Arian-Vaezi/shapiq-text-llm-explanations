"""Streamlit styling extracted mechanically from app.py."""

from __future__ import annotations

# Mechanical re-export chain preserves the monolith's shared global namespace.
from .models import *  # noqa: F403

CSS = """
<style>
.stApp {
    background: #f8f6f0;
}
section[data-testid="stSidebar"] {
    background: #f7f5ef;
    border-right: 1px solid #ddd6c7;
}
.main .block-container {
    max-width: 1220px;
    padding-top: 2rem;
}
/* ---- App header: logo + title + subtitle -------------------------- */
.app-header {
    align-items: center;
    display: flex;
    gap: 0.95rem;
    margin: 0 0 0.3rem 0;
}
.app-logo {
    align-items: center;
    background: linear-gradient(135deg, #2d6f73 0%, #1a4245 100%);
    border-radius: 11px;
    box-shadow: 0 3px 10px rgba(45, 111, 115, 0.35);
    color: #ffffff;
    display: flex;
    flex-shrink: 0;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 1.2rem;
    font-weight: 800;
    height: 44px;
    justify-content: center;
    width: 44px;
}
.app-header h1 {
    color: #1f1f1f;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 1.95rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    line-height: 1.15;
    margin: 0;
}
.app-header p {
    color: #6d6658;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 0.92rem;
    margin: 0.25rem 0 0 0;
}
.floating-robot-wrap {
    position: absolute;
    top: 8px;
    right: 115px;
    z-index: 10;
    pointer-events: auto;
}
.floating-robot {
    width: 145px;
    height: auto;
    opacity: 0.97;
    user-select: none;
    -webkit-user-drag: none;
}
/* ---- Mode selector: two pill/card options -------------------------
   st.radio's own DOM wrapper picks up `.st-key-agentic_mode_card_row`
   (Streamlit's documented key-to-CSS-class mechanism) since the Python
   API has no way to attach a literal class to an individual widget's
   option labels. `.mode-card`/`.mode-card.selected` are defined as real,
   reusable classes in parallel -- any element carrying them gets the
   same look, they are not just inert placeholders. */
.st-key-agentic_mode_card_row {
    margin: 0.85rem 0 1rem 0;
}
.mode-card,
.st-key-agentic_mode_card_row div[data-testid="stRadio"] label {
    background: #ffffff !important;
    border: 1.5px solid #ded6c4 !important;
    border-radius: 12px !important;
    padding: 0.75rem 1.05rem !important;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
.mode-card.selected,
.st-key-agentic_mode_card_row div[data-testid="stRadio"] label:has(input:checked),
.st-key-agentic_mode_card_row div[data-testid="stRadio"] label:has([aria-checked="true"]) {
    background: #eaf4f1 !important;
    border-color: #2d6f73 !important;
    font-weight: 700 !important;
}
/* ---- Shared input bar: request + example + run button ------------- */
.input-bar,
.st-key-agentic_input_bar {
    background: #ffffff;
    border: 1px solid #ded6c4;
    border-radius: 16px;
    box-shadow: 0 4px 14px rgba(45, 41, 35, 0.06);
    margin: 0 0 1.1rem 0;
    padding: 1rem 1.1rem 0.7rem 1.1rem;
}
.input-bar-label {
    color: #6d6658;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin: 0 0 0.3rem 0;
    text-transform: uppercase;
}
.primary-run-button,
.st-key-agentic_run_full_pipeline button {
    background: #2f6fed !important;
    border-color: #2f6fed !important;
    color: #ffffff !important;
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    min-height: 80px !important;
    padding: 1rem 1.25rem !important;
}
.primary-run-button:hover,
.st-key-agentic_run_full_pipeline button:hover {
    background: #2558c4 !important;
    border-color: #2558c4 !important;
}
.st-key-agentic_try_example_select div[data-baseweb="select"] > div {
    height: auto !important;
    min-height: 2.5rem !important;
}
.st-key-agentic_try_example_select div[data-baseweb="select"] span {
    overflow: visible !important;
    text-overflow: clip !important;
    white-space: normal !important;
}
body:has(.st-key-agentic_try_example_select [aria-expanded="true"])
    div[data-baseweb="popover"] [role="listbox"] {
    overflow-x: hidden !important;
    overflow-y: auto !important;
}
body:has(.st-key-agentic_try_example_select [aria-expanded="true"])
    div[data-baseweb="popover"] [role="option"] {
    align-items: flex-start !important;
    display: flex !important;
    height: auto !important;
    line-height: 1.35 !important;
    min-height: 2.75rem !important;
    overflow: visible !important;
    overflow-wrap: anywhere !important;
    padding-bottom: 0.65rem !important;
    padding-top: 0.65rem !important;
    text-overflow: clip !important;
    white-space: normal !important;
}
body:has(.st-key-agentic_try_example_select [aria-expanded="true"])
    div[data-baseweb="popover"] [role="option"] div,
body:has(.st-key-agentic_try_example_select [aria-expanded="true"])
    div[data-baseweb="popover"] [role="option"] span {
    height: auto !important;
    max-height: none !important;
    min-width: 0 !important;
    overflow: visible !important;
    overflow-wrap: anywhere !important;
    text-overflow: clip !important;
    white-space: normal !important;
}
/* ---- Agent Result: compact product-style cards --------------------- */
.result-card {
    background: #ffffff;
    border: 1px solid #ded6c4;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(45, 41, 35, 0.06);
    margin: 0.4rem 0 1.1rem 0;
    padding: 1.1rem 1.3rem;
}
.result-card-header {
    color: #1f1f1f;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 1.1rem;
    font-weight: 800;
    margin: 0;
}
.result-card-header code {
    background: #eaf4f1;
    border-radius: 5px;
    color: #1f554c;
    padding: 0.08rem 0.4rem;
}
.result-card-subtitle {
    color: #6d6658;
    font-size: 0.84rem;
    margin: 0.2rem 0 0.9rem 0;
}
.result-meta-row {
    color: #403d37;
    font-size: 0.86rem;
    margin: 0.25rem 0;
}
.result-meta-row code {
    background: #f3efe2;
    border-radius: 4px;
    padding: 0.05rem 0.35rem;
}
.agent-flow-line {
    color: #8a8372;
    font-size: 0.8rem;
    margin: 0.65rem 0;
}
.agent-status-pill {
    background: #eaf4f1;
    border: 1px solid #bfd4c8;
    border-radius: 999px;
    color: #1f554c;
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 0.45rem 0 0.25rem 0;
    padding: 0.2rem 0.65rem;
}
.agent-model-line {
    border-top: 1px solid #ede7d8;
    color: #6d6658;
    font-size: 0.82rem;
    margin-top: 0.85rem;
    padding-top: 0.65rem;
}
.probability-row {
    align-items: center;
    display: grid;
    gap: 0.6rem;
    grid-template-columns: 8.5rem 1fr 3rem;
    margin: 0.32rem 0;
}
.probability-label {
    color: #403d37;
    font-size: 0.84rem;
    font-weight: 600;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.probability-track {
    background: #f0ece0;
    border-radius: 999px;
    height: 0.6rem;
    overflow: hidden;
    width: 100%;
}
.probability-fill {
    background: #bcb5a2;
    border-radius: 999px;
    display: block;
    height: 100%;
}
.probability-fill.selected {
    background: linear-gradient(90deg, #2d6f73, #45a196);
}
.probability-value {
    color: #403d37;
    font-size: 0.82rem;
    font-variant-numeric: tabular-nums;
    text-align: right;
}
.kv-table {
    margin: 0.35rem 0 0.15rem 0;
}
.kv-row {
    border-bottom: 1px solid #f0ece0;
    display: grid;
    gap: 0.6rem;
    grid-template-columns: 8rem 1fr;
    padding: 0.35rem 0;
}
.kv-row:last-child {
    border-bottom: none;
}
.kv-row .kv-key {
    color: #6d6658;
    font-size: 0.82rem;
    font-weight: 700;
}
.kv-row .kv-value {
    color: #2d2923;
    font-size: 0.88rem;
}
.agent-response-block {
    background: #f9f7f0;
    border-radius: 8px;
    color: #2d2923;
    font-size: 0.9rem;
    line-height: 1.5;
    margin-top: 0.3rem;
    padding: 0.7rem 0.9rem;
}
.empty-state-card {
    background: #fffdf8;
    border: 1px dashed #ded6c4;
    border-radius: 14px;
    margin: 0.4rem 0 1.1rem 0;
    padding: 1.7rem 1.4rem;
    text-align: center;
}
.empty-state-card h3 {
    color: #1f1f1f;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 1.1rem;
    margin: 0 0 0.45rem 0;
}
.empty-state-card p {
    color: #6d6658;
    font-size: 0.9rem;
    margin: 0 0 0.6rem 0;
}
.empty-state-card .pipeline-hint {
    color: #8a8372;
    font-size: 0.8rem;
    font-style: italic;
}
.error-card {
    background: #fdf2f0;
    border: 1px solid #e3b7ac;
    border-left: 4px solid #b2472f;
    border-radius: 10px;
    margin: 0.4rem 0 1.1rem 0;
    padding: 0.95rem 1.15rem;
}
.error-card h3 {
    color: #7c2c1c;
    font-size: 1rem;
    margin: 0 0 0.3rem 0;
}
.error-card p {
    color: #5c3a32;
    font-size: 0.88rem;
    margin: 0;
}
.scenario-panel {
    background: #fffef9;
    border: 1px solid #d9d2bf;
    border-radius: 7px;
    display: grid;
    gap: 0.9rem;
    grid-template-columns: 1.15fr 0.85fr;
    margin: 0 0 1rem 0;
    padding: 0.9rem 1rem;
}
.scenario-panel h3 {
    color: #202020;
    font-family: Georgia, serif;
    font-size: 1.25rem;
    margin: 0 0 0.35rem 0;
}
.scenario-panel p {
    color: #403d37;
    line-height: 1.4;
    margin: 0;
}
.scenario-tag {
    background: #e5efe9;
    border: 1px solid #bfd4c8;
    border-radius: 999px;
    color: #1f554c;
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 0.45rem;
    padding: 0.18rem 0.58rem;
}
.scenario-hint {
    align-self: center;
    border-left: 1px solid #ded6c4;
    color: #5f584b;
    font-size: 0.9rem;
    line-height: 1.45;
    padding-left: 0.9rem;
}
.section-label {
    color: #5f584b;
    font-size: 0.76rem;
    letter-spacing: 0.06em;
    margin: 0.3rem 0 0.45rem 0;
    text-transform: uppercase;
}
.segment-box {
    background: #fffdf7;
    border: 1px solid #d6cab2;
    border-left: 4px solid #2d6f73;
    border-radius: 6px;
    margin-bottom: 0.55rem;
    padding: 0.62rem 0.78rem;
}
.segment-box.user {
    border-left-color: #b15d3b;
}
.segment-box h4 {
    color: #222;
    font-size: 0.88rem;
    margin: 0 0 0.25rem 0;
}
.segment-box p {
    color: #403d37;
    font-size: 0.9rem;
    line-height: 1.4;
    margin: 0;
}
/* ---- XAI top summary: target, evidence, and score change ------------- */
.xai-top-summary {
    margin: 0.25rem 0 1.1rem 0;
}
.xai-top-title {
    color: #1f1f1f;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    line-height: 1.25;
    margin: 0;
}
.xai-top-tool-question {
    white-space: nowrap;
}
.xai-top-title .xai-top-tool-pill {
    font-size: 0.82em;
    vertical-align: 0.04em;
}
.xai-top-subtitle {
    color: #6d6658;
    font-size: 0.92rem;
    margin: 0.25rem 0 0.85rem 0;
}
.xai-top-tool-pill {
    background: #e5f1ed;
    border-radius: 8px;
    color: #19745b;
    display: inline-block;
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-weight: 600;
    line-height: 1.3;
    overflow-wrap: anywhere;
    padding: 0.12rem 0.55rem;
}
.xai-top-card {
    background: #ffffff;
    border: 1px solid #ded6c4;
    border-radius: 16px;
    box-shadow: 0 3px 12px rgba(45, 41, 35, 0.05);
    display: grid;
    grid-template-columns: minmax(0, 1.05fr) minmax(0, 1.35fr) minmax(0, 0.9fr);
    padding: 1.1rem 1.25rem;
}
.xai-top-column {
    min-width: 0;
    padding: 0 1.15rem;
}
.xai-top-column:first-child {
    padding-left: 0;
}
.xai-top-column + .xai-top-column {
    border-left: 1px solid #ddd6c7;
}
.xai-top-heading,
.xai-top-score-heading {
    color: #787267;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.xai-top-heading {
    margin-bottom: 0.65rem;
}
.xai-top-detail-row {
    align-items: baseline;
    color: #403d37;
    display: grid;
    font-size: 0.84rem;
    gap: 0.65rem;
    grid-template-columns: minmax(6.4rem, auto) minmax(0, 1fr);
    margin: 0.45rem 0;
}
.xai-top-detail-row > span:first-child {
    color: #787267;
    font-weight: 700;
}
.xai-top-detail-row strong {
    font-weight: 500;
    min-width: 0;
    overflow-wrap: anywhere;
}
.xai-top-finding {
    border-radius: 10px;
    color: #403d37;
    margin: 0.45rem 0;
    padding: 0.62rem 0.75rem;
}
/* Positive/negative tints derived from shapiq's own RED/BLUE plot palette
   (see shapiq.plot._config), so support/oppose color-coding agrees with the
   bar plot and heatmap: red = support, blue = oppose. */
.xai-top-finding-segment.is-positive {
    background: #ffe2eb;
}
.xai-top-finding-segment.is-negative {
    background: #e4f1fc;
}
.xai-top-finding-segment.is-neutral {
    background: #f2f0e9;
}
.xai-top-finding-pair {
    background: #f5eddd;
}
.xai-top-finding.is-disabled {
    color: #817a6d;
    opacity: 0.8;
}
.xai-top-finding-line {
    align-items: baseline;
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem 0.65rem;
    justify-content: space-between;
}
.xai-top-finding-line strong {
    color: #655e52;
    font-size: 0.82rem;
}
.xai-top-finding-segment.is-positive .xai-top-finding-line strong {
    color: #ff0d57;
}
.xai-top-finding-segment.is-negative .xai-top-finding-line strong {
    color: #1e88e5;
}
.xai-top-finding-segment.is-neutral .xai-top-finding-line strong {
    color: #655e52;
}
.xai-top-finding-pair .xai-top-finding-line strong {
    color: #a96a13;
}
.xai-top-finding-line span {
    font-size: 0.82rem;
    font-variant-numeric: tabular-nums;
    font-weight: 700;
}
.xai-top-segment-text {
    font-size: 0.83rem;
    line-height: 1.4;
    margin-top: 0.38rem;
    overflow-wrap: anywhere;
}
.xai-top-pair-text {
    align-items: baseline;
    display: flex;
    flex-wrap: wrap;
    font-size: 0.83rem;
    gap: 0.25rem 0.45rem;
    line-height: 1.4;
    margin-top: 0.38rem;
    overflow-wrap: anywhere;
}
.xai-top-pair-text > span:not(.xai-top-pair-times) {
    min-width: 0;
}
.xai-top-pair-times {
    color: #817a6d;
    flex: 0 0 auto;
    font-weight: 700;
}
.xai-top-score-column {
    display: flex;
    padding-right: 0;
}
.xai-top-score-card {
    align-items: center;
    background: #f2f0e9;
    border: 1.5px solid #a8a090;
    border-radius: 14px;
    display: flex;
    flex: 1;
    flex-direction: column;
    justify-content: center;
    min-width: 0;
    padding: 0.75rem;
    text-align: center;
}
.xai-top-score-heading {
    color: #655e52;
}
.xai-top-score-value {
    color: #655e52;
    font-size: 1.8rem;
    font-variant-numeric: tabular-nums;
    font-weight: 800;
    line-height: 1.15;
    margin-top: 0.55rem;
}
.xai-top-score-formula {
    color: #787267;
    font-size: 0.78rem;
    margin-top: 0.3rem;
}
.xai-top-effect-label {
    color: #655e52;
    font-size: 0.84rem;
    font-weight: 800;
    margin-top: 0.35rem;
}
.xai-top-score-card.is-positive {
    background: #ffe2eb;
    border-color: #ff0d57;
}
.xai-top-score-card.is-positive .xai-top-score-heading,
.xai-top-score-card.is-positive .xai-top-score-value,
.xai-top-score-card.is-positive .xai-top-effect-label {
    color: #ff0d57;
}
.xai-top-score-card.is-negative {
    background: #e4f1fc;
    border-color: #1e88e5;
}
.xai-top-score-card.is-negative .xai-top-score-heading,
.xai-top-score-card.is-negative .xai-top-score-value,
.xai-top-score-card.is-negative .xai-top-effect-label {
    color: #1e88e5;
}
.xai-top-score-card.is-neutral {
    background: #f2f0e9;
    border-color: #a8a090;
}
.xai-top-score-card.is-neutral .xai-top-score-heading,
.xai-top-score-card.is-neutral .xai-top-score-value,
.xai-top-score-card.is-neutral .xai-top-effect-label {
    color: #655e52;
}
/* ---- Legacy XAI metric styles retained for developer-only helpers ---- */
.xai-summary-card {
    background: #ffffff;
    border: 1px solid #ded6c4;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(45, 41, 35, 0.06);
    column-gap: 1.3rem;
    display: grid;
    grid-template-columns: minmax(110px, 0.8fr) minmax(240px, 1.7fr) minmax(230px, 1.2fr);
    margin: 0.4rem 0 1.1rem 0;
    padding: 1.1rem 1.3rem;
    row-gap: 0.85rem;
}
/* ---- Left: icon + "Why <tool>?" -------------------------------------- */
.xai-summary-left {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}
.xai-summary-icon {
    align-items: center;
    background: linear-gradient(135deg, #2d6f73 0%, #1a4245 100%);
    border-radius: 9px;
    color: #ffffff;
    display: flex;
    flex-shrink: 0;
    font-size: 1.4rem;
    font-weight: 800;
    height: 34px;
    justify-content: center;
    line-height: 1;
    width: 34px;
}
.xai-summary-title {
    color: #1f1f1f;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    line-height: 1.3;
    margin: 0;
}
.xai-summary-title .target-highlight {
    background: #eaf4f1;
    border-radius: 5px;
    color: #1f554c;
    padding: 0.06rem 0.4rem;
}
/* ---- Middle: evidence chips + interpretation ------------------------- */
.xai-summary-main {
    border-left: 1px solid #ede7d8;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    padding-left: 1.1rem;
}
.evidence-row {
    align-items: baseline;
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin: 0.3rem 0;
}
.evidence-row-label {
    color: #8a8372;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.evidence-chip {
    background: #f3efe2;
    border-radius: 999px;
    color: #403d37;
    display: inline-block;
    font-size: 0.82rem;
    padding: 0.18rem 0.65rem;
}
.xai-summary-interpretation {
    color: #403d37;
    font-size: 0.88rem;
    margin: 0.45rem 0 0 0;
}
/* ---- Right: prominent score metric cluster ---------------------------- */
.xai-summary-metrics {
    align-content: flex-start;
    border-left: 1px solid #ede7d8;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    padding-left: 1.1rem;
}
.xai-metric {
    background: #fffdf8;
    border: 1px solid #ded6c4;
    border-radius: 10px;
    flex: 1 1 88px;
    min-width: 88px;
    padding: 0.5rem 0.6rem;
    text-align: center;
}
.xai-metric-label {
    color: #8a8372;
    display: block;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.xai-metric-value {
    color: #1f1f1f;
    display: block;
    font-size: 1.05rem;
    font-weight: 800;
    margin-top: 0.2rem;
}
.xai-metric.delta {
    background: #eaf4f1;
    border-color: #2d6f73;
}
.xai-metric.delta .xai-metric-label {
    color: #1f554c;
}
.xai-metric.delta .xai-metric-value {
    color: #197a52;
    font-size: 1.3rem;
}
.xai-metric.delta.negative {
    background: #fbeceb;
    border-color: #b3261e;
}
.xai-metric.delta.negative .xai-metric-label {
    color: #8a271f;
}
.xai-metric.delta.negative .xai-metric-value {
    color: #b3261e;
}
/* ---- Δ Support magnitude tier badge: low/moderate/high, direction-aware --- */
.delta-tier-badge {
    border-radius: 999px;
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    margin-top: 0.35rem;
    padding: 0.14rem 0.55rem;
    text-transform: uppercase;
}
.delta-tier-badge.low {
    background: #f0ece0;
    color: #6d6658;
}
.delta-tier-badge.moderate {
    background: #fdf1de;
    color: #9a6a1a;
}
.delta-tier-badge.high.support {
    background: #eaf4f1;
    color: #197a52;
}
.delta-tier-badge.high.opposition {
    background: #fbeceb;
    color: #b3261e;
}
/* ---- Sign arrows: never rely on color alone for direction ------------ */
.dir-arrow {
    font-weight: 800;
}
.dir-arrow.up {
    color: #ff0d57;
}
.dir-arrow.down {
    color: #1e88e5;
}
/* ---- Light-weight confidence-% caption next to a log-prob value ------ */
.log-score-confidence {
    color: #8a8372;
    font-size: 0.72rem;
    font-weight: 500;
    margin-top: 0.15rem;
    text-transform: none;
}
.log-score-rounded {
    cursor: help;
    text-decoration: underline dotted #b8b0a0;
}
/* ---- Secondary compact score recap, spans the full card width -------- */
.xai-score-flow {
    color: #6d6658;
    font-size: 0.82rem;
    grid-column: 1 / -1;
}
.xai-score-flow strong {
    color: #403d37;
}
.xai-score-flow .delta-highlight {
    color: #1f554c;
    font-weight: 800;
    margin-left: 0.5rem;
}
.xai-score-flow .delta-highlight.negative {
    color: #b3261e;
}
.xai-score-flow .native-metric-note {
    color: #8a8372;
    font-size: 0.78rem;
    margin-left: 0.4rem;
    white-space: nowrap;
}
/* ---- Native H(&empty;)/H(N) recap nested inside the Δ Support card, below
   its tier badge, instead of as a standalone element elsewhere in the card -- */
.xai-metric.delta .delta-native-note {
    display: block;
    grid-column: auto;
    margin-top: 0.5rem;
}
/* ---- Unified Method note, shared by every fidelity branch's footer note -- */
.method-note {
    background: #fbfaf6;
    border: 1px solid #ede7d8;
    border-radius: 8px;
    color: #6d6658;
    font-size: 0.8rem;
    line-height: 1.45;
    margin: 0.3rem 0 1rem 0;
    padding: 0.55rem 0.75rem;
}
.method-note strong {
    color: #403d37;
}
/* ---- Setup chips: compact user-facing run metadata ------------------ */
.setup-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0 0 1rem 0;
}
.setup-chip {
    background: #fffdf8;
    border: 1px solid #ded6c4;
    border-radius: 999px;
    color: #403d37;
    font-size: 0.82rem;
    padding: 0.32rem 0.8rem;
}
.setup-chip strong {
    color: #1f554c;
    font-weight: 700;
}
/* ---- Player segmentation card ---------------------------------------- */
.st-key-agentic_player_card,
.player-card {
    background: #ffffff;
    border: 1px solid #ded6c4;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(45, 41, 35, 0.06);
    margin: 0.4rem 0 1.1rem 0;
    padding: 1.05rem 1.25rem 0.2rem 1.25rem;
}
.player-card-header {
    color: #1f1f1f;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 1.05rem;
    font-weight: 800;
    margin: 0 0 0.75rem 0;
}
.player-chip-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    margin-bottom: 0.9rem;
}
.player-chip {
    align-items: center;
    background: #fffdf7;
    border: 1px solid #d6cab2;
    border-radius: 999px;
    cursor: default;
    display: inline-flex;
    gap: 0.45rem;
    max-width: 230px;
    padding: 0.28rem 0.75rem 0.28rem 0.28rem;
}
.player-badge {
    align-items: center;
    background: linear-gradient(135deg, #2d6f73, #1a4245);
    border-radius: 999px;
    color: #ffffff;
    display: flex;
    flex-shrink: 0;
    font-size: 0.7rem;
    font-weight: 800;
    height: 22px;
    justify-content: center;
    min-width: 22px;
    padding: 0 0.15rem;
}
.player-chip-text {
    color: #403d37;
    font-size: 0.84rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
/* ---- Evidence cards: SV / k-SII ---------------------------------------- */
.evidence-card,
.st-key-agentic_sv_card,
.st-key-agentic_ksii_card {
    background: #ffffff;
    border: 1px solid #ded6c4;
    border-radius: 14px;
    box-shadow: 0 4px 14px rgba(45, 41, 35, 0.06);
    padding: 1.05rem 1.2rem;
}
.evidence-card-header {
    margin-bottom: 0.2rem;
}
.evidence-card-title {
    color: #1f1f1f;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 1.02rem;
    font-weight: 800;
    margin: 0;
}
.evidence-card-caption {
    color: #6d6658;
    font-size: 0.82rem;
    margin: 0.15rem 0 0.7rem 0;
}
.mini-table {
    margin: 0.5rem 0 0.2rem 0;
}
.mini-table-row {
    align-items: center;
    border-bottom: 1px solid #f0ece0;
    display: grid;
    gap: 0.5rem;
    padding: 0.4rem 0.1rem;
}
.mini-table-row:last-child {
    border-bottom: none;
}
.mini-table-row.header {
    color: #8a8372;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.mini-table-row.sv-row {
    grid-template-columns: 3.5rem 1fr 3.5rem;
}
.mini-table-row.ksii-row {
    grid-template-columns: 4.5rem 1fr 3.2rem 5.5rem;
}
.mini-table-row .cell-segment {
    color: #1f554c;
    font-size: 0.84rem;
    font-weight: 700;
}
.mini-table-row .cell-text {
    color: #403d37;
    font-size: 0.84rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.mini-table-row .cell-value {
    color: #1f1f1f;
    font-size: 0.86rem;
    font-variant-numeric: tabular-nums;
    font-weight: 700;
    text-align: right;
}
.interaction-pill {
    border-radius: 999px;
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.08rem 0.55rem;
    text-align: center;
    white-space: nowrap;
}
.interaction-pill.positive {
    background: #ffe2eb;
    color: #ff0d57;
}
.interaction-pill.negative {
    background: #e4f1fc;
    color: #1e88e5;
}
.interaction-pill.weak {
    background: #f0ece0;
    color: #6d6658;
}
.player-legend {
    background: #fffdf8;
    border: 1px solid #ede7d8;
    border-radius: 10px;
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem 0.9rem;
    margin: 0.6rem 0;
    padding: 0.55rem 0.75rem;
}
.player-legend-row {
    color: #403d37;
    font-size: 0.78rem;
}
.player-legend-row strong {
    color: #1f554c;
}
.interpretation-card {
    background: #fffdf8;
    border: 1px solid #ded6c4;
    border-left: 4px solid #b15d3b;
    border-radius: 10px;
    color: #2d2923;
    font-size: 0.94rem;
    line-height: 1.55;
    margin: 0.3rem 0 1rem 0;
    padding: 0.9rem 1.15rem;
}
.note-box {
    background: #fffdf7;
    border: 1px solid #ded6c4;
    border-radius: 6px;
    margin-bottom: 1rem;
    padding: 0.8rem 0.95rem;
}
.note-box h4 {
    color: #202020;
    font-size: 0.95rem;
    margin: 0 0 0.4rem 0;
}
.note-box ol {
    margin-bottom: 0;
    padding-left: 1.15rem;
}
.note-box li {
    color: #3f3a32;
    line-height: 1.45;
    margin: 0.25rem 0;
}
.mock-chat {
    background: #fffdf8;
    border: 1px solid #ded6c4;
    border-radius: 7px;
    display: grid;
    gap: 0.75rem;
    grid-template-columns: 1fr 1fr;
    margin: 0 0 1rem 0;
    padding: 0.9rem 1rem;
}
.mock-message {
    border-left: 4px solid #b15d3b;
    padding-left: 0.75rem;
}
.mock-message.assistant {
    border-left-color: #2d6f73;
}
.mock-message span {
    color: #6d6658;
    display: block;
    font-size: 0.72rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
    text-transform: uppercase;
}
.mock-message p {
    color: #403d37;
    line-height: 1.42;
    margin: 0;
}
@media (max-width: 850px) {
    .floating-robot-wrap {
        display: none;
    }
    .xai-top-title {
        font-size: 1.8rem;
    }
    .scenario-panel,
    .mock-chat,
    .xai-summary-card,
    .xai-top-card {
        grid-template-columns: 1fr;
    }
    .scenario-hint,
    .xai-summary-main,
    .xai-summary-metrics {
        border-left: 0;
        padding-left: 0;
    }
    .xai-top-column,
    .xai-top-column:first-child {
        padding: 0.9rem 0;
    }
    .xai-top-column:first-child {
        padding-top: 0;
    }
    .xai-top-column:last-child {
        padding-bottom: 0;
    }
    .xai-top-column + .xai-top-column {
        border-left: 0;
        border-top: 1px solid #ddd6c7;
    }
}
@media (max-width: 560px) {
    .xai-top-title {
        font-size: 1.5rem;
    }
}
</style>
"""


__all__ = [name for name in globals() if not name.startswith("__")]
