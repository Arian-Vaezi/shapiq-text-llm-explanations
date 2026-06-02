"""Compact Streamlit dashboard for RAG retrieval attribution with shapiq."""

from __future__ import annotations

import html
import itertools
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import shapiq

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from evaluation import (  # noqa: E402
    deletion_evaluation,
    insertion_evaluation,
    random_deletion_baseline,
    random_insertion_baseline,
    retrieval_score_deletion_baseline,
    summarize_faithfulness,
)
from rag_game import RAGRetrievalGame, RetrievedChunk, budget_for_exactish_demo  # noqa: E402
from rag_pipeline import (  # noqa: E402
    chunk_pdf_pages,
    extract_pdf_pages,
    generate_answer_from_chunks,
    retrieve_relevant_chunks_with_debug,
)
from sample_data import SAMPLE_TRACES  # noqa: E402
from value_functions import make_value_function  # noqa: E402

st.set_page_config(
    page_title="RAG Retrieval Explanation",
    page_icon="R",
    layout="wide",
)


CSS = """
<style>
:root {
    --bg: #f5f7fa;
    --surface: #ffffff;
    --surface-muted: #f8fafc;
    --border: #d9e0ea;
    --text: #182230;
    --text-muted: #667085;
    --accent: #4169c8;
    --positive: #23735d;
    --negative: #b6423a;
}
html,
body,
.stApp {
    background: var(--bg);
    color: var(--text);
}
section[data-testid="stSidebar"] {
    display: none;
}
.main .block-container {
    max-width: 1280px;
    padding: 0.85rem 1.25rem 2rem 1.25rem;
}
div[data-testid="stVerticalBlock"] {
    gap: 0.75rem;
}
div[data-testid="stHorizontalBlock"] {
    gap: 1rem;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 6px 6px 0 0;
    color: var(--text-muted);
    font-size: 0.84rem;
    font-weight: 650;
    height: 2.15rem;
    padding: 0 0.75rem;
}
.stTabs [aria-selected="true"] {
    background: var(--surface-muted);
    color: var(--text);
}
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextInput"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stFileUploader"] label,
div[data-testid="stTextArea"] label {
    color: var(--text-muted);
    font-size: 0.72rem;
    font-weight: 760;
    letter-spacing: 0.04em;
    line-height: 1.1;
    text-transform: uppercase;
}
.page-title {
    color: var(--text);
    font-size: 1.5rem;
    font-weight: 760;
    letter-spacing: 0;
    line-height: 1.15;
    margin: 0 0 0.65rem 0;
}
.topbar,
.st-key-top_controls {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
    margin-bottom: 0.65rem;
    padding: 0.62rem 0.75rem;
}
.st-key-top_controls div[data-testid="stHorizontalBlock"] {
    align-items: center;
    gap: 0.75rem;
}
.st-key-top_controls div[data-testid="column"] {
    align-self: center;
}
.st-key-top_controls div[data-testid="column"] > div,
.st-key-top_controls div[data-testid="stVerticalBlock"],
.st-key-top_controls div[data-testid="stElementContainer"] {
    height: auto;
    min-height: 0;
}
.control-card,
.top-summary {
    background: var(--surface-muted);
    border: 1px solid #e3e8f0;
    border-radius: 7px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    height: auto;
    justify-content: center;
    margin: 0;
    min-height: 2.85rem;
    overflow: visible;
    padding: 0.48rem 0.62rem;
}
.st-key-top_controls div[data-testid="stSelectbox"],
.st-key-top_controls div[data-testid="stSelectbox"] > div,
.st-key-top_controls div[data-testid="stSelectbox"] [data-baseweb="select"],
.st-key-top_controls .stButton > button {
    height: 3.05rem;
    min-height: 3.05rem;
}
.st-key-top_controls div[data-testid="stSelectbox"] {
    margin: 0;
}
.st-key-top_controls div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
    background: var(--surface-muted);
    border-color: #e3e8f0;
    border-radius: 7px;
    min-height: 3.05rem;
}
.st-key-top_controls div[data-testid="stSelectbox"] label {
    display: none;
}
.st-key-top_controls .stButton > button {
    font-weight: 750;
    margin: 0;
}
.top-summary span,
.card-label,
.badge-label,
.label {
    color: var(--text-muted);
    display: block;
    font-size: 0.69rem;
    font-weight: 750;
    letter-spacing: 0.04em;
    line-height: 1.1;
    text-transform: uppercase;
}
.top-summary strong {
    color: var(--text);
    display: block;
    font-size: 0.88rem;
    line-height: 1.25;
    margin-top: 0.18rem;
}
.card,
.side-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.025);
}
.card {
    padding: 0.75rem 0.85rem;
}
.side-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.025);
    margin: 0 0 0.75rem 0;
    padding: 0.72rem 0.78rem;
}
.side-card + .side-card {
    margin-top: 0.65rem;
}
.card + .card {
    margin-top: 0.65rem;
}
.card-title,
.side-card-title {
    color: var(--text);
    font-size: 1rem;
    font-weight: 760;
    line-height: 1.25;
    margin: 0 0 0.55rem 0;
}
.side-card h3.side-card-title,
h3.side-card-title {
    color: var(--text) !important;
    font-size: 1rem !important;
    font-weight: 760 !important;
    line-height: 1.25 !important;
    margin: 0 0 0.55rem 0 !important;
    padding: 0 !important;
}
.question-text {
    color: var(--text);
    font-size: 1rem;
    font-weight: 650;
    line-height: 1.42;
    margin: 0.18rem 0 0 0;
}
.target-text,
.muted-text {
    color: var(--text-muted);
    font-size: 0.86rem;
    line-height: 1.45;
    margin: 0.28rem 0 0 0;
}
.settings-grid {
    display: grid;
    gap: 0.48rem;
    grid-template-columns: repeat(2, minmax(0, 1fr));
}
.setting-cell,
.kv {
    background: var(--surface-muted);
    border: 1px solid #e3e8f0;
    border-radius: 7px;
    min-width: 0;
    padding: 0.45rem 0.52rem;
}
.setting-cell-full {
    grid-column: 1 / -1;
}
.setting-cell strong,
.kv strong {
    color: var(--text);
    display: block;
    font-size: 0.88rem;
    line-height: 1.25;
    margin-top: 0.16rem;
    overflow-wrap: anywhere;
}
.chip-list {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 0.28rem;
    margin-top: 0.26rem;
}
.vf-chip,
.score-badge {
    border-radius: 999px;
    display: inline-flex;
    font-size: 0.74rem;
    font-weight: 760;
    line-height: 1.1;
    white-space: nowrap;
}
.vf-chip {
    background: #eef3ff;
    border: 1px solid #d2dcf7;
    color: #2f4f9f;
    padding: 0.18rem 0.42rem;
}
.vf-chip.muted {
    background: #f2f4f7;
    border-color: #e3e8f0;
    color: var(--text-muted);
}
.run-state {
    background: var(--surface-muted);
    border: 1px solid #e3e8f0;
    border-left: 3px solid var(--accent);
    border-radius: 7px;
    color: #344054;
    font-size: 0.86rem;
    line-height: 1.4;
    padding: 0.5rem 0.58rem;
}
.run-state.error {
    border-left-color: var(--negative);
    background: #fff8f7;
}
.run-state.success {
    border-left-color: var(--positive);
    background: #f6fbf8;
}
.side-card .run-state {
    margin-bottom: 0.48rem;
}
.side-card .run-state:last-child {
    margin-bottom: 0;
}
.metric-row {
    align-items: baseline;
    border-top: 1px solid #eef2f6;
    display: grid;
    gap: 0.3rem;
    grid-template-columns: minmax(82px, 0.85fr) minmax(0, 1.15fr);
    padding: 0.46rem 0 0.38rem 0;
}
.metric-row:first-of-type {
    border-top: 0;
    padding-top: 0;
}
.metric-row strong {
    color: var(--text);
    display: block;
    font-size: 0.95rem;
    line-height: 1.25;
    overflow-wrap: anywhere;
}
.formula-block {
    background: var(--surface-muted);
    border: 1px solid #e3e8f0;
    border-radius: 7px;
    color: var(--text);
    font-size: 0.84rem;
    line-height: 1.45;
    margin-top: 0.45rem;
    padding: 0.42rem 0.5rem;
}
.formula-label {
    color: var(--text-muted);
    display: block;
    font-size: 0.68rem;
    font-weight: 750;
    letter-spacing: 0.04em;
    margin-top: 0.48rem;
    text-transform: uppercase;
}
.evidence-card,
.chunk-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--chunk-border);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.025);
    margin-bottom: 0.56rem;
    overflow: hidden;
}
.chunk-card header {
    align-items: center;
    background: linear-gradient(0deg, var(--chunk-bg), var(--chunk-bg)), var(--surface-muted);
    display: flex;
    gap: 0.55rem;
    justify-content: space-between;
    padding: 0.54rem 0.68rem 0.42rem 0.68rem;
}
.chunk-title-row {
    align-items: center;
    display: flex;
    gap: 0.45rem;
    min-width: 0;
}
.chunk-number {
    background: #eef2f6;
    border: 1px solid #dde4ed;
    border-radius: 999px;
    color: #475467;
    flex: 0 0 auto;
    font-size: 0.68rem;
    font-weight: 780;
    letter-spacing: 0.04em;
    line-height: 1;
    padding: 0.18rem 0.38rem;
    text-transform: uppercase;
}
.chunk-card h4 {
    color: var(--text);
    font-size: 0.92rem;
    font-weight: 740;
    line-height: 1.25;
    margin: 0;
    overflow-wrap: anywhere;
}
.score-badge {
    background: var(--badge-bg);
    border: 1px solid var(--badge-border);
    color: var(--badge-fg);
    flex: 0 0 auto;
    font-variant-numeric: tabular-nums;
    padding: 0.16rem 0.44rem;
}
.chunk-meta {
    background: var(--chunk-bg);
    color: #667085;
    font-size: 0.74rem;
    font-weight: 650;
    line-height: 1.35;
    padding: 0.34rem 0.68rem 0 0.68rem;
}
.chunk-card p {
    background: var(--chunk-bg);
    color: #344054;
    font-size: 0.86rem;
    line-height: 1.52;
    margin: 0;
    padding: 0.2rem 0.68rem 0.64rem 0.68rem;
}
.section-title {
    color: var(--text);
    font-size: 1.18rem;
    font-weight: 760;
    line-height: 1.25;
    margin: 0.25rem 0 0.48rem 0;
}
.chart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.025);
    padding: 0.52rem 0.62rem 0.22rem 0.62rem;
}
.empty-state {
    background: var(--surface-muted);
    border: 1px dashed #cfd8e5;
    border-radius: 8px;
    color: var(--text-muted);
    font-size: 0.88rem;
    line-height: 1.45;
    margin-top: 0.7rem;
    padding: 0.75rem 0.85rem;
}
.advanced-heading {
    color: var(--text);
    font-size: 1.05rem;
    font-weight: 760;
    line-height: 1.25;
    margin: 1rem 0 0.45rem 0;
}
.analysis-layer {
    border-top: 1px solid var(--border);
    margin-top: 1rem;
    padding-top: 0.85rem;
}
.analysis-layer-header {
    align-items: start;
    display: grid;
    gap: 0.75rem;
    grid-template-columns: minmax(0, 1fr) auto;
    margin: 0 0 0.48rem 0;
}
.analysis-layer-title {
    color: var(--text);
    display: block;
    font-size: 1rem;
    font-weight: 760;
    line-height: 1.25;
    margin: 0;
}
.analysis-layer-index {
    color: var(--accent);
    display: block;
    font-size: 0.68rem;
    font-weight: 780;
    letter-spacing: 0.05em;
    line-height: 1.1;
    margin-bottom: 0.18rem;
    text-transform: uppercase;
}
.analysis-layer-subtitle {
    color: var(--text-muted);
    font-size: 0.84rem;
    line-height: 1.42;
    margin: 0.24rem 0 0 0;
    max-width: 760px;
}
.analysis-layer-items {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
    justify-content: flex-end;
}
.analysis-layer-pill {
    background: var(--surface);
    border: 1px solid #dfe6f0;
    border-radius: 999px;
    color: var(--text-muted);
    display: inline-flex;
    font-size: 0.72rem;
    font-weight: 720;
    line-height: 1.1;
    padding: 0.22rem 0.46rem;
    white-space: nowrap;
}
.analysis-tab-note {
    color: var(--text-muted);
    font-size: 0.84rem;
    line-height: 1.42;
    margin: 0.15rem 0 0.65rem 0;
}
.audit-metrics {
    display: grid;
    gap: 0.65rem;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    margin: 0.1rem 0 0.8rem 0;
}
.audit-metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.025);
    padding: 0.68rem 0.78rem;
}
.audit-metric-card span {
    color: var(--text-muted);
    display: block;
    font-size: 0.68rem;
    font-weight: 760;
    letter-spacing: 0.04em;
    line-height: 1.1;
    text-transform: uppercase;
}
.audit-metric-card strong {
    color: var(--text);
    display: block;
    font-size: 1.08rem;
    font-variant-numeric: tabular-nums;
    line-height: 1.2;
    margin-top: 0.2rem;
}
.audit-metric-card p {
    color: var(--text-muted);
    font-size: 0.8rem;
    line-height: 1.38;
    margin: 0.35rem 0 0 0;
}
.audit-order-label {
    align-items: center;
    background: #eef3ff;
    border: 1px solid #d2dcf7;
    border-radius: 999px;
    color: #2f4f9f;
    display: inline-flex;
    font-size: 0.76rem;
    font-weight: 760;
    margin: 0 0 0.65rem 0;
    padding: 0.22rem 0.52rem;
}
.page-intro {
    color: var(--text-muted);
    font-size: 0.9rem;
    line-height: 1.45;
    margin: -0.22rem 0 0.7rem 0;
}
.method-table {
    margin-bottom: 0.85rem;
}
div[class*="st-key-vf_card_"],
div[class*="st-key-rag_step_"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-sizing: border-box;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.025);
    height: auto;
    min-height: 0;
    overflow: visible;
    padding: 1rem 1.05rem 1.15rem 1.05rem;
}
div[class*="st-key-vf_card_"] div[data-testid="stVerticalBlock"],
div[class*="st-key-vf_card_"] div[data-testid="stElementContainer"],
div[class*="st-key-rag_step_"] div[data-testid="stVerticalBlock"],
div[class*="st-key-rag_step_"] div[data-testid="stElementContainer"] {
    height: auto;
    min-height: 0;
    overflow: visible;
}
div[class*="st-key-vf_card_"] [data-testid="stMarkdownContainer"] p,
div[class*="st-key-rag_step_"] [data-testid="stMarkdownContainer"] p {
    margin-bottom: 0;
}
div[class*="st-key-vf_card_"] [data-testid="stLatex"],
div[class*="st-key-rag_step_"] [data-testid="stLatex"] {
    background: var(--surface-muted);
    border: 1px solid #e3e8f0;
    border-radius: 7px;
    box-sizing: border-box;
    height: auto;
    margin: 0.62rem 0 0.75rem 0;
    min-height: 0;
    overflow-y: visible;
    overflow-x: auto;
    padding: 0.62rem 0.7rem;
}
div[class*="st-key-vf_card_"] [data-testid="stLatex"] *,
div[class*="st-key-rag_step_"] [data-testid="stLatex"] * {
    max-height: none;
    overflow: visible;
}
.method-card-title {
    color: var(--text);
    font-size: 1rem;
    font-weight: 760;
    line-height: 1.25;
    margin: 0 0 0.5rem 0;
}
.method-badges {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    margin-bottom: 0.72rem;
}
.meta-badge {
    background: #f2f4f7;
    border: 1px solid #e3e8f0;
    border-radius: 999px;
    color: #475467;
    display: inline-flex;
    font-size: 0.71rem;
    font-weight: 750;
    line-height: 1.1;
    padding: 0.18rem 0.42rem;
}
.meta-badge.positive {
    background: #eef8f4;
    border-color: #c8e4d8;
    color: var(--positive);
}
.meta-badge.negative {
    background: #fff1ef;
    border-color: #f2c4be;
    color: var(--negative);
}
.method-detail {
    border-top: 1px solid #eef2f6;
    display: grid;
    gap: 0.45rem;
    grid-template-columns: 4.4rem minmax(0, 1fr);
    padding-top: 0.68rem;
}
.method-detail + .method-detail {
    margin-top: 0.68rem;
}
.method-detail strong {
    color: var(--text-muted);
    font-size: 0.69rem;
    font-weight: 750;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.method-detail span {
    color: #344054;
    font-size: 0.84rem;
    line-height: 1.42;
}
.method-detail:last-child {
    padding-bottom: 0.1rem;
}
.compact-table [data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 8px;
}
.story-step {
    align-items: center;
    display: flex;
    gap: 0.6rem;
    margin: 1.4rem 0 0.22rem 0;
}
.story-step-num {
    background: var(--accent);
    border-radius: 999px;
    color: #fff;
    font-size: 0.68rem;
    font-weight: 780;
    letter-spacing: 0.06em;
    padding: 0.18rem 0.52rem;
    text-transform: uppercase;
    white-space: nowrap;
}
.story-step-title {
    color: var(--text);
    font-size: 1.08rem;
    font-weight: 760;
    line-height: 1.25;
}
.story-step-sub {
    color: var(--text-muted);
    font-size: 0.86rem;
    line-height: 1.45;
    margin: 0 0 0.65rem 0;
}
.role-tag {
    border-radius: 999px;
    display: inline-block;
    font-size: 0.66rem;
    font-weight: 760;
    letter-spacing: 0.03em;
    padding: 0.14rem 0.38rem;
    white-space: nowrap;
}
.role-tag.evidence {
    background: #eef3ff;
    border: 1px solid #c8d5f5;
    color: #2f4f9f;
}
.role-tag.redundant {
    background: #f3f0ff;
    border: 1px solid #d4c9f7;
    color: #5b3fa6;
}
.role-tag.distractor {
    background: #fef9ee;
    border: 1px solid #f0d894;
    color: #8a6300;
}
.role-tag.background {
    background: #f2f4f7;
    border: 1px solid #e3e8f0;
    color: #475467;
}
.negative-callout {
    background: #fff8f2;
    border: 1px solid #f5c99b;
    border-left: 3px solid #d97706;
    border-radius: 7px;
    color: #344054;
    font-size: 0.86rem;
    line-height: 1.48;
    margin: 0.5rem 0 0.6rem 0;
    padding: 0.52rem 0.7rem;
}
.negative-callout strong { color: #92400e; }
.interaction-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    margin-bottom: 0.75rem;
    padding: 0.68rem 0.85rem;
}
.interaction-card-pair {
    color: var(--text);
    font-size: 0.95rem;
    font-weight: 760;
    line-height: 1.25;
    margin: 0 0 0.3rem 0;
}
.interaction-card-meta {
    align-items: center;
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-bottom: 0.38rem;
}
.interaction-card-text {
    color: var(--text-muted);
    font-size: 0.86rem;
    line-height: 1.45;
    margin: 0;
}
.ins-story {
    border: 1px solid var(--border);
    border-radius: 8px;
    margin: 0.55rem 0 0.9rem 0;
    overflow: hidden;
}
.ins-step {
    align-items: center;
    border-bottom: 1px solid #f0f4f8;
    display: grid;
    font-size: 0.84rem;
    gap: 0.5rem;
    grid-template-columns: 2.2rem 1fr auto auto;
    line-height: 1.4;
    padding: 0.4rem 0.72rem;
}
.ins-step:last-child { border-bottom: 0; }
.ins-step.highlight { background: #f6fbf8; }
.ins-step-num {
    color: var(--text-muted);
    font-size: 0.72rem;
    font-weight: 780;
    letter-spacing: 0.03em;
}
.ins-step-text { color: var(--text); overflow-wrap: anywhere; }
.ins-step-score {
    color: var(--text-muted);
    font-size: 0.82rem;
    font-variant-numeric: tabular-nums;
    text-align: right;
}
.ins-step-delta {
    font-size: 0.82rem;
    font-variant-numeric: tabular-nums;
    font-weight: 760;
    min-width: 4.2rem;
    text-align: right;
}
.ins-step-delta.pos { color: var(--positive); }
.ins-step-delta.neg { color: var(--negative); }
.ins-step-delta.neu { color: var(--text-muted); }
.pdf-chat-shell {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.025);
    margin-bottom: 0.9rem;
    padding: 0.75rem 0.85rem;
}
.pdf-chat-head {
    align-items: center;
    border-bottom: 1px solid #eef2f6;
    display: flex;
    justify-content: space-between;
    gap: 0.75rem;
    padding-bottom: 0.55rem;
}
.pdf-chat-title {
    color: var(--text);
    display: block;
    font-size: 0.98rem;
    font-weight: 760;
    line-height: 1.25;
}
.pdf-chat-file {
    color: var(--text-muted);
    font-size: 0.78rem;
    line-height: 1.25;
    overflow-wrap: anywhere;
}
.pdf-chat-body {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    padding-top: 0.7rem;
}
.chat-row {
    display: flex;
}
.chat-row.user {
    justify-content: flex-end;
}
.chat-row.assistant {
    justify-content: flex-start;
}
.chat-bubble {
    border-radius: 8px;
    color: var(--text);
    font-size: 0.88rem;
    line-height: 1.48;
    max-width: min(760px, 82%);
    padding: 0.62rem 0.72rem;
}
.chat-row.user .chat-bubble {
    background: #eaf0ff;
    border: 1px solid #d2dcf7;
}
.chat-row.assistant .chat-bubble {
    background: var(--surface-muted);
    border: 1px solid #e3e8f0;
}
.chat-bubble.muted {
    color: var(--text-muted);
}
</style>
"""


VALUE_FUNCTION_DISPLAY = {
    "lexical_grounding": {
        "name": "Lexical grounding",
        "category": "Lexical baseline",
        "speed": "Fast",
        "requires_model": False,
        "recommended_for_iteration": True,
        "description": "Fast local baseline for answer-term grounding.",
        "use_case": "Good for quick debugging and checking whether answer terms appear in retrieved chunks.",
        "caveat": "Can overrate keyword overlap and does not verify semantic entailment.",
        "formula": "coverage(target terms in context)^1.35 + question-overlap bonus - length penalty",
        "latex": (
            r"V(C)=\operatorname{coverage}(\text{target terms}, C)^{1.35}"
            r"+b_{\text{question-overlap}}(C)-p_{\text{length}}(C)"
        ),
    },
    "hf_target_likelihood": {
        "name": "HF target likelihood",
        "category": "Model likelihood scorer",
        "speed": "Slow",
        "requires_model": True,
        "recommended_for_iteration": False,
        "description": "Measures whether selected context makes the exact target answer likely.",
        "use_case": "Useful for measuring prompt-conditioned support for a known target answer.",
        "caveat": "Expensive because shapiq evaluates many coalitions; it also rewards likelihood, not factual entailment by itself.",
        "formula": "sigmoid(mean log p(target answer tokens | prompt, selected chunks) / T)",
        "latex": (
            r"V(C)=\sigma\left(\frac{1}{T}\operatorname{mean}_{t \in A}"
            r"\log p(t \mid Q, C, A_{<t})\right)"
        ),
    },
    "hf_contrastive_likelihood": {
        "name": "HF contrastive likelihood",
        "category": "Contrastive HF scorer",
        "speed": "Slow",
        "requires_model": True,
        "recommended_for_iteration": False,
        "description": "Measures target-answer likelihood gain over the model's no-context prior.",
        "use_case": "Useful when you want attribution to reflect what the retrieved context adds beyond the model's prior.",
        "caveat": "Requires two likelihood evaluations per coalition and can be sensitive to calibration and prompt formatting.",
        "formula": "sigmoid((LL_target(selected context) - LL_target(no context)) / T)",
        "latex": (
            r"V(C)=\sigma\left(\frac{\operatorname{LL}(A \mid Q,C)"
            r"-\operatorname{LL}(A \mid Q,\varnothing)}{T}\right)"
        ),
    },
    "hf_generated_answer_overlap": {
        "name": "HF generated answer overlap",
        "category": "Generation-based HF scorer",
        "speed": "Slow",
        "requires_model": True,
        "recommended_for_iteration": False,
        "description": "Generates an answer and compares it with the target answer by token overlap.",
        "use_case": "Useful for checking whether selected context leads the model to generate target-answer terms.",
        "caveat": "Generation is noisy and token overlap can miss paraphrases or reward shallow wording matches.",
        "formula": "|tokens(generated answer) intersect tokens(target answer)| / |tokens(target answer)|",
        "latex": (
            r"V(C)=\frac{|\operatorname{tokens}(\hat{A}(Q,C))"
            r"\cap \operatorname{tokens}(A)|}{|\operatorname{tokens}(A)|}"
        ),
    },
}

VALUE_FUNCTION_DEFINITIONS = {
    metadata["name"]: {
        "category": metadata["category"],
        "speed": metadata["speed"],
        "requires_model": metadata["requires_model"],
        "recommended_for_iteration": metadata["recommended_for_iteration"],
        "formula": metadata["formula"],
        "latex": metadata["latex"],
        "short": metadata["description"],
        "use_case": metadata["use_case"],
        "caveat": metadata["caveat"],
    }
    for metadata in VALUE_FUNCTION_DISPLAY.values()
}


@dataclass
class ExplanationRunResult:
    """Display-ready data produced by one value-function run."""

    scenario_name: str
    value_mode: str
    value_name: str
    value_description: str
    value_formula: str
    full_score: float
    empty_score: float
    first_order_frame: pd.DataFrame
    pairwise_matrix: np.ndarray
    audit_frame: pd.DataFrame
    pair_label: str
    pair_value: float
    faithfulness: object
    deletion_frame: pd.DataFrame
    insertion_frame: pd.DataFrame
    sentence_labels: list[str]
    sentence_meta: pd.DataFrame
    sentence_first_order_frame: pd.DataFrame
    random_deletion_baseline: np.ndarray | None = None
    random_insertion_baseline: np.ndarray | None = None
    retrieval_deletion_baseline: np.ndarray | None = None
    loo_scores: list[float] | None = None

    @property
    def top_chunk(self) -> str:
        """Return the highest-ranked chunk label."""
        if self.first_order_frame.empty:
            return "No chunk"
        row = self.first_order_frame.iloc[0]
        return f"{int(row['player'])}. {row['chunk']}"

    @property
    def top_score(self) -> float:
        """Return the highest first-order attribution."""
        if self.first_order_frame.empty:
            return 0.0
        return float(self.first_order_frame.iloc[0]["attribution"])

    @property
    def direct_evidence_rank(self) -> int | None:
        """Return the fixture convention rank for chunk 1."""
        if self.first_order_frame.empty:
            return None
        players = [int(row.player) for row in self.first_order_frame.itertuples(index=False)]
        return players.index(1) + 1 if 1 in players else None


class ProgressHandle(Protocol):
    """Subset of Streamlit progress handle used by this app."""

    def progress(self, value: float) -> None:
        """Update progress to a value between 0 and 1."""


class MessageHandle(Protocol):
    """Subset of Streamlit message placeholder used by this app."""

    def info(self, body: str) -> None:
        """Show an informational message."""

    def success(self, body: str) -> None:
        """Show a success message."""

    def error(self, body: str) -> None:
        """Show an error message."""


class ProgressScorer:
    """Wrap a value function and report scoring progress to Streamlit."""

    def __init__(
        self,
        scorer: object,
        *,
        label: str,
        progress: ProgressHandle,
        message: MessageHandle,
        expected_calls: int,
        base_progress: float,
        progress_span: float,
    ) -> None:
        """Store progress reporting state."""
        self.scorer = scorer
        self.label = label
        self.progress = progress
        self.message = message
        self.expected_calls = max(1, expected_calls)
        self.base_progress = base_progress
        self.progress_span = progress_span
        self.calls = 0
        self.name = getattr(scorer, "name", "Value function")
        self.description = getattr(scorer, "description", "")

    def __call__(
        self,
        question: str,
        target_answer: str,
        selected_chunks: list[RetrievedChunk],
    ) -> float:
        """Score selected chunks and update visible progress."""
        self.calls += 1
        fraction = min(1.0, self.calls / self.expected_calls)
        self.progress.progress(self.base_progress + self.progress_span * fraction)
        self.message.info(
            f"{self.label}: scoring coalition {self.calls} "
            f"({len(selected_chunks)} selected chunks)."
        )
        return self.scorer(question, target_answer, selected_chunks)

    def warmup(self) -> None:
        """Warm up the wrapped scorer if it exposes a warmup hook."""
        warmup = getattr(self.scorer, "warmup", None)
        if warmup is not None:
            warmup()


def split_sentences(text: str) -> list[str]:
    """Split a retrieved chunk into simple sentence-like units."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def trace_chunks(trace: dict[str, object]) -> list[RetrievedChunk]:
    """Convert trace fixture chunks to retrieved chunk objects."""
    return [
        RetrievedChunk(
            title=str(chunk["title"]),
            text=str(chunk["text"]),
            page_number=chunk.get("page_number"),
            chunk_type=str(chunk.get("chunk_type", "body_text")),
            section_title=str(chunk.get("section_title", "")),
            text_length=int(chunk.get("text_length", 0) or 0),
            retrieval_score=chunk.get("rerank_score"),
            dense_score=chunk.get("dense_score"),
            keyword_score=chunk.get("keyword_score"),
            rerank_score=chunk.get("rerank_score"),
            flags=tuple(chunk.get("flags", [])),
        )
        for chunk in trace["chunks"]
    ]


def short_text(text: str, limit: int = 150) -> str:
    """Return compact text for chart tooltips."""
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[: limit - 1]}..."


def get_setting(key: str, default: object) -> object:
    """Read a persistent Streamlit setting with a default."""
    if key not in st.session_state:
        st.session_state[key] = default
    return st.session_state[key]


def effective_max_order() -> int:
    """Return the max order that applies to the selected index."""
    if st.session_state.interaction_index == "SV":
        return 1
    return int(st.session_state.max_order)


def normalize_value_modes() -> None:
    """Keep the value-function selection non-empty and valid."""
    valid_modes = list(VALUE_FUNCTION_DEFINITIONS)
    modes = [mode for mode in st.session_state.value_modes if mode in valid_modes]
    st.session_state.value_modes = modes or ["Lexical grounding"]


def interaction_values_to_frame(
    values: shapiq.InteractionValues,
    labels: list[str],
) -> pd.DataFrame:
    """Convert first-order interaction values to a display frame."""
    rows = []
    for interaction, score in values.dict_values.items():
        if len(interaction) == 1:
            idx = interaction[0]
            rows.append(
                {
                    "chunk": labels[idx],
                    "player": idx + 1,
                    "attribution": float(score),
                    "abs_attribution": abs(float(score)),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("abs_attribution", ascending=False).drop(columns=["abs_attribution"])


def make_approximator(index: str, n_players: int, max_order: int) -> object:
    """Create an approximator compatible with the selected interaction index."""
    if index == "SV":
        return shapiq.KernelSHAP(n=n_players, random_state=42)
    if index == "STII":
        return shapiq.PermutationSamplingSTII(
            n=n_players,
            max_order=max_order,
            random_state=42,
        )
    if index == "FSII":
        return shapiq.RegressionFSII(
            n=n_players,
            max_order=max_order,
            random_state=42,
        )
    return shapiq.KernelSHAPIQ(
        n=n_players,
        index=index,
        max_order=max_order,
        random_state=42,
    )


@st.cache_resource(show_spinner=False)
def cached_value_function(
    mode: str,
    model_id: str,
    device_map: str,
    torch_dtype: str,
    max_new_tokens: int,
) -> object:
    """Cache model-backed value functions across Streamlit reruns."""
    return make_value_function(
        mode,
        model_id=model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
    )


def strongest_pair(matrix: np.ndarray, labels: list[str]) -> tuple[str, float]:
    """Return the strongest non-diagonal pairwise interaction."""
    if matrix.shape[0] < 2:
        return "No pair", 0.0
    best_pair = (0, 1)
    best_value = matrix[0, 1]
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            if abs(matrix[i, j]) > abs(best_value):
                best_pair = (i, j)
                best_value = matrix[i, j]
    pair_label = (
        f"{best_pair[0] + 1}. {labels[best_pair[0]]} + {best_pair[1] + 1}. {labels[best_pair[1]]}"
    )
    return pair_label, float(best_value)


def pairwise_matrix_from_explanation(
    explanation: shapiq.InteractionValues,
    n_players: int,
) -> np.ndarray:
    """Extract the second-order interaction matrix from shapiq results."""
    if explanation.max_order < 2:
        return np.zeros((n_players, n_players), dtype=float)
    return explanation.get_n_order_values(2)


def build_sentence_players(
    chunks: list[RetrievedChunk],
) -> tuple[list[RetrievedChunk], list[str], pd.DataFrame]:
    """Split all retrieved chunks into sentence-level players."""
    sentence_chunks = []
    sentence_labels = []
    metadata_rows = []
    sentence_id = 1
    for chunk_idx, chunk in enumerate(chunks):
        for sentence in split_sentences(chunk.text):
            label = f"S{sentence_id}"
            sentence_chunks.append(RetrievedChunk(title=label, text=sentence))
            sentence_labels.append(label)
            metadata_rows.append(
                {
                    "sentence": label,
                    "source_chunk": f"{chunk_idx + 1}. {chunk.title}",
                    "text": sentence,
                }
            )
            sentence_id += 1
    return sentence_chunks, sentence_labels, pd.DataFrame(metadata_rows)


def coalition_audit_frame(game: RAGRetrievalGame) -> pd.DataFrame:
    """Build a small exact coalition audit table for transparency."""
    rows = []
    n_players = game.n_players
    for size in range(min(n_players, 3) + 1):
        for combo in itertools.combinations(range(n_players), size):
            coalition = np.zeros((1, n_players), dtype=bool)
            coalition[0, list(combo)] = True
            score = float(game(coalition)[0])
            rows.append(
                {
                    "coalition": ", ".join(str(idx + 1) for idx in combo) or "empty",
                    "selected_chunks": size,
                    "normalized_score": score,
                }
            )
    return pd.DataFrame(rows).sort_values("normalized_score", ascending=False)


def attribution_lookup(attribution_frame: pd.DataFrame) -> dict[int, float]:
    """Map zero-based chunk ids to attribution scores."""
    if attribution_frame.empty:
        return {}
    return {
        int(row.player) - 1: float(row.attribution)
        for row in attribution_frame.itertuples(index=False)
    }


def attribution_style(score: float | None, max_abs: float) -> dict[str, str]:
    """Return subtle HTML colors for an attribution score."""
    if score is None:
        return {
            "border": "#cbd5e1",
            "bg": "#ffffff",
            "badge_bg": "#f8fafc",
            "badge_border": "#d8dee7",
            "badge_fg": "#475467",
        }
    scale = min(1.0, abs(score) / max(max_abs, 0.001))
    alpha = 0.035 + 0.08 * scale
    if score >= 0:
        return {
            "border": "#26735f",
            "bg": f"rgba(38, 115, 95, {alpha:.3f})",
            "badge_bg": "#eef8f4",
            "badge_border": "#b7ddce",
            "badge_fg": "#16614e",
        }
    return {
        "border": "#b5473f",
        "bg": f"rgba(181, 71, 63, {alpha:.3f})",
        "badge_bg": "#fff1ef",
        "badge_border": "#f2b8b2",
        "badge_fg": "#9d312b",
    }


def prepare_attribution_frame(
    result: ExplanationRunResult,
    chunks: list[RetrievedChunk],
) -> pd.DataFrame:
    """Add chart metadata to one attribution frame."""
    frame = result.first_order_frame.copy()
    if frame.empty:
        return frame
    frame["chunk_id"] = frame["player"].astype(int)
    frame["title"] = frame["chunk"]
    frame["preview"] = frame["chunk_id"].map(
        {idx + 1: short_text(chunk.text) for idx, chunk in enumerate(chunks)}
    )
    frame["value_function"] = result.value_name
    if result.loo_scores is not None:
        loo_map = {i + 1: score for i, score in enumerate(result.loo_scores)}
        frame["loo"] = frame["chunk_id"].map(loo_map)
    return frame


def comparison_frame(results: list[ExplanationRunResult]) -> pd.DataFrame:
    """Build the value-function comparison table."""
    rows = []
    for result in results:
        rank = result.direct_evidence_rank
        rows.append(
            {
                "value_function": result.value_name,
                "description": result.value_description,
                "formula": result.value_formula,
                "support_score": result.full_score,
                "top_evidence": result.top_chunk,
                "direct_evidence_rank": rank if rank is not None else "n/a",
                "direct_above_distractors": rank == 1,
            }
        )
    return pd.DataFrame(rows)


def value_mode_chips(modes: list[str]) -> str:
    """Render selected value-function labels as compact chips."""
    if not modes:
        return '<span class="vf-chip muted">None selected</span>'
    return "".join(f'<span class="vf-chip">{html.escape(mode)}</span>' for mode in modes)


def selected_value_modes_require_model() -> bool:
    """Return whether the selected value functions need a local/HF model."""
    return any(
        bool(VALUE_FUNCTION_DEFINITIONS[mode]["requires_model"])
        for mode in st.session_state.value_modes
        if mode in VALUE_FUNCTION_DEFINITIONS
    )


def estimate_run_workload(chunks: list[RetrievedChunk]) -> dict[str, int | bool]:
    """Estimate model/scorer calls before the user commits to a run."""
    n_chunks = len(chunks)
    n_sentences = len(build_sentence_players(chunks)[0]) if chunks else 0
    chunk_budget = int(st.session_state.budget)
    chunk_overhead = n_chunks + 8
    sentence_calls = 0
    if st.session_state.run_sentence_drilldown and n_sentences >= 2:
        sentence_calls = budget_for_exactish_demo(n_sentences)
    return {
        "chunks": n_chunks,
        "sentences": n_sentences,
        "chunk_calls_per_value": chunk_budget + chunk_overhead,
        "sentence_calls_per_value": sentence_calls,
        "value_functions": len(st.session_state.value_modes),
        "uses_model": selected_value_modes_require_model(),
        "run_sentence_drilldown": bool(st.session_state.run_sentence_drilldown),
    }


def render_run_preview(chunks: list[RetrievedChunk]) -> None:
    """Show a compact estimate of how much work the next run will do."""
    workload = estimate_run_workload(chunks)
    value_count = int(workload["value_functions"])
    chunk_calls = int(workload["chunk_calls_per_value"]) * max(1, value_count)
    sentence_calls = int(workload["sentence_calls_per_value"]) * max(1, value_count)
    total_calls = chunk_calls + sentence_calls
    model_note = "model calls" if workload["uses_model"] else "scorer calls"
    sentence_label = (
        f"{sentence_calls:,} sentence drilldown calls"
        if workload["run_sentence_drilldown"]
        else "sentence drilldown skipped"
    )
    st.info(
        "Next run estimate: "
        f"{int(workload['chunks'])} retrieved chunks, "
        f"{int(workload['sentences'])} sentence players, "
        f"about {total_calls:,} {model_note} "
        f"({chunk_calls:,} chunk attribution + {sentence_label})."
    )
    if "HF contrastive likelihood" in st.session_state.get("value_modes", []):
        st.warning(
            "HF contrastive likelihood scores each coalition with two separate model calls "
            "(one with context, one without). Expect roughly 2x the wall-clock time "
            "compared with the other HF value functions."
        )


def render_page_title() -> None:
    """Render the compact dashboard title."""
    st.markdown('<h1 class="page-title">RAG Retrieval Attribution</h1>', unsafe_allow_html=True)


def render_top_controls() -> bool:
    """Render compact top controls and return whether Run was clicked."""
    scenario_options = list(SAMPLE_TRACES)
    if st.session_state.scenario_name not in scenario_options:
        st.session_state.scenario_name = scenario_options[0]
    modes = value_mode_chips(st.session_state.value_modes)
    index_label = f"{st.session_state.interaction_index}, order {effective_max_order()}"
    with st.container(key="top_controls"):
        source_col, scenario_col, value_col, index_col, action_col = st.columns(
            [0.95, 1.55, 1.45, 0.95, 0.62],
            vertical_alignment="center",
        )
        with source_col:
            st.selectbox(
                "Input source",
                ["Sample traces", "PDF upload"],
                key="input_source",
                label_visibility="collapsed",
            )
        with scenario_col:
            if st.session_state.input_source == "Sample traces":
                st.selectbox(
                    "Scenario",
                    scenario_options,
                    key="scenario_name",
                    label_visibility="collapsed",
                )
            else:
                label = st.session_state.get("pdf_name") or "Upload a PDF in settings"
                st.markdown(
                    f"""
                    <div class="top-summary">
                        <span>PDF RAG source</span>
                        <strong>{html.escape(label)}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with value_col:
            st.markdown(
                f"""
                <div class="top-summary">
                    <span>Value functions</span>
                    <div class="chip-list">{modes}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with index_col:
            st.markdown(
                f"""
                <div class="top-summary">
                    <span>Index</span>
                    <strong>{html.escape(index_label)}</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with action_col:
            if st.session_state.input_source == "Sample traces":
                run = st.button("Run", type="primary", width="stretch", key="run_button")
            else:
                run = False
                st.markdown(
                    """
                    <div class="top-summary">
                        <span>Run action</span>
                        <strong>Ask in chat</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    return run


def render_navigation() -> str:
    """Render primary navigation and return the selected page."""
    return st.radio(
        "Page",
        ["Dashboard", "Explanation Settings", "Value Functions", "RAG Pipeline"],
        key="active_page",
        horizontal=True,
        label_visibility="collapsed",
    )


def render_settings_summary() -> None:
    """Render compact current configuration summary."""
    mode_chips = value_mode_chips(st.session_state.value_modes)
    model_value = (
        html.escape(st.session_state.model_id)
        if selected_value_modes_require_model()
        else "Not used by selected value function"
    )
    st.markdown(
        f"""
        <div class="side-card">
            <h3 class="side-card-title">Current settings</h3>
            <div class="settings-grid">
                <div class="setting-cell">
                    <span class="badge-label">Value functions</span>
                    <div class="chip-list">{mode_chips}</div>
                </div>
                <div class="setting-cell">
                    <span class="badge-label">Index</span>
                    <strong>{html.escape(st.session_state.interaction_index)}</strong>
                </div>
                <div class="setting-cell">
                    <span class="badge-label">Max order</span>
                    <strong>{effective_max_order()}</strong>
                </div>
                <div class="setting-cell">
                    <span class="badge-label">Budget</span>
                    <strong>{int(st.session_state.budget)}</strong>
                </div>
                <div class="setting-cell setting-cell-full">
                    <span class="badge-label">Model</span>
                    <strong>{model_value}</strong>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_result_summary(result: ExplanationRunResult | None) -> None:
    """Render a compact light result summary."""
    status = st.session_state.get("run_status", {"state": "idle", "message": "No run yet."})
    status_class = (
        "success" if status["state"] == "success" else "error" if status["state"] == "error" else ""
    )
    if result is None:
        st.markdown(
            f"""
            <div class="side-card">
                <h3 class="side-card-title">Result</h3>
                <div class="run-state {status_class}">{html.escape(status["message"])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    interaction = "First-order only"
    if effective_max_order() >= 2:
        interaction = f"{result.pair_label} ({result.pair_value:+.3f})"
    st.markdown(
        f"""
        <div class="side-card">
            <h3 class="side-card-title">Result</h3>
            <div class="run-state {status_class}">{html.escape(status["message"])}</div>
            <div class="metric-row"><span class="card-label">Value function</span><strong>{html.escape(result.value_name)}</strong></div>
            <div class="metric-row"><span class="card-label">Full-context support</span><strong>{result.full_score:.3f}</strong></div>
            <div class="metric-row"><span class="card-label">Top evidence chunk</span><strong>{html.escape(result.top_chunk)} ({result.top_score:+.3f})</strong></div>
            <div class="metric-row"><span class="card-label">Strongest interaction</span><strong>{html.escape(interaction)}</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def value_functions_overview_frame() -> pd.DataFrame:
    """Build the value-function documentation overview table."""
    return pd.DataFrame(
        [
            {
                "Value function": metadata["name"],
                "Type": metadata["category"],
                "Speed": metadata["speed"],
                "Requires model": "Yes" if metadata["requires_model"] else "No",
                "Best used for": metadata["use_case"],
                "Caveat": metadata["caveat"],
            }
            for metadata in VALUE_FUNCTION_DISPLAY.values()
        ]
    )


def metadata_badges(metadata: dict[str, object]) -> str:
    """Render compact badges for a value-function card."""
    requires_model = bool(metadata["requires_model"])
    recommended = bool(metadata["recommended_for_iteration"])
    model_class = "negative" if requires_model else "positive"
    recommended_class = "positive" if recommended else "negative"
    model_label = "Requires model" if requires_model else "Local only"
    recommended_label = "Quick iteration" if recommended else "Batch / audit"
    return (
        f'<span class="meta-badge">{html.escape(str(metadata["category"]))}</span>'
        f'<span class="meta-badge">{html.escape(str(metadata["speed"]))}</span>'
        f'<span class="meta-badge {model_class}">{model_label}</span>'
        f'<span class="meta-badge {recommended_class}">{recommended_label}</span>'
    )


def render_value_function_card(key: str, metadata: dict[str, object]) -> None:
    """Render one documented value function."""
    with st.container(key=f"vf_card_{key}"):
        st.markdown(
            f"""
            <h3 class="method-card-title">{html.escape(str(metadata["name"]))}</h3>
            <div class="method-badges">{metadata_badges(metadata)}</div>
            <p class="muted-text">{html.escape(str(metadata["description"]))}</p>
            <span class="formula-label">Scoring definition</span>
            """,
            unsafe_allow_html=True,
        )
        st.latex(str(metadata["latex"]))
        st.markdown(
            f"""
            <div class="method-detail">
                <strong>Use</strong>
                <span>{html.escape(str(metadata["use_case"]))}</span>
            </div>
            <div class="method-detail">
                <strong>Caveat</strong>
                <span>{html.escape(str(metadata["caveat"]))}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_value_functions_page() -> None:
    """Render documentation for all available value functions."""
    st.markdown('<div class="section-title">Value Functions</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="page-intro">These functions score how strongly a selected coalition of retrieved chunks supports the target answer.</p>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        value_functions_overview_frame(),
        width="stretch",
        hide_index=True,
        column_config={
            "Value function": st.column_config.TextColumn(width="medium"),
            "Type": st.column_config.TextColumn(width="medium"),
            "Speed": st.column_config.TextColumn(width="small"),
            "Requires model": st.column_config.TextColumn(width="small"),
            "Best used for": st.column_config.TextColumn(width="large"),
            "Caveat": st.column_config.TextColumn(width="large"),
        },
    )

    cards = list(VALUE_FUNCTION_DISPLAY.items())
    for row_start in range(0, len(cards), 2):
        cols = st.columns(2, gap="large")
        for col, (key, metadata) in zip(cols, cards[row_start : row_start + 2], strict=False):
            with col:
                render_value_function_card(key, metadata)


RAG_PIPELINE_STEPS = [
    {
        "name": "1. PDF Loading",
        "purpose": "Extract selectable text page by page while preserving page numbers for provenance.",
        "formula": "",
        "schema": "",
        "details": (
            "The uploaded PDF is parsed with pypdf/PyPDF2. Each page becomes a tuple containing "
            "the 1-indexed page number and normalized page text. OCR-scanned PDFs are rejected "
            "because they do not expose selectable text."
        ),
    },
    {
        "name": "2. Overlapping Chunking",
        "purpose": "Split pages into retrieval units while keeping local continuity.",
        "formula": "",
        "schema": "",
        "details": (
            "Chunks are word windows within a page. Overlap reduces boundary loss: evidence that "
            "falls near a chunk edge can still appear in the next candidate chunk."
        ),
    },
    {
        "name": "3. Chunk Metadata",
        "purpose": "Keep retrieval provenance and suppress low-value evidence types.",
        "formula": "",
        "schema": """{
  "title": "Page 26, chunk 44",
  "text": "...",
  "page_number": 26,
  "section_title": "Methods",
  "chunk_type": "body_text",
  "text_length": 180,
  "flags": [],
  "dense_score": 0.684,
  "keyword_score": 0.150,
  "rerank_score": 0.969
}""",
        "details": (
            "Each candidate is heuristically typed as body text, reference, caption, table, or "
            "heading-only. Reference sections, caption-only chunks, short chunks, and citation-heavy "
            "chunks are downweighted unless the query explicitly asks for them."
        ),
    },
    {
        "name": "4. Dense Embedding Retrieval",
        "purpose": "Recall semantically similar evidence even when wording differs.",
        "formula": r"\operatorname{dense}(q,c)=\cos(\mathbf{e}_q,\mathbf{e}_c)=\frac{\mathbf{e}_q^\top \mathbf{e}_c}{\lVert \mathbf{e}_q\rVert\lVert \mathbf{e}_c\rVert}",
        "schema": "",
        "details": (
            "The default embedding model is sentence-transformers/all-MiniLM-L6-v2. The query and "
            "chunks are embedded with mean-pooled Transformer hidden states, then compared with "
            "cosine similarity."
        ),
    },
    {
        "name": "5. Sparse Keyword Retrieval",
        "purpose": "Recover exact scientific terms that dense embeddings may blur.",
        "formula": r"\operatorname{tfidf}(t,d)=\operatorname{tf}(t,d)\log\frac{N}{1+\operatorname{df}(t)}",
        "schema": "",
        "details": (
            "A local TF-IDF retriever scores expanded query terms against all chunks. For broad "
            "questions, expansions add generic synthesis phrases such as overview, main reasons, "
            "major themes, objectives, motivations, and scientific value."
        ),
    },
    {
        "name": "6. Transparent Reranking",
        "purpose": "Select answer-useful context rather than merely semantically nearby text.",
        "formula": r"R(c)=\alpha \hat{s}_{dense}(c)+\beta \hat{s}_{kw}(c)+\gamma O(q,c)+\Delta_{meta}(c)",
        "schema": "",
        "details": (
            "Dense and keyword scores are normalized and combined with query-term overlap, metadata "
            "quality, and low-value chunk penalties. References and caption-only chunks receive "
            "penalties unless the query explicitly asks for them."
        ),
    },
    {
        "name": "7. Intent-Aware Context Selection",
        "purpose": "Prevent a narrow but high-scoring chunk from dominating broad answers.",
        "formula": r"S(c\mid C)=\lambda R(c)+(1-\lambda)N(c,C)+Q(c)-P_{narrow}(c)",
        "schema": "",
        "details": (
            "The selector first classifies the query as narrow factual or broad synthesis. Narrow "
            "queries use high relevance weight; broad queries use lower relevance weight and more "
            "novelty from new sections, query-term coverage, and low redundancy. Narrow chunks remain "
            "eligible, but they are discouraged from becoming the dominant first context chunk."
        ),
    },
    {
        "name": "8. Grounded Generation",
        "purpose": "Ask Gemma to answer only from the selected context.",
        "formula": "",
        "schema": "",
        "details": (
            "The prompt instructs the model to synthesize across all selected evidence, avoid "
            "anchoring on the first chunk, say when the context is insufficient, and avoid claims "
            "not grounded in the retrieved context."
        ),
    },
    {
        "name": "9. Shapley Attribution",
        "purpose": "Explain which retrieved chunks support the current generated answer/value function.",
        "formula": r"\phi_i=\sum_{S\subseteq N\setminus\{i\}}\frac{|S|!(n-|S|-1)!}{n!}\left[v(S\cup\{i\})-v(S)\right]",
        "schema": "",
        "details": (
            "After the answer exists, chunks become players in a cooperative game. Shapley values "
            "measure marginal contribution to the selected value function, not ideal factual "
            "relevance. A high score can reveal that the generated answer leaned heavily on a chunk."
        ),
    },
]


def rag_pipeline_overview_frame() -> pd.DataFrame:
    """Return compact RAG pipeline overview rows."""
    return pd.DataFrame(
        [
            {
                "Stage": step["name"],
                "Why it matters": step["purpose"],
                "Key output": [
                    "pages",
                    "candidate chunks",
                    "typed chunks",
                    "dense scores",
                    "keyword scores",
                    "rerank scores",
                    "balanced context",
                    "generated answer",
                    "chunk attributions",
                ][idx],
            }
            for idx, step in enumerate(RAG_PIPELINE_STEPS)
        ]
    )


def render_rag_pipeline_step(step: dict[str, str]) -> None:
    """Render one RAG pipeline technical detail card."""
    formula = step.get("formula", "")
    schema = step.get("schema", "")
    st.markdown(
        f"""
        <div class="method-card-title">{html.escape(step["name"])}</div>
        <div class="method-badges">
            <span class="meta-badge positive">RAG pipeline</span>
            <span class="meta-badge">implemented in this demo</span>
        </div>
        <p class="muted-text">{html.escape(step["purpose"])}</p>
        """,
        unsafe_allow_html=True,
    )
    if formula:
        st.markdown('<span class="formula-label">Formula</span>', unsafe_allow_html=True)
        st.latex(formula)
    if schema:
        st.markdown('<span class="formula-label">Metadata schema</span>', unsafe_allow_html=True)
        st.code(schema, language="json")
    st.markdown(
        f"""
        <div class="method-detail">
            <strong>Detail</strong>
            <span>{html.escape(step["details"])}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_rag_pipeline_page() -> None:
    """Render documentation for the local PDF RAG pipeline."""
    st.markdown('<div class="section-title">RAG Pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="page-intro">This page documents the real PDF RAG path used by the demo: from uploaded report to retrieved context, grounded generation, and Shapley-style attribution.</p>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        rag_pipeline_overview_frame(),
        width="stretch",
        hide_index=True,
        column_config={
            "Stage": st.column_config.TextColumn(width="medium"),
            "Why it matters": st.column_config.TextColumn(width="large"),
            "Key output": st.column_config.TextColumn(width="medium"),
        },
    )

    for row_start in range(0, len(RAG_PIPELINE_STEPS), 2):
        cols = st.columns(2, gap="large")
        for offset, (col, step) in enumerate(
            zip(cols, RAG_PIPELINE_STEPS[row_start : row_start + 2], strict=False)
        ):
            with col, st.container(key=f"rag_step_{row_start}_{offset}", border=False):
                render_rag_pipeline_step(step)


def _role_css_class(role: str) -> str:
    """Map a role label string to a CSS modifier class."""
    low = role.lower()
    if "redundant" in low or "duplicate" in low:
        return "redundant"
    if "distractor" in low or "contrast" in low or "wrong" in low:
        return "distractor"
    if "background" in low or "tangential" in low:
        return "background"
    return "evidence"


def render_chunk_card(
    idx: int,
    chunk: RetrievedChunk,
    score: float | None,
    max_abs: float,
    role: str | None = None,
) -> None:
    """Render one retrieved chunk as a compact evidence card."""
    style = attribution_style(score, max_abs)
    score_label = "not scored" if score is None else f"φᵢ {score:+.3f}"
    role_html = ""
    if role:
        css = _role_css_class(role)
        role_html = f'<span class="role-tag {css}">{html.escape(role)}</span>'
    meta_parts = []
    if chunk.page_number:
        meta_parts.append(f"page {chunk.page_number}")
    if chunk.chunk_type:
        meta_parts.append(chunk.chunk_type)
    if chunk.rerank_score is not None:
        meta_parts.append(f"rerank {chunk.rerank_score:.3f}")
    if chunk.flags:
        meta_parts.append("flags: " + ", ".join(chunk.flags[:4]))
    meta_html = ""
    if meta_parts:
        meta_html = f'<div class="chunk-meta">{html.escape(" · ".join(meta_parts))}</div>'
    st.markdown(
        f"""
        <div class="evidence-card chunk-card" style="--chunk-border:{style["border"]}; --chunk-bg:{style["bg"]};
             --badge-bg:{style["badge_bg"]}; --badge-border:{style["badge_border"]}; --badge-fg:{style["badge_fg"]};">
            <header>
                <div class="chunk-title-row">
                    <span class="chunk-number">Chunk {idx + 1}</span>
                    <h4>{html.escape(chunk.title)}</h4>
                    {role_html}
                </div>
                <span class="score-badge">{html.escape(score_label)}</span>
            </header>
            {meta_html}
            <p>{html.escape(chunk.text)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_context_panel(
    trace: dict[str, object],
    chunks: list[RetrievedChunk],
    result: ExplanationRunResult | None,
) -> None:
    """Render retrieved chunks with attribution scores."""
    st.markdown('<div class="section-title">Retrieved chunks</div>', unsafe_allow_html=True)
    st.caption(
        "Shapley contribution explains the current generated answer/value function, "
        "not ideal factual relevance. A high φᵢ can mean the answer leaned too heavily "
        "on that chunk."
    )
    scores = attribution_lookup(result.first_order_frame) if result is not None else {}
    max_abs = max([abs(score) for score in scores.values()] + [0.01])
    role_lookup = {
        str(c["title"]): str(c["evidence_role"])
        for c in trace.get("chunks", [])
        if "evidence_role" in c
    }
    for idx, chunk in enumerate(chunks):
        render_chunk_card(idx, chunk, scores.get(idx), max_abs, role=role_lookup.get(chunk.title))


def render_retrieval_debug(trace: dict[str, object]) -> None:
    """Render transparent retrieval/reranking diagnostics for PDF runs."""
    debug = trace.get("retrieval_debug")
    if not isinstance(debug, dict):
        return
    with st.expander("Retrieval debug", expanded=False):
        st.caption("Query intent")
        st.code(str(debug.get("query_intent", "")))
        st.caption("Original query")
        st.code(str(debug.get("original_query", "")))
        expanded = debug.get("expanded_queries", [])
        if expanded:
            st.caption("Expanded queries")
            st.write(expanded)
        coverage = debug.get("coverage_summary", {})
        if coverage:
            st.caption("Coverage summary")
            st.json(coverage)
        for title, key in [
            ("Raw dense retrieval", "raw_dense_results"),
            ("Raw keyword retrieval", "raw_keyword_results"),
            ("Raw rerank order", "raw_rerank_order"),
            ("Final prompt order", "selected_context"),
        ]:
            rows = debug.get(key, [])
            if rows:
                st.caption(title)
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


def render_attribution_chart(
    result: ExplanationRunResult | None,
    chunks: list[RetrievedChunk],
) -> None:
    """Render attribution bars with LOO marginal tick marks overlaid."""
    if result is None:
        st.markdown(
            '<div class="empty-state">Run the explanation to view attribution charts and audit details.</div>',
            unsafe_allow_html=True,
        )
        return
    st.markdown('<div class="section-title">Attribution chart</div>', unsafe_allow_html=True)
    frame = prepare_attribution_frame(result, chunks)
    if frame.empty:
        st.markdown(
            '<div class="empty-state">No first-order attribution values were returned.</div>',
            unsafe_allow_html=True,
        )
        return
    chart_frame = frame.sort_values("attribution").copy()
    chart_frame["zero"] = 0.0
    row_order = list(chart_frame["title"])

    has_loo = "loo" in chart_frame.columns and chart_frame["loo"].notna().any()
    loo_vals = chart_frame["loo"].dropna().tolist() if has_loo else []
    all_vals = list(chart_frame["attribution"]) + loo_vals
    max_abs = max((abs(v) for v in all_vals), default=0.01)
    x_domain = [-1.08 * max_abs, 1.08 * max_abs]
    height = min(360, max(260, 58 * len(chart_frame)))

    if has_loo:
        # Grouped bars: melt to long format, use yOffset for side-by-side layout
        shap_rows = pd.DataFrame(
            {
                "title": chart_frame["title"],
                "value": chart_frame["attribution"],
                "zero": 0.0,
                "metric": "Shapley φᵢ",
                "chunk_id": chart_frame["chunk_id"],
                "value_function": chart_frame["value_function"],
                "preview": chart_frame["preview"],
            }
        )
        shap_rows["color_key"] = shap_rows["value"].apply(
            lambda v: "shap_pos" if v >= 0 else "shap_neg"
        )
        loo_rows = pd.DataFrame(
            {
                "title": chart_frame["title"],
                "value": chart_frame["loo"].fillna(0.0),
                "zero": 0.0,
                "metric": "LOO marginal",
                "chunk_id": chart_frame["chunk_id"],
                "value_function": chart_frame["value_function"],
                "preview": chart_frame["preview"],
            }
        )
        loo_rows["color_key"] = loo_rows["value"].apply(
            lambda v: "loo_pos" if v >= 0 else "loo_neg"
        )
        long_frame = pd.concat([shap_rows, loo_rows], ignore_index=True)

        color_scale = alt.Scale(
            domain=["shap_pos", "shap_neg", "loo_pos", "loo_neg"],
            range=["#26735f", "#b5473f", "#9ecec0", "#e8b4b0"],
        )
        chart = (
            alt.Chart(long_frame)
            .mark_bar(cornerRadius=2, opacity=0.88)
            .encode(
                x=alt.X("value:Q", title=None, scale=alt.Scale(domain=x_domain)),
                x2="zero:Q",
                y=alt.Y("title:N", title=None, sort=row_order),
                yOffset=alt.YOffset("metric:N", sort=["Shapley φᵢ", "LOO marginal"]),
                color=alt.Color("color_key:N", scale=color_scale, legend=None),
                tooltip=[
                    alt.Tooltip("chunk_id:O", title="Chunk"),
                    alt.Tooltip("title:N", title="Title"),
                    alt.Tooltip("metric:N", title="Metric"),
                    alt.Tooltip("value:Q", title="Score", format="+.3f"),
                    alt.Tooltip("preview:N", title="Preview"),
                ],
            )
            .properties(height=height)
            .interactive()
        )
        legend_html = (
            '<p style="color:var(--text-muted);font-size:0.78rem;margin:0.35rem 0 0 0;">'
            '<span style="display:inline-block;width:11px;height:11px;'
            'background:#26735f;border-radius:2px;vertical-align:middle;margin-right:4px;"></span>'
            "Shapley φᵢ &nbsp;·&nbsp;"
            '<span style="display:inline-block;width:11px;height:11px;'
            'background:#9ecec0;border-radius:2px;vertical-align:middle;margin:0 4px 0 8px;"></span>'
            "LOO marginal V(all) - V(all without i). "
            "When Shapley bar extends past LOO, credit is shared across subsets that LOO misses."
            "</p>"
        )
    else:
        chart = (
            alt.Chart(chart_frame)
            .mark_bar(cornerRadius=3, size=18, opacity=0.92)
            .encode(
                x=alt.X("attribution:Q", title=None, scale=alt.Scale(domain=x_domain)),
                x2="zero:Q",
                y=alt.Y("title:N", title=None, sort=row_order),
                color=alt.condition(
                    alt.datum.attribution >= 0,
                    alt.value("#26735f"),
                    alt.value("#b5473f"),
                ),
                tooltip=[
                    alt.Tooltip("chunk_id:O", title="Chunk"),
                    alt.Tooltip("title:N", title="Title"),
                    alt.Tooltip("value_function:N", title="Value function"),
                    alt.Tooltip("attribution:Q", title="Shapley φᵢ", format="+.3f"),
                    alt.Tooltip("preview:N", title="Preview"),
                ],
            )
            .properties(height=height)
            .interactive()
        )
        legend_html = ""

    with st.container(border=True):
        st.altair_chart(chart, width="stretch")
        if legend_html:
            st.markdown(legend_html, unsafe_allow_html=True)


def render_interaction_map(
    result: ExplanationRunResult | None, chunks: list[RetrievedChunk]
) -> None:
    """Render a compact pairwise interaction heatmap."""
    if result is None:
        st.info("Run an explanation to inspect pairwise interactions.")
        return
    if effective_max_order() < 2:
        st.info("SV is first-order only. Select k-SII, STII, or FSII for pairwise effects.")
        return
    labels = [f"{idx + 1}. {chunk.title}" for idx, chunk in enumerate(chunks)]
    rows = []
    for i, x_label in enumerate(labels):
        for j, y_label in enumerate(labels):
            rows.append(
                {
                    "x": x_label,
                    "y": y_label,
                    "pair": f"{x_label} + {y_label}",
                    "interaction": float(result.pairwise_matrix[i, j]),
                }
            )
    frame = pd.DataFrame(rows)
    max_abs = max(float(np.max(np.abs(result.pairwise_matrix))), 0.01)
    chart = (
        alt.Chart(frame)
        .mark_rect()
        .encode(
            x=alt.X("x:N", title=None, sort=labels),
            y=alt.Y("y:N", title=None, sort=labels),
            color=alt.Color(
                "interaction:Q",
                title="Interaction",
                scale=alt.Scale(
                    domain=[-max_abs, 0, max_abs], range=["#b5473f", "#f8fafc", "#26735f"]
                ),
            ),
            tooltip=[
                alt.Tooltip("pair:N", title="Pair"),
                alt.Tooltip("interaction:Q", title="Second-order interaction", format="+.3f"),
            ],
        )
        .properties(height=min(420, max(320, 76 * len(labels))))
        .interactive()
    )
    with st.container(border=True):
        st.altair_chart(chart, width="stretch")


def render_value_function_comparison(
    results: list[ExplanationRunResult],
    chunks: list[RetrievedChunk],
) -> None:
    """Render central value-function comparison section."""
    if not results:
        st.info("Run one or more value functions to compare support scores and evidence rankings.")
        return
    frame = comparison_frame(results)
    display_frame = frame.rename(
        columns={
            "value_function": "Value function",
            "support_score": "Full-context score",
            "top_evidence": "Top evidence",
            "direct_evidence_rank": "Direct evidence rank",
            "direct_above_distractors": "Direct evidence above distractors",
            "description": "What it measures",
        }
    )[
        [
            "Value function",
            "Full-context score",
            "Top evidence",
            "Direct evidence rank",
            "Direct evidence above distractors",
            "What it measures",
        ]
    ]
    st.dataframe(
        display_frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Value function": st.column_config.TextColumn(width="medium"),
            "Full-context score": st.column_config.NumberColumn(format="%.3f", width="small"),
            "Top evidence": st.column_config.TextColumn(width="large"),
            "Direct evidence rank": st.column_config.TextColumn(width="small"),
            "Direct evidence above distractors": st.column_config.CheckboxColumn(width="small"),
            "What it measures": st.column_config.TextColumn(width="large"),
        },
    )
    combined = pd.concat(
        [prepare_attribution_frame(result, chunks) for result in results],
        ignore_index=True,
    )
    if combined.empty:
        return
    combined["zero"] = 0.0
    chunk_order = (
        combined.assign(abs_attr=combined["attribution"].abs())
        .groupby("title", as_index=False)["abs_attr"]
        .max()
        .sort_values("abs_attr", ascending=True)["title"]
        .tolist()
    )
    max_abs = max(float(combined["attribution"].abs().max()), 0.01)
    chart = (
        alt.Chart(combined)
        .mark_bar(cornerRadius=2, size=14, opacity=0.9)
        .encode(
            x=alt.X(
                "attribution:Q",
                title="Attribution",
                scale=alt.Scale(domain=[-1.08 * max_abs, 1.08 * max_abs]),
            ),
            x2="zero:Q",
            y=alt.Y("title:N", title=None, sort=chunk_order),
            yOffset=alt.YOffset("value_function:N"),
            color=alt.Color("value_function:N", title="Value function"),
            tooltip=[
                alt.Tooltip("chunk_id:O", title="Chunk"),
                alt.Tooltip("title:N", title="Title"),
                alt.Tooltip("value_function:N", title="Value function"),
                alt.Tooltip("attribution:Q", title="Attribution", format="+.3f"),
                alt.Tooltip("preview:N", title="Preview"),
            ],
        )
        .properties(height=min(380, max(280, 70 * len(chunk_order))))
        .interactive()
    )
    with st.container(border=True):
        st.altair_chart(chart, width="stretch")


def render_layer_header(
    *,
    number: int,
    layer: str,
    title: str,
    subtitle: str,
    items: list[str],
) -> None:
    """Render a compact layer header directly above its analysis views."""
    pills = "".join(
        f'<span class="analysis-layer-pill">{html.escape(item)}</span>' for item in items
    )
    st.markdown(
        f"""
        <div class="analysis-layer">
            <div class="analysis-layer-header">
                <div>
                    <span class="analysis-layer-index">Layer {number} · {html.escape(layer)}</span>
                    <strong class="analysis-layer-title">{html.escape(title)}</strong>
                    <p class="analysis-layer-subtitle">{html.escape(subtitle)}</p>
                </div>
                <div class="analysis-layer-items">{pills}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def attribution_order_rows(result: ExplanationRunResult) -> list[dict[str, object]]:
    """Return chunk ranking rows in the actual audit order."""
    rows = []
    if result.first_order_frame.empty:
        return rows
    for rank, row in enumerate(result.first_order_frame.itertuples(index=False), start=1):
        rows.append(
            {
                "rank": rank,
                "player_id": int(row.player),
                "chunk": str(row.chunk),
                "attribution": float(row.attribution),
            }
        )
    return rows


def audit_order_frame(result: ExplanationRunResult) -> pd.DataFrame:
    """Build the audit order table shown to users."""
    return pd.DataFrame(
        [
            {
                "Rank": row["rank"],
                "Chunk": row["chunk"],
                "Attribution": row["attribution"],
                "Deletion action": f"removed at step {row['rank']}",
                "Insertion action": f"added at step {row['rank']}",
            }
            for row in attribution_order_rows(result)
        ]
    )


def audit_trace_frames(
    result: ExplanationRunResult,
    chunks: list[RetrievedChunk],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build self-describing deletion and insertion traces for plotting."""
    all_titles = [chunk.title for chunk in chunks]
    ordered = attribution_order_rows(result)
    ordered_titles = [str(row["chunk"]) for row in ordered]

    deletion_rows = []
    previous_score: float | None = None
    for row in result.deletion_frame.itertuples(index=False):
        step = int(row.step)
        removed_titles = ordered_titles[:step]
        remaining_titles = [title for title in all_titles if title not in removed_titles]
        changed = "" if step == 0 else str(row.removed_chunk)
        score = float(row.score)
        step_drop = 0.0 if previous_score is None else previous_score - score
        previous_score = score
        deletion_rows.append(
            {
                "step": step,
                "axis_label": "0 removed" if step == 0 else f"{step} removed",
                "audit": "Deletion",
                "action": "Initial: all chunks visible"
                if step == 0
                else f"Removed chunk: {changed}",
                "changed_chunk": changed or "none",
                "selected_chunks": ", ".join(remaining_titles) if remaining_titles else "none",
                "remaining_chunks": ", ".join(remaining_titles) if remaining_titles else "none",
                "support_score": score,
                "step_delta": step_drop,
                "step_delta_label": "" if step == 0 else f"drop {step_drop:+.3f}",
            }
        )

    insertion_rows = []
    previous_score = None
    for row in result.insertion_frame.itertuples(index=False):
        step = int(row.step)
        selected_titles = ordered_titles[:step]
        remaining_titles = [title for title in all_titles if title not in selected_titles]
        changed = "" if step == 0 else str(row.added_chunk)
        score = float(row.score)
        step_gain = 0.0 if previous_score is None else score - previous_score
        previous_score = score
        insertion_rows.append(
            {
                "step": step,
                "axis_label": "No chunks"
                if step == 0
                else f"Add {step}: {short_text(changed, 30)}",
                "audit": "Insertion",
                "action": "Initial: no chunks visible" if step == 0 else f"Added chunk: {changed}",
                "changed_chunk": changed or "none",
                "selected_chunks": ", ".join(selected_titles) if selected_titles else "none",
                "remaining_chunks": ", ".join(remaining_titles) if remaining_titles else "none",
                "support_score": score,
                "step_delta": step_gain,
                "step_delta_label": "" if step == 0 else f"gain {step_gain:+.3f}",
            }
        )

    return pd.DataFrame(deletion_rows), pd.DataFrame(insertion_rows)


def render_audit_metric_cards(result: ExplanationRunResult) -> None:
    """Render faithfulness metrics with definitions."""
    st.markdown(
        f'<p style="color:var(--text-muted);font-size:0.84rem;margin:0 0 0.55rem 0;">'
        f"Reference &nbsp;·&nbsp; full-context score&nbsp;=&nbsp;"
        f'<strong style="color:var(--text);">{result.full_score:.3f}</strong>'
        f"&nbsp;&nbsp;empty-context score&nbsp;=&nbsp;"
        f'<strong style="color:var(--text);">{result.empty_score:.3f}</strong>. '
        f"All metrics below are relative to these bounds.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="audit-metrics">
            <div class="audit-metric-card">
                <span>Deletion drop</span>
                <strong>{result.faithfulness.deletion_drop:.3f}</strong>
                <p>Full-context score minus score after removing top evidence. Larger drop means top evidence was important.</p>
            </div>
            <div class="audit-metric-card">
                <span>Top-only score</span>
                <strong>{result.faithfulness.score_with_top_only:.3f}</strong>
                <p>Score using only the top-ranked evidence chunk.</p>
            </div>
            <div class="audit-metric-card">
                <span>Sufficiency gap</span>
                <strong>{result.faithfulness.sufficiency_gap:.3f}</strong>
                <p>Full-context score minus top-only score. Smaller gap means top evidence is sufficient.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def audit_line_chart(
    frame: pd.DataFrame,
    *,
    title: str,
    color: str,
    delta_title: str,
    random_baseline: np.ndarray | None = None,
    retrieval_baseline: np.ndarray | None = None,
) -> alt.Chart:
    """Return one audit curve with optional random/retrieval baselines."""
    label_order = frame["axis_label"].tolist()
    base = alt.Chart(frame).encode(
        x=alt.X(
            "axis_label:N",
            title=None,
            sort=label_order,
            axis=alt.Axis(labelAngle=-25, labelLimit=120),
        ),
        y=alt.Y("support_score:Q", title="Support score"),
        tooltip=[
            alt.Tooltip("audit:N", title="Audit type"),
            alt.Tooltip("step:O", title="Step"),
            alt.Tooltip("action:N", title="Action"),
            alt.Tooltip("changed_chunk:N", title="Chunk changed"),
            alt.Tooltip("selected_chunks:N", title="Selected chunks"),
            alt.Tooltip("remaining_chunks:N", title="Remaining chunks"),
            alt.Tooltip("support_score:Q", title="Support score", format=".3f"),
            alt.Tooltip("step_delta:Q", title=delta_title, format="+.3f"),
        ],
    )
    line = base.mark_line(point=True, color=color)
    labels = base.mark_text(
        align="center", baseline="bottom", dy=-10, fontSize=11, color=color
    ).encode(text=alt.Text("step_delta_label:N"))
    chart = line + labels

    if random_baseline is not None and len(random_baseline) == len(frame):
        baseline_frame = pd.DataFrame(
            {
                "axis_label": label_order,
                "support_score": random_baseline.tolist(),
            }
        )
        baseline_line = (
            alt.Chart(baseline_frame)
            .mark_line(point=False, color="#94a3b8", opacity=0.7, strokeDash=[5, 4])
            .encode(
                x=alt.X("axis_label:N", sort=label_order),
                y=alt.Y("support_score:Q"),
                tooltip=[alt.Tooltip("support_score:Q", title="Random baseline", format=".3f")],
            )
        )
        chart = chart + baseline_line

    if retrieval_baseline is not None and len(retrieval_baseline) == len(frame):
        retrieval_frame = pd.DataFrame(
            {
                "axis_label": label_order,
                "support_score": retrieval_baseline.tolist(),
            }
        )
        retrieval_line = (
            alt.Chart(retrieval_frame)
            .mark_line(point=True, color="#4169c8", opacity=0.8, strokeDash=[2, 3])
            .encode(
                x=alt.X("axis_label:N", sort=label_order),
                y=alt.Y("support_score:Q"),
                tooltip=[
                    alt.Tooltip(
                        "support_score:Q",
                        title="Retrieval-score deletion",
                        format=".3f",
                    )
                ],
            )
        )
        chart = chart + retrieval_line

    return chart.properties(title=title, height=315).interactive()


def curve_auc(values: np.ndarray | None) -> float | None:
    """Return normalized area under a score curve."""
    if values is None or len(values) < 2:
        return None
    return float(np.trapezoid(values, dx=1.0) / (len(values) - 1))


def render_deletion_comparison_summary(result: ExplanationRunResult) -> None:
    """Summarize Shapley, retrieval-score, and random deletion curves."""
    shapley_values = result.deletion_frame["score"].to_numpy(dtype=float)
    shapley_auc = curve_auc(shapley_values)
    retrieval_auc = curve_auc(result.retrieval_deletion_baseline)
    random_auc = curve_auc(result.random_deletion_baseline)
    rows = [
        {
            "Deletion order": "Shapley attribution",
            "AUC": shapley_auc,
            "Interpretation": "Lower is better: answer support drops quickly.",
        },
        {
            "Deletion order": "Retrieval score",
            "AUC": retrieval_auc,
            "Interpretation": "Removes chunks the retriever thought were most relevant.",
        },
        {
            "Deletion order": "Random average",
            "AUC": random_auc,
            "Interpretation": "Average over random removal orders.",
        },
    ]
    frame = pd.DataFrame(rows).dropna(subset=["AUC"])
    if frame.empty:
        return
    st.dataframe(
        frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Deletion order": st.column_config.TextColumn(width="medium"),
            "AUC": st.column_config.NumberColumn(format="%.3f", width="small"),
            "Interpretation": st.column_config.TextColumn(width="large"),
        },
    )


def render_audit_trace_tables(deletion_trace: pd.DataFrame, insertion_trace: pd.DataFrame) -> None:
    """Render readable deletion and insertion trace tables."""
    deletion_display = deletion_trace.rename(
        columns={
            "step": "Step",
            "action": "Action",
            "changed_chunk": "Removed chunk",
            "support_score": "Support score",
            "step_delta": "Step drop",
            "remaining_chunks": "Remaining chunks",
        }
    )[["Step", "Action", "Removed chunk", "Support score", "Step drop", "Remaining chunks"]]
    insertion_display = insertion_trace.rename(
        columns={
            "step": "Step",
            "action": "Action",
            "changed_chunk": "Added chunk",
            "support_score": "Support score",
            "step_delta": "Step gain",
            "selected_chunks": "Selected chunks",
        }
    )[["Step", "Action", "Added chunk", "Support score", "Step gain", "Selected chunks"]]

    deletion_col, insertion_col = st.columns(2, gap="large")
    with deletion_col:
        st.dataframe(
            deletion_display,
            width="stretch",
            hide_index=True,
            column_config={
                "Step": st.column_config.NumberColumn(width="small"),
                "Action": st.column_config.TextColumn(width="medium"),
                "Removed chunk": st.column_config.TextColumn(width="medium"),
                "Support score": st.column_config.NumberColumn(format="%.3f", width="small"),
                "Step drop": st.column_config.NumberColumn(format="%+.3f", width="small"),
                "Remaining chunks": st.column_config.TextColumn(width="large"),
            },
        )
    with insertion_col:
        st.dataframe(
            insertion_display,
            width="stretch",
            hide_index=True,
            column_config={
                "Step": st.column_config.NumberColumn(width="small"),
                "Action": st.column_config.TextColumn(width="medium"),
                "Added chunk": st.column_config.TextColumn(width="medium"),
                "Support score": st.column_config.NumberColumn(format="%.3f", width="small"),
                "Step gain": st.column_config.NumberColumn(format="%+.3f", width="small"),
                "Selected chunks": st.column_config.TextColumn(width="large"),
            },
        )


def sentence_attribution_display_frame(result: ExplanationRunResult) -> pd.DataFrame:
    """Build a checked sentence-attribution display frame."""
    if result.sentence_first_order_frame.empty or result.sentence_meta.empty:
        return pd.DataFrame()
    frame = result.sentence_first_order_frame.merge(
        result.sentence_meta,
        left_on="chunk",
        right_on="sentence",
        how="left",
    )
    frame["attribution"] = pd.to_numeric(frame["attribution"], errors="coerce")
    frame["sentence_id"] = frame["sentence"].fillna(frame["chunk"]).astype(str)
    frame["source_chunk"] = frame["source_chunk"].fillna("Unknown source").astype(str)
    frame["sentence_text"] = frame["text"].fillna("").astype(str)
    frame["sentence_preview"] = frame["sentence_text"].map(lambda text: short_text(text, 96))
    frame["zero"] = 0.0
    return frame[
        [
            "sentence_id",
            "source_chunk",
            "sentence_text",
            "sentence_preview",
            "attribution",
            "zero",
        ]
    ]


def render_sentence_attribution_table(frame: pd.DataFrame) -> None:
    """Render a compact numeric sentence attribution table."""
    display_frame = frame[
        ["sentence_id", "source_chunk", "attribution", "sentence_preview"]
    ].rename(
        columns={
            "sentence_id": "Sentence id",
            "source_chunk": "Source chunk",
            "attribution": "Attribution",
            "sentence_preview": "Sentence preview",
        }
    )
    st.dataframe(
        display_frame,
        width="stretch",
        hide_index=True,
        column_config={
            "Sentence id": st.column_config.TextColumn(width="small"),
            "Source chunk": st.column_config.TextColumn(width="medium"),
            "Attribution": st.column_config.NumberColumn(format="%+.3f", width="small"),
            "Sentence preview": st.column_config.TextColumn(width="large"),
        },
    )


def render_sentence_attribution_chart(frame: pd.DataFrame) -> None:
    """Render sentence attribution as visible horizontal bars."""
    values = frame["attribution"].dropna()
    if values.empty:
        st.markdown(
            '<div class="empty-state">Sentence attribution was not computed for this run.</div>',
            unsafe_allow_html=True,
        )
        return
    if bool((values == 0).all()):
        st.markdown(
            '<div class="empty-state">All sentence attributions are zero under the selected value function.</div>',
            unsafe_allow_html=True,
        )
        render_sentence_attribution_table(frame)
        return

    chart_frame = frame.dropna(subset=["attribution"]).sort_values("attribution").copy()
    max_abs = max(float(chart_frame["attribution"].abs().max()), 1e-9)
    height = min(460, max(280, 34 * len(chart_frame)))
    chart = (
        alt.Chart(chart_frame)
        .mark_bar(cornerRadius=3, size=17, opacity=0.92)
        .encode(
            x=alt.X(
                "attribution:Q",
                title="Sentence attribution",
                scale=alt.Scale(domain=[-1.12 * max_abs, 1.12 * max_abs]),
            ),
            x2="zero:Q",
            y=alt.Y(
                "sentence_id:N",
                title=None,
                sort=list(chart_frame["sentence_id"]),
            ),
            color=alt.condition(
                alt.datum.attribution >= 0,
                alt.value("#26735f"),
                alt.value("#b5473f"),
            ),
            tooltip=[
                alt.Tooltip("sentence_id:N", title="Sentence id"),
                alt.Tooltip("source_chunk:N", title="Source chunk"),
                alt.Tooltip("sentence_text:N", title="Sentence text"),
                alt.Tooltip("attribution:Q", title="Attribution", format="+.4f"),
            ],
        )
        .properties(height=height)
        .interactive()
    )
    with st.container(border=True):
        st.altair_chart(chart, width="stretch")
    render_sentence_attribution_table(frame.sort_values("sentence_id"))


def render_negative_score_callout(result: ExplanationRunResult | None) -> None:
    """Show a contextual explanation when any first-order Shapley value is negative."""
    if result is None or result.first_order_frame.empty:
        return
    if not (result.first_order_frame["attribution"] < -0.001).any():
        return
    st.markdown(
        """
        <div class="negative-callout">
            <strong>Why negative?</strong>
            A negative score means this chunk's average marginal contribution was negative
            across the context subsets sampled — not because it is factually wrong, but
            because the value function penalises it in most partial contexts under the
            current scoring mode. It may still be essential in specific combinations.
            See <strong>Step 2</strong> for pairwise interaction effects.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_interaction_card(result: ExplanationRunResult | None) -> None:
    """Show an auto-generated natural language summary of the strongest pairwise interaction."""
    if result is None or effective_max_order() < 2:
        return
    if abs(result.pair_value) < 0.005:
        return
    v = result.pair_value
    if v > 0.05:
        label, label_color, explanation = (
            "Synergy",
            "var(--positive)",
            "Together these chunks support the answer more than either does alone — "
            "likely because they cover complementary answer parts.",
        )
    elif v < -0.05:
        label, label_color, explanation = (
            "Redundancy",
            "var(--negative)",
            "Each chunk covers similar ground. Having both adds little over either alone — "
            "a signal that retrieval fetched near-duplicate evidence.",
        )
    else:
        label, label_color, explanation = (
            "Near-independent",
            "var(--text-muted)",
            "Their joint contribution is approximately the sum of their individual effects.",
        )
    st.markdown(
        f"""
        <div class="interaction-card">
            <p class="interaction-card-pair">Strongest interaction: {html.escape(result.pair_label)}</p>
            <div class="interaction-card-meta">
                <span style="font-size:0.84rem;font-weight:760;color:{label_color};">{label}</span>
                <span style="font-size:0.84rem;color:var(--text-muted);font-variant-numeric:tabular-nums;">
                    φᵢ,ⱼ = {v:+.3f}
                </span>
            </div>
            <p class="interaction-card-text">{html.escape(explanation)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_story_step(step: str, title: str, subtitle: str) -> None:
    """Render a numbered narrative section heading."""
    st.markdown(
        f"""
        <div class="story-step">
            <span class="story-step-num">{html.escape(step)}</span>
            <span class="story-step-title">{html.escape(title)}</span>
        </div>
        <p class="story-step-sub">{html.escape(subtitle)}</p>
        """,
        unsafe_allow_html=True,
    )


def render_advanced_audit_tabs(
    result: ExplanationRunResult | None,
    results: list[ExplanationRunResult],
    chunks: list[RetrievedChunk],
) -> None:
    """Render deep-dive auxiliary tabs (coalition scores, comparison, sentence detail)."""
    if not results:
        return

    st.markdown('<div class="advanced-heading">Deep dive</div>', unsafe_allow_html=True)

    scores_tab, compare_tab, sentence_tab = st.tabs(
        ["Coalition Scores", "Cross-function Comparison", "Sentence Detail"]
    )

    with scores_tab:
        st.markdown(
            '<p class="analysis-tab-note">V(C) for all coalitions of size 0-3 - a transparency sample of the raw game values feeding the Shapley computation. Larger coalitions are omitted to keep the table readable.</p>',
            unsafe_allow_html=True,
        )
        if result is None:
            st.info("Run an explanation to inspect coalition scores.")
        else:
            display_frame = result.audit_frame.rename(
                columns={
                    "coalition": "Coalition",
                    "selected_chunks": "Selected chunks",
                    "normalized_score": "Value function score V(C)",
                }
            )
            st.dataframe(
                display_frame,
                width="stretch",
                hide_index=True,
                column_config={
                    "Coalition": st.column_config.TextColumn(width="medium"),
                    "Selected chunks": st.column_config.NumberColumn(width="small"),
                    "Value function score V(C)": st.column_config.NumberColumn(
                        format="%.3f",
                        width="medium",
                    ),
                },
            )

    with compare_tab:
        st.markdown(
            '<p class="analysis-tab-note">Compare attribution rankings across all value functions run in this session.</p>',
            unsafe_allow_html=True,
        )
        render_value_function_comparison(results, chunks)

    with sentence_tab:
        st.markdown(
            '<p class="analysis-tab-note">Sentence-level attribution: re-runs Shapley with individual sentences as players inside the top chunks.</p>',
            unsafe_allow_html=True,
        )
        if result is None:
            st.info("Run an explanation to inspect sentence-level attribution.")
        else:
            sentence_frame = sentence_attribution_display_frame(result)
            if sentence_frame.empty or sentence_frame["attribution"].isna().all():
                st.markdown(
                    '<div class="empty-state">'
                    "Sentence attribution was not computed for this run. "
                    "Enable <strong>Run sentence drilldown</strong> in Explanation Settings "
                    "and re-run to see per-sentence Shapley values."
                    "</div>",
                    unsafe_allow_html=True,
                )
            else:
                render_sentence_attribution_chart(sentence_frame)

    _ = None


def _render_insertion_step_story(
    insertion_trace: pd.DataFrame,
    trace: dict | None,
) -> None:
    """Render a per-step natural language story of the insertion curve."""
    role_lookup: dict[str, str] = {}
    if trace:
        for c in trace.get("chunks", []):
            if "evidence_role" in c:
                role_lookup[str(c["title"])] = str(c["evidence_role"])

    gains = {
        int(row.step): float(row.step_delta)
        for row in insertion_trace.itertuples(index=False)
        if int(row.step) > 0
    }
    top_step = max(gains, key=lambda s: gains[s]) if gains else -1

    parts: list[str] = []
    for row in insertion_trace.itertuples(index=False):
        step = int(row.step)
        score = float(row.support_score)
        delta = float(row.step_delta)
        chunk_name = str(row.changed_chunk) if step > 0 else ""
        role = role_lookup.get(chunk_name)

        role_html = ""
        if role:
            css = _role_css_class(role)
            role_html = f' <span class="role-tag {css}">{html.escape(role)}</span>'

        highlight = " highlight" if step == top_step else ""

        if step == 0:
            parts.append(
                f'<div class="ins-step{highlight}">'
                f'<span class="ins-step-num">—</span>'
                f'<span class="ins-step-text">Empty context</span>'
                f'<span class="ins-step-score">{score:.3f}</span>'
                f'<span class="ins-step-delta neu"></span>'
                f"</div>"
            )
        else:
            dcls = "pos" if delta > 0.001 else ("neg" if delta < -0.001 else "neu")
            parts.append(
                f'<div class="ins-step{highlight}">'
                f'<span class="ins-step-num">+{step}</span>'
                f'<span class="ins-step-text">{html.escape(chunk_name)}{role_html}</span>'
                f'<span class="ins-step-score">{score:.3f}</span>'
                f'<span class="ins-step-delta {dcls}">{delta:+.3f}</span>'
                f"</div>"
            )

    st.markdown(
        '<div class="ins-story">' + "".join(parts) + "</div>",
        unsafe_allow_html=True,
    )


def _render_validation_inline(
    result: ExplanationRunResult | None,
    chunks: list[RetrievedChunk],
    trace: dict | None = None,
) -> None:
    """Render validation: insertion first (with step story), deletion in expander."""
    if result is None:
        st.markdown(
            '<div class="empty-state">Run an explanation to see the validation curves.</div>',
            unsafe_allow_html=True,
        )
        return
    render_audit_metric_cards(result)
    deletion_trace, insertion_trace = audit_trace_frames(result, chunks)

    # Insertion (primary view)
    st.altair_chart(
        audit_line_chart(
            insertion_trace,
            title="Insertion: build answer support from empty context",
            color="#26735f",
            delta_title="Step gain",
            random_baseline=result.random_insertion_baseline,
        ),
        width="stretch",
    )
    _render_insertion_step_story(insertion_trace, trace)

    with st.expander("Deletion curve", expanded=False):
        st.markdown(
            '<p class="analysis-tab-note" style="margin-top:0.3rem;">'
            "Starts with all chunks and removes evidence in three orders: Shapley attribution "
            "(red), retriever relevance score (blue dashed), and random removal average "
            "(grey dashed). A faithful explanation should make the score drop faster than "
            "retrieval score or random removal."
            "</p>",
            unsafe_allow_html=True,
        )
        st.altair_chart(
            audit_line_chart(
                deletion_trace,
                title="Deletion: remove evidence from full context",
                color="#b5473f",
                delta_title="Step drop",
                random_baseline=result.random_deletion_baseline,
                retrieval_baseline=result.retrieval_deletion_baseline,
            ),
            width="stretch",
        )
        render_deletion_comparison_summary(result)
        order_frame = audit_order_frame(result)
        st.dataframe(
            order_frame,
            width="stretch",
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "Chunk": st.column_config.TextColumn(width="large"),
                "Attribution": st.column_config.NumberColumn(format="%+.3f", width="small"),
                "Deletion action": st.column_config.TextColumn(width="medium"),
                "Insertion action": st.column_config.TextColumn(width="medium"),
            },
        )


def render_settings_page() -> None:
    """Render persistent explanation settings."""
    st.markdown('<div class="section-title">Explanation settings</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.0, 1.0, 1.15])
    with c1:
        _ix_opts = ["SV", "k-SII", "STII", "FSII"]
        _ix_cur = st.session_state.get("interaction_index", "SV")
        _ix_sel = st.selectbox(
            "Primary interaction index",
            _ix_opts,
            index=_ix_opts.index(_ix_cur) if _ix_cur in _ix_opts else 0,
            help=(
                "SV — Shapley values: first-order attribution only, no pairwise interaction effects.\n"
                "k-SII — k-Shapley Interaction Index: captures synergy and redundancy between pairs of chunks (recommended).\n"
                "STII — Shapley-Taylor Interaction Index: alternative higher-order formulation.\n"
                "FSII — Faithful Shapley Interaction Index: regression-based, consistent with SV at order 1.\n"
                "See the Value Functions page for full descriptions."
            ),
        )
        st.session_state.interaction_index = _ix_sel
        if st.session_state.interaction_index == "SV":
            st.number_input(
                "Max interaction order",
                min_value=1,
                max_value=1,
                value=1,
                disabled=True,
                key="max_order_sv_display",
            )
        else:
            st.slider("Max interaction order", 2, 3, key="max_order")
        st.slider("Approximation budget", 16, 512, step=8, key="budget")
    with c2:
        st.multiselect(
            "Value functions",
            list(VALUE_FUNCTION_DEFINITIONS),
            key="value_modes",
        )
        if not st.session_state.value_modes:
            st.warning("Select at least one value function before running.")
        if st.button("Reset settings", key="reset_settings"):
            st.session_state.interaction_index = "SV"
            st.session_state.max_order = 2
            st.session_state.budget = 16
            st.session_state.value_modes = ["Lexical grounding"]
            st.session_state.model_id = "google/gemma-4-E2B-it"
            st.session_state.device_map = "auto"
            st.session_state.torch_dtype = "auto"
            st.session_state.max_new_tokens = 96
            clear_run_outputs()
            st.rerun()
    with c3:
        if selected_value_modes_require_model():
            st.text_input(
                "Hugging Face model id", key="model_id", placeholder="google/gemma-4-E2B-it"
            )
            inner1, inner2 = st.columns(2)
            with inner1:
                st.text_input("device_map", key="device_map", placeholder="auto")
            with inner2:
                st.selectbox(
                    "torch_dtype",
                    ["auto", "float16", "bfloat16", "float32"],
                    key="torch_dtype",
                )
            st.slider("max_new_tokens", 16, 256, step=16, key="max_new_tokens")
        else:
            st.markdown(
                """
                <div class="card">
                    <span class="card-label">Model settings</span>
                    <p class="muted-text">The selected value function runs locally and does not use a Hugging Face model.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.checkbox(
        "Run sentence drilldown",
        key="run_sentence_drilldown",
        help=(
            "Sentence drilldown treats every retrieved sentence as a player. "
            "With HF value functions this can require hundreds or thousands of model calls."
        ),
    )
    st.markdown(
        """
        <div class="card">
            <span class="card-label">Current configuration</span>
            <p class="muted-text">Settings are stored in session state and apply to the next Run. HF model settings apply only to HF-backed value functions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.input_source != "PDF upload":
        return

    st.markdown('<div class="section-title">PDF setup</div>', unsafe_allow_html=True)
    pdf_col, retrieval_col = st.columns([1.2, 1.0], gap="large")
    with pdf_col:
        st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            key="uploaded_pdf",
            on_change=activate_dashboard_after_pdf_upload,
        )
    with retrieval_col:
        st.selectbox("Retrieval method", ["Dense embeddings", "TF-IDF"], key="retrieval_method")
        if st.session_state.retrieval_method == "Dense embeddings":
            st.text_input(
                "Embedding model",
                key="embedding_model_id",
                placeholder="sentence-transformers/all-MiniLM-L6-v2",
            )
            st.selectbox("Embedding device", ["auto", "cpu", "mps", "cuda"], key="embedding_device")
        st.slider("Retrieved chunks", 1, 8, key="retrieval_top_k")
        st.slider("Words per chunk", 80, 320, step=20, key="rag_chunk_words")
        max_overlap = max(10, int(st.session_state.rag_chunk_words) - 20)
        if int(st.session_state.rag_chunk_overlap) > max_overlap:
            st.session_state.rag_chunk_overlap = max_overlap
        st.slider("Chunk overlap words", 0, max_overlap, step=5, key="rag_chunk_overlap")
        st.markdown(
            """
            <div class="card">
                <span class="card-label">PDF run path</span>
                <p class="muted-text">Run parses the PDF, retrieves chunks with dense embeddings by default, asks the configured Hugging Face model to answer from those chunks, then explains those retrieved chunks.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _run_sentence_drilldown(
    *,
    value_function: object,
    value_mode: str,
    question: str,
    target_answer: str,
    sentence_chunks: list[RetrievedChunk],
    sentence_labels: list[str],
    progress: ProgressHandle,
    message: MessageHandle,
    progress_base: float,
    progress_span: float,
) -> pd.DataFrame:
    """Run sentence-level attribution and return the first-order frame."""
    sentence_budget = budget_for_exactish_demo(len(sentence_chunks))
    message.info(
        f"{value_mode}: sentence drilldown has {len(sentence_chunks)} sentence players "
        f"and will score about {sentence_budget} coalitions."
    )
    sentence_scorer = ProgressScorer(
        value_function,
        label=f"{value_mode} sentence drilldown",
        progress=progress,
        message=message,
        expected_calls=sentence_budget + 12,
        base_progress=progress_base + 0.58 * progress_span,
        progress_span=0.35 * progress_span,
    )
    sentence_game = RAGRetrievalGame(
        question=question,
        target_answer=target_answer,
        chunks=sentence_chunks,
        scorer=sentence_scorer,
    )
    if st.session_state.interaction_index == "SV":
        sentence_approximator = shapiq.KernelSHAP(n=sentence_game.n_players, random_state=42)
    else:
        sentence_approximator = shapiq.KernelSHAPIQ(
            n=sentence_game.n_players, index="k-SII", max_order=2, random_state=42
        )
    sentence_explanation = sentence_approximator.approximate(
        budget=sentence_budget, game=sentence_game
    )
    return interaction_values_to_frame(sentence_explanation.get_n_order(order=1), sentence_labels)


def run_single_explanation(
    *,
    scenario_name: str,
    value_mode: str,
    question: str,
    target_answer: str,
    chunks: list[RetrievedChunk],
    labels: list[str],
    retrieval_scores: list[float],
    progress: ProgressHandle,
    message: MessageHandle,
    progress_base: float,
    progress_span: float,
) -> ExplanationRunResult:
    """Run one configured value function end-to-end."""
    value_function = cached_value_function(
        value_mode,
        st.session_state.model_id,
        st.session_state.device_map,
        st.session_state.torch_dtype,
        st.session_state.max_new_tokens,
    )
    metadata = VALUE_FUNCTION_DEFINITIONS[value_mode]
    message.info(f"{value_mode}: loading value function.")
    progress.progress(progress_base)
    value_function.warmup()

    chunk_scorer = ProgressScorer(
        value_function,
        label=value_mode,
        progress=progress,
        message=message,
        expected_calls=st.session_state.budget + 12,
        base_progress=progress_base + 0.05 * progress_span,
        progress_span=0.45 * progress_span,
    )
    game = RAGRetrievalGame(
        question=question,
        target_answer=target_answer,
        chunks=chunks,
        scorer=chunk_scorer,
    )
    full_score = float(game(game.grand_coalition)[0])
    empty_score = float(game(game.empty_coalition)[0])

    message.info(f"{value_mode}: estimating chunk attributions.")
    approximator = make_approximator(
        st.session_state.interaction_index,
        game.n_players,
        effective_max_order(),
    )
    explanation = approximator.approximate(budget=st.session_state.budget, game=game)
    first_order = explanation.get_n_order(order=1)
    first_order_frame = interaction_values_to_frame(first_order, labels)
    pairwise_matrix = pairwise_matrix_from_explanation(explanation, game.n_players)
    audit_frame = coalition_audit_frame(game)
    pair_label, pair_value = strongest_pair(pairwise_matrix, labels)

    message.info(f"{value_mode}: running audit checks.")
    faithfulness = summarize_faithfulness(game, first_order_frame)
    deletion_frame = deletion_evaluation(game, first_order_frame)
    insertion_frame = insertion_evaluation(game, first_order_frame)
    del_baseline = random_deletion_baseline(game)
    ins_baseline = random_insertion_baseline(game)
    retrieval_del_baseline = retrieval_score_deletion_baseline(game, retrieval_scores)

    # LOO marginal: V(N) - V(N \ {i}) for each chunk i, batched in one call.
    loo_batch = np.ones((game.n_players, game.n_players), dtype=bool)
    for _i in range(game.n_players):
        loo_batch[_i, _i] = False
    loo_scores = [full_score - float(v) for v in game(loo_batch)]

    sentence_first_order_frame = pd.DataFrame()
    sentence_chunks, sentence_labels, sentence_meta = build_sentence_players(chunks)
    if st.session_state.run_sentence_drilldown and len(sentence_chunks) >= 2:
        sentence_first_order_frame = _run_sentence_drilldown(
            value_function=value_function,
            value_mode=value_mode,
            question=question,
            target_answer=target_answer,
            sentence_chunks=sentence_chunks,
            sentence_labels=sentence_labels,
            progress=progress,
            message=message,
            progress_base=progress_base,
            progress_span=progress_span,
        )

    progress.progress(progress_base + progress_span)
    return ExplanationRunResult(
        scenario_name=scenario_name,
        value_mode=value_mode,
        value_name=getattr(value_function, "name", value_mode),
        value_description=metadata["short"],
        value_formula=metadata["formula"],
        full_score=full_score,
        empty_score=empty_score,
        first_order_frame=first_order_frame,
        pairwise_matrix=pairwise_matrix,
        audit_frame=audit_frame,
        pair_label=pair_label,
        pair_value=pair_value,
        faithfulness=faithfulness,
        deletion_frame=deletion_frame,
        insertion_frame=insertion_frame,
        sentence_labels=sentence_labels,
        sentence_meta=sentence_meta,
        sentence_first_order_frame=sentence_first_order_frame,
        random_deletion_baseline=del_baseline,
        random_insertion_baseline=ins_baseline,
        retrieval_deletion_baseline=retrieval_del_baseline,
        loo_scores=loo_scores,
    )


def execute_run(trace_name: str, trace: dict[str, object], chunks: list[RetrievedChunk]) -> None:
    """Execute the explanation pipeline and persist results in session state."""
    if not st.session_state.value_modes:
        st.session_state.run_status = {
            "state": "error",
            "message": "Run failed: select at least one value function.",
        }
        return
    if not chunks:
        st.session_state.run_status = {
            "state": "error",
            "message": "Run failed: no retrieved chunks are available.",
        }
        return
    st.session_state.run_results = []
    st.session_state.run_result = None
    st.session_state.active_result_index = 0
    if "active_result_selector" in st.session_state:
        del st.session_state["active_result_selector"]
    st.session_state.run_status = {
        "state": "running",
        "message": f"Running {len(st.session_state.value_modes)} value function(s).",
    }

    labels = [chunk.title for chunk in chunks]
    retrieval_scores = [float(score) for score in trace.get("retrieval_scores", [])]
    workload = estimate_run_workload(chunks)
    progress = st.progress(0.0)
    message = st.empty()
    try:
        results = []
        modes = list(st.session_state.value_modes)
        message.info(
            "Starting run: "
            f"{workload['chunks']} chunks, {workload['sentences']} sentence players, "
            f"{workload['chunk_calls_per_value']} chunk-level calls per value function, "
            f"{workload['sentence_calls_per_value']} sentence-level calls per value function."
        )
        span = 1.0 / len(modes)
        for idx, value_mode in enumerate(modes):
            message.info(f"Running {value_mode} ({idx + 1}/{len(modes)}).")
            results.append(
                run_single_explanation(
                    scenario_name=trace_name,
                    value_mode=value_mode,
                    question=str(trace["question"]),
                    target_answer=str(trace["target_answer"]),
                    chunks=chunks,
                    labels=labels,
                    retrieval_scores=retrieval_scores,
                    progress=progress,
                    message=message,
                    progress_base=idx * span,
                    progress_span=span,
                )
            )
        progress.progress(1.0)
        message.success("Run complete. Dashboard updated.")
        st.session_state.run_results = results
        st.session_state.run_result = results[0] if results else None
        st.session_state.run_status = {
            "state": "success",
            "message": (
                f"Run complete for {trace_name} using "
                f"{', '.join(result.value_name for result in results)}."
            ),
        }
    except (RuntimeError, ValueError, OSError, ImportError) as error:
        st.session_state.run_results = []
        st.session_state.run_result = None
        st.session_state.run_status = {
            "state": "error",
            "message": f"Run failed: {error}",
        }
        message.error(f"Run failed: {error}")
        st.exception(error)


def build_uploaded_pdf_trace() -> tuple[str, dict[str, object], list[RetrievedChunk]] | None:
    """Build and cache a real RAG trace from the uploaded PDF widget state."""
    pdf_name = str(st.session_state.get("pdf_name", "")).strip()
    pdf_bytes = st.session_state.get("pdf_bytes")
    question = str(st.session_state.get("rag_question", "")).strip()
    if not pdf_name or not pdf_bytes:
        st.session_state.run_status = {
            "state": "error",
            "message": "Run failed: upload a PDF before running PDF RAG.",
        }
        return None
    if not question:
        st.session_state.run_status = {
            "state": "error",
            "message": "Run failed: enter a question for the uploaded PDF.",
        }
        return None

    status = st.status("Starting PDF RAG pipeline...", expanded=True)
    try:
        st.session_state.run_status = {
            "state": "running",
            "message": "Parsing uploaded PDF.",
        }
        status.write("1/5 Parsing selectable text from the PDF.")
        pages = extract_pdf_pages(pdf_bytes)

        st.session_state.run_status = {
            "state": "running",
            "message": f"Chunking {len(pages)} extracted PDF page(s).",
        }
        status.write(f"2/5 Building overlapping text chunks from {len(pages)} extracted page(s).")
        candidates = chunk_pdf_pages(
            pages,
            words_per_chunk=int(st.session_state.rag_chunk_words),
            overlap_words=int(st.session_state.rag_chunk_overlap),
        )

        st.session_state.run_status = {
            "state": "running",
            "message": f"Retrieving top {int(st.session_state.retrieval_top_k)} chunks.",
        }
        retrieval_detail = (
            f"dense embeddings from {st.session_state.embedding_model_id}"
            if st.session_state.retrieval_method == "Dense embeddings"
            else "TF-IDF"
        )
        chunk_type_counts = (
            pd.Series([chunk.chunk_type for chunk in candidates]).value_counts().to_dict()
        )
        status.write(
            f"3/5 Retrieving and reranking top {int(st.session_state.retrieval_top_k)} chunks "
            f"from {len(candidates)} candidates with {retrieval_detail}. "
            f"Candidate types: {chunk_type_counts}."
        )
        ranked, retrieval_debug = retrieve_relevant_chunks_with_debug(
            question,
            candidates,
            top_k=int(st.session_state.retrieval_top_k),
            method=st.session_state.retrieval_method,
            embedding_model_id=st.session_state.embedding_model_id,
            embedding_device=st.session_state.embedding_device,
        )
        chunks = [item.chunk for item in ranked]
        if len(chunks) < int(st.session_state.retrieval_top_k):
            st.warning(
                f"Only {len(chunks)} chunk(s) were retrieved — fewer than the configured "
                f"{int(st.session_state.retrieval_top_k)}. "
                "The PDF may be too short, or try reducing 'Words per chunk' in Settings."
            )

        st.session_state.run_status = {
            "state": "running",
            "message": (
                f"Generating grounded answer with {st.session_state.model_id}. "
                "First model load can take several minutes."
            ),
        }
        status.write(
            "4/5 Generating the answer with the configured Hugging Face model. "
            "If this is the first Gemma run, model loading can take several minutes."
        )
        answer = generate_answer_from_chunks(
            question=question,
            chunks=chunks,
            model_id=st.session_state.model_id,
            device_map=st.session_state.device_map,
            torch_dtype=st.session_state.torch_dtype,
            max_new_tokens=int(st.session_state.max_new_tokens),
        )

        status.write("5/5 Preparing retrieved chunks for shapiq attribution.")
        trace = {
            "page": "Uploaded PDF RAG",
            "question": question,
            "target_answer": answer,
            "takeaway": "Retrieved chunks are produced from the uploaded PDF before Gemma generation.",
            "source": "pdf_upload",
            "pdf_name": pdf_name,
            "retrieval_method": st.session_state.retrieval_method,
            "embedding_model_id": st.session_state.embedding_model_id,
            "retrieval_scores": [item.score for item in ranked],
            "retrieval_debug": asdict(retrieval_debug),
            "chunks": [
                {
                    "title": item.chunk.title,
                    "text": item.chunk.text,
                    "page_number": item.chunk.page_number,
                    "chunk_type": item.chunk.chunk_type,
                    "section_title": item.chunk.section_title,
                    "text_length": item.chunk.text_length,
                    "dense_score": item.dense_score,
                    "keyword_score": item.keyword_score,
                    "rerank_score": item.rerank_score,
                    "flags": list(item.chunk.flags),
                    "evidence_role": (
                        f"{item.chunk.chunk_type} · rerank {item.rerank_score:.3f} · "
                        f"dense {item.dense_score:.3f} · keyword {item.keyword_score:.3f}"
                    ),
                }
                for item in ranked
            ],
        }
        status.update(label="PDF RAG trace ready. Starting attribution.", state="complete")
    except (RuntimeError, ValueError, OSError, ImportError) as error:
        st.session_state.pdf_rag_trace = None
        st.session_state.run_status = {
            "state": "error",
            "message": f"PDF RAG failed: {error}",
        }
        status.update(label=f"PDF RAG failed: {error}", state="error")
        st.exception(error)
        return None

    st.session_state.pdf_rag_trace = trace
    trace_name = f"Uploaded PDF: {pdf_name}"
    return trace_name, trace, chunks


def activate_dashboard_after_pdf_upload() -> None:
    """Store the uploaded PDF and flag a pending redirect to the dashboard."""
    uploaded = st.session_state.get("uploaded_pdf")
    if uploaded is not None:
        st.session_state.pdf_name = uploaded.name
        st.session_state.pdf_bytes = uploaded.getvalue()
        st.session_state.pdf_rag_trace = None
        st.session_state.pdf_chat_history = []
        clear_run_outputs()
        st.session_state.pdf_upload_pending_redirect = True
    st.session_state.input_source = "PDF upload"


def run_pdf_chat_question(question: str) -> None:
    """Run PDF RAG from a dashboard chat prompt."""
    st.session_state.rag_question = question.strip()
    built = build_uploaded_pdf_trace()
    if built is None:
        return
    trace_name, trace, chunks = built
    execute_run(trace_name, trace, chunks)
    st.session_state.pdf_chat_history.append(
        {
            "question": question.strip(),
            "answer": str(trace.get("target_answer", "")),
            "chunks": chunks,
            "results": list(st.session_state.run_results),
        }
    )


def current_trace_and_chunks() -> tuple[str, dict[str, object], list[RetrievedChunk]]:
    """Return the trace shown by the dashboard for the selected input source."""
    if st.session_state.input_source == "Sample traces":
        trace_name = st.session_state.scenario_name
        trace = SAMPLE_TRACES[trace_name]
        return trace_name, trace, trace_chunks(trace)

    trace = st.session_state.get("pdf_rag_trace")
    if isinstance(trace, dict):
        name = f"Uploaded PDF: {trace.get('pdf_name', 'PDF')}"
        return name, trace, trace_chunks(trace)

    question = str(st.session_state.get("rag_question", "")).strip()
    placeholder = {
        "page": "Uploaded PDF RAG",
        "question": question or "Ask a question about the uploaded PDF.",
        "target_answer": "Run PDF RAG to generate an answer from retrieved chunks.",
        "takeaway": "No PDF RAG trace has been generated yet.",
        "chunks": [],
    }
    return "Uploaded PDF", placeholder, []


def render_pdf_entry_attribution(entry: dict) -> None:
    """Render full retrieval attribution for a past PDF chat entry inside an expander."""
    entry_results: list[ExplanationRunResult] = entry.get("results", [])
    entry_chunks: list[RetrievedChunk] = entry.get("chunks", [])
    if not entry_results or not entry_chunks:
        st.info("No attribution results were saved for this question.")
        return
    result = entry_results[0]

    scores = attribution_lookup(result.first_order_frame)
    chunk_rows = [
        {
            "Chunk": chunk.title,
            "φᵢ (Shapley value)": f"{scores[idx]:+.3f}" if idx in scores else "—",
        }
        for idx, chunk in enumerate(entry_chunks)
    ]
    st.caption("Retrieved chunks")
    st.dataframe(pd.DataFrame(chunk_rows), hide_index=True, use_container_width=True)

    st.caption("Step 1 · Individual marginal effect")
    render_attribution_chart(result, entry_chunks)
    render_negative_score_callout(result)

    st.caption("Step 2 · Synergy or redundancy?")
    render_top_interaction_card(result)
    render_interaction_map(result, entry_chunks)

    st.caption("Step 3 · Does the ranking hold?")
    _render_validation_inline(result, entry_chunks)


def render_pdf_chat_panel() -> None:
    """Render the PDF-first chat entry point on the dashboard."""
    pdf_name = str(st.session_state.get("pdf_name", "")).strip()
    has_pdf = bool(pdf_name and st.session_state.get("pdf_bytes"))
    file_label = pdf_name or "No PDF uploaded"

    status = "Ready for questions" if has_pdf else "Upload a PDF in Settings"
    st.markdown(
        f"""
        <div class="pdf-chat-shell">
            <div class="pdf-chat-head">
                <div>
                    <span class="pdf-chat-title">PDF RAG chat</span>
                    <span class="pdf-chat-file">{html.escape(file_label)}</span>
                </div>
                <div class="chip-list">
                    <span class="vf-chip">{int(st.session_state.retrieval_top_k)} chunks</span>
                    <span class="vf-chip muted">{html.escape(status)}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = st.session_state.get("pdf_chat_history", [])
    if not history and not has_pdf:
        st.info("Upload a PDF on the Explanation Settings page, then come back here to ask.")
    for i, entry in enumerate(history):
        with st.chat_message("user"):
            st.write(entry["question"])
        with st.chat_message("assistant"):
            st.write(entry["answer"])
        if i < len(history) - 1:
            with st.expander("Retrieval attribution", expanded=False):
                render_pdf_entry_attribution(entry)

    prompt = st.chat_input(
        "Ask a question about the uploaded PDF",
        disabled=not has_pdf,
        key="pdf_chat_input",
    )
    if prompt:
        run_pdf_chat_question(prompt)
        st.rerun()


def active_result_from_state() -> ExplanationRunResult | None:
    """Return the selected display result."""
    results: list[ExplanationRunResult] = st.session_state.run_results
    if not results:
        return None
    idx = min(st.session_state.active_result_index, len(results) - 1)
    st.session_state.active_result_index = idx
    st.session_state.run_result = results[idx]
    return results[idx]


def render_dashboard(trace: dict[str, object], chunks: list[RetrievedChunk]) -> None:
    """Render the main narrative dashboard."""
    if st.session_state.input_source == "PDF upload":
        render_pdf_chat_panel()

    results: list[ExplanationRunResult] = st.session_state.run_results
    active_result = active_result_from_state()
    if chunks and not results:
        render_run_preview(chunks)
    if st.session_state.input_source == "PDF upload" and len(chunks) < 2:
        st.warning(
            "Only one retrieved chunk is selected. First-order bars still work, but pairwise "
            "interaction visualizations need at least two chunks. Increase Retrieved chunks in "
            "Explanation Settings to bring the richer visuals back."
        )

    # ── Setup: question + context left, run settings + result summary right ──
    left_col, right_col = st.columns([0.68, 0.32], gap="large", vertical_alignment="top")
    with left_col:
        render_context_panel(trace, chunks, active_result)
        render_retrieval_debug(trace)
    with right_col:
        render_settings_summary()
        render_result_summary(active_result)

    if not results:
        return

    # Multi-run selector
    if len(results) > 1:
        options = [result.value_name for result in results]
        active_name = st.selectbox(
            "Active value function",
            options,
            index=st.session_state.active_result_index,
            key="active_result_selector",
        )
        st.session_state.active_result_index = options.index(active_name)
        active_result = active_result_from_state()

    if active_result is not None:
        st.markdown(
            f'<p style="color:var(--text-muted);font-size:0.82rem;margin:0.1rem 0 0.2rem 0;">'
            f'Steps 1-3 &nbsp;·&nbsp; <strong style="color:var(--text);">'
            f"{html.escape(active_result.value_name)}</strong></p>",
            unsafe_allow_html=True,
        )

    # ── Step 1: Individual marginal effect ───────────────────────────────────
    render_story_step(
        "Step 1",
        "Individual marginal effect",
        "Bars show Shapley values (φᵢ): each chunk's average marginal contribution across all "
        "possible context subsets — not factual relevance, but model-score attribution under "
        "the selected value function. "
        "When LOO scores are available, a lighter side-by-side bar shows the LOO marginal "
        "V(all) - V(all without chunk i). If the Shapley bar extends beyond the LOO bar, "
        "credit is distributed across partial subsets that LOO misses.",
    )
    render_attribution_chart(active_result, chunks)
    render_negative_score_callout(active_result)

    # ── Step 2: Synergy or redundancy? ───────────────────────────────────────
    render_story_step(
        "Step 2",
        "Synergy or redundancy?",
        "k-SII interaction values (φᵢ,ⱼ) reveal whether two chunks are complementary "
        "(positive — together they exceed the sum of parts) or redundant "
        "(negative — each is dispensable when the other is present). "
        "Switch the index to k-SII in Explanation Settings to enable this view.",
    )
    render_top_interaction_card(active_result)
    render_interaction_map(active_result, chunks)

    # ── Step 3: Does the ranking hold? ───────────────────────────────────────
    render_story_step(
        "Step 3",
        "Does the ranking hold?",
        "Insertion adds chunks one by one in descending attribution-magnitude order (|φᵢ|) "
        "and tracks how support grows. The dashed line is the average curve for random "
        "orderings — if the Shapley ordering rises faster, the attribution is informative.",
    )
    _render_validation_inline(active_result, chunks, trace)

    # ── Deep dive (auxiliary tabs) ────────────────────────────────────────────
    render_advanced_audit_tabs(active_result, results, chunks)


def initialize_state() -> None:
    """Initialize session state without overwriting user choices."""
    first_trace = next(iter(SAMPLE_TRACES))
    get_setting("input_source", "Sample traces")
    get_setting("active_page", "Dashboard")
    get_setting("last_input_source", st.session_state.input_source)
    get_setting("last_uploaded_pdf_key", "")
    get_setting("scenario_name", first_trace)
    get_setting("interaction_index", "SV")
    get_setting("max_order", 2)
    get_setting("budget", 16)
    get_setting("value_modes", ["Lexical grounding"])
    get_setting("model_id", "google/gemma-4-E2B-it")
    get_setting("device_map", "auto")
    get_setting("torch_dtype", "auto")
    get_setting("max_new_tokens", 96)
    if "run_sentence_drilldown" not in st.session_state:
        st.session_state.run_sentence_drilldown = False
    # Restore defaults if session state was hot-reloaded to empty strings.
    if not st.session_state.get("model_id"):
        st.session_state["model_id"] = "google/gemma-4-E2B-it"
    if not st.session_state.get("device_map"):
        st.session_state["device_map"] = "auto"
    get_setting("run_results", [])
    get_setting("run_result", None)
    get_setting("active_result_index", 0)
    get_setting("run_status", {"state": "idle", "message": "No run yet."})
    get_setting("pdf_rag_trace", None)
    get_setting("last_scenario_name", st.session_state.scenario_name)
    get_setting("pdf_chat_history", [])
    if "pdf_upload_pending_redirect" not in st.session_state:
        st.session_state.pdf_upload_pending_redirect = False
    get_setting("pdf_name", "")
    get_setting("pdf_bytes", b"")
    get_setting("rag_question", "")
    get_setting("retrieval_method", "Dense embeddings")
    get_setting("embedding_model_id", "sentence-transformers/all-MiniLM-L6-v2")
    get_setting("embedding_device", "auto")
    get_setting("retrieval_top_k", 4)
    if int(st.session_state.retrieval_top_k) < 2:
        st.session_state.retrieval_top_k = 4
    get_setting("rag_chunk_words", 180)
    get_setting("rag_chunk_overlap", 35)
    normalize_value_modes()


def clear_run_outputs() -> None:
    """Clear run outputs that no longer match the visible input."""
    st.session_state.run_results = []
    st.session_state.run_result = None
    st.session_state.active_result_index = 0
    st.session_state.run_status = {"state": "idle", "message": "No run yet."}


def sync_input_state() -> None:
    """Reset stale results when the visible input source, scenario, or PDF file changes."""
    if st.session_state.input_source != st.session_state.last_input_source:
        clear_run_outputs()
        st.session_state.active_page = "Dashboard"
        st.session_state.last_input_source = st.session_state.input_source

    if st.session_state.scenario_name != st.session_state.last_scenario_name:
        clear_run_outputs()
        st.session_state.last_scenario_name = st.session_state.scenario_name

    uploaded = st.session_state.get("uploaded_pdf")
    uploaded_key = ""
    if uploaded is not None:
        uploaded_key = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
    if uploaded_key != st.session_state.last_uploaded_pdf_key:
        if uploaded is not None:
            st.session_state.pdf_name = uploaded.name
            st.session_state.pdf_bytes = uploaded.getvalue()
            st.session_state.pdf_rag_trace = None
            if st.session_state.input_source == "PDF upload":
                clear_run_outputs()
        st.session_state.last_uploaded_pdf_key = uploaded_key


def main() -> None:
    """Run the Streamlit dashboard."""
    initialize_state()
    st.markdown(CSS, unsafe_allow_html=True)
    render_page_title()

    if st.session_state.get("pdf_upload_pending_redirect"):
        st.session_state.pdf_upload_pending_redirect = False
        pdf_name = str(st.session_state.get("pdf_name", "PDF"))
        st.success(f"Upload successful: {pdf_name}")

    run = render_top_controls()
    sync_input_state()

    if run:
        if st.session_state.input_source == "PDF upload":
            built = build_uploaded_pdf_trace()
            if built is not None:
                trace_name, trace, chunks = built
                execute_run(trace_name, trace, chunks)
        else:
            trace_name, trace, chunks = current_trace_and_chunks()
            execute_run(trace_name, trace, chunks)

    _, trace, chunks = current_trace_and_chunks()

    page = render_navigation()
    if page == "Dashboard":
        render_dashboard(trace, chunks)
    elif page == "Explanation Settings":
        render_settings_page()
    elif page == "Value Functions":
        render_value_functions_page()
    else:
        render_rag_pipeline_page()


if __name__ == "__main__":
    main()
