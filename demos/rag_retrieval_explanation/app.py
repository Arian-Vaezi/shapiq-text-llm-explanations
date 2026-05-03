"""Streamlit app for RAG retrieval explanation with shapiq."""

from __future__ import annotations

import itertools
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle

import shapiq
from shapiq.plot import sentence_interaction_heatmap, token_attribution_bar_plot

from rag_game import (
    RAGRetrievalGame,
    RetrievedChunk,
    budget_for_exactish_demo,
)
from sample_data import SAMPLE_TRACES, SCENARIO_PAGES


st.set_page_config(
    page_title="RAG Retrieval Explanation",
    page_icon="R",
    layout="wide",
)


CSS = """
<style>
section[data-testid="stSidebar"] {
    background: #f7f5ef;
    border-right: 1px solid #ddd6c7;
}
.main .block-container {
    max-width: 1180px;
    padding-top: 2rem;
}
.rag-title {
    border-bottom: 1px solid #252525;
    margin-bottom: 0.85rem;
    padding-bottom: 0.75rem;
}
.rag-title h1 {
    color: #1f1f1f;
    font-family: Georgia, serif;
    font-size: 2.15rem;
    font-weight: 700;
    letter-spacing: 0;
    line-height: 1.05;
    margin: 0;
}
.rag-title p {
    color: #59544a;
    font-size: 0.96rem;
    margin: 0.45rem 0 0 0;
}
.metric-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.75rem;
    margin: 0.75rem 0 1.1rem 0;
}
.metric-card {
    background: #fffdf8;
    border: 1px solid #ded6c4;
    border-radius: 6px;
    padding: 0.75rem 0.9rem;
}
.metric-card span {
    color: #6d6658;
    display: block;
    font-size: 0.75rem;
    text-transform: uppercase;
}
.metric-card strong {
    color: #1f1f1f;
    display: block;
    font-size: 1.25rem;
    margin-top: 0.2rem;
}
.verdict {
    background: #1f2a28;
    border-radius: 7px;
    color: #f7f1e4;
    display: grid;
    gap: 1rem;
    grid-template-columns: 1.2fr 1fr 1fr;
    margin: 0.35rem 0 1rem 0;
    padding: 1rem;
}
.verdict-card {
    border-left: 1px solid rgba(247, 241, 228, 0.22);
    padding-left: 1rem;
}
.verdict-card:first-child {
    border-left: 0;
    padding-left: 0;
}
.verdict-card span {
    color: #c8d3c7;
    display: block;
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    margin-bottom: 0.28rem;
    text-transform: uppercase;
}
.verdict-card strong {
    color: #ffffff;
    display: block;
    font-size: 1.15rem;
    line-height: 1.2;
}
.verdict-card p {
    color: #e7dfcf;
    font-size: 0.86rem;
    line-height: 1.35;
    margin: 0.35rem 0 0 0;
}
.section-label {
    color: #5f584b;
    font-size: 0.76rem;
    letter-spacing: 0.06em;
    margin: 0.3rem 0 0.45rem 0;
    text-transform: uppercase;
}
.scenario-panel {
    background: #fffdf8;
    border: 1px solid #ded6c4;
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
.chunk-box {
    background: #fffdf7;
    border: 1px solid #d6cab2;
    border-left: 4px solid #2d6f73;
    border-radius: 6px;
    margin-bottom: 0.55rem;
    padding: 0.62rem 0.78rem;
}
.chunk-box h4 {
    color: #222;
    font-size: 0.9rem;
    margin: 0 0 0.25rem 0;
}
.chunk-box p {
    color: #403d37;
    font-size: 0.9rem;
    line-height: 1.4;
    margin: 0;
}
.note-box {
    background: #fffdf7;
    border: 1px solid #ded6c4;
    border-radius: 6px;
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
</style>
"""


def clean_key(value: str) -> str:
    """Create a stable Streamlit key fragment."""
    return (
        value.lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(":", "")
    )


def chunk_editor(default_chunks: list[dict[str, str]], trace_key: str) -> list[RetrievedChunk]:
    """Render chunk inputs and return edited chunks."""
    st.sidebar.subheader("Retrieved Chunks")
    chunk_count = st.sidebar.slider(
        "Number of chunks",
        2,
        8,
        len(default_chunks),
        key=f"{trace_key}_chunk_count",
    )
    chunks = []
    for idx in range(chunk_count):
        fallback = default_chunks[idx] if idx < len(default_chunks) else {"title": "", "text": ""}
        title = st.sidebar.text_input(
            f"Chunk {idx + 1} title",
            fallback["title"],
            key=f"{trace_key}_title_{idx}",
        )
        text = st.sidebar.text_area(
            f"Chunk {idx + 1} text",
            fallback["text"],
            height=92,
            key=f"{trace_key}_text_{idx}",
        )
        if title.strip() or text.strip():
            chunks.append(RetrievedChunk(title=title.strip() or f"Chunk {idx + 1}", text=text.strip()))
    return chunks


def split_sentences(text: str) -> list[str]:
    """Split a retrieved chunk into simple sentence-like units."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def interaction_values_to_frame(values: shapiq.InteractionValues, labels: list[str]) -> pd.DataFrame:
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


def make_approximator(index: str, n_players: int, max_order: int):
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
    pair_label = f"{best_pair[0] + 1}. {labels[best_pair[0]]} + {best_pair[1] + 1}. {labels[best_pair[1]]}"
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


def build_interpretation_notes(
    attribution_frame: pd.DataFrame,
    pair_label: str,
    pair_value: float,
    full_score: float,
) -> list[str]:
    """Create plain-language interpretation bullets from the computed result."""
    if attribution_frame.empty:
        return ["No first-order attribution was returned for this run."]

    top = attribution_frame.iloc[0]
    notes = [
        (
            f"Start with chunk {int(top['player'])}, `{top['chunk']}`. "
            f"It has the largest individual attribution ({top['attribution']:.3f}), so it is "
            "the clearest single evidence source for the target answer."
        )
    ]

    if len(attribution_frame) > 1:
        second = attribution_frame.iloc[1]
        gap = abs(float(top["attribution"])) - abs(float(second["attribution"]))
        if gap < 0.05:
            notes.append(
                (
                    f"The lead is not decisive: chunk {int(second['player'])}, "
                    f"`{second['chunk']}`, is close behind ({second['attribution']:.3f}). "
                    "This is a good case to discuss multiple evidence sources."
                )
            )
        else:
            notes.append(
                (
                    f"The attribution gap to the next chunk is {gap:.3f}, so the evidence is "
                    "fairly concentrated rather than evenly spread across retrieval results."
                )
            )

    if abs(pair_value) < 0.03:
        notes.append(
            "Second-order effects are weak in this trace. The result is mostly explainable by individual chunks."
        )
    elif pair_value > 0:
        notes.append(
            (
                f"The strongest pair is `{pair_label}` ({pair_value:.3f}). Positive interaction means "
                "shapiq assigns extra shared support to that chunk pair under the selected index."
            )
        )
    else:
        notes.append(
            (
                f"The strongest pair is `{pair_label}` ({pair_value:.3f}). Negative interaction means "
                "the selected index treats that pair as redundant, saturating, or partly distracting."
            )
        )

    notes.append(
        (
            f"The full-context normalized support score is {full_score:.3f}. Treat this as a demo "
            "support score until you replace the lexical scorer with an LLM or entailment scorer. "
            "Use the sentence drilldown tab to inspect which sentences inside top chunks carry the signal."
        )
    )
    return notes


def coalition_audit_frame(game: RAGRetrievalGame) -> pd.DataFrame:
    """Build a small exact coalition audit table for transparency."""
    rows = []
    n_players = game.n_players
    for size in range(0, min(n_players, 3) + 1):
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


def plot_heatmap(matrix: np.ndarray, labels: list[str]) -> plt.Figure:
    """Render pairwise interaction matrix."""
    fig, ax = plt.subplots(figsize=(6.6, 5.1))
    max_abs = max(float(np.max(np.abs(matrix))), 0.01)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
    short_labels = [f"{idx + 1}. {label[:16]}" for idx, label in enumerate(labels)]
    ax.set_xticks(range(len(labels)), labels=short_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels=short_labels)
    ax.set_title("Second-order chunk interactions")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i != j:
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_prompt_graph(matrix: np.ndarray, attributions: pd.DataFrame, labels: list[str]) -> plt.Figure:
    """Render a compact chunk interaction network without extra dependencies."""
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_axis_off()

    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.column_stack([1.15 * np.cos(angles), 0.8 * np.sin(angles)])

    attr_lookup = {
        int(row.player) - 1: float(row.attribution)
        for row in attributions.itertuples(index=False)
    }
    max_attr = max([abs(v) for v in attr_lookup.values()] + [0.01])
    max_edge = max(float(np.max(np.abs(matrix))), 0.01)
    edge_threshold = max(0.02, 0.18 * max_edge)

    for i in range(n):
        for j in range(i + 1, n):
            weight = matrix[i, j]
            if abs(weight) < edge_threshold:
                continue
            color = "#23735f" if weight > 0 else "#a64f48"
            linewidth = 1.2 + 3.6 * abs(weight) / max_edge
            ax.plot(
                [positions[i, 0], positions[j, 0]],
                [positions[i, 1], positions[j, 1]],
                color=color,
                alpha=0.38,
                linewidth=linewidth,
                solid_capstyle="round",
                zorder=1,
            )

    for idx, (x_pos, y_pos) in enumerate(positions):
        attribution = attr_lookup.get(idx, 0.0)
        node_size = 620 + 1500 * abs(attribution) / max_attr
        color = "#2d6f73" if attribution >= 0 else "#b14b3a"
        ax.scatter(
            [x_pos],
            [y_pos],
            s=node_size,
            color=color,
            alpha=0.94,
            edgecolor="#1f1f1f",
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            x_pos,
            y_pos,
            str(idx + 1),
            color="white",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            zorder=4,
        )
        label_y = y_pos + (0.26 if y_pos >= 0 else -0.26)
        va = "bottom" if y_pos >= 0 else "top"
        ax.text(
            x_pos,
            label_y,
            f"{idx + 1}. {labels[idx][:24]}",
            ha="center",
            va=va,
            fontsize=8.3,
            color="#28251f",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "#fffdf8",
                "edgecolor": "#ded6c4",
                "alpha": 0.92,
            },
            zorder=5,
        )

    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.35, 1.35)
    ax.set_title("Chunk Interaction Network", fontsize=13, pad=8)
    ax.text(
        -1.7,
        -1.22,
        "Node size = attribution strength; edge width = second-order interaction strength.",
        fontsize=8,
        color="#6d6658",
    )
    fig.tight_layout()
    return fig


def polish_sentence_bar_plot(fig: plt.Figure, ax: plt.Axes) -> plt.Figure:
    """Make the package sentence bar plot fit the Streamlit drilldown panel."""
    fig.set_size_inches(6.2, 3.4)
    ax.set_title("", loc="center")
    ax.set_title("Sentence Attribution", loc="left", fontsize=12, pad=8)
    ax.set_xlabel("Support attribution")
    ax.grid(axis="x", color="#ded6c4", alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for patch in ax.patches:
        width = patch.get_width()
        if abs(width) < 0.01:
            continue
        x_pos = width + (0.015 if width >= 0 else -0.015)
        ha = "left" if width >= 0 else "right"
        ax.text(
            x_pos,
            patch.get_y() + patch.get_height() / 2,
            f"{width:.2f}",
            va="center",
            ha=ha,
            fontsize=8,
            color="#403d37",
        )

    fig.tight_layout()
    return fig


def polish_sentence_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    sentence_meta: pd.DataFrame,
) -> plt.Figure:
    """Make the package sentence heatmap fit the Streamlit drilldown panel."""
    fig.set_size_inches(5.8, 4.4)
    ax.set_title("", loc="center")
    ax.set_title("Sentence Interaction Heatmap", loc="left", fontsize=12, pad=8)
    ax.tick_params(axis="x", labelrotation=30)

    group_ranges = []
    for _, group in sentence_meta.groupby("source_chunk", sort=False):
        indices = [int(sentence[1:]) - 1 for sentence in group["sentence"]]
        group_ranges.append((min(indices), max(indices)))

    for start, end in group_ranges:
        size = end - start + 1
        ax.add_patch(
            Rectangle(
                (start - 0.5, start - 0.5),
                size,
                size,
                fill=False,
                edgecolor="#1f554c",
                linewidth=2.2,
                zorder=5,
            )
        )

    ax.text(
        0,
        -0.18,
        "Outlined blocks group sentences from the same retrieved chunk.",
        transform=ax.transAxes,
        fontsize=8,
        color="#5f584b",
        va="top",
    )
    fig.tight_layout()
    return fig


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="rag-title">
            <h1>RAG Retrieval Explanation</h1>
            <p>Which retrieved chunks actually support the answer, and which chunks only look relevant?</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page_name = st.sidebar.radio("Demo page", list(SCENARIO_PAGES), horizontal=False)
    page = SCENARIO_PAGES[page_name]
    trace_options = [
        name for name, trace_data in SAMPLE_TRACES.items() if trace_data["page"] == page_name
    ]
    trace_name = st.sidebar.selectbox("Scenario example", trace_options)
    trace = SAMPLE_TRACES[trace_name]
    trace_key = clean_key(f"{page_name}_{trace_name}")

    st.markdown(
        f"""
        <div class="scenario-panel">
            <div>
                <span class="scenario-tag">{page_name}</span>
                <h3>{page["title"]}</h3>
                <p>{page["goal"]}</p>
            </div>
            <div class="scenario-hint">
                <strong>Best used for:</strong> {page["best_for"]}<br>
                <strong>This example:</strong> {trace["takeaway"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.subheader("Question And Answer")
    question = st.sidebar.text_area(
        "Question",
        trace["question"],
        height=90,
        key=f"{trace_key}_question",
    )
    target_answer = st.sidebar.text_area(
        "Target answer",
        trace["target_answer"],
        height=90,
        key=f"{trace_key}_target_answer",
    )
    chunks = chunk_editor(trace["chunks"], trace_key)

    st.sidebar.subheader("Explanation Settings")
    index = st.sidebar.selectbox("Interaction index", ["SV", "k-SII", "STII", "FSII"], index=1)
    max_order = 1 if index == "SV" else st.sidebar.slider("Max interaction order", 2, 3, 2)
    default_budget = budget_for_exactish_demo(len(chunks))
    budget = st.sidebar.slider("Approximation budget", 16, 512, default_budget, step=8)

    if len(chunks) < 2:
        st.warning("Add at least two chunks to run the retrieval explanation.")
        return

    game = RAGRetrievalGame(question=question, target_answer=target_answer, chunks=chunks)
    labels = [chunk.title for chunk in chunks]

    full_score = float(game(game.grand_coalition)[0])
    empty_score = float(game(game.empty_coalition)[0])
    metric_html = f"""
    <div class="metric-strip">
        <div class="metric-card"><span>Chunks</span><strong>{len(chunks)}</strong></div>
        <div class="metric-card"><span>Empty Context</span><strong>{empty_score:.3f}</strong></div>
        <div class="metric-card"><span>Full Context</span><strong>{full_score:.3f}</strong></div>
    </div>
    """
    st.markdown(metric_html, unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown('<div class="section-label">Input Trace</div>', unsafe_allow_html=True)
        st.markdown(f"**Question**: {question}")
        st.markdown(f"**Target answer**: {target_answer}")
        for idx, chunk in enumerate(chunks, start=1):
            st.markdown(
                f"""
                <div class="chunk-box">
                    <h4>{idx}. {chunk.title}</h4>
                    <p>{chunk.text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with right:
        st.markdown('<div class="section-label">Game Setup</div>', unsafe_allow_html=True)
        st.markdown(
            """
            This demo creates one player per retrieved chunk. A coalition means
            “show only these chunks,” then the scorer estimates how strongly that
            selected context supports the target answer.
            """
        )
        preview_chunks = chunks[: min(2, len(chunks))]
        with st.expander("Prompt preview for a model-backed scorer"):
            st.code(game.build_prompt(preview_chunks), language="text")

    run = st.button("Run explanation", type="primary", use_container_width=True)
    if not run:
        st.info("Use the sidebar to choose or edit a RAG trace, then run the explanation.")
        return

    with st.spinner("Computing shapiq attributions..."):
        approximator = make_approximator(index, game.n_players, max_order)
        explanation = approximator.approximate(budget=budget, game=game)
        first_order = explanation.get_n_order(order=1)
        first_order_frame = interaction_values_to_frame(first_order, labels)
        pairwise_matrix = pairwise_matrix_from_explanation(explanation, game.n_players)
        audit_frame = coalition_audit_frame(game)
        pair_label, pair_value = strongest_pair(pairwise_matrix, labels)
        notes = build_interpretation_notes(
            first_order_frame,
            pair_label,
            pair_value,
            full_score,
        )

    top_chunk = "No chunk"
    top_score = 0.0
    if not first_order_frame.empty:
        top_row = first_order_frame.iloc[0]
        top_chunk = f"{int(top_row['player'])}. {top_row['chunk']}"
        top_score = float(top_row["attribution"])

    st.markdown(
        f"""
        <div class="verdict">
            <div class="verdict-card">
                <span>Main Evidence</span>
                <strong>{top_chunk}</strong>
                <p>First-order attribution: {top_score:.3f}</p>
            </div>
            <div class="verdict-card">
                <span>Strongest Pair</span>
                <strong>{pair_label}</strong>
                <p>Second-order interaction: {pair_value:.3f}</p>
            </div>
            <div class="verdict-card">
                <span>Answer Support</span>
                <strong>{full_score:.3f}</strong>
                <p>Normalized score with all retrieved chunks visible.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='note-box'><h4>How to read this run</h4><ol>"
        + "".join(f"<li>{note}</li>" for note in notes)
        + "</ol></div>",
        unsafe_allow_html=True,
    )

    evidence_tab, interaction_tab, sentence_tab, audit_tab = st.tabs(
        ["Evidence Ranking", "Chunk Interactions", "Sentence Drilldown", "Coalition Audit"],
    )
    with evidence_tab:
        attr_col, chart_col = st.columns([0.95, 1.05])
        with attr_col:
            st.dataframe(first_order_frame, use_container_width=True, hide_index=True)
        with chart_col:
            if not first_order_frame.empty:
                chart_frame = first_order_frame.set_index("chunk")["attribution"]
                st.bar_chart(chart_frame)

    with interaction_tab:
        heat_col, graph_col = st.columns([1, 1])
        with heat_col:
            st.pyplot(plot_heatmap(pairwise_matrix, labels), clear_figure=True)
        with graph_col:
            st.pyplot(plot_prompt_graph(pairwise_matrix, first_order_frame, labels), clear_figure=True)

    with sentence_tab:
        sentence_chunks, sentence_labels, sentence_meta = build_sentence_players(
            chunks,
        )
        st.caption(
            "Sentence-level drilldown splits all retrieved chunks into sentence players. "
            "Interactions can be within the same chunk or across different chunks; use the "
            "source_chunk column to tell which is which. The plots below reuse the package "
            "sentence visualization functions."
        )
        if len(sentence_chunks) < 2:
            st.warning("Need at least two sentence players for sentence-level interaction analysis.")
        else:
            sentence_game = RAGRetrievalGame(
                question=question,
                target_answer=target_answer,
                chunks=sentence_chunks,
            )
            sentence_budget = budget_for_exactish_demo(sentence_game.n_players)
            sentence_approximator = shapiq.KernelSHAPIQ(
                n=sentence_game.n_players,
                index="k-SII",
                max_order=2,
                random_state=42,
            )
            sentence_explanation = sentence_approximator.approximate(
                budget=sentence_budget,
                game=sentence_game,
            )
            sentence_first_order = sentence_explanation.get_n_order(order=1)

            st.dataframe(sentence_meta, use_container_width=True, hide_index=True)
            bar_col, heatmap_col = st.columns([1, 1])
            with bar_col:
                fig_ax = token_attribution_bar_plot(
                    sentence_first_order,
                    sentence_labels,
                    show=False,
                )
                if fig_ax is not None:
                    fig, ax = fig_ax
                    st.pyplot(polish_sentence_bar_plot(fig, ax), clear_figure=True)
            with heatmap_col:
                fig_ax = sentence_interaction_heatmap(
                    sentence_explanation,
                    sentence_labels,
                    show=False,
                )
                if fig_ax is not None:
                    fig, ax = fig_ax
                    st.pyplot(polish_sentence_heatmap(fig, ax, sentence_meta), clear_figure=True)

    with audit_tab:
        st.caption("Exact scores for empty, single-chunk, pair, and small triple coalitions.")
        st.dataframe(audit_frame, use_container_width=True, hide_index=True)

    st.caption(f"Demo path: `{Path(__file__).parent.relative_to(Path.cwd())}`")


if __name__ == "__main__":
    main()
