"""Streamlit demo for explaining agentic tool-use decisions with shapiq."""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle

import shapiq
from shapiq.plot import sentence_interaction_heatmap, token_attribution_bar_plot

from sample_data import SAMPLE_TRACES, TOOLS
from tool_game import ToolUseGame, ToolUseSegment, budget_for_demo


st.set_page_config(
    page_title="Agentic Tool-Use Explanation",
    page_icon="T",
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
.tool-title {
    border-bottom: 1px solid #252525;
    margin-bottom: 0.85rem;
    padding-bottom: 0.75rem;
}
.tool-title h1 {
    color: #1f1f1f;
    font-family: Georgia, serif;
    font-size: 2.15rem;
    font-weight: 700;
    letter-spacing: 0;
    line-height: 1.08;
    margin: 0;
}
.tool-title p {
    color: #59544a;
    font-size: 0.96rem;
    margin: 0.45rem 0 0 0;
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
.verdict {
    background: #1f2a28;
    border-radius: 7px;
    color: #f7f1e4;
    display: grid;
    gap: 1rem;
    grid-template-columns: 1fr 1fr 1fr;
    margin: 0.4rem 0 1rem 0;
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
    font-size: 1.12rem;
    line-height: 1.2;
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
</style>
"""


def clean_key(value: str) -> str:
    """Create a stable Streamlit key fragment."""
    return value.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def segment_editor(default_segments: list[str], prefix: str, source: str) -> list[ToolUseSegment]:
    """Render editable segment inputs."""
    # TODO(final-demo): Manual segmentation is good for a controlled demo.
    # For a model-backed demo, consider adding segmentation modes such as:
    # sentence split, word split, tool-schema split, or message-role split.
    st.sidebar.subheader(f"{source.title()} Segments")
    segment_count = st.sidebar.slider(
        f"Number of {source} segments",
        1,
        8,
        len(default_segments),
        key=f"{prefix}_{source}_count",
    )
    segments = []
    for idx in range(segment_count):
        fallback = default_segments[idx] if idx < len(default_segments) else ""
        text = st.sidebar.text_area(
            f"{source.title()} segment {idx + 1}",
            fallback,
            height=72,
            key=f"{prefix}_{source}_{idx}",
        )
        if text.strip():
            label = f"{source[0].upper()}{idx + 1}"
            segments.append(ToolUseSegment(source=source, label=label, text=text.strip()))
    return segments


def values_to_frame(values: shapiq.InteractionValues, segments: list[ToolUseSegment]) -> pd.DataFrame:
    """Convert first-order values to a display frame."""
    rows = []
    for interaction, score in values.dict_values.items():
        if len(interaction) != 1:
            continue
        idx = interaction[0]
        segment = segments[idx]
        rows.append(
            {
                "segment": segment.label,
                "source": segment.source,
                "text": segment.text,
                "attribution": float(score),
                "abs_attribution": abs(float(score)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("abs_attribution", ascending=False).drop(columns=["abs_attribution"])


def make_approximator(index: str, n_players: int, max_order: int):
    """Create a shapiq approximator for the selected index."""
    if index == "SV":
        return shapiq.KernelSHAP(n=n_players, random_state=42)
    if index == "STII":
        return shapiq.PermutationSamplingSTII(n=n_players, max_order=max_order, random_state=42)
    if index == "FSII":
        return shapiq.RegressionFSII(n=n_players, max_order=max_order, random_state=42)
    return shapiq.KernelSHAPIQ(n=n_players, index=index, max_order=max_order, random_state=42)


def pairwise_matrix_from_explanation(
    explanation: shapiq.InteractionValues,
    n_players: int,
) -> pd.DataFrame:
    """Extract second-order values as a dense matrix."""
    if explanation.max_order < 2:
        return pd.DataFrame([[0.0] * n_players for _ in range(n_players)])
    return pd.DataFrame(explanation.get_n_order_values(2))


def strongest_pair(matrix: pd.DataFrame, labels: list[str]) -> tuple[str, float]:
    """Return the strongest non-diagonal second-order interaction."""
    if matrix.shape[0] < 2:
        return "No pair", 0.0
    best_pair = (0, 1)
    best_value = float(matrix.iloc[0, 1])
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            value = float(matrix.iloc[i, j])
            if abs(value) > abs(best_value):
                best_pair = (i, j)
                best_value = value
    return f"{labels[best_pair[0]]} + {labels[best_pair[1]]}", best_value


def build_interpretation_notes(
    attribution_frame: pd.DataFrame,
    pair_label: str,
    pair_value: float,
    full_score: float,
) -> list[str]:
    """Create plain-language interpretation bullets for the current run."""
    if attribution_frame.empty:
        return ["No first-order attribution was returned for this run."]

    top = attribution_frame.iloc[0]
    notes = [
        (
            f"Start with segment `{top['segment']}` from the {top['source']} prompt. "
            f"It has the largest individual attribution ({top['attribution']:.3f}) for the target tool."
        )
    ]

    source_split = attribution_frame.groupby("source")["attribution"].sum().to_dict()
    notes.append(
        (
            f"System-rule contribution sums to {source_split.get('system', 0.0):.3f}; "
            f"user-request contribution sums to {source_split.get('user', 0.0):.3f}. "
            "This separates policy pressure from request-trigger pressure."
        )
    )

    if abs(pair_value) < 0.03:
        notes.append("Second-order effects are weak; the decision is mostly explained by individual segments.")
    elif pair_value > 0:
        notes.append(
            (
                f"The strongest pair is `{pair_label}` ({pair_value:.3f}). Positive interaction means "
                "the selected index assigns extra shared support to that segment pair."
            )
        )
    else:
        notes.append(
            (
                f"The strongest pair is `{pair_label}` ({pair_value:.3f}). Negative interaction means "
                "the selected index treats the pair as redundant, saturating, or partly conflicting."
            )
        )

    notes.append(
        (
            f"The full-prompt target-tool support score is {full_score:.3f}. "
            "This is still lexical scaffolding until `ToolUseGame.score_segments` is replaced "
            "with an LLM/tool-router scorer."
        )
    )
    return notes


def coalition_audit_frame(game: ToolUseGame, segments: list[ToolUseSegment]) -> pd.DataFrame:
    """Show exact scores for small coalitions."""
    rows = []
    n_players = game.n_players
    for size in range(0, min(n_players, 3) + 1):
        for combo in itertools.combinations(range(n_players), size):
            coalition = np.zeros((1, n_players), dtype=bool)
            coalition[0, list(combo)] = True
            score = float(game(coalition)[0])
            rows.append(
                {
                    "coalition": ", ".join(segments[idx].label for idx in combo) or "empty",
                    "selected_segments": size,
                    "target_tool_score": score,
                }
            )
    return pd.DataFrame(rows).sort_values("target_tool_score", ascending=False)


def polish_bar(fig: plt.Figure, ax: plt.Axes) -> plt.Figure:
    """Make package bar plot fit the Streamlit layout."""
    fig.set_size_inches(6.2, 3.7)
    ax.set_title("", loc="center")
    ax.set_title("Prompt Segment Attribution", loc="left", fontsize=12, pad=8)
    ax.set_xlabel("Target-tool attribution")
    ax.grid(axis="x", color="#d7dfdf", alpha=0.65, linewidth=0.8)
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


def polish_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    segments: list[ToolUseSegment],
) -> plt.Figure:
    """Make package heatmap fit the Streamlit layout."""
    fig.set_size_inches(6.0, 4.7)
    ax.set_title("", loc="center")
    ax.set_title("Prompt Segment Interaction Heatmap", loc="left", fontsize=12, pad=8)
    ax.tick_params(axis="x", labelrotation=30)

    group_ranges = []
    start = 0
    while start < len(segments):
        source = segments[start].source
        end = start
        while end + 1 < len(segments) and segments[end + 1].source == source:
            end += 1
        group_ranges.append((source, start, end))
        start = end + 1

    colors = {"system": "#1f554c", "user": "#b15d3b"}
    for source, start, end in group_ranges:
        size = end - start + 1
        ax.add_patch(
            Rectangle(
                (start - 0.5, start - 0.5),
                size,
                size,
                fill=False,
                edgecolor=colors.get(source, "#1f554c"),
                linewidth=2.2,
                zorder=5,
            )
        )

    ax.text(
        0,
        -0.2,
        "Outlined blocks group system-rule segments and user-request segments.",
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
        <div class="tool-title">
            <h1>Agentic Tool-Use Explanation</h1>
            <p>Attribute an agent's decision to call a tool back to system rules and user-request segments.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    trace_name = st.sidebar.selectbox("Scenario", list(SAMPLE_TRACES))
    trace = SAMPLE_TRACES[trace_name]
    key = clean_key(trace_name)

    st.sidebar.subheader("Target Tool")
    target_tool = st.sidebar.selectbox(
        "Explain decision for",
        list(TOOLS),
        index=list(TOOLS).index(trace["target_tool"]),
        key=f"{key}_target_tool",
    )
    system_segments = segment_editor(trace["system_segments"], key, "system")
    user_segments = segment_editor(trace["user_segments"], key, "user")
    segments = system_segments + user_segments

    st.sidebar.subheader("Explanation Settings")
    index = st.sidebar.selectbox("Interaction index", ["k-SII", "STII", "FSII", "SV"], index=0)
    max_order = 1 if index == "SV" else st.sidebar.slider("Max interaction order", 2, 3, 2)
    budget = st.sidebar.slider(
        "Approximation budget",
        16,
        512,
        budget_for_demo(len(segments)),
        step=8,
    )

    if len(segments) < 2:
        st.warning("Add at least two prompt segments.")
        return

    game = ToolUseGame(target_tool=target_tool, segments=segments)
    labels = [segment.label for segment in segments]
    full_score = float(game(game.grand_coalition)[0])
    empty_score = float(game(game.empty_coalition)[0])

    st.markdown(
        f"""
        <div class="scenario-panel">
            <div>
                <span class="scenario-tag">Agentic tool-use</span>
                <h3>{trace_name}</h3>
                <p>{trace["takeaway"]}</p>
            </div>
            <div class="scenario-hint">
                <strong>Target tool:</strong> {target_tool}<br>
                <strong>Available tools:</strong> {", ".join(TOOLS)}<br>
                <strong>Players:</strong> system and user prompt segments
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_html = f"""
    <div class="metric-strip">
        <div class="metric-card"><span>Segments</span><strong>{len(segments)}</strong></div>
        <div class="metric-card"><span>Empty Prompt</span><strong>{empty_score:.3f}</strong></div>
        <div class="metric-card"><span>Full Prompt</span><strong>{full_score:.3f}</strong></div>
    </div>
    """
    st.markdown(metric_html, unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown('<div class="section-label">Input Trace</div>', unsafe_allow_html=True)
        st.markdown("**System prompt segments**")
        for segment in system_segments:
            st.markdown(
                f"<div class='segment-box'><h4>{segment.label}</h4><p>{segment.text}</p></div>",
                unsafe_allow_html=True,
            )
        st.markdown("**User request segments**")
        for segment in user_segments:
            st.markdown(
                f"<div class='segment-box user'><h4>{segment.label}</h4><p>{segment.text}</p></div>",
                unsafe_allow_html=True,
            )
    with right:
        st.markdown('<div class="section-label">Game Setup</div>', unsafe_allow_html=True)
        st.markdown(
            """
            This demo creates one player per prompt segment. A coalition means
            “show only these system/user segments,” then the scorer estimates how
            strongly the visible prompt supports the target tool call.
            """
        )
        with st.expander("Prompt preview for a model-backed scorer"):
            preview_system = "\n".join(f"- {segment.text}" for segment in system_segments)
            preview_user = " ".join(segment.text for segment in user_segments)
            st.code(
                "Available tools:\n"
                + "\n".join(f"- {name}: {desc}" for name, desc in TOOLS.items())
                + f"\n\nSystem rules:\n{preview_system}\n\nUser request:\n{preview_user}\n\n"
                + "Choose exactly one tool or no_tool.",
                language="text",
            )

    run = st.button("Run explanation", type="primary", use_container_width=True)
    if not run:
        st.info("Choose a scenario and run the explanation.")
        return

    with st.spinner("Computing tool-use attributions..."):
        approximator = make_approximator(index, game.n_players, max_order)
        explanation = approximator.approximate(budget=budget, game=game)
        first_order = explanation.get_n_order(order=1)
        attribution_frame = values_to_frame(first_order, segments)
        pairwise_matrix = pairwise_matrix_from_explanation(explanation, game.n_players)
        audit_frame = coalition_audit_frame(game, segments)
        pair_label, pair_value = strongest_pair(pairwise_matrix, labels)
        notes = build_interpretation_notes(
            attribution_frame,
            pair_label,
            pair_value,
            full_score,
        )

    top = attribution_frame.iloc[0] if not attribution_frame.empty else None
    top_label = "No segment" if top is None else f"{top['segment']} ({top['source']})"
    top_score = 0.0 if top is None else float(top["attribution"])
    source_split = attribution_frame.groupby("source")["attribution"].sum().to_dict()

    st.markdown(
        f"""
        <div class="verdict">
            <div class="verdict-card">
                <span>Decision Explained</span>
                <strong>{target_tool}</strong>
            </div>
            <div class="verdict-card">
                <span>Main Driver</span>
                <strong>{top_label}</strong>
                <p>{top_score:.3f}</p>
            </div>
            <div class="verdict-card">
                <span>System vs User</span>
                <strong>S: {source_split.get("system", 0.0):.3f} / U: {source_split.get("user", 0.0):.3f}</strong>
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

    ranking_tab, interaction_tab, audit_tab = st.tabs(
        ["Attribution Ranking", "Segment Interactions", "Coalition Audit"],
    )
    with ranking_tab:
        table_col, chart_col = st.columns([1.08, 0.92])
        with table_col:
            st.dataframe(attribution_frame, use_container_width=True, hide_index=True)
        with chart_col:
            fig_ax = token_attribution_bar_plot(first_order, labels, show=False)
            if fig_ax is not None:
                fig, ax = fig_ax
                st.pyplot(polish_bar(fig, ax), clear_figure=True)

    with interaction_tab:
        fig_ax = sentence_interaction_heatmap(explanation, labels, show=False)
        if fig_ax is not None:
            fig, ax = fig_ax
            st.pyplot(polish_heatmap(fig, ax, segments), clear_figure=True)

    with audit_tab:
        st.caption("Exact target-tool scores for empty, single-segment, pair, and small triple coalitions.")
        st.dataframe(audit_frame, use_container_width=True, hide_index=True)

    st.caption(f"Demo path: `{Path(__file__).parent.relative_to(Path.cwd())}`")


if __name__ == "__main__":
    main()
