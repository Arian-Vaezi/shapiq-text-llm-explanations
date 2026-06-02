"""Streamlit demo for explaining agentic tool-use decisions with shapiq."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle
from sample_data import SAMPLE_TRACES, TOOLS
from scorers import (
    DEFAULT_CANDIDATE_TEMPLATE,
    DEFAULT_LOGPROB_MODEL_ID,
    LexicalToolRouter,
    LexicalToolScorer,
    LLMToolScorer,
    LogProbToolScorer,
    MockLLM,
    ToolChoice,
)

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    import shapiq

SegmentSource = Literal["system", "user"]


DEFAULT_INDEX = "k-SII"
DEFAULT_MAX_ORDER = 2
DELTA_STATUS_THRESHOLD = 0.01
DEFAULT_MOCK_QUERY = "Will it rain in Berlin tomorrow morning?"
MOCK_SYSTEM_SEGMENTS = [
    "Use weather_tool for weather, rain, temperature, forecast, or city-date questions.",
    "Use calculator_tool for exact arithmetic, totals, percentages, and numeric expressions.",
    "Use web_search_tool when the answer depends on current, latest, recent, or live information.",
    "Use no_tool for stable conceptual explanations that do not require external data.",
]
DESCRIPTIVE_CANDIDATE_TEXTS = {
    "weather_tool": "The assistant should use the weather forecast tool.",
    "calculator_tool": "The assistant should use the calculator tool.",
    "web_search_tool": "The assistant should use the web search tool.",
    "no_tool": "The assistant should answer directly without using an external tool.",
}


st.set_page_config(
    page_title="Explaining tool selection",
    page_icon="T",
    layout="wide",
)


@st.cache_resource
def load_logprob_scorer(
    model_id: str,
    candidate_template: str,
    candidate_texts: dict[str, str] | None,
    normalize_by_length: bool,
) -> LogProbToolScorer:
    """Load and cache the optional local HuggingFace logprob scorer."""
    return LogProbToolScorer(
        model_id=model_id,
        candidate_template=candidate_template,
        candidate_texts=candidate_texts,
        normalize_by_length=normalize_by_length,
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
    grid-template-columns: repeat(4, minmax(0, 1fr));
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
.setup-line {
    background: #fffdf8;
    border: 1px solid #ded6c4;
    border-radius: 7px;
    color: #2d2923;
    margin: 0 0 1rem 0;
    padding: 0.8rem 0.95rem;
}
.setup-line code {
    background: #efe8d9;
    border-radius: 4px;
    padding: 0.08rem 0.22rem;
}
@media (max-width: 850px) {
    .scenario-panel,
    .mock-chat,
    .metric-strip,
    .verdict {
        grid-template-columns: 1fr;
    }
    .scenario-hint,
    .verdict-card {
        border-left: 0;
        padding-left: 0;
    }
}
</style>
"""


@dataclass(frozen=True)
class ToolUseSegment:
    """Lightweight prompt segment for rendering the UI before shapiq loads."""

    source: SegmentSource
    label: str
    text: str


def budget_for_demo(n_players: int) -> int:
    """Small interactive default budget."""
    return int(min(2**n_players, max(48, 8 * n_players * math.log2(n_players + 1))))


def clean_key(value: str) -> str:
    """Create a stable Streamlit key fragment."""
    return value.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def truncate_label(value: str, max_length: int = 72) -> str:
    """Shorten long selectbox labels without changing their underlying value."""
    if len(value) <= max_length:
        return value
    return value[: max_length - 1].rstrip() + "..."


def scenario_prompt_label(trace_name: str) -> str:
    """Display a sample scenario by its user prompt instead of its internal name."""
    user_prompt = " ".join(SAMPLE_TRACES[trace_name]["user_segments"])
    return truncate_label(user_prompt)


def format_attribution(value: float, digits: int = 3) -> str:
    """Format signed attribution values for display."""
    if not math.isfinite(value):
        return ""
    threshold = 0.5 * 10 ** (-digits)
    if 0 < abs(value) < threshold:
        return f"{'+' if value > 0 else '-'}<0.001"
    return f"{value:.{digits}f}"


def attribution_ranking_frame(
    attribution_frame: pd.DataFrame,
    *,
    supporting: bool,
    limit: int | None = None,
) -> pd.DataFrame:
    """Build a compact positive or negative attribution ranking."""
    columns = ["segment", "source", "attribution", "preview"]
    if attribution_frame.empty:
        return pd.DataFrame(columns=columns)

    if supporting:
        frame = attribution_frame[attribution_frame["attribution"] > 0].sort_values(
            "attribution",
            ascending=False,
        )
    else:
        frame = attribution_frame[attribution_frame["attribution"] < 0].sort_values(
            "attribution",
            ascending=True,
        )
    if limit is not None:
        frame = frame.head(limit)
    if frame.empty:
        return pd.DataFrame(columns=columns)

    display_frame = frame.copy()
    display_frame["preview"] = display_frame["text"].map(
        lambda text: truncate_label(str(text), max_length=96)
    )
    display_frame["attribution"] = display_frame["attribution"].map(format_attribution)
    return display_frame[columns]


def build_segments(default_segments: list[str], source: str) -> list[ToolUseSegment]:
    """Create fixed demo segments for a prompt source."""
    return [
        ToolUseSegment(source=source, label=f"{source[0].upper()}{idx + 1}", text=text.strip())
        for idx, text in enumerate(default_segments)
        if text.strip()
    ]


def build_coalition_prompt(selected_segments: list[ToolUseSegment]) -> str:
    """Build a coalition prompt without importing the full shapiq game stack."""
    system_lines = [
        f"- {segment.text}" for segment in selected_segments if segment.source == "system"
    ]
    user_lines = [f"- {segment.text}" for segment in selected_segments if segment.source == "user"]
    return (
        "System rules:\n"
        + ("\n".join(system_lines) if system_lines else "(none)")
        + "\n\nUser request:\n"
        + ("\n".join(user_lines) if user_lines else "(none)")
    )


def split_user_request(user_input: str) -> list[str]:
    """Split a custom user request into a few stable explanation segments."""
    cleaned = " ".join(user_input.strip().split())
    if not cleaned:
        return []

    parts = [part.strip(" .?!,;:") for part in re.split(r"[,;?.!]+", cleaned) if part.strip()]
    if len(parts) > 1:
        return parts[:4]

    words = cleaned.split()
    if len(words) <= 6:
        return [cleaned]

    chunk_size = max(2, math.ceil(len(words) / 3))
    return [" ".join(words[idx : idx + chunk_size]) for idx in range(0, len(words), chunk_size)]


def build_mock_trace(user_input: str, choice: ToolChoice) -> dict[str, object]:
    """Create a trace from a mock-router conversation."""
    return {
        "target_tool": choice.tool,
        "system_segments": MOCK_SYSTEM_SEGMENTS,
        "user_segments": split_user_request(user_input),
        "takeaway": (
            "The mock LLM router only chooses a tool. It does not call external APIs or run "
            "the selected tool; shapiq explains the text evidence behind the chosen route."
        ),
    }


def values_to_frame(
    values: shapiq.InteractionValues, segments: list[ToolUseSegment]
) -> pd.DataFrame:
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
                "direction": "positive"
                if float(score) > 0
                else "negative"
                if float(score) < 0
                else "neutral",
                "abs_attribution": abs(float(score)),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values("abs_attribution", ascending=False).drop(columns=["abs_attribution"])


def make_approximator(index: str, n_players: int, max_order: int) -> object:
    """Create a shapiq approximator for the selected index."""
    import shapiq

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
        f"System-rule contribution sums to {source_split.get('system', 0.0):.3f}; "
        f"user-request contribution sums to {source_split.get('user', 0.0):.3f}. "
        "This separates policy pressure from request-trigger pressure."
    )

    if abs(pair_value) < 0.03:
        notes.append(
            "Second-order effects are weak; the decision is mostly explained by individual segments."
        )
    elif pair_value > 0:
        notes.append(
            f"The strongest pair is `{pair_label}` ({pair_value:.3f}). Positive interaction means "
            "the selected index assigns extra shared support to that segment pair."
        )
    else:
        notes.append(
            f"The strongest pair is `{pair_label}` ({pair_value:.3f}). Negative interaction means "
            "the selected index treats the pair as redundant, saturating, or partly conflicting."
        )

    notes.append(
        f"The full-prompt target-tool support score is {full_score:.3f}. "
        "This is still lexical/mock scoring scaffolding until a real local "
        "tool-router scorer is integrated."
    )
    return notes


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
            <h1>Explaining tool selection</h1>
            <p>Inspect which prompt parts support a tool choice.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if "has_run" not in st.session_state:
        st.session_state.has_run = False
    if "result" not in st.session_state:
        st.session_state.result = None
    if "pending_run" not in st.session_state:
        st.session_state.pending_run = False

    mode = st.sidebar.radio(
        "Input",
        ["Example request", "Custom request"],
        index=0,
    )
    with st.sidebar.expander("How it works", expanded=False):
        st.write("Request -> Segmentation -> Remove players -> Tool support score -> Shapley Explanation")
        st.caption(
            "The app checks which prompt parts support the selected tool and then shows "
            "their importance."
        )
    scorer_backend = st.sidebar.selectbox(
        "Scoring method",
        # Only presentation-ready scoring methods are exposed. Other experimental scorers
        # remain in code but are hidden from the UI.
        [
            "Mock model scorer",
            "Keyword baseline",
            "LLM logprob scorer",
        ],
        index=0,
    )
    logprob_model_id = DEFAULT_LOGPROB_MODEL_ID
    candidate_template = DEFAULT_CANDIDATE_TEMPLATE
    candidate_texts = None
    normalize_by_length = True
    if scorer_backend == "LLM logprob scorer":
        with st.sidebar.expander("Logprob scorer settings", expanded=True):
            logprob_model_id = st.text_input("model id", value=DEFAULT_LOGPROB_MODEL_ID)

    router = LexicalToolRouter()
    if mode == "Example request":
        trace_name = st.sidebar.selectbox(
            "Scenario",
            list(SAMPLE_TRACES),
            format_func=scenario_prompt_label,
        )
        trace = SAMPLE_TRACES[trace_name]
        key = clean_key(trace_name)
        default_target = str(trace["target_tool"])
        mock_choice = None
    else:
        st.markdown('<div class="section-label">Request</div>', unsafe_allow_html=True)
        mock_input = st.text_area(
            "Request text",
            value=DEFAULT_MOCK_QUERY,
            height=86,
            help=(
                "This local mock router chooses a tool only. It does not call a real model "
                "or any tool."
            ),
        )
        mock_choice = router.choose_tool(mock_input, TOOLS)
        trace_name = "Custom request"
        trace = build_mock_trace(mock_input, mock_choice)
        key = "mock_llm_router"
        default_target = mock_choice.tool

    target_tool = st.sidebar.selectbox(
        "Tool to explain",
        list(TOOLS),
        index=list(TOOLS).index(default_target),
        key=f"{key}_target_tool",
    )

    system_segments = build_segments(trace["system_segments"], "system")
    user_segments = build_segments(trace["user_segments"], "user")
    segments = system_segments + user_segments
    labels = [segment.label for segment in segments]
    budget = budget_for_demo(len(segments))

    with st.sidebar.expander("More options", expanded=False):
        st.caption(f"index: fixed `{DEFAULT_INDEX}`")
        st.caption(f"max_order: fixed `{DEFAULT_MAX_ORDER}`")
        st.caption(f"budget: `{budget}` auto")
        show_prompt_segments = st.checkbox("show prompt segments", value=False)
        show_value_function_details = st.checkbox("show value function details", value=False)
        show_scoring_prompt_preview = st.checkbox("show scoring prompt preview", value=False)
        show_lexical_comparison = st.checkbox("show keyword comparison", value=False)

    if len(segments) < 2:
        st.warning("Add at least two prompt segments.")
        return

    user_request = " ".join(trace["user_segments"])
    players_text = f"{len(system_segments)} system segments + {len(user_segments)} user segments"
    full_prompt = build_coalition_prompt(segments)
    empty_prompt = build_coalition_prompt([])
    index = DEFAULT_INDEX
    max_order = DEFAULT_MAX_ORDER
    signature = (
        mode,
        trace_name,
        user_request,
        target_tool,
        scorer_backend,
        logprob_model_id,
        candidate_template,
        bool(candidate_texts),
        normalize_by_length,
        show_lexical_comparison,
    )
    if st.session_state.get("result_signature") != signature:
        st.session_state.has_run = False
        st.session_state.result = None
        st.session_state.pending_run = False
        st.session_state.result_signature = signature

    st.markdown('<div class="section-label">Setup</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="scenario-panel">
            <div>
                <span class="scenario-tag">Tool selection</span>
                <h3>{escape(trace_name)}</h3>
                <p>{escape(user_request)}</p>
            </div>
            <div class="scenario-hint">
                <strong>Tool to explain:</strong> {escape(target_tool)}<br>
                <strong>Available tools:</strong> {escape(", ".join(TOOLS))}<br>
                <strong>Players:</strong> {escape(players_text)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    run = False
    if not st.session_state.has_run and not st.session_state.pending_run:
        if st.button("Run explanation", type="primary"):
            st.session_state.pending_run = True
            st.rerun()
        st.info("Enter a prompt and click Run to see the explanation.")
        return

    st.markdown('<div class="section-label">Initial tool suggestion</div>', unsafe_allow_html=True)
    st.caption(
        "This is only a setup preview. Click Run explanation to compute segment attributions."
    )
    if mock_choice is None:
        st.info(f"Example target: `{trace['target_tool']}`. {trace['takeaway']}")
    else:
        router_left, router_right = st.columns([0.85, 1.15])
        with router_left:
            st.metric("Suggested tool", mock_choice.tool, f"{mock_choice.score:.3f}")
            st.caption(mock_choice.reason)
        with router_right:
            score_frame = pd.DataFrame(
                [
                    {"tool": tool, "score": score}
                    for tool, score in sorted(
                        mock_choice.scores.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ]
            )
            st.dataframe(score_frame, use_container_width=True, hide_index=True, height=178)

    st.markdown(
        '<div class="section-label">How the explanation is computed</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="setup-line">
            <strong>Value function:</strong>
            <code>v(S) = score(target tool | selected prompt parts S)</code><br>
            S is a subset of prompt parts / players. For each S, the app builds a
            partial prompt using only those selected parts, and the scorer returns how
            strongly that partial prompt supports the selected target tool.
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Why this is a Shapley game", expanded=False):
        st.markdown(
            """
            Shapley values compare tool-support scores across different prompt-part
            combinations to estimate each prompt part's contribution.

            A positive contribution means the prompt part supports the target tool.
            A negative contribution means it weakens support for the target tool.
            """
        )

    with st.expander("Show prompt segments / players", expanded=show_prompt_segments):
        segment_left, segment_right = st.columns(2)
        with segment_left:
            st.markdown("**System prompt segments**")
            for segment in system_segments:
                st.markdown(
                    (
                        "<div class='segment-box'>"
                        f"<h4>{segment.label}</h4><p>{escape(segment.text)}</p></div>"
                    ),
                    unsafe_allow_html=True,
                )
        with segment_right:
            st.markdown("**User request segments**")
            for segment in user_segments:
                st.markdown(
                    (
                        "<div class='segment-box user'>"
                        f"<h4>{segment.label}</h4><p>{escape(segment.text)}</p></div>"
                    ),
                    unsafe_allow_html=True,
                )

    run = st.session_state.pending_run
    st.session_state.pending_run = False

    try:
        from shapiq.plot import sentence_interaction_heatmap, token_attribution_bar_plot
    except Exception as error:  # noqa: BLE001
        st.error(f"The result exists, but plotting could not be imported: {error}")
        return

    if run:
        try:
            from tool_game import ToolUseGame
        except Exception as error:  # noqa: BLE001
            st.error(
                "The demo controls are ready, but the full shapiq explanation stack "
                f"could not be imported in this local environment: {error}"
            )
            return

    if run:
        llm_scorer = LLMToolScorer(llm=MockLLM())
        lexical_scorer = LexicalToolScorer()
        if scorer_backend == "Keyword baseline":
            primary_scorer = lexical_scorer
            primary_label = "Keyword baseline"
        elif scorer_backend == "LLM logprob scorer":
            with st.spinner(f"Loading logprob scorer `{logprob_model_id}`..."):
                try:
                    primary_scorer = load_logprob_scorer(
                        logprob_model_id,
                        candidate_template,
                        candidate_texts,
                        bool(normalize_by_length),
                    )
                except Exception as error:  # noqa: BLE001
                    st.error(
                        "Could not load the logprob-based scorer. "
                        "Try a smaller causal language model or check your environment. "
                        f"Details: {error}"
                    )
                    return
            primary_label = "LLM logprob scorer"
        else:
            primary_scorer = llm_scorer
            primary_label = "Mock model scorer"

        full_score = primary_scorer.score_batch(
            [full_prompt],
            target_tool=target_tool,
            tool_descriptions=TOOLS,
        )[0]
        empty_score = primary_scorer.score_batch(
            [empty_prompt],
            target_tool=target_tool,
            tool_descriptions=TOOLS,
        )[0]

        with st.spinner("Computing tool-use attributions..."):
            game = ToolUseGame(
                target_tool=target_tool,
                segments=segments,
                scorer=primary_scorer,
                tool_descriptions=TOOLS,
            )
            approximator = make_approximator(index, game.n_players, max_order)
            explanation = approximator.approximate(budget=budget, game=game)
            first_order = explanation.get_n_order(order=1)
            attribution_frame = values_to_frame(first_order, segments)
            pairwise_matrix = pairwise_matrix_from_explanation(explanation, game.n_players)
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
        interpretation_sentence = (
            notes[0] if notes else "No interpretation is available for this run."
        )

        compare_with_lexical = show_lexical_comparison and primary_label != "Keyword baseline"
        llm_debug_outputs = getattr(primary_scorer, "last_debug_outputs", [])
        lexical_result = None
        if compare_with_lexical:
            with st.spinner("Computing lexical baseline comparison..."):
                lexical_game = ToolUseGame(
                    target_tool=target_tool,
                    segments=segments,
                    scorer=lexical_scorer,
                    tool_descriptions=TOOLS,
                )
                lexical_approximator = make_approximator(
                    index,
                    lexical_game.n_players,
                    max_order,
                )
                lexical_explanation = lexical_approximator.approximate(
                    budget=budget,
                    game=lexical_game,
                )
                lexical_first_order = lexical_explanation.get_n_order(order=1)
                lexical_frame = values_to_frame(lexical_first_order, segments)
                lexical_matrix = pairwise_matrix_from_explanation(
                    lexical_explanation,
                    lexical_game.n_players,
                )
                lexical_pair_label, lexical_pair_value = strongest_pair(lexical_matrix, labels)
                lexical_full_score = lexical_scorer.score_batch(
                    [full_prompt],
                    target_tool=target_tool,
                    tool_descriptions=TOOLS,
                )[0]
                lexical_empty_score = lexical_scorer.score_batch(
                    [empty_prompt],
                    target_tool=target_tool,
                    tool_descriptions=TOOLS,
                )[0]
                lexical_top = lexical_frame.iloc[0] if not lexical_frame.empty else None
                lexical_result = {
                    "label": "Keyword baseline",
                    "full_score": lexical_full_score,
                    "empty_score": lexical_empty_score,
                    "top": "No segment"
                    if lexical_top is None
                    else f"{lexical_top['segment']} ({lexical_top['source']})",
                    "top_value": 0.0
                    if lexical_top is None
                    else float(lexical_top["attribution"]),
                    "pair": lexical_pair_label,
                    "pair_value": lexical_pair_value,
                }

        st.session_state.has_run = True
        st.session_state.result = {
            "primary_label": primary_label,
            "full_score": full_score,
            "empty_score": empty_score,
            "first_order": first_order,
            "attribution_frame": attribution_frame,
            "explanation": explanation,
            "pair_label": pair_label,
            "pair_value": pair_value,
            "top_label": top_label,
            "top_score": top_score,
            "interpretation_sentence": interpretation_sentence,
            "compare_with_lexical": compare_with_lexical,
            "llm_debug_outputs": llm_debug_outputs,
            "lexical_result": lexical_result,
            "scoring_prompt": llm_scorer.build_scoring_prompt(
                full_prompt,
                target_tool=target_tool,
                tool_descriptions=TOOLS,
            ),
        }

    result = st.session_state.result
    if result is None:
        st.error(
            "No explanation result is available. Click Run explanation to compute one."
        )
        return
    primary_label = result["primary_label"]
    full_score = result["full_score"]
    empty_score = result["empty_score"]
    first_order = result["first_order"]
    attribution_frame = result["attribution_frame"]
    explanation = result["explanation"]
    pair_label = result["pair_label"]
    pair_value = result["pair_value"]
    top_label = result["top_label"]
    top_score = result["top_score"]
    interpretation_sentence = result["interpretation_sentence"]
    compare_with_lexical = result["compare_with_lexical"]
    llm_debug_outputs = result["llm_debug_outputs"]
    lexical_result = result["lexical_result"]
    delta_support = float(full_score) - float(empty_score)
    if delta_support > DELTA_STATUS_THRESHOLD:
        support_status = "Supported by the prompt"
        support_interpretation = (
            "The complete prompt increases support for the target tool compared with the baseline."
        )
    elif delta_support < -DELTA_STATUS_THRESHOLD:
        support_status = "Reduced by the prompt"
        support_interpretation = (
            "The complete prompt reduces support for the target tool compared with the baseline."
        )
    else:
        support_status = "Neutral / weak evidence"
        support_interpretation = (
            "The complete prompt does not change support much compared with the baseline."
        )
    supporting_frame = attribution_ranking_frame(attribution_frame, supporting=True)
    reducing_frame = attribution_ranking_frame(attribution_frame, supporting=False)

    debug_requested = (
        show_value_function_details
        or show_scoring_prompt_preview
        or compare_with_lexical
        or bool(llm_debug_outputs)
    )
    tab_names = ["Summary", "Attribution", "Interactions"]
    if debug_requested:
        tab_names.append("Debug")
    tabs = st.tabs(tab_names)

    with tabs[0]:
        st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
        st.markdown("**Tool support overview**")
        st.markdown(
            f"""
            <div class="metric-strip">
                <div class="metric-card"><span>Target tool</span><strong>{escape(target_tool)}</strong></div>
                <div class="metric-card"><span>Full support score = v(all segments)</span><strong>{full_score:.3f}</strong></div>
                <div class="metric-card"><span>Baseline / empty-prompt score = v(empty)</span><strong>{empty_score:.3f}</strong></div>
                <div class="metric-card"><span>Delta support</span><strong>{delta_support:.3f}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.info(f"**{support_status}.** {support_interpretation}")

        st.caption("See the Attribution tab for the full first-order segment ranking.")

    with tabs[1]:
        st.markdown("**First-order attribution ranking**")
        st.dataframe(attribution_frame, use_container_width=True, hide_index=True)
        fig_ax = token_attribution_bar_plot(first_order, labels, show=False)
        if fig_ax is not None:
            fig, ax = fig_ax
            st.pyplot(polish_bar(fig, ax), clear_figure=True)

    with tabs[2]:
        st.markdown("**Pairwise interaction heatmap**")
        fig_ax = sentence_interaction_heatmap(explanation, labels, show=False)
        if fig_ax is not None:
            fig, ax = fig_ax
            st.pyplot(polish_heatmap(fig, ax, segments), clear_figure=True)
        st.write(f"Strongest interaction pair: `{pair_label}` ({pair_value:.3f})")

    if debug_requested:
        with tabs[3]:
            if llm_debug_outputs:
                with st.expander("Model output diagnostics", expanded=False):
                    displayed_debug_outputs = llm_debug_outputs[:10]
                    if (
                        primary_label == "LLM logprob scorer"
                        and displayed_debug_outputs
                        and all(row.get("used_fallback") for row in displayed_debug_outputs)
                    ):
                        st.warning(
                            "The local model did not return numeric scores for this run, "
                            "so the keyword baseline was used as fallback."
                        )
                    debug_frame = pd.DataFrame(displayed_debug_outputs)
                    debug_columns = [
                        "raw_output",
                        "parsed_score",
                        "used_fallback",
                        "fallback_score",
                        "candidate_tools",
                        "candidate_continuations",
                        "candidate_logprobs",
                        "candidate_probs",
                        "final_score",
                        "prompt_preview",
                    ]
                    st.dataframe(
                        debug_frame[[column for column in debug_columns if column in debug_frame]],
                        use_container_width=True,
                        hide_index=True,
                    )
            if lexical_result is not None:
                st.markdown("**Scorer comparison**")
                comparison_rows = [
                    {
                        "scorer": primary_label,
                        "full_score": full_score,
                        "empty_score": empty_score,
                        "top_segment": top_label,
                        "top_attribution": top_score,
                        "strongest_pair": pair_label,
                        "pair_value": pair_value,
                    },
                    {
                        "scorer": lexical_result["label"],
                        "full_score": lexical_result["full_score"],
                        "empty_score": lexical_result["empty_score"],
                        "top_segment": lexical_result["top"],
                        "top_attribution": lexical_result["top_value"],
                        "strongest_pair": lexical_result["pair"],
                        "pair_value": lexical_result["pair_value"],
                    },
                ]
                comparison_frame = pd.DataFrame(comparison_rows)
                st.dataframe(comparison_frame, use_container_width=True, hide_index=True)
            if show_scoring_prompt_preview:
                st.markdown("**Scoring prompt preview**")
                st.code(result["scoring_prompt"], language="text")
            if show_value_function_details:
                st.markdown("**Value function details**")
                st.write(f"Index: `{index}`")
                st.write(f"Max order: `{max_order}`")
                st.write(f"Budget: `{budget}`")
                st.write("Full coalition prompt:")
                st.code(full_prompt, language="text")
                st.write("Empty coalition prompt:")
                st.code(empty_prompt, language="text")

    st.caption(f"Demo path: `{Path(__file__).parent.relative_to(Path.cwd())}`")


if __name__ == "__main__":
    main()
