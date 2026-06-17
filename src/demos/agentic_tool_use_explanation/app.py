"""Streamlit demo for explaining agentic tool-use decisions with shapiq."""

from __future__ import annotations

import importlib.util
import itertools
import math
import os
import sys
import types
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Rectangle
from gemini_agent import list_available_gemini_models, run_gemini_tool_inference
from groq_agent import run_groq_tool_inference
from hf_router import DEFAULT_LOCAL_HF_ROUTER_MODEL_ID, LocalHFRouter
from router_scorers import (
    DEFAULT_GROQ_ROUTER_MODEL_ID,
    DEFAULT_GROQ_SOFT_VOTE_MAX_RETRIES,
    DEFAULT_GROQ_SOFT_VOTE_N_SAMPLES,
    DEFAULT_GROQ_SOFT_VOTE_TEMPERATURE,
    GroqDeterministicRouterScorer,
    GroqSoftVoteToolScorer,
    ToolTrajectory,
    TrajectoryArgumentMatchScorer,
    build_groq_inference_trajectory_provider,
)
from sample_data import SAMPLE_TRACES, TOOLS
from scorers import (
    DEFAULT_CANDIDATE_TEMPLATE,
    DEFAULT_LOGPROB_MODEL_ID,
    LexicalToolScorer,
    LLMToolScorer,
    LogProbToolScorer,
    MockLLM,
    ToolChoice,
)
from semantic_segmenter import SemanticSegmenter
from tool_schemas import get_executable_tool_schemas

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
    "no_tool": "The assistant should answer directly without calling a tool.",
}
TEXT_PLOT_PACKAGE = "_agentic_text_plot"


st.set_page_config(
    page_title="Explaining tool selection",
    page_icon="T",
    layout="wide",
)


@st.cache_resource
def load_local_hf_router(
    model_name: str,
    max_new_tokens: int,
    *,
    trust_remote_code: bool,
) -> LocalHFRouter:
    """Load and cache the optional local HuggingFace router."""
    return LocalHFRouter(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        trust_remote_code=trust_remote_code,
    )


@st.cache_resource
def load_logprob_scorer(
    model_id: str,
    candidate_template: str,
    candidate_texts: dict[str, str] | None,
    *,  # future-proof: make scoring-related args keyword-only
    normalize_by_length: bool,
) -> LogProbToolScorer:
    """Load and cache the optional local HuggingFace logprob scorer."""
    return LogProbToolScorer(
        model_id=model_id,
        candidate_template=candidate_template,
        candidate_texts=candidate_texts,
        normalize_by_length=normalize_by_length,
        max_pairs_per_batch=1,
    )


@st.cache_resource
def load_semantic_segmenter(
    threshold: float,
    window: int,
    min_segment_words: int,
) -> SemanticSegmenter:
    """Load and cache the semantic segmenter model."""
    return SemanticSegmenter(
        device="auto",
        threshold=threshold,
        window=window,
        min_segment_words=min_segment_words,
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


def format_tool_context(tool_descriptions: dict[str, str]) -> str:
    """Render tool definitions as fixed prompt context."""
    return "\n".join(
        f"- {tool_name}: {description}" for tool_name, description in tool_descriptions.items()
    )


def build_system_prompt(system_segments: list[ToolUseSegment]) -> str:
    """Render system prompt segments as fixed prompt context."""
    return "\n".join(f"- {segment.text}" for segment in system_segments)


def build_coalition_prompt(
    selected_user_segments: list[ToolUseSegment],
    *,
    system_prompt: str,
    tool_context: str,
) -> str:
    """Build a coalition prompt with fixed context and selected user-request segments."""
    user_request = " ".join(segment.text.strip() for segment in selected_user_segments)
    return (
        f"{system_prompt.strip()}\n\n"
        f"Available tools:\n{tool_context.strip()}\n\n"
        f"User request:\n{user_request}\n\n"
        "Assistant:"
    )


def segment_user_request(
    segmenter: SemanticSegmenter,
    user_request: str,
) -> tuple[list[str], list[dict[str, object]]]:
    """Segment only the user request; fixed context is not a Shapley player."""
    return segmenter.segment_with_debug(user_request)


def choose_tool_with_scorer(
    scorer: object,
    prompt: str,
    *,
    tool_descriptions: dict[str, str],
) -> ToolChoice:
    """Choose the highest-scoring candidate tool with the selected scorer."""
    scores = {
        tool_name: float(
            scorer.score_batch(
                [prompt],
                target_tool=tool_name,
                tool_descriptions=tool_descriptions,
            )[0]
        )
        for tool_name in tool_descriptions
    }
    selected_tool = max(scores, key=scores.get)
    return ToolChoice(
        tool=selected_tool,
        score=scores[selected_tool],
        reason="Highest preview score from the selected scoring method.",
        scores=scores,
    )


def build_scoring_prompt_preview(
    scorer: object,
    prompt: str,
    *,
    target_tool: str,
    tool_descriptions: dict[str, str],
) -> str | None:
    """Return the actual scorer's debug prompt preview when available."""
    build_scoring_prompt = getattr(scorer, "build_scoring_prompt", None)
    if not callable(build_scoring_prompt):
        return None
    return build_scoring_prompt(
        prompt,
        target_tool=target_tool,
        tool_descriptions=tool_descriptions,
    )


def build_mock_trace(user_input: str) -> dict[str, object]:
    """Create a trace for a custom request before target-tool selection."""
    return {
        "system_segments": MOCK_SYSTEM_SEGMENTS,
        "user_segments": [" ".join(user_input.strip().split())],
        "takeaway": (
            "The setup preview chooses a tool from the full fixed context and request. It does "
            "not call external APIs or run the selected tool; shapiq explains the text evidence "
            "behind the selected route."
        ),
    }


def values_to_frame(
    values: shapiq.InteractionValues, segments: list[ToolUseSegment]
) -> pd.DataFrame:
    """Convert first-order values to a display frame."""
    rows = []
    value_items = getattr(values, "dict_values", {}).items()
    for interaction, score in value_items:
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


@dataclass
class DemoInteractionValues:
    """Minimal interaction-values object for the local fallback path."""

    first_order: list[float]
    second_order: pd.DataFrame
    index: str = "SV"

    @property
    def max_order(self) -> int:
        return 2

    @property
    def n_players(self) -> int:
        return len(self.first_order)

    @property
    def dict_values(self) -> dict[tuple[int, ...], float]:
        values = {(idx,): value for idx, value in enumerate(self.first_order)}
        for left in range(self.n_players):
            for right in range(left + 1, self.n_players):
                values[(left, right)] = float(self.second_order.iloc[left, right])
        return values

    def __getitem__(self, interaction: tuple[int, ...]) -> float:
        return self.dict_values.get(tuple(sorted(interaction)), 0.0)

    def get_n_order(self, order: int) -> DemoInteractionValues:
        if order == 1:
            empty_matrix = pd.DataFrame([[0.0] * self.n_players for _ in range(self.n_players)])
            return DemoInteractionValues(self.first_order, empty_matrix, self.index)
        if order == 2:
            return DemoInteractionValues([0.0] * self.n_players, self.second_order, self.index)
        empty_matrix = pd.DataFrame([[0.0] * self.n_players for _ in range(self.n_players)])
        return DemoInteractionValues([0.0] * self.n_players, empty_matrix, self.index)

    def get_n_order_values(self, order: int) -> np.ndarray:
        if order == 1:
            return np.asarray(self.first_order, dtype=float)
        if order == 2:
            return self.second_order.to_numpy(dtype=float)
        return np.zeros(tuple([self.n_players] * order), dtype=float)


class ExactFallbackApproximator:
    """Exact local Shapley fallback used when shapiq cannot import optional C extensions."""

    def __init__(self, *, n: int, index: str, max_order: int) -> None:
        self.n = n
        self.index = index
        self.max_order = max_order

    def approximate(self, *, budget: int, game: object) -> DemoInteractionValues:
        del budget
        values = self._evaluate_all_coalitions(game)
        first_order = [self._shapley_value(player, values) for player in range(self.n)]
        second_order = self._pairwise_synergy(values)
        return DemoInteractionValues(first_order, pd.DataFrame(second_order), self.index)

    def _evaluate_all_coalitions(self, game: object) -> dict[int, float]:
        coalitions = []
        masks = []
        for mask in range(1 << self.n):
            masks.append(mask)
            coalitions.append([(mask >> player) & 1 == 1 for player in range(self.n)])
        scores = game.value_function(np.asarray(coalitions, dtype=bool))
        return {mask: float(score) for mask, score in zip(masks, scores, strict=True)}

    def _shapley_value(self, player: int, values: dict[int, float]) -> float:
        other_players = [idx for idx in range(self.n) if idx != player]
        score = 0.0
        for size in range(self.n):
            weight = (
                math.factorial(size) * math.factorial(self.n - size - 1) / math.factorial(self.n)
            )
            for subset in itertools.combinations(other_players, size):
                mask = sum(1 << idx for idx in subset)
                score += weight * (values[mask | (1 << player)] - values[mask])
        return score

    def _pairwise_synergy(self, values: dict[int, float]) -> list[list[float]]:
        full_mask = (1 << self.n) - 1
        matrix = [[0.0] * self.n for _ in range(self.n)]
        for left in range(self.n):
            for right in range(left + 1, self.n):
                without_pair = full_mask & ~(1 << left) & ~(1 << right)
                without_left = full_mask & ~(1 << left)
                without_right = full_mask & ~(1 << right)
                synergy = (
                    values[full_mask]
                    - values[without_left]
                    - values[without_right]
                    + values[without_pair]
                )
                matrix[left][right] = synergy
                matrix[right][left] = synergy
        return matrix


def make_approximator(index: str, n_players: int, max_order: int) -> object:
    """Create a shapiq approximator for the selected index."""
    try:
        import shapiq
    except Exception:  # noqa: BLE001
        return ExactFallbackApproximator(n=n_players, index=index, max_order=max_order)

    try:
        if index == "SV":
            return shapiq.KernelSHAP(n=n_players, random_state=42)
        if index == "STII":
            return shapiq.PermutationSamplingSTII(
                n=n_players,
                max_order=max_order,
                random_state=42,
            )
        if index == "FSII":
            return shapiq.RegressionFSII(n=n_players, max_order=max_order, random_state=42)
        return shapiq.KernelSHAPIQ(
            n=n_players,
            index=index,
            max_order=max_order,
            random_state=42,
        )
    except Exception:  # noqa: BLE001
        return ExactFallbackApproximator(n=n_players, index=index, max_order=max_order)


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
            f"Start with user segment `{top['segment']}`. "
            "It has the largest individual attribution "
            f"({top['attribution']:.3f}) for the target tool."
        )
    ]

    total_user_attribution = float(attribution_frame["attribution"].sum())
    notes.append(
        f"User-request contribution sums to {total_user_attribution:.3f}. "
        "The system prompt and tool definitions remain fixed context for every coalition."
    )

    if abs(pair_value) < 0.03:
        notes.append(
            "Second-order effects are weak; the decision is mostly explained by "
            "individual segments."
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
    ax.set_title("User Request Segment Attribution", loc="left", fontsize=12, pad=8)
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
    ax.set_title("User Request Segment Interaction Heatmap", loc="left", fontsize=12, pad=8)
    ax.tick_params(axis="x", labelrotation=30)

    if segments:
        ax.add_patch(
            Rectangle(
                (-0.5, -0.5),
                len(segments),
                len(segments),
                fill=False,
                edgecolor="#b15d3b",
                linewidth=2.2,
                zorder=5,
            )
        )
    fig.tight_layout()
    return fig


def load_text_plotters() -> tuple[object | None, object | None, str | None]:
    """Load text plotting helpers without importing the full shapiq plotting package."""
    try:
        module = load_sentence_plot_module()
    except Exception as error:  # noqa: BLE001
        return None, None, str(error)
    return (
        module.token_attribution_bar_plot,
        module.sentence_interaction_heatmap,
        None,
    )


def load_sentence_plot_module() -> types.ModuleType:
    """Load shapiq.plot.sentence directly to avoid optional tree C extensions."""
    plot_dir = Path(__file__).resolve().parents[2] / "shapiq" / "plot"
    package = sys.modules.get(TEXT_PLOT_PACKAGE)
    if package is None:
        package = types.ModuleType(TEXT_PLOT_PACKAGE)
        package.__path__ = [str(plot_dir)]  # type: ignore[attr-defined]
        sys.modules[TEXT_PLOT_PACKAGE] = package

    config_name = f"{TEXT_PLOT_PACKAGE}._config"
    if config_name not in sys.modules:
        config_spec = importlib.util.spec_from_file_location(
            config_name,
            plot_dir / "_config.py",
        )
        if config_spec is None or config_spec.loader is None:
            msg = "Could not load shapiq sentence plot color configuration."
            raise ImportError(msg)
        config_module = importlib.util.module_from_spec(config_spec)
        sys.modules[config_name] = config_module
        config_spec.loader.exec_module(config_module)

    sentence_name = f"{TEXT_PLOT_PACKAGE}.sentence"
    sentence_module = sys.modules.get(sentence_name)
    if sentence_module is not None:
        return sentence_module

    sentence_spec = importlib.util.spec_from_file_location(
        sentence_name,
        plot_dir / "sentence.py",
    )
    if sentence_spec is None or sentence_spec.loader is None:
        msg = "Could not load shapiq sentence plotting helpers."
        raise ImportError(msg)
    sentence_module = importlib.util.module_from_spec(sentence_spec)
    sys.modules[sentence_name] = sentence_module
    sentence_spec.loader.exec_module(sentence_module)
    return sentence_module


def show_fallback_attribution_chart(attribution_frame: pd.DataFrame) -> None:
    """Render a small Streamlit-native attribution chart when matplotlib plots are unavailable."""
    if attribution_frame.empty:
        st.info("No first-order attributions are available to plot.")
        return
    chart_frame = attribution_frame[["segment", "attribution"]].copy()
    chart_frame = chart_frame.sort_values("attribution").set_index("segment")
    st.bar_chart(chart_frame, use_container_width=True)


def show_fallback_interaction_table(pairwise_matrix: pd.DataFrame, labels: list[str]) -> None:
    """Render pairwise interactions as a table when the heatmap helper is unavailable."""
    fallback_matrix = pairwise_matrix.copy()
    fallback_matrix.index = labels
    fallback_matrix.columns = labels
    st.dataframe(fallback_matrix, use_container_width=True)


def display_demo_path() -> str:
    """Return a stable demo path caption for Streamlit and test runners."""
    demo_path = Path(__file__).resolve().parent
    try:
        return str(demo_path.relative_to(Path.cwd()))
    except ValueError:
        return str(demo_path)


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="tool-title">
            <h1>Explaining tool selection</h1>
            <p>Inspect which user-request parts support a tool choice.</p>
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
    if "agentic_inferred_tool" not in st.session_state:
        st.session_state["agentic_inferred_tool"] = None
    if "agentic_inference_result" not in st.session_state:
        st.session_state["agentic_inference_result"] = None
    if "agentic_inference_signature" not in st.session_state:
        st.session_state["agentic_inference_signature"] = None
    if "agentic_request_text" not in st.session_state:
        st.session_state["agentic_request_text"] = DEFAULT_MOCK_QUERY

    example_placeholder = "Choose an example..."
    pending_example_request = st.session_state.pop("agentic_pending_example_request", None)
    if pending_example_request is not None:
        st.session_state["agentic_request_text"] = pending_example_request
        st.session_state["agentic_inferred_tool"] = None
        st.session_state["agentic_inference_result"] = None
        st.session_state.has_run = False
        st.session_state.result = None
        st.session_state.result_signature = None
        st.session_state["agentic_try_example_select"] = example_placeholder

    with st.sidebar.expander("How it works", expanded=False):
        st.write(
            "Request -> Segmentation -> Remove players -> Tool support score "
            "-> Shapley Explanation"
        )
        st.caption(
            "The app keeps system/tool context fixed, removes user-request players, "
            "and then shows their importance."
        )

    user_request = st.session_state["agentic_request_text"]
    trace_name = "Custom request"
    trace = build_mock_trace(user_request)
    system_segments = build_segments(trace["system_segments"], "system")
    system_prompt = build_system_prompt(system_segments)
    tool_context = format_tool_context(TOOLS)

    current_inference_signature = (
        user_request,
        system_prompt,
        tool_context,
    )
    if st.session_state.get("agentic_inference_signature") != current_inference_signature:
        st.session_state["agentic_inference_signature"] = current_inference_signature
        st.session_state["agentic_inferred_tool"] = None
        st.session_state["agentic_inference_result"] = None
        st.session_state["agentic_inference_backend"] = None
        st.session_state["agentic_inference_model"] = None

    inference_tab, explanation_tab = st.tabs(["Inference", "Explanation"])

    with inference_tab:
        st.markdown('<div class="section-label">Inference</div>', unsafe_allow_html=True)
        st.markdown("### User request")
        st.text_area(
            "User request",
            height=96,
            key="agentic_request_text",
            label_visibility="collapsed",
            help=(
                "This preview chooses a tool from the fixed context and request. "
                "It does not call the selected tool."
            ),
        )
        user_request = st.session_state["agentic_request_text"]
        trace = build_mock_trace(user_request)
        system_segments = build_segments(trace["system_segments"], "system")
        system_prompt = build_system_prompt(system_segments)
        tool_context = format_tool_context(TOOLS)

        def apply_selected_example() -> None:
            selected = st.session_state.get("agentic_try_example_select")
            if selected and selected != example_placeholder:
                example_request = " ".join(SAMPLE_TRACES[selected]["user_segments"])
                st.session_state["agentic_pending_example_request"] = example_request

        example_options = [example_placeholder] + list(SAMPLE_TRACES)
        st.selectbox(
            "Try example",
            example_options,
            format_func=lambda name: name
            if name == example_placeholder
            else scenario_prompt_label(name),
            key="agentic_try_example_select",
            on_change=apply_selected_example,
        )
        if st.session_state.get("agentic_pending_example_request") is not None:
            st.rerun()

        with st.expander("Inference settings", expanded=False):
            inference_backend = st.selectbox(
                "Inference backend",
                ["Groq", "Gemini", "HF local"],
                index=0,
                key="agentic_inference_backend_choice",
            )
            if inference_backend == "Groq":
                inference_model_name = st.text_input(
                    "Groq model",
                    value="llama-3.1-8b-instant",
                    key="agentic_groq_inference_model",
                )
                if not os.getenv("GROQ_API_KEY"):
                    st.warning("GROQ_API_KEY is not set. Add it to run Groq inference.")
            elif inference_backend == "Gemini":
                inference_model_name = st.text_input(
                    "Gemini model",
                    value="gemini-2.5-flash",
                    key="agentic_gemini_inference_model",
                )
                with st.expander("Check available Gemini models", expanded=False):
                    if st.button("Check available Gemini models", key="check_gemini_models"):
                        st.session_state["agentic_available_gemini_models"] = (
                            list_available_gemini_models()
                        )
                    available_models = st.session_state.get("agentic_available_gemini_models")
                    if available_models is None:
                        st.caption(
                            "Model listing is optional and may fail under quota or SDK differences."
                        )
                    elif available_models:
                        st.write(available_models)
                    else:
                        st.warning("No available Gemini models were returned.")
                if not os.getenv("GEMINI_API_KEY"):
                    st.warning("GEMINI_API_KEY is not set. Add it to run Gemini inference.")
            elif inference_backend == "HF local":
                inference_model_name = st.text_input(
                    "HF model",
                    value=DEFAULT_LOCAL_HF_ROUTER_MODEL_ID,
                    key="agentic_hf_inference_model",
                )
                hf_max_new_tokens = st.number_input(
                    "HF max_new_tokens",
                    min_value=16,
                    max_value=1024,
                    value=256,
                    step=16,
                    key="agentic_hf_max_new_tokens",
                )
                hf_trust_remote_code = st.checkbox(
                    "trust remote code",
                    value=False,
                    key="agentic_hf_trust_remote_code",
                )
                st.caption(
                    "Routes with a local transformers causal LM. No real tools are executed."
                )

        def execute_tool_inference() -> object:
            if inference_backend == "Groq":
                return run_groq_tool_inference(
                    user_request,
                    get_executable_tool_schemas(),
                    inference_model_name,
                    system_prompt=system_prompt,
                    tool_context=tool_context,
                )
            if inference_backend == "Gemini":
                return run_gemini_tool_inference(
                    user_request,
                    get_executable_tool_schemas(),
                    inference_model_name,
                    system_prompt=system_prompt,
                    tool_context=tool_context,
                )
            try:
                hf_router = load_local_hf_router(
                    inference_model_name,
                    int(hf_max_new_tokens),
                    trust_remote_code=bool(hf_trust_remote_code),
                )
                return hf_router.choose_tool(user_request, TOOLS)
            except Exception as error:  # noqa: BLE001
                return types.SimpleNamespace(
                    selected_tool=None,
                    tool_arguments={},
                    agent_response="",
                    raw_response="",
                    debug_prompt=None,
                    error=f"HF local inference failed: {error}",
                    available=False,
                )

        _, run_column = st.columns([3, 1])
        if run_column.button(
            "Run inference",
            type="primary",
            key="run_inference",
            use_container_width=True,
        ):
            if hasattr(st, "status"):
                with st.status("Running agent inference...", expanded=True) as status:
                    st.write(f"Backend: {inference_backend}")
                    st.write(f"Model: {inference_model_name}")
                    inference_result = execute_tool_inference()
                    status.update(label="Agent inference complete.", state="complete")
            else:
                with st.spinner("Running agent inference..."):
                    inference_result = execute_tool_inference()
            setattr(inference_result, "backend", inference_backend)
            setattr(inference_result, "model", inference_model_name)
            st.session_state["agentic_inference_backend"] = inference_backend
            st.session_state["agentic_inference_model"] = inference_model_name
            st.session_state["agentic_inference_result"] = inference_result
            if inference_result.selected_tool in TOOLS:
                st.session_state["agentic_inferred_tool"] = inference_result.selected_tool
                st.session_state.has_run = False
                st.session_state.result = None
                st.rerun()
            st.session_state["agentic_inferred_tool"] = None
            st.session_state.has_run = False
            st.session_state.result = None

        inference_result = st.session_state.get("agentic_inference_result")
        if inference_result is None:
            st.info("Run inference to select a tool before explaining it.")
        else:
            inference_error = getattr(inference_result, "error", None)
            if inference_error:
                st.warning(inference_error)
            selected_tool = getattr(inference_result, "selected_tool", None)
            assistant_message = (
                getattr(inference_result, "agent_response", "")
                or getattr(inference_result, "assistant_answer", "")
                or getattr(inference_result, "final_answer", "")
                or f"I recommend `{selected_tool}` for this request."
            )
            with st.chat_message("user"):
                st.write(user_request)
            with st.chat_message("assistant"):
                st.write(assistant_message)

            st.divider()
            result_backend = getattr(
                inference_result,
                "backend",
                st.session_state.get("agentic_inference_backend", inference_backend),
            )
            result_model = getattr(
                inference_result,
                "model",
                st.session_state.get("agentic_inference_model", inference_model_name),
            )
            with st.expander("Model information and arguments", expanded=False):
                st.write(f"Backend: `{result_backend}`")
                st.write(f"Model: `{result_model}`")
                st.metric("Selected tool", selected_tool or "No tool selected")
                st.markdown("**Tool arguments**")
                st.json(getattr(inference_result, "tool_arguments", {}))
            raw_trace = getattr(inference_result, "raw_trace", None)
            if raw_trace is None:
                raw_trace = {
                    "debug_prompt": getattr(inference_result, "debug_prompt", None),
                    "raw_response": getattr(inference_result, "raw_response", ""),
                }
            with st.expander("Debug trace", expanded=False):
                st.json(raw_trace)

    with explanation_tab:
        st.markdown('<div class="section-label">Explanation controls</div>', unsafe_allow_html=True)
        current_inference_backend = st.session_state.get(
            "agentic_inference_backend_choice",
            "Groq",
        )
        latest_inference_backend = st.session_state.get("agentic_inference_backend")
        has_groq_inference_result = (
            latest_inference_backend == "Groq"
            and st.session_state.get("agentic_inference_result") is not None
        )
        groq_reference_result = (
            st.session_state.get("agentic_inference_result") if has_groq_inference_result else None
        )
        # Only offer trajectory matching when a real Groq inference result selected a
        # known tool AND that tool was actually called with non-empty arguments --
        # otherwise the argument-match part of the score would be meaningless.
        trajectory_match_available = (
            groq_reference_result is not None
            and st.session_state.get("agentic_inferred_tool") in TOOLS
            and bool(getattr(groq_reference_result, "tool_arguments", None))
        )
        show_developer_scorers = st.checkbox(
            "Show developer scoring methods",
            value=False,
            key="agentic_show_developer_scorers",
        )
        scorer_options = ["LLM logprob scorer"]
        if current_inference_backend == "Groq" or has_groq_inference_result:
            scorer_options.append("Groq soft-vote scorer")
            scorer_options.append("Groq deterministic router")
        if trajectory_match_available:
            scorer_options.append("Trajectory match: tool + normalized args")
        if show_developer_scorers:
            st.caption(
                "Developer scorers are intended for debugging and should not be used for "
                "final demo results."
            )
            scorer_options.extend(["Keyword scorer", "Mock model scorer"])
        scorer_backend_key = "agentic_explanation_scorer_backend"
        default_scorer = "Groq soft-vote scorer" if has_groq_inference_result else scorer_options[0]
        if scorer_backend_key not in st.session_state:
            st.session_state[scorer_backend_key] = default_scorer
        # Only reset the stored selection when it is no longer a valid option -- never
        # overwrite a still-valid manual selection on every rerun (that previously made
        # selecting any scorer impossible whenever a Groq inference result was present).
        if st.session_state[scorer_backend_key] not in scorer_options:
            st.session_state[scorer_backend_key] = default_scorer
        scorer_backend = st.selectbox(
            "Scoring method",
            scorer_options,
            key=scorer_backend_key,
        )
        logprob_model_id = DEFAULT_LOGPROB_MODEL_ID
        candidate_template = DEFAULT_CANDIDATE_TEMPLATE
        candidate_texts = None
        normalize_by_length = True
        max_pairs_per_batch = 1
        router_model_id = DEFAULT_GROQ_ROUTER_MODEL_ID
        soft_vote_model_id = DEFAULT_GROQ_ROUTER_MODEL_ID
        soft_vote_n_samples = DEFAULT_GROQ_SOFT_VOTE_N_SAMPLES
        soft_vote_temperature = DEFAULT_GROQ_SOFT_VOTE_TEMPERATURE
        soft_vote_max_retries = DEFAULT_GROQ_SOFT_VOTE_MAX_RETRIES
        soft_vote_seed = None
        if scorer_backend == "LLM logprob scorer":
            with st.expander("Logprob scorer settings", expanded=True):
                logprob_model_id = st.text_input(
                    "model id",
                    value=DEFAULT_LOGPROB_MODEL_ID,
                    key="agentic_logprob_model_id",
                )
                max_pairs_per_batch = st.number_input(
                    "HF pair batch size",
                    min_value=1,
                    max_value=16,
                    value=1,
                    step=1,
                    key="agentic_logprob_pair_batch_size",
                    help=(
                        "Number of prompt/candidate pairs scored per local-model forward pass. "
                        "Use 1 on Colab T4 to avoid CUDA out-of-memory errors."
                    ),
                )
        elif scorer_backend == "Groq deterministic router":
            with st.expander("Groq router scorer settings", expanded=True):
                router_model_id = st.text_input(
                    "Groq router model",
                    value=DEFAULT_GROQ_ROUTER_MODEL_ID,
                    key="agentic_groq_router_scorer_model",
                )
                st.caption(
                    "Calls the real Groq API once per distinct coalition prompt to ask which "
                    "tool it would route to, and scores 1.0 if that matches the target tool, "
                    "else 0.0. Every app interaction re-runs the setup preview, so this "
                    "issues real Groq calls even before clicking Run explanation."
                )
                if not os.getenv("GROQ_API_KEY"):
                    st.warning("GROQ_API_KEY is not set. Add it to use the Groq router scorer.")
        elif scorer_backend == "Groq soft-vote scorer":
            with st.expander("Groq soft-vote scorer settings", expanded=True):
                soft_vote_model_id = st.text_input(
                    "Groq soft-vote model",
                    value=DEFAULT_GROQ_ROUTER_MODEL_ID,
                    key="agentic_groq_soft_vote_model",
                )
                soft_vote_n_samples = st.number_input(
                    "Groq soft-vote samples",
                    min_value=1,
                    max_value=25,
                    value=DEFAULT_GROQ_SOFT_VOTE_N_SAMPLES,
                    step=1,
                    key="agentic_groq_soft_vote_samples",
                )
                soft_vote_temperature = st.slider(
                    "Groq soft-vote temperature",
                    min_value=0.0,
                    max_value=1.5,
                    value=DEFAULT_GROQ_SOFT_VOTE_TEMPERATURE,
                    step=0.05,
                    key="agentic_groq_soft_vote_temperature",
                )
                soft_vote_max_retries = st.number_input(
                    "Groq soft-vote max retries",
                    min_value=0,
                    max_value=5,
                    value=DEFAULT_GROQ_SOFT_VOTE_MAX_RETRIES,
                    step=1,
                    key="agentic_groq_soft_vote_max_retries",
                )
                use_soft_vote_seed = st.checkbox(
                    "set Groq soft-vote seed",
                    value=False,
                    key="agentic_groq_soft_vote_use_seed",
                )
                if use_soft_vote_seed:
                    soft_vote_seed = int(
                        st.number_input(
                            "Groq soft-vote seed",
                            min_value=0,
                            max_value=2_147_483_647,
                            value=42,
                            step=1,
                            key="agentic_groq_soft_vote_seed",
                        )
                    )
                st.caption(
                    "Soft-vote score: empirical target-tool selection frequency across sampled "
                    "Groq router calls."
                )
                if not os.getenv("GROQ_API_KEY"):
                    st.warning("GROQ_API_KEY is not set. Add it to use the Groq soft-vote scorer.")
        elif scorer_backend == "Trajectory match: tool + normalized args":
            with st.expander("Trajectory match scorer settings", expanded=True):
                st.warning(
                    "This scorer re-runs the real Groq agent once per distinct coalition "
                    "prompt to get an actual tool call (name + arguments), not just a routing "
                    "decision, then compares it against the recorded inference result. This "
                    "can be slow and costly: expect one real Groq API call per coalition "
                    "shapiq samples."
                )
                if groq_reference_result is not None:
                    st.caption(
                        f"Reference tool: `{groq_reference_result.selected_tool}` with "
                        f"arguments {dict(groq_reference_result.tool_arguments)}."
                    )
                if not os.getenv("GROQ_API_KEY"):
                    st.warning(
                        "GROQ_API_KEY is not set. Add it to use the trajectory match scorer."
                    )

        with st.expander("Segmentation settings", expanded=False):
            segment_threshold = st.slider(
                "semantic threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.72,
                step=0.01,
                key="agentic_segment_threshold",
            )
            segment_window = st.slider(
                "context window",
                min_value=1,
                max_value=10,
                value=3,
                step=1,
                key="agentic_segment_window",
            )
            min_segment_words = st.slider(
                "min words per segment",
                min_value=1,
                max_value=8,
                value=1,
                step=1,
                key="agentic_min_segment_words",
            )

        try:
            segmenter = load_semantic_segmenter(
                segment_threshold,
                segment_window,
                min_segment_words,
            )
            semantic_user_texts, segment_debug_rows = segment_user_request(
                segmenter,
                user_request,
            )
        except Exception as error:  # noqa: BLE001
            st.error(f"Could not segment the user request with MPNet: {error}")
            return
        user_segments = build_segments(semantic_user_texts, "user")
        labels = [segment.label for segment in user_segments]
        budget = budget_for_demo(len(user_segments))

        with st.expander("Shapley and debug settings", expanded=False):
            st.caption(f"index: fixed `{DEFAULT_INDEX}`")
            st.caption(f"max_order: fixed `{DEFAULT_MAX_ORDER}`")
            st.caption(f"budget: `{budget}` auto")
            show_prompt_segments = st.checkbox(
                "show prompt segments",
                value=False,
                key="agentic_show_prompt_segments",
            )
            show_value_function_details = st.checkbox(
                "show value function details",
                value=False,
                key="agentic_show_value_function_details",
            )
            show_scoring_prompt_preview = st.checkbox(
                "show scoring prompt preview",
                value=False,
                key="agentic_show_scoring_prompt_preview",
            )
            show_lexical_comparison = st.checkbox(
                "show keyword comparison",
                value=False,
                key="agentic_show_lexical_comparison",
            )
            enable_fallback_target_selection = st.checkbox(
                "enable fallback target selection",
                value=False,
                key="agentic_enable_fallback_target_selection",
                help=(
                    "Use the selected explanation scorer to choose a target tool when no "
                    "inference result is available."
                ),
            )

        if len(user_segments) < 1:
            st.warning("Add a user request with at least one segment.")
            return

        full_prompt = build_coalition_prompt(
            user_segments,
            system_prompt=system_prompt,
            tool_context=tool_context,
        )
        empty_prompt = build_coalition_prompt(
            [],
            system_prompt=system_prompt,
            tool_context=tool_context,
        )

        inferred_tool = st.session_state.get("agentic_inferred_tool")
        using_inferred_tool = inferred_tool in TOOLS
        inference_source = st.session_state.get("agentic_inference_backend")
        result_target_tool = None
        result_target_source = None
        result = st.session_state.result
        if isinstance(result, dict):
            result_target_tool = result.get("target_tool")
            result_target_source = result.get("target_source")
        if using_inferred_tool:
            target_tool = inferred_tool
            target_source = f"{inference_source} inference" if inference_source else "Inference"
            pending_fallback_target = False
        elif result_target_tool in TOOLS and result_target_source == "fallback explanation scorer":
            target_tool = str(result_target_tool)
            target_source = "fallback explanation scorer"
            pending_fallback_target = False
        else:
            target_tool = None
            target_source = "fallback explanation scorer"
            pending_fallback_target = bool(enable_fallback_target_selection)

        if not using_inferred_tool:
            if enable_fallback_target_selection:
                st.info(
                    "No inference result is available. Fallback target selection is enabled; "
                    "the explanation scorer will choose the target when you run explanation."
                )
                st.caption("Target tool source: fallback explanation scorer")
            else:
                st.info(
                    "No inference result is available. Run inference first, or enable fallback "
                    "target selection in advanced settings."
                )

        index = DEFAULT_INDEX
        max_order = DEFAULT_MAX_ORDER
        signature_target = target_tool if target_tool is not None else "__pending_target__"
        trajectory_reference_signature = (
            tuple(sorted(groq_reference_result.tool_arguments.items()))
            if scorer_backend == "Trajectory match: tool + normalized args"
            and groq_reference_result is not None
            else None
        )
        signature = (
            trace_name,
            user_request,
            signature_target,
            scorer_backend,
            logprob_model_id,
            candidate_template,
            bool(candidate_texts),
            normalize_by_length,
            int(max_pairs_per_batch),
            router_model_id,
            soft_vote_model_id,
            int(soft_vote_n_samples),
            float(soft_vote_temperature),
            int(soft_vote_max_retries),
            soft_vote_seed,
            trajectory_reference_signature,
            bool(enable_fallback_target_selection),
            show_lexical_comparison,
            segment_threshold,
            segment_window,
            min_segment_words,
            tuple(semantic_user_texts),
        )
        if st.session_state.get("result_signature") != signature:
            st.session_state.has_run = False
            st.session_state.result = None
            st.session_state.pending_run = False
            st.session_state.result_signature = signature
            result = None
            if not using_inferred_tool:
                target_tool = None
                pending_fallback_target = bool(enable_fallback_target_selection)
                st.session_state.result_signature = (
                    trace_name,
                    user_request,
                    "__pending_target__",
                    scorer_backend,
                    logprob_model_id,
                    candidate_template,
                    bool(candidate_texts),
                    normalize_by_length,
                    int(max_pairs_per_batch),
                    router_model_id,
                    soft_vote_model_id,
                    int(soft_vote_n_samples),
                    float(soft_vote_temperature),
                    int(soft_vote_max_retries),
                    soft_vote_seed,
                    trajectory_reference_signature,
                    bool(enable_fallback_target_selection),
                    show_lexical_comparison,
                    segment_threshold,
                    segment_window,
                    min_segment_words,
                    tuple(semantic_user_texts),
                )
    
        st.markdown('<div class="section-label">Setup</div>', unsafe_allow_html=True)
        target_tool_label = target_tool if target_tool is not None else "Pending"
        target_line = (
            f"<strong>Explaining why agent selected:</strong> {escape(target_tool_label)}"
            if using_inferred_tool
            else f"<strong>Fallback target tool:</strong> {escape(target_tool_label)}"
        )
        st.markdown(
            f"""
            <div class="scenario-panel">
                <div>
                    <span class="scenario-tag">Tool selection</span>
                    <h3>{escape(trace_name)}</h3>
                    <p>{escape(user_request)}</p>
                </div>
                <div class="scenario-hint">
                    {target_line}<br>
                    <strong>Target tool source:</strong> {escape(target_source)}<br>
                    <strong>Available tools:</strong> {escape(", ".join(TOOLS))}<br>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
        st.markdown('<div class="section-label">Explanation target</div>', unsafe_allow_html=True)
        target_left, target_right = st.columns([0.85, 1.15])
        with target_left:
            st.markdown(f"### `{target_tool_label}`")
        with target_right:
            st.metric("Target tool source", target_source)

        if pending_fallback_target:
            st.caption(
                "Fallback target selection will run with the selected explanation scorer after "
                "you click Run explanation."
            )
        elif (
            target_source == "fallback explanation scorer"
            and isinstance(result, dict)
            and result.get("fallback_choice_scores")
        ):
            st.markdown("**Fallback scorer diagnostic**")
            st.caption(
                "These scores are not from Groq, Gemini, or HF local inference. They come "
                "from the selected explanation scorer and were used to choose the fallback "
                "target tool."
            )
            score_frame = pd.DataFrame(
                [
                    {"tool": tool, "score": score}
                    for tool, score in sorted(
                        result["fallback_choice_scores"].items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )
                ]
            )
            st.dataframe(score_frame, use_container_width=True, hide_index=True, height=178)
    
        with st.expander("Show fixed context and Shapley players", expanded=show_prompt_segments):
            segment_left, segment_right = st.columns(2)
            with segment_left:
                st.markdown("**Fixed context, not explained**")
                st.caption("System prompt")
                for segment in system_segments:
                    st.markdown(
                        (
                            "<div class='segment-box'>"
                            f"<h4>{segment.label}</h4><p>{escape(segment.text)}</p></div>"
                        ),
                        unsafe_allow_html=True,
                    )
                st.caption("Tool definitions")
                for tool_name, description in TOOLS.items():
                    st.markdown(
                        (
                            "<div class='segment-box'>"
                            f"<h4>{escape(tool_name)}</h4><p>{escape(description)}</p></div>"
                        ),
                        unsafe_allow_html=True,
                    )
            with segment_right:
                st.markdown("**Shapley players: user request segments**")
                st.caption(
                    f"{len(user_segments)} user segments from `{segmenter.model_id}` on "
                    f"`{segmenter.device}`. threshold={segmenter.threshold:.2f}, "
                    f"window={segmenter.window}, min words={segmenter.min_segment_words}."
                )
                for segment in user_segments:
                    st.markdown(
                        (
                            "<div class='segment-box user'>"
                            f"<h4>{segment.label}</h4><p>{escape(segment.text)}</p></div>"
                        ),
                        unsafe_allow_html=True,
                    )
                if segment_debug_rows:
                    st.markdown("**Semantic boundary diagnostics**")
                    st.dataframe(
                        pd.DataFrame(segment_debug_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

        if not st.session_state.has_run and not st.session_state.pending_run:
            can_run_explanation = target_tool is not None or pending_fallback_target
            if st.button(
                "Run explanation",
                type="primary",
                key="agentic_run_explanation",
                disabled=not can_run_explanation,
            ):
                st.session_state.pending_run = True
                st.rerun()
            if can_run_explanation:
                st.info("Enter a prompt and click Run explanation to compute the explanation.")
            else:
                st.info(
                    "Run inference first, or enable fallback target selection in advanced "
                    "settings before running explanation."
                )
            return
    
        run = st.session_state.pending_run
        st.session_state.pending_run = False
    
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
            lexical_scorer = LexicalToolScorer()
            if scorer_backend == "Keyword scorer":
                primary_scorer = lexical_scorer
                primary_label = "Keyword scorer"
            elif scorer_backend == "LLM logprob scorer":
                with st.spinner("Loading logprob scorer model..."):
                    try:
                        primary_scorer = load_logprob_scorer(
                            logprob_model_id,
                            candidate_template,
                            candidate_texts,
                            normalize_by_length=bool(normalize_by_length),
                        )
                    except Exception as error:  # noqa: BLE001
                        st.error(
                            "Could not load the logprob-based scorer. Install/check "
                            "`transformers` and `torch`, try a smaller causal language model, "
                            f"or check your environment. Details: {error}"
                        )
                        return
                primary_scorer.max_pairs_per_batch = int(max_pairs_per_batch)
                primary_label = "LLM logprob scorer"
            elif scorer_backend == "Groq deterministic router":
                if not os.getenv("GROQ_API_KEY"):
                    st.error(
                        "GROQ_API_KEY is not set. Add it to the environment to use the Groq "
                        "deterministic router scorer."
                    )
                    return
                primary_scorer = GroqDeterministicRouterScorer(model_name=router_model_id)
                primary_label = "Groq deterministic router"
            elif scorer_backend == "Groq soft-vote scorer":
                if not os.getenv("GROQ_API_KEY"):
                    st.error(
                        "GROQ_API_KEY is not set. Add it to the environment to use the Groq "
                        "soft-vote scorer."
                    )
                    return
                primary_scorer = GroqSoftVoteToolScorer(
                    model_name=soft_vote_model_id,
                    n_samples=int(soft_vote_n_samples),
                    temperature=float(soft_vote_temperature),
                    max_retries=int(soft_vote_max_retries),
                    seed=soft_vote_seed,
                )
                primary_label = "Groq soft-vote scorer"
            elif scorer_backend == "Trajectory match: tool + normalized args":
                if not os.getenv("GROQ_API_KEY"):
                    st.error(
                        "GROQ_API_KEY is not set. Add it to the environment to use the "
                        "trajectory match scorer."
                    )
                    return
                if groq_reference_result is None or not groq_reference_result.tool_arguments:
                    st.error(
                        "No real Groq inference result with tool arguments is available. "
                        "Run Groq inference first."
                    )
                    return
                reference_trajectory = ToolTrajectory(
                    selected_tool=groq_reference_result.selected_tool,
                    tool_arguments=dict(groq_reference_result.tool_arguments),
                )
                trajectory_provider = build_groq_inference_trajectory_provider(
                    getattr(groq_reference_result, "model", DEFAULT_GROQ_ROUTER_MODEL_ID),
                    get_executable_tool_schemas(),
                    tool_context=tool_context,
                )
                primary_scorer = TrajectoryArgumentMatchScorer(
                    reference_trajectory=reference_trajectory,
                    trajectory_provider=trajectory_provider,
                )
                primary_label = "Trajectory match: tool + normalized args"
            else:
                primary_scorer = LLMToolScorer(llm=MockLLM())
                primary_label = "Mock model scorer"

            fallback_choice = None
            if target_tool is None:
                if not enable_fallback_target_selection:
                    st.error(
                        "No inference result is available. Run inference first, or enable "
                        "fallback target selection in advanced settings."
                    )
                    return
                with st.spinner("Selecting fallback target tool with the explanation scorer..."):
                    fallback_choice = choose_tool_with_scorer(
                        primary_scorer,
                        full_prompt,
                        tool_descriptions=TOOLS,
                    )
                target_tool = fallback_choice.tool
                target_source = "fallback explanation scorer"
    
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
                    user_segments=user_segments,
                    system_prompt=system_prompt,
                    tool_context=tool_context,
                    scorer=primary_scorer,
                    tool_descriptions=TOOLS,
                )
                approximator = make_approximator(index, game.n_players, max_order)
                explanation = approximator.approximate(budget=budget, game=game)
                first_order = explanation.get_n_order(order=1)
                attribution_frame = values_to_frame(first_order, user_segments)
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
    
            compare_with_lexical = show_lexical_comparison and primary_label != "Keyword scorer"
            llm_debug_outputs = getattr(primary_scorer, "last_debug_outputs", [])
            lexical_result = None
            if compare_with_lexical:
                with st.spinner("Computing lexical baseline comparison..."):
                    lexical_game = ToolUseGame(
                        target_tool=target_tool,
                        user_segments=user_segments,
                        system_prompt=system_prompt,
                        tool_context=tool_context,
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
                    lexical_frame = values_to_frame(lexical_first_order, user_segments)
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
                        "label": "Keyword scorer",
                        "full_score": lexical_full_score,
                        "empty_score": lexical_empty_score,
                        "top": "No segment"
                        if lexical_top is None
                        else f"{lexical_top['segment']} ({lexical_top['source']})",
                        "top_value": 0.0 if lexical_top is None else float(lexical_top["attribution"]),
                        "pair": lexical_pair_label,
                        "pair_value": lexical_pair_value,
                    }
    
            scoring_prompt_preview = build_scoring_prompt_preview(
                primary_scorer,
                full_prompt,
                target_tool=target_tool,
                tool_descriptions=TOOLS,
            )
            st.session_state.has_run = True
            result_signature = (
                trace_name,
                user_request,
                target_tool,
                scorer_backend,
                logprob_model_id,
                candidate_template,
                bool(candidate_texts),
                normalize_by_length,
                int(max_pairs_per_batch),
                router_model_id,
                soft_vote_model_id,
                int(soft_vote_n_samples),
                float(soft_vote_temperature),
                int(soft_vote_max_retries),
                soft_vote_seed,
                trajectory_reference_signature,
                bool(enable_fallback_target_selection),
                show_lexical_comparison,
                segment_threshold,
                segment_window,
                min_segment_words,
                tuple(semantic_user_texts),
            )
            st.session_state.result_signature = result_signature
            st.session_state.result = {
                "target_tool": target_tool,
                "target_source": target_source,
                "fallback_choice_scores": None
                if fallback_choice is None
                else fallback_choice.scores,
                "primary_label": primary_label,
                "full_score": full_score,
                "empty_score": empty_score,
                "first_order": first_order,
                "attribution_frame": attribution_frame,
                "explanation": explanation,
                "pairwise_matrix": pairwise_matrix,
                "pair_label": pair_label,
                "pair_value": pair_value,
                "top_label": top_label,
                "top_score": top_score,
                "interpretation_sentence": interpretation_sentence,
                "compare_with_lexical": compare_with_lexical,
                "llm_debug_outputs": llm_debug_outputs,
                "lexical_result": lexical_result,
                "scoring_prompt": scoring_prompt_preview,
            }
    
        result = st.session_state.result
        if result is None:
            st.error("No explanation result is available. Click Run explanation to compute one.")
            return
        primary_label = result["primary_label"]
        full_score = result["full_score"]
        empty_score = result["empty_score"]
        first_order = result["first_order"]
        attribution_frame = result["attribution_frame"]
        explanation = result["explanation"]
        pairwise_matrix = result.get("pairwise_matrix")
        if pairwise_matrix is None:
            pairwise_matrix = pairwise_matrix_from_explanation(explanation, len(user_segments))
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
        token_attribution_bar_plot, sentence_interaction_heatmap, plot_import_error = (
            load_text_plotters()
        )
    
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
                    <div class="metric-card">
                        <span>Target tool</span><strong>{escape(target_tool)}</strong>
                    </div>
                    <div class="metric-card">
                        <span>Full support score = v(all user segments)</span>
                        <strong>{full_score:.3f}</strong>
                    </div>
                    <div class="metric-card">
                        <span>Baseline = fixed context + empty user request</span>
                        <strong>{empty_score:.3f}</strong>
                    </div>
                    <div class="metric-card">
                        <span>Delta support</span><strong>{delta_support:.3f}</strong>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.info(f"**{support_status}.** {support_interpretation}")
    
            st.caption("See the Attribution tab for the full first-order segment ranking.")
    
        with tabs[1]:
            st.markdown("**First-order attribution ranking**")
            st.dataframe(attribution_frame, use_container_width=True, hide_index=True)
            if token_attribution_bar_plot is None:
                st.warning(
                    "The shapiq text attribution plot is unavailable in this environment. "
                    f"Showing a simple fallback chart instead. Details: {plot_import_error}"
                )
                show_fallback_attribution_chart(attribution_frame)
            else:
                try:
                    fig_ax = token_attribution_bar_plot(first_order, labels, show=False)
                except Exception as error:  # noqa: BLE001
                    st.warning(
                        "The shapiq text attribution plot failed. "
                        f"Showing a simple fallback chart instead. Details: {error}"
                    )
                    show_fallback_attribution_chart(attribution_frame)
                else:
                    if fig_ax is not None:
                        fig, ax = fig_ax
                        st.pyplot(polish_bar(fig, ax), clear_figure=True)
    
        with tabs[2]:
            st.markdown("**Pairwise interaction heatmap**")
            if sentence_interaction_heatmap is None:
                st.warning(
                    "The shapiq text interaction heatmap is unavailable in this environment. "
                    f"Showing a fallback interaction table instead. Details: {plot_import_error}"
                )
                show_fallback_interaction_table(pairwise_matrix, labels)
            else:
                try:
                    fig_ax = sentence_interaction_heatmap(explanation, labels, show=False)
                except Exception as error:  # noqa: BLE001
                    st.warning(
                        "The shapiq text interaction heatmap failed. "
                        f"Showing a fallback interaction table instead. Details: {error}"
                    )
                    show_fallback_interaction_table(pairwise_matrix, labels)
                else:
                    if fig_ax is not None:
                        fig, ax = fig_ax
                        st.pyplot(polish_heatmap(fig, ax, user_segments), clear_figure=True)
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
                            "score_kind",
                            "score_description",
                            "selected_tools",
                            "target_matches",
                            "n_samples",
                            "temperature",
                            "raw_output",
                            "raw_outputs",
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
                    scoring_prompt = result.get("scoring_prompt")
                    if scoring_prompt:
                        st.code(scoring_prompt, language="text")
                    else:
                        st.caption(
                            "A separate scoring-prompt preview is not available for this scoring "
                            "backend."
                        )
                if show_value_function_details:
                    st.markdown("**Value function details**")
                    st.write(f"Index: `{index}`")
                    st.write(f"Max order: `{max_order}`")
                    st.write(f"Budget: `{budget}`")
                    st.write("Full coalition prompt:")
                    st.code(full_prompt, language="text")
                    st.write("Empty coalition prompt:")
                    st.code(empty_prompt, language="text")
    
        st.markdown(
            '<div class="section-label">How the explanation is computed</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="setup-line">
                <strong>Value function:</strong>
                <code>v(S) = score(target tool | fixed context + selected user segments S)</code><br>
                S is a subset of user-request segment players. The system prompt and tool
                definitions are included unchanged for every coalition.
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("Why this is a Shapley game", expanded=False):
            st.markdown(
                """
                Shapley values compare tool-support scores across different user-segment
                combinations to estimate each user segment's contribution.
    
                A positive contribution means the user segment supports the target tool.
                A negative contribution means it weakens support for the target tool.
                """
            )
    
        st.caption(f"Demo path: `{display_demo_path()}`")


if __name__ == "__main__":
    main()
