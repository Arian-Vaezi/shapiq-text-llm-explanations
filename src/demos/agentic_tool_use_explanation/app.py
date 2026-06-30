"""Streamlit demo for explaining agentic tool-use decisions with shapiq."""

from __future__ import annotations

import importlib.util
import math
import os
import sys
import types
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd
import streamlit as st
from coalition_evaluation import (
    DEFAULT_RETRY_POLICY,
    CoalitionEvaluationIncompleteError,
    evaluate_game_exactly,
)
from exact_interactions import (
    MAX_EXACT_DEMO_PLAYERS,
    ExactComputationLimitError,
    UnsupportedExactIndexError,
    compute_exact_interactions,
)
from final_answer_similarity_scorer import (
    DEFAULT_FINAL_ANSWER_EMBEDDING_MODEL_ID,
    FinalAnswerSimilarityScorer,
    SentenceTransformerAnswerEmbedder,
    extract_final_answer,
)
from gemini_agent import list_available_gemini_models, run_gemini_tool_inference
from groq_agent import run_groq_tool_inference
from hf_router import DEFAULT_LOCAL_HF_ROUTER_MODEL_ID, LocalHFRouter
from linguistic_segmenter import LinguisticSegmenter
from matplotlib.patches import Rectangle
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
    build_coalition_prompt as canonical_coalition_prompt,
    join_user_request_segments,
)
from semantic_segmenter import SemanticSegmenter
from tool_schemas import get_executable_tool_schemas

if TYPE_CHECKING:
    from collections.abc import Callable

    import matplotlib.pyplot as plt

    import shapiq

SegmentSource = Literal["system", "user"]


DEFAULT_INDEX = "k-SII"
DEFAULT_MAX_ORDER = 2
DELTA_STATUS_THRESHOLD = 0.01
DEFAULT_MOCK_QUERY = "Will it rain in Berlin tomorrow morning?"
FINAL_ANSWER_SIMILARITY_LABEL = "Final answer semantic similarity"
EFFICIENCY_RESIDUAL_TOLERANCE = 1e-4
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


@st.cache_resource
def load_linguistic_segmenter() -> LinguisticSegmenter:
    """Load and cache the optional spaCy linguistic segmenter."""
    return LinguisticSegmenter()


@st.cache_resource
def load_final_answer_embedder(model_id: str) -> SentenceTransformerAnswerEmbedder:
    """Load and cache the optional embedding model for the final-answer similarity scorer.

    The underlying sentence-transformers model is only downloaded/loaded the first
    time the embedder is called, not when this cached wrapper is constructed.
    """
    return SentenceTransformerAnswerEmbedder(model_id=model_id, device="auto")


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
    """Build a coalition prompt with fixed context and selected user-request segments.

    Delegates to ``scorers.build_coalition_prompt``, the single canonical prompt
    format shared with ``tool_game.ToolUseGame.build_prompt``, so the same
    coalition (including the empty one used as the Shapley normalization
    baseline) always produces byte-identical prompt text regardless of call site.
    """
    user_request = join_user_request_segments(selected_user_segments)
    return canonical_coalition_prompt(
        user_request,
        system_prompt=system_prompt,
        tool_context=tool_context,
    )


def segment_user_request(
    segmenter: SemanticSegmenter | LinguisticSegmenter,
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


def budget_for_demo(n_players: int) -> int:
    """Sampling budget for the official shapiq approximator used above the exact limit."""
    return int(min(2**n_players, max(48, 8 * n_players * math.log2(n_players + 1))))


def make_approximator(index: str, n_players: int, max_order: int) -> object:
    """Create a real shapiq approximator for player counts above ``MAX_EXACT_DEMO_PLAYERS``.

    This is only reached when exact computation is infeasible. It must never silently
    substitute a hand-rolled heuristic: if shapiq is unavailable or construction fails,
    the caller is expected to surface the error instead of falling back to one.
    """
    import shapiq

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


def compute_interaction_explanation(
    *,
    game: object,
    index: str,
    max_order: int,
    budget: int | None,
) -> tuple[shapiq.InteractionValues, str]:
    """Compute interaction values, preferring exact computation over approximation.

    For ``game.n_players <= MAX_EXACT_DEMO_PLAYERS`` this always uses the real
    ``shapiq.ExactComputer`` so the returned values are genuinely the requested official
    index, not a heuristic. Above that limit it falls back to a real, clearly labelled
    shapiq approximator instead of silently substituting a different computation.

    Returns:
        A tuple of the native ``shapiq.InteractionValues`` result and an algorithm label
        describing how it was computed, for display in the UI.

    Raises:
        ExactComputationLimitError: Propagated from ``compute_exact_interactions`` if
            ``game.n_players`` unexpectedly exceeds the exact limit.
        UnsupportedExactIndexError: Propagated from ``compute_exact_interactions`` if
            ``index`` is not supported by ``ExactComputer``.
        CoalitionEvaluationIncompleteError: If any of the ``2**game.n_players``
            coalitions could not be resolved to a real score (after retries for
            transient failures). ``ExactComputer`` is never invoked in that case.
    """
    if game.n_players <= MAX_EXACT_DEMO_PLAYERS:
        evaluate_game_exactly(game, retry_policy=DEFAULT_RETRY_POLICY)
        explanation, metadata = compute_exact_interactions(
            game=game,
            index=index,
            max_order=max_order,
        )
        algorithm_label = (
            f"shapiq ExactComputer (exact evaluation: {metadata.coalition_count} / "
            f"{metadata.coalition_count} coalitions)"
        )
        return explanation, algorithm_label

    if not getattr(game, "_normalization_resolved", True):
        msg = (
            "Internal error: this game was constructed with "
            "defer_empty_coalition_evaluation=True but its normalization baseline "
            "was never resolved. The approximate path (above MAX_EXACT_DEMO_PLAYERS) "
            "must only ever receive a game with an already-initialized baseline; "
            "refusing to approximate against an uninitialized placeholder."
        )
        raise RuntimeError(msg)
    approximator = make_approximator(index, game.n_players, max_order)
    explanation = approximator.approximate(budget=budget, game=game)
    return explanation, f"Official shapiq approximation: {type(approximator).__name__}"


def pairwise_matrix_from_explanation(
    explanation: shapiq.InteractionValues,
    n_players: int,
) -> pd.DataFrame:
    """Extract second-order values as a dense matrix."""
    if explanation.max_order < 2:
        return pd.DataFrame([[0.0] * n_players for _ in range(n_players)])
    return pd.DataFrame(explanation.get_n_order_values(2))


def interaction_order_diagnostics(
    explanation: shapiq.InteractionValues,
    *,
    full_value: float,
    empty_value: float,
) -> dict[str, float]:
    """Summarize order-1/order-2 values and the k-SII efficiency residual."""
    order_1_sum = 0.0
    unique_order_2: dict[tuple[int, int], float] = {}
    for interaction, value in getattr(explanation, "dict_values", {}).items():
        interaction_tuple = tuple(sorted(interaction))
        if len(interaction_tuple) == 1:
            order_1_sum += float(value)
        elif len(interaction_tuple) == 2:
            left, right = interaction_tuple
            if left < right:
                unique_order_2[(left, right)] = float(value)

    order_2_sum = float(sum(unique_order_2.values()))
    total_game_value = float(full_value) - float(empty_value)
    residual = (order_1_sum + order_2_sum) - total_game_value
    return {
        "full_value": float(full_value),
        "empty_value": float(empty_value),
        "order_1_sum": order_1_sum,
        "order_2_sum": order_2_sum,
        "total_game_value": total_game_value,
        "residual": residual,
    }


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


def polish_bar(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    xlabel: str = "Target-tool attribution",
) -> plt.Figure:
    """Make package bar plot fit the Streamlit layout."""
    fig.set_size_inches(6.2, 3.7)
    ax.set_title("", loc="center")
    ax.set_title("User Request Segment Attribution", loc="left", fontsize=12, pad=8)
    ax.set_xlabel(xlabel)
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
    ax.set_title(
        "First- and Second-Order Interaction Heatmap",
        loc="left",
        fontsize=12,
        pad=8,
    )
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


def build_complete_agent_callable(
    *,
    inference_backend: str,
    inference_model_name: str,
    system_prompt: str,
    tool_context: str,
    hf_max_new_tokens: int = 256,
    hf_trust_remote_code: bool = False,
) -> Callable[[str], object]:
    """Build a backend-agnostic callable that runs the complete tool-calling agent.

    Mirrors the Inference tab's "Run inference" action (router/tool-choice -> tool
    execution -> final answer) exactly, but parameterized over the user request so
    it can be re-run once per Shapley coalition by
    ``final_answer_similarity_scorer.FinalAnswerSimilarityScorer``.
    """

    def run(user_request: str) -> object:
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

    return run


def resolve_full_run_reference_answer(
    *,
    agent_callable: Callable[[str], object],
    user_request: str,
    inference_result: object | None,
    inference_result_is_current: bool,
) -> tuple[str, bool, str | None]:
    """Return (reference final answer, whether an existing result was reused, error reason).

    Reuses the answer already produced by the normal "Run inference" action when it
    matches the current request, system/tool configuration, and backend/model
    configuration. Otherwise runs the full prompt once through the same agent
    pipeline and uses that answer as the reference. The full-run reference answer
    must never be missing or a placeholder: if it cannot be obtained, the third
    return value carries a concrete, user-actionable reason so the caller can fail
    the explanation clearly instead of silently computing meaningless scores.
    """
    if (
        inference_result_is_current
        and inference_result is not None
        and not getattr(inference_result, "error", None)
    ):
        answer = extract_final_answer(inference_result)
        if answer:
            return answer, True, None

    try:
        full_run_result = agent_callable(user_request)
    except Exception as error:  # noqa: BLE001
        return "", False, f"The agent raised an error while answering the full request: {error}"

    backend_error = getattr(full_run_result, "error", None)
    answer = extract_final_answer(full_run_result)
    if answer:
        return answer, False, None
    if backend_error:
        return "", False, str(backend_error)
    return "", False, "The agent returned no final-answer text for the full request."


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

        example_options = [example_placeholder, *list(SAMPLE_TRACES)]
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
            agent_callable = build_complete_agent_callable(
                inference_backend=inference_backend,
                inference_model_name=inference_model_name,
                system_prompt=system_prompt,
                tool_context=tool_context,
                hf_max_new_tokens=int(hf_max_new_tokens)
                if inference_backend == "HF local"
                else 256,
                hf_trust_remote_code=bool(hf_trust_remote_code)
                if inference_backend == "HF local"
                else False,
            )
            return agent_callable(user_request)

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
            inference_result.backend = inference_backend
            inference_result.model = inference_model_name
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
        if current_inference_backend in {"Groq", "Gemini", "HF local"}:
            # All three backends can run the complete tool-calling agent end to end, which
            # is what this scorer needs (it is not a routing-only scorer like the ones below).
            scorer_options.append(FINAL_ANSWER_SIMILARITY_LABEL)
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
        final_answer_embedding_model_id = DEFAULT_FINAL_ANSWER_EMBEDDING_MODEL_ID
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
        elif scorer_backend == FINAL_ANSWER_SIMILARITY_LABEL:
            with st.expander("Final answer similarity scorer settings", expanded=True):
                final_answer_embedding_model_id = st.text_input(
                    "Embedding model",
                    value=DEFAULT_FINAL_ANSWER_EMBEDDING_MODEL_ID,
                    key="agentic_final_answer_embedding_model_id",
                    help=(
                        "A sentence-transformers model used only to embed final answers for "
                        "this scorer. Loaded lazily the first time you run this scorer, not "
                        "at app startup."
                    ),
                )
                if current_inference_backend == "Groq" and not os.getenv("GROQ_API_KEY"):
                    st.warning("GROQ_API_KEY is not set. Add it to use this scorer with Groq.")
                elif current_inference_backend == "Gemini" and not os.getenv("GEMINI_API_KEY"):
                    st.warning("GEMINI_API_KEY is not set. Add it to use this scorer with Gemini.")
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
            segmenter_choice = st.selectbox(
                "Segmenter",
                [
                    "Embedding (semantic similarity)",
                    "Linguistic (spaCy chunking)",
                ],
                index=1,
                key="agentic_segmenter",
            )
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
            if segmenter_choice == "Linguistic (spaCy chunking)":
                segmenter = load_linguistic_segmenter()
            else:
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
            st.error(f"Could not segment the user request with {segmenter_choice}: {error}")
            return
        user_segments = build_segments(semantic_user_texts, "user")
        labels = [segment.label for segment in user_segments]
        using_exact_computation = len(user_segments) <= MAX_EXACT_DEMO_PLAYERS
        budget = budget_for_demo(len(user_segments)) if not using_exact_computation else None

        with st.expander("Shapley and debug settings", expanded=False):
            st.caption(f"index: fixed `{DEFAULT_INDEX}`")
            st.caption(f"max_order: fixed `{DEFAULT_MAX_ORDER}`")
            if using_exact_computation:
                coalition_count = 2 ** len(user_segments)
                st.caption(
                    "Algorithm: `shapiq ExactComputer` "
                    f"(exact evaluation: `{coalition_count}` / `{coalition_count}` coalitions)"
                )
            else:
                st.caption(
                    f"`{len(user_segments)}` players exceeds the exact limit of "
                    f"`{MAX_EXACT_DEMO_PLAYERS}`. Algorithm: official shapiq approximation, "
                    f"budget: `{budget}` auto."
                )
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
            final_answer_embedding_model_id,
            bool(enable_fallback_target_selection),
            show_lexical_comparison,
            segmenter_choice,
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
                    final_answer_embedding_model_id,
                    bool(enable_fallback_target_selection),
                    show_lexical_comparison,
                    segmenter_choice,
                    segment_threshold,
                    segment_window,
                    min_segment_words,
                    tuple(semantic_user_texts),
                )

        st.markdown('<div class="section-label">Setup</div>', unsafe_allow_html=True)
        is_final_answer_similarity_scorer = scorer_backend == FINAL_ANSWER_SIMILARITY_LABEL
        target_tool_label = target_tool if target_tool is not None else "Pending"
        if is_final_answer_similarity_scorer:
            # This scorer compares final answers, not tool choice, so show the actual
            # reference answer it computed instead of the legacy target-tool wording/value.
            reference_answer_preview = None
            if isinstance(result, dict):
                final_answer_scorer_meta = result.get("final_answer_scorer_meta")
                if final_answer_scorer_meta:
                    reference_answer_preview = truncate_label(
                        final_answer_scorer_meta["reference_answer"],
                        max_length=96,
                    )
            target_line = (
                "<strong>Reference answer from the full request:</strong> "
                f"{escape(reference_answer_preview or 'Pending')}"
            )
        elif using_inferred_tool:
            target_line = (
                f"<strong>Explaining why agent selected:</strong> {escape(target_tool_label)}"
            )
        else:
            target_line = f"<strong>Fallback target tool:</strong> {escape(target_tool_label)}"
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
                if isinstance(segmenter, LinguisticSegmenter):
                    st.caption(
                        f"{len(user_segments)} user segments from `{segmenter.model_id}` "
                        f"with stray_merge=`{segmenter.stray_merge}`."
                    )
                else:
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
                    diagnostic_label = (
                        "Linguistic segment diagnostics"
                        if isinstance(segmenter, LinguisticSegmenter)
                        else "Semantic boundary diagnostics"
                    )
                    st.markdown(f"**{diagnostic_label}**")
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
            elif scorer_backend == FINAL_ANSWER_SIMILARITY_LABEL:
                with st.spinner("Loading embedding model..."):
                    try:
                        embedder = load_final_answer_embedder(final_answer_embedding_model_id)
                    except Exception as error:  # noqa: BLE001
                        st.error(
                            "Could not load embedding model "
                            f"{final_answer_embedding_model_id!r}: {error}"
                        )
                        return
                agent_callable = build_complete_agent_callable(
                    inference_backend=inference_backend,
                    inference_model_name=inference_model_name,
                    system_prompt=system_prompt,
                    tool_context=tool_context,
                    hf_max_new_tokens=int(hf_max_new_tokens)
                    if inference_backend == "HF local"
                    else 256,
                    hf_trust_remote_code=bool(hf_trust_remote_code)
                    if inference_backend == "HF local"
                    else False,
                )
                inference_result_is_current = (
                    st.session_state.get("agentic_inference_signature")
                    == current_inference_signature
                    and st.session_state.get("agentic_inference_backend") == inference_backend
                    and st.session_state.get("agentic_inference_model") == inference_model_name
                )
                with st.spinner("Resolving full-run reference answer..."):
                    reference_answer, reused_existing_inference, reference_error = (
                        resolve_full_run_reference_answer(
                            agent_callable=agent_callable,
                            user_request=user_request,
                            inference_result=inference_result,
                            inference_result_is_current=inference_result_is_current,
                        )
                    )
                if not reference_answer:
                    st.error(
                        "Could not obtain a final answer for the full user request, so the "
                        "final answer similarity scorer cannot run. "
                        f"Reason: {reference_error or 'unknown error'}"
                    )
                    return
                primary_scorer = FinalAnswerSimilarityScorer(
                    agent_callable=agent_callable,
                    embedder=embedder,
                    reference_answer=reference_answer,
                    empty_prompt=empty_prompt,
                )
                primary_label = FINAL_ANSWER_SIMILARITY_LABEL
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
                    defer_empty_coalition_evaluation=using_exact_computation,
                )
                try:
                    explanation, algorithm_label = compute_interaction_explanation(
                        game=game,
                        index=index,
                        max_order=max_order,
                        budget=budget,
                    )
                except (ExactComputationLimitError, UnsupportedExactIndexError) as error:
                    st.error(f"Could not compute the {index} explanation: {error}")
                    return
                except CoalitionEvaluationIncompleteError as error:
                    st.error(str(error))
                    metrics = error.metrics
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {"metric": "coalition_total", "value": metrics.coalition_total},
                                {"metric": "real_count", "value": metrics.real_count},
                                {"metric": "fallback_count", "value": metrics.fallback_count},
                                {
                                    "metric": "retry_triggered_count",
                                    "value": metrics.retry_triggered_count,
                                },
                                {
                                    "metric": "retry_success_count",
                                    "value": metrics.retry_success_count,
                                },
                                {
                                    "metric": "retry_exhausted_count",
                                    "value": metrics.retry_exhausted_count,
                                },
                                {
                                    "metric": "semantic_failure_count",
                                    "value": metrics.semantic_failure_count,
                                },
                                {
                                    "metric": "remote_request_count",
                                    "value": metrics.remote_request_count,
                                },
                                {
                                    "metric": "embedding_call_count",
                                    "value": metrics.embedding_call_count,
                                },
                            ]
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                    return
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
                        defer_empty_coalition_evaluation=using_exact_computation,
                    )
                    try:
                        lexical_explanation, _lexical_algorithm_label = (
                            compute_interaction_explanation(
                                game=lexical_game,
                                index=index,
                                max_order=max_order,
                                budget=budget,
                            )
                        )
                    except (
                        ExactComputationLimitError,
                        UnsupportedExactIndexError,
                        CoalitionEvaluationIncompleteError,
                    ) as error:
                        st.warning(f"Could not compute the keyword-scorer comparison: {error}")
                        compare_with_lexical = False
                        lexical_result = None
                    else:
                        lexical_first_order = lexical_explanation.get_n_order(order=1)
                        lexical_frame = values_to_frame(lexical_first_order, user_segments)
                        lexical_matrix = pairwise_matrix_from_explanation(
                            lexical_explanation,
                            lexical_game.n_players,
                        )
                        lexical_pair_label, lexical_pair_value = strongest_pair(
                            lexical_matrix,
                            labels,
                        )
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
                            "top_value": 0.0
                            if lexical_top is None
                            else float(lexical_top["attribution"]),
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
                final_answer_embedding_model_id,
                bool(enable_fallback_target_selection),
                show_lexical_comparison,
                segmenter_choice,
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
                "algorithm_label": algorithm_label,
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
                "final_answer_scorer_meta": (
                    {
                        "embedding_model_id": final_answer_embedding_model_id,
                        "reference_answer": primary_scorer.reference_answer,
                        "reused_existing_inference": reused_existing_inference,
                        "empty_raw_similarity": primary_scorer.last_empty_raw_similarity,
                    }
                    if primary_label == FINAL_ANSWER_SIMILARITY_LABEL
                    else None
                ),
            }

        result = st.session_state.result
        if result is None:
            st.error("No explanation result is available. Click Run explanation to compute one.")
            return
        primary_label = result["primary_label"]
        algorithm_label = result["algorithm_label"]
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
        final_answer_scorer_meta = result.get("final_answer_scorer_meta")
        delta_support = float(full_score) - float(empty_score)
        is_final_answer_result = primary_label == FINAL_ANSWER_SIMILARITY_LABEL
        if is_final_answer_result:
            if delta_support > DELTA_STATUS_THRESHOLD:
                support_status = "Higher semantic fidelity"
                support_interpretation = (
                    "The complete prompt's final answer is more similar to the full-run "
                    "reference answer than the empty-request baseline."
                )
            elif delta_support < -DELTA_STATUS_THRESHOLD:
                support_status = "Lower semantic fidelity"
                support_interpretation = (
                    "The complete prompt's final answer is less similar to the full-run "
                    "reference answer than the empty-request baseline."
                )
            else:
                support_status = "Neutral / weak change"
                support_interpretation = (
                    "The complete prompt does not change final-answer similarity much compared "
                    "with the empty-request baseline."
                )
        elif delta_support > DELTA_STATUS_THRESHOLD:
            support_status = "Supported by the prompt"
            support_interpretation = (
                "The complete prompt increases support for the target tool compared with the "
                "baseline."
            )
        elif delta_support < -DELTA_STATUS_THRESHOLD:
            support_status = "Reduced by the prompt"
            support_interpretation = (
                "The complete prompt reduces support for the target tool compared with the "
                "baseline."
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
            or final_answer_scorer_meta is not None
        )
        tab_names = ["Summary", "Attribution", "Interactions"]
        if debug_requested:
            tab_names.append("Debug")
        tabs = st.tabs(tab_names)

        with tabs[0]:
            st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
            summary_heading = (
                "**Final answer similarity overview**"
                if is_final_answer_result
                else "**Tool support overview**"
            )
            metric_labels = (
                {
                    "target": "Scorer",
                    "full": "Normalized full-coalition value = v(all user segments)",
                    "empty": "Normalized empty-coalition value = v(empty user request)",
                    "delta": "Semantic fidelity gain",
                }
                if is_final_answer_result
                else {
                    "target": "Target tool",
                    "full": "Full support score = v(all user segments)",
                    "empty": "Baseline = fixed context + empty user request",
                    "delta": "Delta support",
                }
            )
            metric_target_value = (
                "Final-answer semantic similarity" if is_final_answer_result else target_tool
            )
            st.markdown(summary_heading)
            st.markdown(
                f"""
                <div class="metric-strip">
                    <div class="metric-card">
                        <span>{metric_labels["target"]}</span>
                        <strong>{escape(metric_target_value)}</strong>
                    </div>
                    <div class="metric-card">
                        <span>{metric_labels["full"]}</span>
                        <strong>{full_score:.3f}</strong>
                    </div>
                    <div class="metric-card">
                        <span>{metric_labels["empty"]}</span>
                        <strong>{empty_score:.3f}</strong>
                    </div>
                    <div class="metric-card">
                        <span>{metric_labels["delta"]}</span><strong>{delta_support:.3f}</strong>
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
            if is_final_answer_result:
                st.caption(
                    "With k-SII up to order 2, first-order scores alone do not sum to the "
                    "total game value; pairwise interactions account for the remaining "
                    "contribution."
                )
            bar_xlabel = (
                "Final-answer semantic-fidelity attribution"
                if is_final_answer_result
                else "Target-tool attribution"
            )
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
                        st.pyplot(polish_bar(fig, ax, xlabel=bar_xlabel), clear_figure=True)

        with tabs[2]:
            st.markdown("**First- and second-order interaction heatmap**")
            st.caption(
                "Diagonal: first-order segment attribution. Off-diagonal: pairwise k-SII "
                "interaction."
            )
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
                            "masked_user_request",
                            "raw_similarity",
                            "normalized_score",
                            "execution_status",
                            "execution_error",
                        ]
                        st.dataframe(
                            debug_frame[
                                [column for column in debug_columns if column in debug_frame]
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                if final_answer_scorer_meta is not None:
                    with st.expander("Final answer similarity details", expanded=False):
                        diagnostics = interaction_order_diagnostics(
                            explanation,
                            full_value=full_score,
                            empty_value=empty_score,
                        )
                        st.write(
                            f"Embedding model: `{final_answer_scorer_meta['embedding_model_id']}`"
                        )
                        st.write("Normalized by empty-coalition raw similarity: `True`")
                        empty_raw_similarity = final_answer_scorer_meta["empty_raw_similarity"]
                        st.write(
                            f"Raw empty-coalition similarity: `{empty_raw_similarity:.3f}`"
                            if empty_raw_similarity is not None
                            else "Raw empty-coalition similarity: not yet computed"
                        )
                        st.write(
                            "Reused the existing Inference tab result as the reference answer: "
                            f"`{final_answer_scorer_meta['reused_existing_inference']}`"
                        )
                        raw_full_similarity = (
                            float(diagnostics["full_value"]) + float(empty_raw_similarity)
                            if empty_raw_similarity is not None
                            else None
                        )
                        diagnostic_frame = pd.DataFrame(
                            [
                                {
                                    "quantity": "raw empty-coalition cosine similarity",
                                    "value": empty_raw_similarity,
                                },
                                {
                                    "quantity": "raw full-coalition cosine similarity",
                                    "value": raw_full_similarity,
                                },
                                {
                                    "quantity": "normalized full-coalition game value",
                                    "value": diagnostics["full_value"],
                                },
                                {
                                    "quantity": "sum of order-1 interaction values",
                                    "value": diagnostics["order_1_sum"],
                                },
                                {
                                    "quantity": "sum of unique order-2 pairwise interaction values",
                                    "value": diagnostics["order_2_sum"],
                                },
                                {
                                    "quantity": "k-SII efficiency residual",
                                    "value": diagnostics["residual"],
                                },
                            ]
                        )
                        diagnostic_frame["value"] = diagnostic_frame["value"].map(
                            lambda value: "n/a" if value is None else f"{float(value):.6f}"
                        )
                        st.dataframe(
                            diagnostic_frame,
                            use_container_width=True,
                            hide_index=True,
                        )
                        if abs(diagnostics["residual"]) > EFFICIENCY_RESIDUAL_TOLERANCE:
                            st.warning(
                                "k-SII efficiency residual exceeds tolerance "
                                f"{EFFICIENCY_RESIDUAL_TOLERANCE:g}: "
                                f"{diagnostics['residual']:.6g}"
                            )
                        failed_coalition_rows = [
                            row
                            for row in llm_debug_outputs
                            if row.get("execution_status") not in (None, "ok")
                        ]
                        if failed_coalition_rows:
                            st.warning(
                                f"{len(failed_coalition_rows)} of {len(llm_debug_outputs)} "
                                "sampled coalitions did not produce a usable final answer "
                                "(agent error or empty answer) and were scored with the "
                                "configured fallback raw similarity instead of being embedded. "
                                "See the model output diagnostics table above for per-coalition "
                                "execution_status/execution_error."
                            )
                        else:
                            st.caption("No coalition execution failures for this run.")
                        st.markdown("**Reference final answer (full request)**")
                        st.code(final_answer_scorer_meta["reference_answer"], language="text")
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
                    st.write(f"Algorithm: `{algorithm_label}`")
                    st.write(f"Index: `{index}`")
                    st.write(f"Max order: `{max_order}`")
                    if not using_exact_computation:
                        st.write(f"Budget: `{budget}`")
                    st.write("Full coalition prompt:")
                    st.code(full_prompt, language="text")
                    st.write("Empty coalition prompt:")
                    st.code(empty_prompt, language="text")

        st.caption(f"Demo path: `{display_demo_path()}`")


if __name__ == "__main__":
    main()
