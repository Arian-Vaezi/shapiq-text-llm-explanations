"""Streamlit demo for explaining agentic tool-use decisions with shapiq."""

from __future__ import annotations

import os

# Must be set before torch/transformers/spaCy/sentence-transformers are
# imported (directly below, and transitively by linguistic_segmenter /
# semantic_segmenter / final_answer_similarity_scorer). Several of this app's
# ML dependencies each bundle their own OpenMP runtime (libomp.dylib); loading
# more than one in the same process on macOS causes a native SIGSEGV inside
# libomp's thread-barrier code (confirmed via ~/Library/Logs/DiagnosticReports
# .ips crash reports) with no Python traceback -- Streamlit just exits.
# KMP_DUPLICATE_LIB_OK=TRUE alone does NOT prevent this crash (verified): it
# only suppresses the *graceful* "duplicate runtime" abort message, not the
# underlying multi-threaded barrier corruption. OMP_NUM_THREADS=1 is what
# actually avoids the crash, by keeping every OpenMP-linked library
# (PyTorch, spaCy/thinc, NumPy/OpenBLAS, sentence-transformers) to a single
# worker thread, so the conflicting thread-pool barrier code never runs.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import importlib.util
import math
import sys
import threading
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
from gemini_agent import run_gemini_tool_inference
from groq_agent import run_groq_tool_inference
from hf_router import (
    HFArgumentExtractor,
    LocalHFClassificationRouter,
    LocalHFRouter,
)
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
    DEFAULT_LOGPROB_MODEL_ID,
    CalibratedToolLogOddsScorer,
    LexicalToolScorer,
    LLMToolScorer,
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


SV_INDEX = "SV"
SV_MAX_ORDER = 1
KSII_INDEX = "k-SII"
KSII_MAX_ORDER = 2
DELTA_STATUS_THRESHOLD = 0.01
DEFAULT_MOCK_QUERY = "Will it rain in Berlin tomorrow morning?"
FINAL_ANSWER_SIMILARITY_LABEL = "Final answer semantic similarity"
LOGPROB_SCORER_LABEL = "Calibrated multiclass tool log-odds (HF local)"
# Shared session-state key for the Explanation tab's "Scoring method" selectbox,
# read from the Inference tab too so "HF local" inference can detect that the
# calibrated log-odds scorer is selected and route through the same selected HF
# model instance instead of the separate structured-JSON router.
SCORER_BACKEND_SESSION_KEY = "agentic_explanation_scorer_backend"
HF_LOCAL_MODEL_OPTIONS = ["Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-3B-Instruct"]
# Single canonical session-state key for the "HF local" backend's model dropdown,
# read from both the Inference tab (router/calibrated inference) and the
# Explanation tab (calibrated log-odds scorer settings, read-only there) so the
# two can never silently diverge onto two different loaded model instances.
HF_MODEL_ID_SESSION_KEY = "agentic_hf_model_id"
SAME_HF_MODEL_EXPLANATION = (
    "In calibrated HF-local mode, inference and explanation use the same selected "
    "HF model/tokenizer instance."
)
LOGPROB_SCORER_HELP = (
    "Scores each retained-segment coalition by the local router model's calibrated "
    "log-odds for the target tool against all other available routing decisions. "
    "The score is model-internal routing preference, not trajectory correctness "
    "or tool-execution success."
)
# Temporary HF model-lifecycle tracing for the "two Qwen models resident at
# once" / MPS crash investigation. Set to True only for local debugging; keep
# False in normal runs. Remove once the investigation is closed out.
DEBUG_HF_LIFECYCLE = False


def _hf_lifecycle_log(*parts: object) -> None:
    """Print a temporary HF model-lifecycle trace line, gated by DEBUG_HF_LIFECYCLE."""
    if not DEBUG_HF_LIFECYCLE:
        return
    print(
        "[HF LIFECYCLE]",
        f"pid={os.getpid()}",
        f"thread={threading.current_thread().name}",
        *parts,
    )


EFFICIENCY_RESIDUAL_TOLERANCE = 1e-4
MOCK_SYSTEM_SEGMENTS = [
    "Use weather_tool for weather, rain, temperature, forecast, or city-date questions.",
    "Use calculator_tool for exact arithmetic, totals, percentages, and numeric expressions.",
    "Use web_search_tool when the answer depends on current, latest, recent, or live information.",
    "Use no_tool for stable conceptual explanations that do not require external data.",
]
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
    _hf_lifecycle_log("[HF LOAD] entering load_local_hf_router", f"model_name={model_name}")
    router = LocalHFRouter(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        trust_remote_code=trust_remote_code,
    )
    _hf_lifecycle_log("[HF LOAD] LocalHFRouter loaded", f"model_name={model_name}")
    if DEBUG_HF_LIFECYCLE:
        first_param = next(router.model.parameters())
        _hf_lifecycle_log(
            "[HF MODEL]",
            f"device={first_param.device}",
            f"dtype={first_param.dtype}",
            f"model_id={model_name}",
        )
    return router


@st.cache_resource
def load_logprob_scorer(
    model_id: str,
    *,
    max_pairs_per_batch: int,
    device: str | None = None,
    dtype: str = "auto",
) -> CalibratedToolLogOddsScorer:
    """Load and cache the optional local HuggingFace calibrated log-odds scorer.

    ``max_pairs_per_batch`` (and ``device``/``dtype``) are part of the cache
    key, not mutated on the returned instance afterwards -- a different batch
    size loads (or reuses) a distinct cached scorer instead of reaching into a
    shared cached object and changing its configuration in place.
    """
    _hf_lifecycle_log("[HF LOAD] entering load_logprob_scorer", f"model_id={model_id}")
    scorer = CalibratedToolLogOddsScorer(
        model_id=model_id,
        device=device,
        dtype=dtype,
        max_pairs_per_batch=max_pairs_per_batch,
    )
    _hf_lifecycle_log("[HF LOAD] CalibratedToolLogOddsScorer loaded", f"model_id={model_id}")
    if DEBUG_HF_LIFECYCLE:
        first_param = next(scorer.model.parameters())
        _hf_lifecycle_log(
            "[HF MODEL]",
            f"device={first_param.device}",
            f"dtype={first_param.dtype}",
            f"model_id={model_id}",
        )
    return scorer


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
.segment-chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 0 0 0.9rem 0;
}
.segment-chip {
    background: #fffdf7;
    border: 1px solid #d6cab2;
    border-left: 4px solid #b15d3b;
    border-radius: 999px;
    color: #403d37;
    font-size: 0.85rem;
    max-width: 360px;
    padding: 0.35rem 0.75rem 0.35rem 0.65rem;
}
.segment-chip strong {
    color: #222;
    margin-right: 0.3rem;
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
        # Skip re-evaluation if this game was already precomputed by an earlier call
        # (e.g. computing SV and k-SII from the same game): every one of the
        # 2**n_players coalitions was already scored exactly once, so a second call
        # here must read from that cached table instead of invoking the scorer again.
        if not getattr(game, "precomputed", False):
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


def compute_dual_index_explanations(
    *,
    game: object,
    budget: int | None,
) -> tuple[shapiq.InteractionValues, str, shapiq.InteractionValues, str]:
    """Compute standard Shapley Values (SV) and pairwise k-SII from the same game.

    The bar plot's individual segment effects and the heatmap's pairwise interactions
    are theoretically distinct indices (``SV`` at ``max_order=1`` vs. ``k-SII`` at
    ``max_order=2``), not two views of one k-SII result. Both are computed here from
    the same ``game``/players/target tool/value function via two
    :func:`compute_interaction_explanation` calls.

    For ``game.n_players <= MAX_EXACT_DEMO_PLAYERS`` the second call reuses the first
    call's precomputed coalition table (see the ``precomputed`` guard in
    :func:`compute_interaction_explanation`), so every coalition's value function
    (the scorer/API call) is still only evaluated once, not once per index. Above
    that limit, SV and k-SII each get their own real shapiq approximator, since
    approximators sample coalitions independently and cannot share evaluations.

    Returns:
        A 4-tuple of ``(sv_explanation, sv_algorithm_label, ksii_explanation,
        ksii_algorithm_label)``.
    """
    sv_explanation, sv_algorithm_label = compute_interaction_explanation(
        game=game,
        index=SV_INDEX,
        max_order=SV_MAX_ORDER,
        budget=budget,
    )
    ksii_explanation, ksii_algorithm_label = compute_interaction_explanation(
        game=game,
        index=KSII_INDEX,
        max_order=KSII_MAX_ORDER,
        budget=budget,
    )
    return sv_explanation, sv_algorithm_label, ksii_explanation, ksii_algorithm_label


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


def top_pairwise_interactions(
    pairwise_matrix: pd.DataFrame,
    user_segments: list[ToolUseSegment],
    n: int = 5,
) -> list[dict[str, object]]:
    """Return the top N off-diagonal pairwise k-SII interactions sorted by |value|."""
    rows: list[dict[str, object]] = []
    n_players = pairwise_matrix.shape[0]
    for i in range(n_players):
        for j in range(i + 1, n_players):
            value = float(pairwise_matrix.iloc[i, j])
            rows.append(
                {
                    "segment_i": user_segments[i].label,
                    "segment_j": user_segments[j].label,
                    "text_i": user_segments[i].text,
                    "text_j": user_segments[j].text,
                    "value": value,
                    "type": "complementary" if value > 0 else "redundant",
                }
            )
    rows.sort(key=lambda r: abs(float(r["value"])), reverse=True)
    return rows[:n]


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
    title: str = "Individual segment effects — Shapley Values (SV)",
) -> plt.Figure:
    """Make package bar plot fit the Streamlit layout."""
    fig.set_size_inches(6.2, 3.7)
    ax.set_title("", loc="center")
    ax.set_title(title, loc="left", fontsize=12, pad=8)
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
    *,
    title: str = "Pairwise segment interactions — k-SII, order 2",
) -> plt.Figure:
    """Make package heatmap fit the Streamlit layout."""
    fig.set_size_inches(6.0, 4.7)
    ax.set_title("", loc="center")
    ax.set_title(
        title,
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
    calibrated_hf_mode: bool = False,
    logprob_model_id: str = DEFAULT_LOGPROB_MODEL_ID,
    logprob_max_pairs_per_batch: int = 1,
) -> Callable[[str], object]:
    """Build a backend-agnostic callable that runs the complete tool-calling agent.

    Mirrors the Inference tab's "Run inference" action (router/tool-choice -> tool
    execution -> final answer) exactly, but parameterized over the user request so
    it can be re-run once per Shapley coalition by
    ``final_answer_similarity_scorer.FinalAnswerSimilarityScorer``.

    ``calibrated_hf_mode`` selects the calibrated HF-local path: when
    ``inference_backend == "HF local"`` and the Explanation tab's scorer is set
    to ``LOGPROB_SCORER_LABEL``, inference must use the exact same
    ``CalibratedToolLogOddsScorer`` model/tokenizer instance (via
    ``LocalHFClassificationRouter`` for routing and ``HFArgumentExtractor`` for
    arguments) instead of loading a separate structured-JSON ``LocalHFRouter``.
    Either way, ``inference_model_name`` is always the single ``HF_MODEL_ID_SESSION_KEY``
    dropdown selection from the Inference tab -- there is no second, independently
    editable model id for the structured-JSON path.
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
        if calibrated_hf_mode:
            try:
                primary_scorer = load_logprob_scorer(
                    logprob_model_id,
                    max_pairs_per_batch=logprob_max_pairs_per_batch,
                )
                classification_router = LocalHFClassificationRouter(primary_scorer)
                _hf_lifecycle_log(
                    "[HF ROUTING] LocalHFClassificationRouter reusing scorer instance",
                    f"model_id={logprob_model_id}",
                )
                target_choice = classification_router.choose_tool(
                    user_request,
                    system_prompt=system_prompt,
                    tool_descriptions=TOOLS,
                )
                argument_extractor = HFArgumentExtractor(
                    model=primary_scorer.model,
                    tokenizer=primary_scorer.tokenizer,
                    device=primary_scorer.device,
                )
                _hf_lifecycle_log(
                    "[HF ARGUMENTS] HFArgumentExtractor reusing scorer model/tokenizer",
                    f"model_id={logprob_model_id}",
                )
                tool_arguments = argument_extractor.extract_arguments(
                    user_request=user_request,
                    selected_tool=target_choice.tool,
                    tool_descriptions=TOOLS,
                )
                agent_response = (
                    f"I would use {target_choice.tool} for this request (calibrated "
                    "A/B/C/D routing evidence, same model instance as the coalition scorer)."
                )
                return types.SimpleNamespace(
                    selected_tool=target_choice.tool,
                    tool_arguments=tool_arguments,
                    agent_response=agent_response,
                    raw_response=f"calibrated_scores={target_choice.scores}",
                    debug_prompt=None,
                    error=None,
                    available=True,
                    calibrated_hf_mode=True,
                    # Display-only: the router's raw calibrated log-odds per tool, exposed so
                    # the UI can render a presentation-level probability distribution (via a
                    # local softmax) without re-deriving it from scorer internals.
                    tool_scores=dict(target_choice.scores),
                )
            except Exception as error:  # noqa: BLE001
                return types.SimpleNamespace(
                    selected_tool=None,
                    tool_arguments={},
                    agent_response="",
                    raw_response="",
                    debug_prompt=None,
                    error=f"HF calibrated inference failed: {error}",
                    available=False,
                )
        try:
            _hf_lifecycle_log(
                "[HF LOAD] using structured LocalHFRouter", f"model_name={inference_model_name}"
            )
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


SV_EFFICIENCY_TOLERANCE = 1e-6


def _as_finite_float(value: object) -> float | None:
    """Return ``value`` as a finite ``float``, or ``None`` if that is not possible."""
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _extract_raw_log_odds_summary(debug_outputs: object) -> tuple[float, float] | None:
    """Return ``(raw_full_score, raw_empty_score)`` from a scorer's full-request debug record.

    Only the ``CalibratedToolLogOddsScorer`` currently populates
    ``target_vs_all_log_odds`` (``h(N)``, the raw target-vs-all log-odds for the full
    prompt) and ``empty_coalition_log_odds`` (``h(empty)``, the same raw log-odds for
    the empty coalition) on its debug records, so this returns ``None`` for every
    other scorer's debug output shape instead of fabricating raw values.

    For this scorer, the normalized Shapley game value ``V(S) = h(S) - h(empty)`` is
    zero at the empty coalition *by construction* -- so the main-UI baseline metric
    must show these raw, non-normalized scores instead, or it would always read
    0.000 regardless of the actual routing evidence.
    """
    if not isinstance(debug_outputs, dict):
        return None
    raw_full_score = _as_finite_float(debug_outputs.get("target_vs_all_log_odds"))
    raw_empty_score = _as_finite_float(debug_outputs.get("empty_coalition_log_odds"))
    if raw_full_score is None or raw_empty_score is None:
        return None
    return raw_full_score, raw_empty_score


def softmax_dict(scores: dict[str, float]) -> dict[str, float]:
    """Convert log-odds scores into a display-only softmax probability distribution.

    Presentation-only: mirrors how ``CalibratedToolLogOddsScorer`` derives its own
    diagnostic ``calibrated_probabilities`` from raw calibrated log-odds, so the
    Agent Result routing chart can show the same kind of distribution without
    re-deriving it from scoring internals.
    """
    if not scores:
        return {}
    max_score = max(scores.values())
    exp_scores = {tool: math.exp(score - max_score) for tool, score in scores.items()}
    total = sum(exp_scores.values())
    if total <= 0:
        return dict.fromkeys(scores, 0.0)
    return {tool: value / total for tool, value in exp_scores.items()}


def build_takeaway_sentence(
    *,
    target_tool: str,
    attribution_frame: pd.DataFrame,
    pair_label: str,
    pair_value: float,
) -> str:
    """Generate a conservative one-sentence takeaway from already-computed SV/k-SII results.

    Uses only the top Shapley Value segment(s) and the strongest k-SII pair already
    computed for this run as a template -- it does not run any separate free-form
    generation step.
    """
    if attribution_frame.empty:
        return (
            f"No individual segment attributions are available to summarize the "
            f"`{target_tool}` decision."
        )

    positive = attribution_frame[attribution_frame["attribution"] > 0]
    driver_frame = positive if not positive.empty else attribution_frame
    driver_labels = list(driver_frame["segment"].head(2))
    driver_text = " and ".join(driver_labels) if driver_labels else "no single segment"

    if pair_label == "No pair" or abs(pair_value) < 0.03:
        return (
            f"The decision is mainly driven by {driver_text}. No pairwise interaction is "
            f"strong enough to stand out, so individual segment effects dominate the "
            f"`{target_tool}` decision."
        )
    relation = "jointly strengthen" if pair_value > 0 else "are partly redundant for"
    return (
        f"The decision is mainly driven by {driver_text}. The strongest interaction is "
        f"{pair_label}, suggesting these segments {relation} the `{target_tool}` decision."
    )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="tool-title">
            <h1>Agentic Tool-Use Explanation</h1>
            <p>Explain why an agent selected a tool by attributing the decision to user-request segments.</p>
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

    trace_name = "Custom request"

    # ------------------------------------------------------------------
    # 1. Top-level mode selector + developer mode toggle
    # ------------------------------------------------------------------
    mode_col, dev_col = st.columns([3, 1])
    with mode_col:
        st.markdown('<div class="section-label">Explanation mode</div>', unsafe_allow_html=True)
        explanation_mode = st.radio(
            "Explanation mode",
            ["HF Local", "API Agent"],
            captions=[
                "Model-internal routing evidence",
                "Black-box Groq tool-use trajectory",
            ],
            horizontal=True,
            key="agentic_explanation_mode",
            label_visibility="collapsed",
        )
    with dev_col:
        st.markdown('<div class="section-label">&nbsp;</div>', unsafe_allow_html=True)
        developer_mode = st.toggle(
            "Developer mode",
            value=False,
            key="agentic_developer_mode",
            help="Show segmentation, scorer, and raw-diagnostic controls for debugging.",
        )

    inference_backend = "HF local" if explanation_mode == "HF Local" else "Groq"

    if inference_backend == "HF local":
        inference_model_name = st.selectbox(
            "HF model",
            HF_LOCAL_MODEL_OPTIONS,
            key=HF_MODEL_ID_SESSION_KEY,
            help=(
                "Local HF model used for both routing and the calibrated log-odds "
                "explanation scorer."
            ),
        )
    else:
        inference_model_name = st.text_input(
            "Groq model",
            value="llama-3.1-8b-instant",
            key="agentic_groq_inference_model",
        )
        if not os.getenv("GROQ_API_KEY"):
            st.warning("GROQ_API_KEY is not set. Add it to run Groq inference.")

    # ------------------------------------------------------------------
    # 2. Shared input area
    # ------------------------------------------------------------------
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
        format_func=lambda name: (
            name if name == example_placeholder else scenario_prompt_label(name)
        ),
        key="agentic_try_example_select",
        on_change=apply_selected_example,
    )
    if st.session_state.get("agentic_pending_example_request") is not None:
        st.rerun()

    run_full_pipeline_clicked = st.button(
        "Run full pipeline",
        type="primary",
        key="agentic_run_full_pipeline",
        use_container_width=True,
        help="Runs the agent, then prepares the shapiq explanation for the selected tool.",
    )

    current_inference_signature = (user_request, system_prompt, tool_context)
    if st.session_state.get("agentic_inference_signature") != current_inference_signature:
        st.session_state["agentic_inference_signature"] = current_inference_signature
        st.session_state["agentic_inferred_tool"] = None
        st.session_state["agentic_inference_result"] = None
        st.session_state["agentic_inference_backend"] = None
        st.session_state["agentic_inference_model"] = None

    # ------------------------------------------------------------------
    # 6. Developer mode: grouped controls (hidden entirely when OFF)
    # ------------------------------------------------------------------
    latest_inference_backend = st.session_state.get("agentic_inference_backend")
    has_groq_inference_result = (
        latest_inference_backend == "Groq"
        and st.session_state.get("agentic_inference_result") is not None
    )
    groq_reference_result = (
        st.session_state.get("agentic_inference_result") if has_groq_inference_result else None
    )
    trajectory_match_available = (
        groq_reference_result is not None
        and st.session_state.get("agentic_inferred_tool") in TOOLS
        and bool(getattr(groq_reference_result, "tool_arguments", None))
    )

    scorer_backend_key = SCORER_BACKEND_SESSION_KEY
    logprob_model_id = DEFAULT_LOGPROB_MODEL_ID
    max_pairs_per_batch = 1
    router_model_id = DEFAULT_GROQ_ROUTER_MODEL_ID
    soft_vote_model_id = DEFAULT_GROQ_ROUTER_MODEL_ID
    soft_vote_n_samples = DEFAULT_GROQ_SOFT_VOTE_N_SAMPLES
    soft_vote_temperature = DEFAULT_GROQ_SOFT_VOTE_TEMPERATURE
    soft_vote_max_retries = DEFAULT_GROQ_SOFT_VOTE_MAX_RETRIES
    soft_vote_seed = None
    final_answer_embedding_model_id = DEFAULT_FINAL_ANSWER_EMBEDDING_MODEL_ID
    show_lexical_comparison = False
    enable_fallback_target_selection = False
    segmenter_choice = "Linguistic (spaCy chunking)"
    segment_threshold = 0.72
    segment_window = 3
    min_segment_words = 1
    show_value_function_details = False
    show_scoring_prompt_preview = False

    forced_default_scorer = (
        LOGPROB_SCORER_LABEL
        if inference_backend == "HF local"
        else ("Groq soft-vote scorer" if has_groq_inference_result else LOGPROB_SCORER_LABEL)
    )

    if not developer_mode:
        st.session_state[scorer_backend_key] = forced_default_scorer
        scorer_backend = forced_default_scorer
    else:
        st.markdown('<div class="section-label">Developer settings</div>', unsafe_allow_html=True)
        with st.expander("Segmentation", expanded=False):
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

        with st.expander("Value function / scorer diagnostics", expanded=False):
            show_developer_scorers = st.checkbox(
                "Show developer scoring methods",
                value=False,
                key="agentic_show_developer_scorers",
            )
            scorer_options = [LOGPROB_SCORER_LABEL, FINAL_ANSWER_SIMILARITY_LABEL]
            if inference_backend == "Groq" or has_groq_inference_result:
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
            if scorer_backend_key not in st.session_state:
                st.session_state[scorer_backend_key] = forced_default_scorer
            if st.session_state[scorer_backend_key] not in scorer_options:
                st.session_state[scorer_backend_key] = forced_default_scorer
            scorer_backend = st.selectbox(
                "Scoring method",
                scorer_options,
                key=scorer_backend_key,
            )
            if scorer_backend == LOGPROB_SCORER_LABEL:
                logprob_model_id = st.session_state.get(
                    HF_MODEL_ID_SESSION_KEY, DEFAULT_LOGPROB_MODEL_ID
                )
                st.caption(f"Inherited from HF model selection: `{logprob_model_id}`")
                st.caption(SAME_HF_MODEL_EXPLANATION)
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
                st.caption(LOGPROB_SCORER_HELP)
            elif scorer_backend == FINAL_ANSWER_SIMILARITY_LABEL:
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
                if inference_backend == "Groq" and not os.getenv("GROQ_API_KEY"):
                    st.warning("GROQ_API_KEY is not set. Add it to use this scorer with Groq.")
            elif scorer_backend == "Groq deterministic router":
                router_model_id = st.text_input(
                    "Groq router model",
                    value=DEFAULT_GROQ_ROUTER_MODEL_ID,
                    key="agentic_groq_router_scorer_model",
                )
                st.caption(
                    "Calls the real Groq API once per distinct coalition prompt to ask which "
                    "tool it would route to, and scores 1.0 if that matches the target tool, "
                    "else 0.0."
                )
                if not os.getenv("GROQ_API_KEY"):
                    st.warning("GROQ_API_KEY is not set. Add it to use the Groq router scorer.")
            elif scorer_backend == "Groq soft-vote scorer":
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

        with st.expander("Computation diagnostics", expanded=False):
            st.caption(f"Individual effects: `{SV_INDEX}`")
            st.caption(f"Pairwise interactions: `{KSII_INDEX}`, max_order=`{KSII_MAX_ORDER}`")
            st.caption("Players: user-request segments")
            st.caption("Fixed context: system prompt + tool definitions")
            st.caption("Value function: selected-tool support under the chosen scorer")
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

    def execute_tool_inference() -> object:
        calibrated_hf_mode = (
            inference_backend == "HF local"
            and st.session_state.get(SCORER_BACKEND_SESSION_KEY) == LOGPROB_SCORER_LABEL
        )
        agent_callable = build_complete_agent_callable(
            inference_backend=inference_backend,
            inference_model_name=inference_model_name,
            system_prompt=system_prompt,
            tool_context=tool_context,
            hf_max_new_tokens=256,
            hf_trust_remote_code=False,
            calibrated_hf_mode=calibrated_hf_mode,
            logprob_model_id=inference_model_name
            if calibrated_hf_mode
            else DEFAULT_LOGPROB_MODEL_ID,
            logprob_max_pairs_per_batch=int(
                st.session_state.get("agentic_logprob_pair_batch_size", 1)
            ),
        )
        return agent_callable(user_request)

    if run_full_pipeline_clicked:
        if hasattr(st, "status"):
            with st.status("Running agent...", expanded=True) as status:
                st.write(f"Mode: {explanation_mode}")
                st.write(f"Model: {inference_model_name}")
                inference_result = execute_tool_inference()
                status.update(
                    label="Agent step complete. Preparing explanation...", state="complete"
                )
        else:
            with st.spinner("Running agent..."):
                inference_result = execute_tool_inference()
        inference_result.backend = inference_backend
        inference_result.model = inference_model_name
        st.session_state["agentic_inference_backend"] = inference_backend
        st.session_state["agentic_inference_model"] = inference_model_name
        st.session_state["agentic_inference_result"] = inference_result
        st.session_state["agentic_inferred_tool"] = (
            inference_result.selected_tool if inference_result.selected_tool in TOOLS else None
        )
        st.session_state.has_run = False
        st.session_state.result = None
        st.session_state.pending_run = True
        st.rerun()

    # ------------------------------------------------------------------
    # 3. Two tabs, feeling like two stages of one pipeline
    # ------------------------------------------------------------------
    agent_tab, xai_tab = st.tabs(["1. Agent Result", "2. XAI Explanation"])

    # ------------------------------------------------------------------
    # 4. Tab 1: Agent Result
    # ------------------------------------------------------------------
    with agent_tab:
        st.markdown('<div class="section-label">Agent Result</div>', unsafe_allow_html=True)
        inference_result = st.session_state.get("agentic_inference_result")
        if inference_result is None:
            st.info("Click **Run full pipeline** above to run the agent and select a tool.")
        else:
            inference_error = getattr(inference_result, "error", None)
            if inference_error:
                st.warning(inference_error)
            selected_tool = getattr(inference_result, "selected_tool", None)
            result_backend = getattr(inference_result, "backend", inference_backend)
            result_model = getattr(inference_result, "model", inference_model_name)

            if result_backend == "HF local":
                st.metric("Model routed to", selected_tool or "No tool selected")
                tool_scores = getattr(inference_result, "tool_scores", None)
                if tool_scores:
                    probabilities = softmax_dict(tool_scores)
                    prob_frame = pd.DataFrame(
                        [
                            {"tool": tool_name, "probability": probabilities.get(tool_name, 0.0)}
                            for tool_name in TOOLS
                        ]
                    ).sort_values("probability", ascending=False)
                    st.bar_chart(prob_frame.set_index("tool"), use_container_width=True)
                    st.dataframe(
                        prob_frame.assign(
                            probability=prob_frame["probability"].map(lambda v: f"{v:.3f}")
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )
                st.caption(
                    "Decision and explanation use the same local HF model and calibrated "
                    "log-odds scorer."
                )
            else:
                st.write("**Backend:** Groq")
                st.write(f"**Model:** `{result_model}`")
                st.metric("Selected tool", selected_tool or "No tool selected")
                st.markdown("**Tool arguments**")
                st.json(getattr(inference_result, "tool_arguments", {}))
                assistant_message = (
                    getattr(inference_result, "agent_response", "")
                    or getattr(inference_result, "assistant_answer", "")
                    or getattr(inference_result, "final_answer", "")
                    or f"I recommend `{selected_tool}` for this request."
                )
                st.markdown("**Agent response**")
                st.write(assistant_message)

            if developer_mode:
                with st.expander("Raw backend diagnostics (agent)", expanded=False):
                    st.write(f"Backend: `{result_backend}`")
                    st.write(f"Model: `{result_model}`")
                    st.markdown("**Tool arguments**")
                    st.json(getattr(inference_result, "tool_arguments", {}))
                    raw_trace = getattr(inference_result, "raw_trace", None)
                    if raw_trace is None:
                        raw_trace = {
                            "debug_prompt": getattr(inference_result, "debug_prompt", None),
                            "raw_response": getattr(inference_result, "raw_response", ""),
                        }
                    st.json(raw_trace)

    # ------------------------------------------------------------------
    # 5. Tab 2: XAI Explanation
    # ------------------------------------------------------------------
    with xai_tab:
        st.markdown('<div class="section-label">XAI Explanation</div>', unsafe_allow_html=True)

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

        if developer_mode:
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
        result = st.session_state.result
        result_target_tool = None
        result_target_source = None
        if isinstance(result, dict):
            result_target_tool = result.get("target_tool")
            result_target_source = result.get("target_source")
        if using_inferred_tool:
            target_tool = inferred_tool
            target_source = "Agent Result"
        elif result_target_tool in TOOLS and result_target_source == "fallback explanation scorer":
            target_tool = str(result_target_tool)
            target_source = "fallback explanation scorer"
        else:
            target_tool = None
            target_source = "fallback explanation scorer"

        if not using_inferred_tool and target_tool is None:
            if enable_fallback_target_selection:
                st.info(
                    "No agent result is available. Fallback target selection is enabled; "
                    "the explanation scorer will choose the target when the pipeline runs."
                )
            else:
                st.info(
                    "Click **Run full pipeline** above to run the agent and explain its decision."
                )

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
            # Note: unlike a stale-settings mismatch, `pending_run` is deliberately left
            # untouched here. It is only ever set True immediately before a rerun, right
            # after this exact signature was (re)computed from the freshly stored agent
            # result -- so a mismatch at this point reflects that legitimate change, not
            # a stale request that should cancel the run the user just triggered.
            st.session_state.has_run = False
            st.session_state.result = None
            st.session_state.result_signature = signature
            result = None
            if not using_inferred_tool:
                target_tool = None
                st.session_state.result_signature = (
                    trace_name,
                    user_request,
                    "__pending_target__",
                    scorer_backend,
                    logprob_model_id,
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

        is_final_answer_similarity_scorer = scorer_backend == FINAL_ANSWER_SIMILARITY_LABEL
        if is_final_answer_similarity_scorer:
            reference_answer_preview = None
            if isinstance(result, dict):
                final_answer_scorer_meta_preview = result.get("final_answer_scorer_meta")
                if final_answer_scorer_meta_preview:
                    reference_answer_preview = truncate_label(
                        final_answer_scorer_meta_preview["reference_answer"],
                        max_length=96,
                    )
            explaining_value = reference_answer_preview or "Pending (final-answer similarity)"
        else:
            explaining_value = target_tool if target_tool is not None else "Pending"

        # ---- Compact explanation setup panel ----
        st.markdown(
            f"""
            <div class="setup-line">
                <strong>Explaining:</strong> <code>{escape(str(explaining_value))}</code><br>
                <strong>Target source:</strong> {escape(target_source)}<br>
                <strong>Players:</strong> user-request segments<br>
                <strong>Fixed context:</strong> system prompt + tool definitions<br>
                <strong>Masking:</strong> remove absent segments<br>
                <strong>Individual effects:</strong> SV<br>
                <strong>Pairwise interactions:</strong> k-SII
            </div>
            """,
            unsafe_allow_html=True,
        )

        if (
            target_source == "fallback explanation scorer"
            and isinstance(result, dict)
            and result.get("fallback_choice_scores")
        ):
            with st.expander("Fallback scorer diagnostic", expanded=False):
                st.caption(
                    "These scores come from the selected explanation scorer and were used to "
                    "choose the fallback target tool (no agent result was available)."
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

        # ---- A. Player segmentation ----
        st.markdown('<div class="section-label">Player segmentation</div>', unsafe_allow_html=True)
        chip_html = "".join(
            (
                "<div class='segment-chip'>"
                f"<strong>{escape(segment.label)}:</strong> {escape(segment.text)}</div>"
            )
            for segment in user_segments
        )
        st.markdown(f"<div class='segment-chip-row'>{chip_html}</div>", unsafe_allow_html=True)

        with st.expander("Fixed context: system prompt + tool definitions", expanded=False):
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
            if developer_mode and segment_debug_rows:
                # Duck-typed rather than isinstance(...): Streamlit's local file watcher can
                # hot-reload a same-directory module while an older cached segmenter instance
                # from st.cache_resource survives the rerun, which breaks isinstance checks
                # against the freshly reloaded class.
                is_linguistic_segmenter = hasattr(segmenter, "stray_merge")
                diagnostic_label = (
                    "Linguistic segment diagnostics"
                    if is_linguistic_segmenter
                    else "Semantic boundary diagnostics"
                )
                st.markdown(f"**{diagnostic_label}**")
                st.dataframe(
                    pd.DataFrame(segment_debug_rows),
                    use_container_width=True,
                    hide_index=True,
                )

        if not st.session_state.has_run and not st.session_state.pending_run:
            st.info("Click **Run full pipeline** above to compute the SV / k-SII explanation.")
            return

        run = st.session_state.pending_run
        st.session_state.pending_run = False

        if run:
            try:
                from tool_game import (
                    ToolUseGame,
                )
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
            elif scorer_backend == LOGPROB_SCORER_LABEL:
                with st.spinner("Loading calibrated log-odds scorer model..."):
                    try:
                        primary_scorer = load_logprob_scorer(
                            logprob_model_id,
                            max_pairs_per_batch=int(max_pairs_per_batch),
                        )
                    except Exception as error:  # noqa: BLE001
                        st.error(
                            "Could not load the calibrated log-odds scorer. Install/check "
                            "`transformers` and `torch`, try a smaller causal language model, "
                            f"or check your environment. Details: {error}"
                        )
                        return
                primary_label = LOGPROB_SCORER_LABEL
                _hf_lifecycle_log(
                    "[HF ROUTING] LocalHFClassificationRouter reusing scorer instance",
                    f"model_id={logprob_model_id}",
                )
                classification_router = LocalHFClassificationRouter(primary_scorer)
                classifier_choice = classification_router.choose_tool(
                    user_request,
                    system_prompt=system_prompt,
                    tool_descriptions=TOOLS,
                )
                target_tool = classifier_choice.tool
                target_source = "HF classifier argmax (A/B/C/D)"
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
                    hf_max_new_tokens=256,
                    hf_trust_remote_code=False,
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
                        "No agent result is available. Run the full pipeline first, or enable "
                        "fallback target selection in developer settings."
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
            logprob_full_diagnostics = (
                dict(primary_scorer.last_debug_outputs[0])
                if primary_label == LOGPROB_SCORER_LABEL and primary_scorer.last_debug_outputs
                else None
            )
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
                    sv_explanation, sv_algorithm_label, ksii_explanation, ksii_algorithm_label = (
                        compute_dual_index_explanations(game=game, budget=budget)
                    )
                except (ExactComputationLimitError, UnsupportedExactIndexError) as error:
                    st.error(f"Could not compute the {SV_INDEX}/{KSII_INDEX} explanations: {error}")
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
                first_order_sv = sv_explanation.get_n_order(order=1)
                attribution_frame = values_to_frame(first_order_sv, user_segments)
                pairwise_ksii = ksii_explanation.get_n_order(order=2)
                pairwise_matrix = pairwise_matrix_from_explanation(pairwise_ksii, game.n_players)
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
                        (
                            lexical_sv_explanation,
                            _lexical_sv_algorithm_label,
                            lexical_ksii_explanation,
                            _lexical_ksii_algorithm_label,
                        ) = compute_dual_index_explanations(
                            game=lexical_game,
                            budget=budget,
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
                        lexical_first_order_sv = lexical_sv_explanation.get_n_order(order=1)
                        lexical_frame = values_to_frame(lexical_first_order_sv, user_segments)
                        lexical_pairwise_ksii = lexical_ksii_explanation.get_n_order(order=2)
                        lexical_matrix = pairwise_matrix_from_explanation(
                            lexical_pairwise_ksii,
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
                "sv_algorithm_label": sv_algorithm_label,
                "ksii_algorithm_label": ksii_algorithm_label,
                "full_score": full_score,
                "empty_score": empty_score,
                "first_order_sv": first_order_sv,
                "attribution_frame": attribution_frame,
                "sv_explanation": sv_explanation,
                "ksii_explanation": ksii_explanation,
                "pairwise_ksii": pairwise_ksii,
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
                "logprob_full_diagnostics": logprob_full_diagnostics,
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
            st.error("No explanation result is available. Click Run full pipeline to compute one.")
            return
        primary_label = result["primary_label"]
        sv_algorithm_label = result["sv_algorithm_label"]
        ksii_algorithm_label = result["ksii_algorithm_label"]
        full_score = result["full_score"]
        empty_score = result["empty_score"]
        first_order_sv = result["first_order_sv"]
        attribution_frame = result["attribution_frame"]
        sv_explanation = result["sv_explanation"]
        ksii_explanation = result["ksii_explanation"]
        pairwise_ksii = result.get("pairwise_ksii")
        if pairwise_ksii is None:
            pairwise_ksii = ksii_explanation.get_n_order(order=2)
        pairwise_matrix = result.get("pairwise_matrix")
        if pairwise_matrix is None:
            pairwise_matrix = pairwise_matrix_from_explanation(pairwise_ksii, len(user_segments))
        pair_label = result["pair_label"]
        pair_value = result["pair_value"]
        compare_with_lexical = result["compare_with_lexical"]
        llm_debug_outputs = result["llm_debug_outputs"]
        lexical_result = result["lexical_result"]
        final_answer_scorer_meta = result.get("final_answer_scorer_meta")
        logprob_full_diagnostics = result.get("logprob_full_diagnostics")
        is_final_answer_result = primary_label == FINAL_ANSWER_SIMILARITY_LABEL
        target_tool = result["target_tool"]

        # ---- B. Summary metrics (exactly four) ----
        st.markdown('<div class="section-label">Summary metrics</div>', unsafe_allow_html=True)
        metric_target_value = (
            "Final-answer semantic similarity"
            if is_final_answer_result
            else (target_tool or "Pending")
        )
        # The calibrated log-odds scorer normalizes its Shapley game value against the
        # empty coalition by construction (V(empty) = h(empty) - h(empty) == 0), so the
        # normalized baseline is always trivially zero for this scorer specifically. Show
        # its raw, non-normalized h(∅)/h(N) log-odds instead -- otherwise this card would
        # always read 0.000 for HF Local mode's default scorer regardless of the actual
        # routing evidence.
        raw_log_odds_summary = (
            None
            if is_final_answer_result
            else _extract_raw_log_odds_summary(logprob_full_diagnostics)
        )
        if raw_log_odds_summary is not None:
            raw_full_score, raw_empty_score = raw_log_odds_summary
            explained_increase = raw_full_score - raw_empty_score
            empty_label, empty_display = "Raw empty score h(&empty;)", f"{raw_empty_score:.3f}"
            full_label, full_display = "Raw full score h(N)", f"{raw_full_score:.3f}"
            delta_label = "Explained increase h(N)-h(&empty;)"
        else:
            explained_increase = float(full_score) - float(empty_score)
            empty_label, empty_display = "Baseline support V(&empty;)", f"{empty_score:.3f}"
            full_label, full_display = "Full-request support V(N)", f"{full_score:.3f}"
            delta_label = "Explained increase V(N)-V(&empty;)"
        st.markdown(
            f"""
            <div class="metric-strip">
                <div class="metric-card">
                    <span>Explained decision</span>
                    <strong>{escape(str(metric_target_value))}</strong>
                </div>
                <div class="metric-card">
                    <span>{empty_label}</span>
                    <strong>{empty_display}</strong>
                </div>
                <div class="metric-card">
                    <span>{full_label}</span>
                    <strong>{full_display}</strong>
                </div>
                <div class="metric-card">
                    <span>{delta_label}</span>
                    <strong>{explained_increase:.3f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- C. Individual segment effects -- Shapley Values ----
        st.markdown(
            '<div class="section-label">Individual segment effects — Shapley Values (SV)</div>',
            unsafe_allow_html=True,
        )
        token_attribution_bar_plot, sentence_interaction_heatmap, plot_import_error = (
            load_text_plotters()
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
                fig_ax = token_attribution_bar_plot(first_order_sv, labels, show=False)
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
        st.dataframe(attribution_frame, use_container_width=True, hide_index=True)
        st.caption("Which individual request segments push the agent toward the selected tool?")

        # ---- D. Pairwise interactions -- k-SII ----
        st.markdown(
            '<div class="section-label">Pairwise interactions — k-SII</div>',
            unsafe_allow_html=True,
        )
        if sentence_interaction_heatmap is None:
            st.warning(
                "The shapiq text interaction heatmap is unavailable in this environment. "
                f"Showing a fallback interaction table instead. Details: {plot_import_error}"
            )
            show_fallback_interaction_table(pairwise_matrix, labels)
        else:
            try:
                fig_ax = sentence_interaction_heatmap(ksii_explanation, labels, show=False)
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

        if pairwise_matrix.shape[0] >= 2:
            top_pairs = top_pairwise_interactions(pairwise_matrix, user_segments)
            if top_pairs:
                pairs_frame = pd.DataFrame(
                    [
                        {
                            "segments": f"{row['segment_i']} + {row['segment_j']}",
                            "text": f"{row['text_i']}  |  {row['text_j']}",
                            "k-SII": format_attribution(row["value"]),
                            "type": row["type"],
                        }
                        for row in top_pairs
                    ]
                )
                st.dataframe(pairs_frame, use_container_width=True, hide_index=True)
        st.caption(
            "Which segment pairs jointly trigger or redundantly encode the tool-use decision?"
        )

        # ---- E. Takeaway ----
        takeaway_target = str(metric_target_value)
        takeaway_sentence = build_takeaway_sentence(
            target_tool=takeaway_target,
            attribution_frame=attribution_frame,
            pair_label=pair_label,
            pair_value=pair_value,
        )
        st.success(takeaway_sentence)

        # ---- Developer mode: raw backend diagnostics ----
        if developer_mode:
            with st.expander("Raw backend diagnostics", expanded=False):
                sv_sum = float(sum(getattr(first_order_sv, "dict_values", {}).values()))
                sv_residual = abs(sv_sum - explained_increase)
                sv_efficiency_status = (
                    "passes" if sv_residual <= SV_EFFICIENCY_TOLERANCE else "exceeds tolerance"
                )
                st.caption(
                    f"SV efficiency check: sum(SV)={sv_sum:.3f}, "
                    f"explained increase={explained_increase:.3f}, "
                    f"residual={sv_residual:.6g} ({sv_efficiency_status})"
                )
                diag = interaction_order_diagnostics(
                    ksii_explanation,
                    full_value=full_score,
                    empty_value=empty_score,
                )
                diag_frame = pd.DataFrame(
                    [
                        {
                            "quantity": "sum of order-1 values",
                            "value": f"{diag['order_1_sum']:.6f}",
                        },
                        {
                            "quantity": "sum of order-2 pairwise values",
                            "value": f"{diag['order_2_sum']:.6f}",
                        },
                        {
                            "quantity": "total game value v(N) - v(empty)",
                            "value": f"{diag['total_game_value']:.6f}",
                        },
                        {
                            "quantity": "efficiency residual",
                            "value": f"{diag['residual']:.6g}",
                        },
                    ]
                )
                st.dataframe(diag_frame, use_container_width=True, hide_index=True)
                if abs(diag["residual"]) <= EFFICIENCY_RESIDUAL_TOLERANCE:
                    st.caption(
                        "k-SII order-2 efficiency holds: first- and second-order values "
                        "sum to the total game value."
                    )
                else:
                    st.warning(
                        f"Efficiency residual {diag['residual']:.6g} exceeds tolerance "
                        f"{EFFICIENCY_RESIDUAL_TOLERANCE:g}. "
                        "This may indicate approximation error above the exact-computation limit."
                    )

                if llm_debug_outputs:
                    st.markdown("**Model output diagnostics**")
                    displayed_debug_outputs = llm_debug_outputs[:10]
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
                        "target_tool",
                        "target_label",
                        "candidate_tools",
                        "candidate_labels",
                        "candidate_continuations",
                        "raw_log_scores",
                        "calibration_log_scores",
                        "calibrated_scores",
                        "target_vs_all_log_odds",
                        "empty_coalition_log_odds",
                        "calibrated_probabilities",
                        "argmax_tool",
                        "final_score",
                        "prompt_preview",
                        "masked_user_request",
                        "raw_similarity",
                        "normalized_score",
                        "execution_status",
                        "execution_error",
                    ]
                    st.dataframe(
                        debug_frame[[column for column in debug_columns if column in debug_frame]],
                        use_container_width=True,
                        hide_index=True,
                    )

                if primary_label == LOGPROB_SCORER_LABEL and logprob_full_diagnostics is not None:
                    st.markdown("**Calibrated multiclass tool log-odds (full user request)**")
                    argmax_tool = logprob_full_diagnostics.get("argmax_tool")
                    is_router_argmax = argmax_tool == target_tool
                    calibrated_probability = logprob_full_diagnostics[
                        "calibrated_probabilities"
                    ].get(target_tool)
                    st.caption(LOGPROB_SCORER_HELP)
                    logprob_metric_left, logprob_metric_mid, logprob_metric_right = st.columns(3)
                    logprob_metric_left.metric(
                        "Calibrated probability",
                        f"{calibrated_probability:.3f}"
                        if calibrated_probability is not None
                        else "n/a",
                    )
                    logprob_metric_mid.metric(
                        "Target-vs-all log-odds",
                        f"{logprob_full_diagnostics['target_vs_all_log_odds']:.3f}",
                    )
                    logprob_metric_right.metric(
                        "Local-router argmax match",
                        "Yes" if is_router_argmax else "No",
                        help=f"Local-router argmax candidate: `{argmax_tool}`.",
                    )
                    st.caption(
                        f"{SAME_HF_MODEL_EXPLANATION} Loaded model: `{logprob_model_id}`. "
                        "No second model is loaded for target-tool selection. Argument "
                        "extraction was not executed; this run explains routing only."
                    )

                if final_answer_scorer_meta is not None:
                    st.markdown("**Final answer similarity details**")
                    diagnostics = interaction_order_diagnostics(
                        ksii_explanation,
                        full_value=full_score,
                        empty_value=empty_score,
                    )
                    st.write(f"Embedding model: `{final_answer_scorer_meta['embedding_model_id']}`")
                    st.write("Normalized by empty-coalition raw similarity: `True`")
                    empty_raw_similarity = final_answer_scorer_meta["empty_raw_similarity"]
                    st.write(
                        f"Raw empty-coalition similarity: `{empty_raw_similarity:.3f}`"
                        if empty_raw_similarity is not None
                        else "Raw empty-coalition similarity: not yet computed"
                    )
                    st.write(
                        "Reused the existing agent result as the reference answer: "
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
                            "top_segment": result["top_label"],
                            "top_attribution": result["top_score"],
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
                            "A separate scoring-prompt preview is not available for this "
                            "scoring backend."
                        )

                if show_value_function_details:
                    st.markdown("**Value function details**")
                    st.write(
                        f"Individual effects: `{SV_INDEX}`, max_order=`{SV_MAX_ORDER}`, "
                        f"algorithm: `{sv_algorithm_label}`"
                    )
                    st.write(
                        f"Pairwise interactions: `{KSII_INDEX}`, max_order=`{KSII_MAX_ORDER}`, "
                        f"algorithm: `{ksii_algorithm_label}`"
                    )
                    st.write("Players: user-request segments")
                    st.write("Fixed context: system prompt + tool definitions")
                    st.write("Value function: selected-tool support under the chosen scorer")
                    if not using_exact_computation:
                        st.write(f"Budget: `{budget}`")
                    st.write("Full coalition prompt:")
                    st.code(full_prompt, language="text")
                    st.write("Empty coalition prompt:")
                    st.code(empty_prompt, language="text")

        st.caption(f"Demo path: `{display_demo_path()}`")


if __name__ == "__main__":
    main()
