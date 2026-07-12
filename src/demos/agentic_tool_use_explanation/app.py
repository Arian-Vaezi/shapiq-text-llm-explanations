"""Streamlit demo for explaining agentic tool-use decisions with shapiq."""

from __future__ import annotations

import base64
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
    DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
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
    NativeDirectAnswerScorer,
    NativeToolCallScorer,
    ToolChoice,
    build_canonical_direct_answer_target,
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
# Display-only: per-tool icons for the XAI summary card and Agent Result cards.
# Purely cosmetic -- has no effect on which tool is selected, scored, or explained.
TOOL_ICONS = {
    "weather_tool": "🌦️",
    "calculator_tool": "🧮",
    "web_search_tool": "🌐",
    "no_tool": "💬",
    "parse_failure": "⚠️",
}
TOOL_ROUTE_EXPLANATIONS = {
    "web_search_tool": "This request requires current or externally retrieved information.",
    "weather_tool": "This request requires weather information for a specific place or time.",
    "calculator_tool": "This request requires an exact numerical calculation.",
}
FINAL_ANSWER_SIMILARITY_LABEL = "Final answer semantic similarity"
NATIVE_HF_SCORER_LABEL = "Canonical native tool-identity continuation likelihood"
NATIVE_HF_SCORER_SHORT_LABEL = "Native tool-identity likelihood"
NATIVE_DIRECT_ANSWER_SCORER_LABEL = "Canonical direct-answer continuation likelihood"
NATIVE_DIRECT_ANSWER_SCORER_SHORT_LABEL = "Direct-answer continuation likelihood"
NO_TOOL_SURROGATE_SCORER_LABEL = "Legacy A/B/C/D no-tool probe — surrogate ablation"
LOGPROB_SCORER_LABEL = "Calibrated A/B/C/D classification (Developer ablation)"
# Shared session-state key for the Explanation tab's "Scoring method" selectbox.
SCORER_BACKEND_SESSION_KEY = "agentic_explanation_scorer_backend"
# Single canonical session-state key for the "HF local" backend's model dropdown.
HF_MODEL_ID_SESSION_KEY = "agentic_hf_model_id"
SAME_HF_MODEL_EXPLANATION = (
    "In HF-local mode, inference and explanation use the same selected HF model/tokenizer instance."
)
NATIVE_HF_SCORER_HELP = (
    "Scores each retained-segment coalition by teacher-forced log-probability of a "
    "canonical native-format continuation for the frozen Agent Result tool identity. "
    "The continuation is derived from the selected tokenizer's chat template, stops "
    "at the tool name, and excludes free-form argument tokens."
)
NATIVE_DIRECT_ANSWER_SCORER_HELP = (
    "Scores each retained-segment coalition by the teacher-forced mean "
    "log-probability of a frozen deterministic fragment extracted from the "
    "full-context direct answer. This explains support for that answer content; "
    "it is not a probability of choosing no tool."
)
NO_TOOL_SURROGATE_HELP = (
    "This direct-answer explanation uses the legacy A/B/C/D forced-choice surrogate. "
    "`NoTool` is an artificial candidate for that probe only: it is not executable, "
    "not native, and is never emitted by the native agent."
)
LOGPROB_SCORER_HELP = (
    "Developer ablation: rewrites routing into artificial A/B/C/D labels and scores "
    "calibrated log-odds for the target label. This is not the primary native "
    "tool-calling pipeline."
)
# ---- Unified "Method note" shown at the foot of the summary card ---------------
# Every value-function fidelity gets the *same* note format
# ("Method note — <fidelity name>: <description>"), so a reader comparing runs
# across scorer backends sees one consistent voice instead of three differently
# worded/placed asides (a plain caption for one branch, a warning box for
# another, nothing at all for a third).
NATIVE_TOOL_IDENTITY_METHOD_NOTE = ("Native tool-identity VF", NATIVE_HF_SCORER_HELP)
NATIVE_DIRECT_ANSWER_METHOD_NOTE = ("Native direct-answer VF", NATIVE_DIRECT_ANSWER_SCORER_HELP)
LEGACY_ABCD_METHOD_NOTE = (
    "Legacy A/B/C/D ablation (developer-only)",
    "Rewrites routing into artificial A/B/C/D labels and scores calibrated "
    "log-odds for the target label (or the artificial `NoTool` candidate when "
    "explaining a no-tool/direct-answer decision). This is not the primary native "
    "tool-calling pipeline; none of the A/B/C/D candidates are executable or ever "
    "emitted by the native agent.",
)
NATIVE_HF_CONTINUATION_DIAGNOSTICS = {
    "Target source": "full-context native inference",
    "Continuation type": "canonical native template",
    "Continuation scope": "tool identity only",
    "Arguments included": "no",
}
NATIVE_DIRECT_ANSWER_CONTINUATION_DIAGNOSTICS = {
    "Target source": "full-context native direct answer",
    "Continuation type": "canonical deterministic answer fragment",
    "Continuation scope": "bounded direct-answer text",
    "Target regenerated per coalition": "no",
}
REQUIRE_TRUSTED_DIRECT_ANSWER_TARGET_MESSAGE = (
    "A trusted current HF-native direct answer is required before the "
    "direct-answer continuation explanation can run. Re-run the full pipeline."
)
# Temporary HF model-lifecycle tracing for the "two Qwen models resident at
# once" / MPS crash investigation. Set to True only for local debugging; keep
# False in normal runs. Remove once the investigation is closed out.
DEBUG_HF_LIFECYCLE = False
EXECUTABLE_TOOL_NAMES = tuple(
    schema["function"]["name"] for schema in get_executable_tool_schemas()
)


@dataclass(frozen=True)
class SelectedHFModelConfig:
    """Single source of truth for the selected local HF model pipeline."""

    model_id: str
    model_family: str
    quantization_mode: str
    device: str
    dtype: str
    supports_native_tools: bool

    @property
    def tokenizer_id(self) -> str:
        return self.model_id

    def cache_key(self, *, scorer_mode: str) -> tuple[str, str, str, str, str, str]:
        return (
            self.model_id,
            self.tokenizer_id,
            self.quantization_mode,
            self.device,
            self.dtype,
            scorer_mode,
        )


HF_LOCAL_MODEL_CONFIGS: dict[str, SelectedHFModelConfig] = {
    "Qwen/Qwen2.5-1.5B-Instruct": SelectedHFModelConfig(
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        model_family="qwen2.5",
        quantization_mode="none",
        device="auto",
        dtype="auto",
        supports_native_tools=True,
    ),
    "Qwen/Qwen2.5-3B-Instruct": SelectedHFModelConfig(
        model_id="Qwen/Qwen2.5-3B-Instruct",
        model_family="qwen2.5",
        quantization_mode="none",
        device="auto",
        dtype="auto",
        supports_native_tools=True,
    ),
    "Qwen/Qwen3-4B-Instruct-2507": SelectedHFModelConfig(
        model_id="Qwen/Qwen3-4B-Instruct-2507",
        model_family="qwen3",
        quantization_mode="none",
        device="auto",
        dtype="auto",
        supports_native_tools=True,
    ),
}
HF_LOCAL_MODEL_OPTIONS = list(HF_LOCAL_MODEL_CONFIGS)


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
    "Answer stable conceptual explanations directly without calling an external tool.",
]
TEXT_PLOT_PACKAGE = "_agentic_text_plot"
ASSET_DIR = Path(__file__).parent / "assets"
ROBOT_IMAGE_PATH = ASSET_DIR / "robot_canva.png"


st.set_page_config(
    page_title="Explaining tool selection",
    page_icon="T",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading local HF router...")
def load_local_hf_router(
    model_name: str,
    max_new_tokens: int,
    *,
    trust_remote_code: bool,
    quantization_mode: str = "none",
    device: str = "auto",
    dtype: str = "auto",
) -> LocalHFRouter:
    """Load and cache the optional local HuggingFace router."""
    _hf_lifecycle_log(
        "[HF LOAD] entering load_local_hf_router",
        f"model_name={model_name}",
        f"quantization={quantization_mode}",
        f"device={device}",
        f"dtype={dtype}",
    )
    router = LocalHFRouter(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        trust_remote_code=trust_remote_code,
        quantization_mode=quantization_mode,
        device=device,
        dtype=dtype,
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


@st.cache_resource(show_spinner="Preparing local HF scorer...")
def load_logprob_scorer(
    model_id: str,
    *,
    max_pairs_per_batch: int,
    device: str | None = None,
    dtype: str = "auto",
    quantization_mode: str = "none",
    tokenizer_id: str | None = None,
) -> CalibratedToolLogOddsScorer:
    """Load and cache the optional local HuggingFace calibrated log-odds scorer.

    ``max_pairs_per_batch`` (and ``device``/``dtype``) are part of the cache
    key, not mutated on the returned instance afterwards -- a different batch
    size loads (or reuses) a distinct cached scorer instead of reaching into a
    shared cached object and changing its configuration in place.
    """
    _hf_lifecycle_log(
        "[HF LOAD] entering load_logprob_scorer",
        f"model_id={model_id}",
        f"tokenizer_id={tokenizer_id or model_id}",
        f"quantization={quantization_mode}",
        f"device={device}",
        f"dtype={dtype}",
    )
    if quantization_mode != "none":
        msg = (
            f"Calibrated log-odds scorer does not currently support quantization mode "
            f"{quantization_mode!r}."
        )
        raise RuntimeError(msg)
    if tokenizer_id is not None and tokenizer_id != model_id:
        msg = "Calibrated log-odds scorer requires tokenizer_id to match model_id."
        raise RuntimeError(msg)
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


def build_native_hf_scorer_from_router(
    router: LocalHFRouter,
    *,
    max_pairs_per_batch: int,
    selected_config: SelectedHFModelConfig | None = None,
) -> NativeToolCallScorer:
    """Wrap an already-cached local HF router as the native coalition scorer."""
    model_id = getattr(router, "model_name", None) or (
        selected_config.model_id if selected_config is not None else DEFAULT_LOGPROB_MODEL_ID
    )
    tokenizer_id = getattr(router, "tokenizer_id", None) or model_id
    quantization_mode = getattr(
        router,
        "requested_quantization_mode",
        selected_config.quantization_mode if selected_config is not None else "none",
    )
    return NativeToolCallScorer(
        model=router.model,
        tokenizer=router.tokenizer,
        device=router.device,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        quantization_mode=quantization_mode,
        actual_quantization_mode=getattr(router, "actual_quantization_mode", quantization_mode),
        dtype=getattr(router, "dtype", selected_config.dtype if selected_config else "auto"),
        max_pairs_per_batch=max_pairs_per_batch,
    )


def build_native_direct_answer_scorer_from_router(
    router: LocalHFRouter,
    *,
    direct_answer_target: str,
    max_pairs_per_batch: int,
    selected_config: SelectedHFModelConfig | None = None,
) -> NativeDirectAnswerScorer:
    """Wrap an already-cached local HF router as the direct-answer coalition scorer.

    Reuses the router's already-loaded ``model``/``tokenizer`` (never loads a
    second HF model), exactly like :func:`build_native_hf_scorer_from_router`.
    """
    model_id = getattr(router, "model_name", None) or (
        selected_config.model_id if selected_config is not None else DEFAULT_LOGPROB_MODEL_ID
    )
    tokenizer_id = getattr(router, "tokenizer_id", None) or model_id
    quantization_mode = getattr(
        router,
        "requested_quantization_mode",
        selected_config.quantization_mode if selected_config is not None else "none",
    )
    return NativeDirectAnswerScorer(
        model=router.model,
        tokenizer=router.tokenizer,
        device=router.device,
        model_id=model_id,
        tokenizer_id=tokenizer_id,
        quantization_mode=quantization_mode,
        actual_quantization_mode=getattr(router, "actual_quantization_mode", quantization_mode),
        dtype=getattr(router, "dtype", selected_config.dtype if selected_config else "auto"),
        max_pairs_per_batch=max_pairs_per_batch,
        direct_answer_target=direct_answer_target,
    )


def hf_value_function_label_for_target(target_tool: str | None, requested_label: str) -> str:
    """Return the HF-local value-function label for a frozen Agent Result target.

    ``no_tool`` Agent Results default to the native direct-answer continuation
    scorer, the same unified continuation-likelihood family used for
    executable tools. The legacy A/B/C/D surrogate (``NO_TOOL_SURROGATE_SCORER_LABEL``)
    remains available only as a manually selected developer ablation; it is no
    longer substituted in automatically.
    """
    if target_tool == "no_tool" and requested_label == NATIVE_HF_SCORER_LABEL:
        return NATIVE_DIRECT_ANSWER_SCORER_LABEL
    return requested_label


def value_function_metadata(label: str) -> tuple[str, str]:
    """Return ``(value_function_type, value_function_fidelity)`` for result metadata."""
    if label == NATIVE_HF_SCORER_LABEL:
        return "native_target_tool_continuation_likelihood", "native_tool_call"
    if label == NATIVE_DIRECT_ANSWER_SCORER_LABEL:
        return "native_direct_answer_continuation_likelihood", "native_direct_answer"
    if label == NO_TOOL_SURROGATE_SCORER_LABEL:
        return "legacy_abcd_no_tool_probe", "surrogate_ablation"
    if label == LOGPROB_SCORER_LABEL:
        return "legacy_abcd_forced_choice_probe", "developer_ablation"
    return label.lower().replace(" ", "_"), "diagnostic"


def extract_direct_answer_text(inference_result: object) -> str | None:
    """Return the first non-empty natural-language direct answer, in priority order.

    Priority: ``cleaned_direct_answer`` (sentinel-stripped by native parsing),
    then ``direct_answer``, then ``agent_response``. Never falls back to
    ``raw_response``/``raw_response_original`` or an internal sentinel -- those
    are excluded on purpose so a malformed or sentinel-only native output can
    never become a direct-answer continuation target.
    """
    for attribute in ("cleaned_direct_answer", "direct_answer", "agent_response"):
        value = getattr(inference_result, attribute, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def require_trusted_direct_answer_target(
    inference_result: object | None,
    *,
    inference_backend: str,
) -> str:
    """Return the frozen direct-answer text, or raise when no trusted target exists.

    Requires an HF-local, parse-error-free, ``no_tool`` full-context inference
    result with a non-empty natural-language answer. The existing inference
    signature invalidation (switching HF model/backend or editing the user
    request clears ``agentic_inference_result``) already guarantees that any
    non-``None`` result reaching this point is current for the active
    configuration, so no separate staleness check is needed here.
    """
    if inference_backend != "HF local" or inference_result is None:
        raise RuntimeError(REQUIRE_TRUSTED_DIRECT_ANSWER_TARGET_MESSAGE)
    if getattr(inference_result, "parse_error", None) is not None:
        raise RuntimeError(REQUIRE_TRUSTED_DIRECT_ANSWER_TARGET_MESSAGE)
    if getattr(inference_result, "selected_tool", None) != "no_tool":
        raise RuntimeError(REQUIRE_TRUSTED_DIRECT_ANSWER_TARGET_MESSAGE)
    answer_text = extract_direct_answer_text(inference_result)
    if not answer_text:
        raise RuntimeError(REQUIRE_TRUSTED_DIRECT_ANSWER_TARGET_MESSAGE)
    return answer_text


def selected_hf_model_config(model_id: str) -> SelectedHFModelConfig:
    """Return the canonical local-HF config for a selectable model id."""
    try:
        return HF_LOCAL_MODEL_CONFIGS[model_id]
    except KeyError:
        model_family = (
            "qwen3" if "Qwen3" in model_id else "qwen2.5" if "Qwen2.5" in model_id else "unknown"
        )
        return SelectedHFModelConfig(
            model_id=model_id,
            model_family=model_family,
            quantization_mode="none",
            device="auto",
            dtype="auto",
            supports_native_tools=True,
        )


def hf_runtime_identity(
    component: object,
    *,
    fallback_config: SelectedHFModelConfig,
) -> dict[str, object]:
    """Extract comparable HF runtime identity from a router or scorer."""
    return {
        "model_id": getattr(component, "model_name", getattr(component, "model_id", None))
        or fallback_config.model_id,
        "tokenizer_id": getattr(component, "tokenizer_id", None) or fallback_config.tokenizer_id,
        "requested_quantization": getattr(
            component,
            "requested_quantization_mode",
            fallback_config.quantization_mode,
        ),
        "actual_quantization": getattr(component, "actual_quantization_mode", None)
        or getattr(component, "requested_quantization_mode", fallback_config.quantization_mode),
        "device": getattr(component, "device", fallback_config.device),
        "dtype": getattr(component, "dtype", fallback_config.dtype),
    }


def hf_consistency_error_message(
    *,
    message: str,
    key: str,
    inference_value: object,
    scorer_value: object,
) -> str:
    """Format a clear HF inference/scorer mismatch error."""
    return f"{message} Inference {key}={inference_value!r}; scorer {key}={scorer_value!r}."


def validate_hf_inference_xai_consistency(
    *,
    selected_config: SelectedHFModelConfig,
    router: object,
    scorer: object,
) -> dict[str, object]:
    """Block XAI if the native router and scorer do not explain the same HF runtime."""
    inference_identity = hf_runtime_identity(router, fallback_config=selected_config)
    scorer_identity = hf_runtime_identity(scorer, fallback_config=selected_config)
    comparisons = (
        ("model_id", "Inference and XAI scorer must use the same model."),
        ("tokenizer_id", "Inference and XAI scorer must use the same tokenizer."),
        ("actual_quantization", "Inference and XAI scorer must use the same quantization."),
        ("device", "Inference and XAI scorer must use the same device."),
    )
    for key, message in comparisons:
        if inference_identity[key] != scorer_identity[key]:
            error_message = hf_consistency_error_message(
                message=message,
                key=key,
                inference_value=inference_identity[key],
                scorer_value=scorer_identity[key],
            )
            raise RuntimeError(error_message)
    return {
        "inference": inference_identity,
        "scorer": scorer_identity,
        "model_family": selected_config.model_family,
        "supports_native_tools": selected_config.supports_native_tools,
        "match": True,
    }


def has_untrusted_native_parse_result(inference_result: object | None) -> bool:
    """Return True when native parser failure leaves no trustworthy XAI target."""
    return bool(getattr(inference_result, "parse_error", None))


def require_selected_hf_config(
    selected_config: SelectedHFModelConfig | None,
) -> SelectedHFModelConfig:
    """Return the selected HF config or fail before native-HF XAI starts."""
    if selected_config is None:
        msg = "Native HF scorer requires an HF-local model configuration."
        raise RuntimeError(msg)
    return selected_config


@st.cache_resource(show_spinner="Loading segmentation model...")
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


@st.cache_resource(show_spinner="Loading segmentation model...")
def load_linguistic_segmenter() -> LinguisticSegmenter:
    """Load and cache the optional spaCy linguistic segmenter."""
    return LinguisticSegmenter()


@st.cache_resource(show_spinner="Loading embedding model...")
def load_final_answer_embedder(model_id: str) -> SentenceTransformerAnswerEmbedder:
    """Load and cache the optional embedding model for the final-answer similarity scorer.

    The underlying sentence-transformers model is only downloaded/loaded the first
    time the embedder is called, not when this cached wrapper is constructed.
    """
    return SentenceTransformerAnswerEmbedder(model_id=model_id, device="auto")


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
    font-weight: 700 !important;
}
.primary-run-button:hover,
.st-key-agentic_run_full_pipeline button:hover {
    background: #2558c4 !important;
    border-color: #2558c4 !important;
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
/* ---- XAI summary card: compact "why <tool>?" explanation ------------ */
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
    color: #197a52;
}
.dir-arrow.down {
    color: #b3261e;
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
.type-pill {
    border-radius: 999px;
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.08rem 0.55rem;
    text-align: center;
}
.type-pill.complementary {
    background: #eaf4f1;
    color: #197a52;
}
.type-pill.redundant {
    background: #eaf1f7;
    color: #2b5f8a;
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
    .scenario-panel,
    .mock-chat,
    .xai-summary-card {
        grid-template-columns: 1fr;
    }
    .scenario-hint,
    .xai-summary-main,
    .xai-summary-metrics {
        border-left: 0;
        padding-left: 0;
    }
}
</style>
"""


def _image_to_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


@dataclass(frozen=True)
class ToolUseSegment:
    """Lightweight prompt segment for rendering the UI before shapiq loads."""

    source: SegmentSource
    label: str
    text: str


def scorer_short_label(label: str) -> str:
    """Return the compact setup-chip label for a value-function scorer label."""
    if label == NATIVE_HF_SCORER_LABEL:
        return NATIVE_HF_SCORER_SHORT_LABEL
    if label == NATIVE_DIRECT_ANSWER_SCORER_LABEL:
        return NATIVE_DIRECT_ANSWER_SCORER_SHORT_LABEL
    return str(label)


def truncate_label(value: str, max_length: int = 72) -> str:
    """Shorten long selectbox labels without changing their underlying value."""
    if len(value) <= max_length:
        return value
    return value[: max_length - 1].rstrip() + "..."


def scenario_prompt_label(trace_name: str) -> str:
    """Display a sample scenario by its user prompt instead of its internal name."""
    user_prompt = " ".join(SAMPLE_TRACES[trace_name]["user_segments"])
    return truncate_label(user_prompt)


def short_player_label(segment: ToolUseSegment, max_chars: int = 18) -> str:
    """Build a short, text-aware label like 'U2: it rain' for plot axes.

    Presentation-only: combines the player's short id with a truncated preview of its
    underlying text, so plot axes are self-explanatory without requiring the reader to
    cross-reference the player chips separately. Never used as a lookup key or player
    identity -- only as display text for matplotlib tick labels.
    """
    preview = " ".join(segment.text.split())
    if len(preview) > max_chars:
        preview = preview[: max_chars - 1].rstrip() + "…"
    return f"{segment.label}: {preview}"


def format_attribution(value: float, digits: int = 3) -> str:
    """Format signed attribution values for display."""
    if not math.isfinite(value):
        return ""
    threshold = 0.5 * 10 ** (-digits)
    if 0 < abs(value) < threshold:
        return f"{'+' if value > 0 else '-'}<0.001"
    return f"{value:.{digits}f}"


# A native h(empty)/h(N) log-probability this close to zero rounds to "0.0000" at
# 4 decimal places, which reads as "exactly zero" even though it never is (a
# teacher-forced log-probability is only exactly 0 in the limit). Below this
# magnitude we show "≈0.0000" plus a tooltip instead of a misleading exact zero.
NEAR_ZERO_LOG_SCORE_THRESHOLD = 0.0005
NEAR_ZERO_LOG_SCORE_TOOLTIP = "Nonzero value, rounded to 0.0000 for display."

# Rough, non-statistical Δ Support magnitude bands for the native log-probability
# value function. Calibrated against real measured deltas from this demo's own
# sample scenarios (Qwen2.5-1.5B-Instruct, native tool-identity/direct-answer
# scorer): F1/web-search +0.67, calculator +0.93, weather +1.13, photosynthesis
# direct-answer +1.14. Only meant to give a reader a quick sense of "is this Δ
# big or small", not a formal effect-size measure.
DELTA_SUPPORT_LOW_MAX = 0.4
DELTA_SUPPORT_MODERATE_MAX = 0.9


def format_log_score(value: float) -> tuple[str, bool]:
    """Format a native log-probability score at 4 decimals.

    Returns ``(display_text, is_rounded_near_zero)``. When ``|value|`` is below
    ``NEAR_ZERO_LOG_SCORE_THRESHOLD`` the display text is "≈0.0000" instead of a
    bare "0.0000"/"-0.0000", so a genuinely nonzero value is never misread as
    exactly zero.
    """
    if not math.isfinite(value):
        return "n/a", False
    if abs(value) < NEAR_ZERO_LOG_SCORE_THRESHOLD:
        return "≈0.0000", True
    return f"{value:.4f}", False


def log_prob_to_confidence_caption(value: float) -> str:
    """Convert a natural-log continuation log-probability into a plain-language %.

    Display-only: ``exp(log_prob)`` recovers the underlying probability, shown as
    a rough confidence percentage for non-technical readers. Clamped to
    ``[0, 1]`` before converting to guard against tiny floating-point overshoot
    past 0.0 for values extremely close to zero.
    """
    if not math.isfinite(value):
        return ""
    probability = min(max(math.exp(min(value, 0.0)), 0.0), 1.0)
    return f"≈{probability * 100:.1f}% confidence"


def build_log_score_display(value: float) -> tuple[str, str]:
    """Return ``(value_html, confidence_caption)`` for one native log-prob value.

    ``value_html`` is the 4-decimal value from :func:`format_log_score`, wrapped
    with a dotted-underline + tooltip when it rounds to "≈0.0000" so a genuinely
    nonzero value is never misread as exactly zero. ``confidence_caption`` is the
    plain-language "≈NN.N% confidence" string for embedding as light secondary
    text next to it (in a big metric card or a compact summary line alike).
    """
    display, rounded_near_zero = format_log_score(value)
    if rounded_near_zero:
        value_html = (
            f"<span class='log-score-rounded' title='{NEAR_ZERO_LOG_SCORE_TOOLTIP}'>"
            f"{display}</span>"
        )
    else:
        value_html = display
    return value_html, log_prob_to_confidence_caption(value)


def delta_support_tier(delta: float) -> tuple[str, str]:
    """Classify a Δ Support value into a coarse low/moderate/high magnitude tier.

    Returns ``(label, css_class)``, e.g. ``("High support", "high")`` or
    ``("Moderate opposition", "moderate")``. Direction (support vs. opposition)
    comes from the sign of ``delta``; the tier comes from ``|delta|`` alone.
    """
    magnitude = abs(delta)
    if magnitude < DELTA_SUPPORT_LOW_MAX:
        tier_css, tier_word = "low", "Low"
    elif magnitude < DELTA_SUPPORT_MODERATE_MAX:
        tier_css, tier_word = "moderate", "Moderate"
    else:
        tier_css, tier_word = "high", "High"
    direction_word = "support" if delta >= 0 else "opposition"
    return f"{tier_word} {direction_word}", tier_css


def direction_arrow_html(value: float, *, css_class: str = "dir-arrow") -> str:
    """Render a colored ↑/↓ arrow marking the sign of ``value``, or "" if zero.

    Positive/negative colors already exist as the only signal in several charts;
    this adds a shape-based cue (arrow direction) so sign is never color-alone,
    e.g. for black-and-white printing or color-vision-deficient readers.
    """
    if value > 0:
        return f"<span class='{css_class} up'>↑</span>"
    if value < 0:
        return f"<span class='{css_class} down'>↓</span>"
    return ""


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
    """Build the combined k-SII matrix: order-1 main effects on the diagonal,
    order-2 pairwise interactions off the diagonal.

    ``explanation`` must be the *unfiltered* k-SII explanation (containing
    both order-1 and order-2 interactions) -- passing an order-2-only
    explanation (e.g. from ``explanation.get_n_order(order=2)``) would
    silently zero out the diagonal, since its singleton interactions have
    already been stripped out.

    Delegates to the shared
    ``shapiq.plot.sentence.interaction_matrix_from_explanation`` (loaded via
    :func:`load_sentence_plot_module`) so this fallback/full-matrix path and
    the visual heatmap (:func:`sentence_interaction_heatmap`) always agree on
    identical values. Use :func:`pairwise_only_matrix_from_explanation` for a
    diagonal-free variant meant for internal ranking only -- do not use a
    zero-diagonal matrix as a fallback for anything displayed to the user.
    """
    module = load_sentence_plot_module()
    matrix = module.interaction_matrix_from_explanation(
        explanation, n_players, include_main_effects=True
    )
    return pd.DataFrame(matrix)


def pairwise_only_matrix_from_explanation(
    explanation: shapiq.InteractionValues,
    n_players: int,
) -> pd.DataFrame:
    """Build a pure order-2 pairwise matrix with a zero diagonal.

    For internal use only (e.g. the lexical-scorer comparison's
    ``strongest_pair`` ranking), where no main effects are needed or
    displayed. Do not use this as a fallback for the displayed combined
    heatmap -- use :func:`pairwise_matrix_from_explanation` for anything shown
    to the user.
    """
    module = load_sentence_plot_module()
    matrix = module.interaction_matrix_from_explanation(
        explanation, n_players, include_main_effects=False
    )
    return pd.DataFrame(matrix)


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
    """Return the strongest off-diagonal (``i < j``) pairwise interaction.

    ``matrix`` may be the combined main-effect + pairwise matrix from
    :func:`pairwise_matrix_from_explanation` (diagonal populated) or the
    diagonal-free variant -- either way, only cells with ``i < j`` are
    inspected, so a populated diagonal never leaks into the ranking.
    """
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
    empty_score: float,
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

    delta = float(full_score) - float(empty_score)
    if delta > DELTA_STATUS_THRESHOLD:
        direction = "increases"
    elif delta < -DELTA_STATUS_THRESHOLD:
        direction = "decreases"
    else:
        direction = "barely changes"
    notes.append(
        f"The full request {direction} selected-tool support relative to the empty "
        f"request ({full_score:.3f} vs {empty_score:.3f})."
    )
    return notes


def polish_bar(
    fig: plt.Figure,
    ax: plt.Axes,
    *,
    xlabel: str = "Target-tool attribution",
    title: str = "",
    figsize: tuple[float, float] = (5.2, 2.6),
) -> plt.Figure:
    """Make package bar plot fit a compact Streamlit layout (e.g. one of two side-by-side columns).

    ``title`` defaults to empty: the Streamlit section header directly above the plot
    (e.g. "2. Individual segment effects -- SV") already states what the plot is, so an
    internal matplotlib title would just repeat it. Pass a non-empty ``title`` to opt
    back into an in-plot title for a call site without its own section header.
    """
    fig.set_size_inches(*figsize)
    ax.set_title("", loc="center")
    if title:
        ax.set_title(title, loc="left", fontsize=9, pad=5)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
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
        # Direction is marked with both an arrow glyph and a color (never color
        # alone), so sign reads correctly in black-and-white print or for
        # color-vision-deficient readers -- the bar fill itself keeps the
        # shapiq library's own red/blue convention untouched.
        arrow, label_color = ("↑", "#197a52") if width >= 0 else ("↓", "#b3261e")
        ax.text(
            x_pos,
            patch.get_y() + patch.get_height() / 2,
            f"{arrow} {width:.2f}",
            va="center",
            ha=ha,
            fontsize=7,
            color=label_color,
        )

    fig.tight_layout()
    return fig


def polish_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    segments: list[ToolUseSegment],
    *,
    title: str = "",
    figsize: tuple[float, float] = (4.3, 3.2),
) -> plt.Figure:
    """Make package heatmap fit a compact Streamlit layout (e.g. one of two side-by-side columns).

    ``title`` defaults to empty: the Streamlit section header directly above the plot
    (e.g. "3. Pairwise interactions -- k-SII") already states what the plot is, so an
    internal matplotlib title would just repeat it. Pass a non-empty ``title`` to opt
    back into an in-plot title for a call site without its own section header.
    """
    fig.set_size_inches(*figsize)
    ax.set_title("", loc="center")
    if title:
        ax.set_title(title, loc="left", fontsize=9, pad=5)
    ax.tick_params(axis="x", labelrotation=30, labelsize=6.5)
    ax.tick_params(axis="y", labelsize=6.5)

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
    hf_model_config: SelectedHFModelConfig | None = None,
    hf_max_new_tokens: int = DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
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

    HF Local uses model-native structured tool schemas through the tokenizer's
    chat template. The selected HF model id is always the single
    ``HF_MODEL_ID_SESSION_KEY`` dropdown selection from the Inference tab.
    """
    if inference_backend == "HF local":
        hf_model_config = hf_model_config or selected_hf_model_config(inference_model_name)

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
                    device=hf_model_config.device if hf_model_config is not None else None,
                    dtype=hf_model_config.dtype if hf_model_config is not None else "auto",
                    quantization_mode=(
                        hf_model_config.quantization_mode if hf_model_config is not None else "none"
                    ),
                    tokenizer_id=(
                        hf_model_config.tokenizer_id if hf_model_config is not None else None
                    ),
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
                "[HF LOAD] using native LocalHFRouter", f"model_name={inference_model_name}"
            )
            hf_router = load_local_hf_router(
                inference_model_name,
                int(hf_max_new_tokens),
                trust_remote_code=bool(hf_trust_remote_code),
                quantization_mode=(
                    hf_model_config.quantization_mode if hf_model_config is not None else "none"
                ),
                device=hf_model_config.device if hf_model_config is not None else "auto",
                dtype=hf_model_config.dtype if hf_model_config is not None else "auto",
            )
            try:
                return hf_router.choose_tool(user_request, TOOLS, system_prompt=system_prompt)
            except TypeError as error:
                if "system_prompt" not in str(error):
                    raise
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


def build_probability_bar_html(tool_name: str, probability: float, *, is_selected: bool) -> str:
    """Render one compact horizontal probability bar row as HTML.

    Presentation-only: formats an already-computed probability (e.g. from
    ``softmax_dict``) as a labeled bar, in place of a full-size ``st.bar_chart`` +
    dataframe pair. Does not compute or alter the probability itself.
    """
    width_pct = max(0.0, min(100.0, float(probability) * 100))
    fill_class = "probability-fill selected" if is_selected else "probability-fill"
    return (
        "<div class='probability-row'>"
        f"<span class='probability-label'>{escape(str(tool_name))}</span>"
        "<span class='probability-track'>"
        f"<span class='{fill_class}' style='width:{width_pct:.1f}%'></span>"
        "</span>"
        f"<span class='probability-value'>{float(probability):.2f}</span>"
        "</div>"
    )


def flatten_markdown_html(html: str) -> str:
    """Collapse a human-indented multi-line HTML template into markdown-safe text.

    Streamlit's markdown renderer treats a blank line followed by a
    4+-space-indented line as the start of an indented code block, rendering
    that HTML literally (as visible source text) instead of parsing it. This
    happens whenever an f-string placeholder that sits alone on its own
    template line evaluates to an empty string (e.g. an optional card section
    that isn't shown this run) -- the line becomes whitespace-only, which
    counts as a blank line, and the next (still-indented) line trips the
    code-block rule. Stripping each line's leading/trailing whitespace and
    dropping now-empty lines removes any such gap; this is safe because
    whitespace between HTML block elements is not meaningful.
    """
    return "\n".join(line.strip() for line in html.strip().splitlines() if line.strip())


def build_kv_table_html(items: dict[str, object]) -> str:
    """Render a compact two-column key/value table as HTML (e.g. tool arguments).

    Presentation-only: formats an already-available mapping (e.g.
    ``inference_result.tool_arguments``); returns an empty string for an empty
    mapping so callers can show their own "no arguments" message instead.
    """
    if not items:
        return ""
    rows = "".join(
        "<div class='kv-row'>"
        f"<span class='kv-key'>{escape(str(key))}</span>"
        f"<span class='kv-value'>{escape(str(value))}</span>"
        "</div>"
        for key, value in items.items()
    )
    return f"<div class='kv-table'>{rows}</div>"


def backend_display_name(backend: object) -> str:
    """Return compact user-facing backend text for Agent Result metadata."""
    backend_text = str(backend or "").strip()
    if backend_text.lower() == "hf local":
        return "HF Local"
    return backend_text or "Unknown backend"


def agent_model_line_html(*, model: object, backend: object) -> str:
    """Render compact model/backend metadata for the Agent Result card."""
    return (
        "<div class='agent-model-line'>"
        f"{escape(str(model or 'Unknown model'))} · {escape(backend_display_name(backend))}"
        "</div>"
    )


def tool_execution_state(inference_result: object) -> tuple[str, object | None]:
    """Return truthful tool execution status and optional output for display."""
    raw_trace = getattr(inference_result, "raw_trace", None)
    if isinstance(raw_trace, dict) and "tool_output" in raw_trace:
        return "Tool executed successfully", raw_trace.get("tool_output")
    return "Tool call prepared", None


def render_tool_arguments_html(tool_arguments: object) -> str:
    """Render parsed tool arguments prominently without falling back to raw JSON."""
    arguments = dict(tool_arguments) if isinstance(tool_arguments, dict) else {}
    if not arguments:
        return "<p class='result-meta-row'>No parsed arguments were prepared.</p>"
    return (
        "<p class='result-meta-row' style='margin-top:0.7rem;'><strong>Arguments</strong></p>"
        f"{build_kv_table_html(arguments)}"
    )


def render_tool_decision_card(
    inference_result: object,
    *,
    backend: object,
    model: object,
) -> str:
    """Render an executable-tool Agent Result card."""
    selected_tool = str(getattr(inference_result, "selected_tool", "") or "")
    icon = TOOL_ICONS.get(selected_tool, "?")
    route_explanation = TOOL_ROUTE_EXPLANATIONS.get(
        selected_tool,
        "The agent prepared an external tool call for this request.",
    )
    status_label, tool_output = tool_execution_state(inference_result)
    output_html = (
        (
            "<p class='result-meta-row' style='margin-top:0.7rem;'>"
            "<strong>Execution result</strong></p>"
            f"<div class='agent-response-block'>{escape(str(tool_output))}</div>"
        )
        if tool_output is not None
        else ""
    )
    flow_html = (
        "<div class='agent-flow-line'>"
        f"User request &rarr; {escape(selected_tool)} &rarr; {escape(status_label)}"
        "</div>"
    )
    return flatten_markdown_html(f"""
        <div class="result-card">
            <div class="result-card-header">
                {icon} Agent selected <code>{escape(selected_tool)}</code>
            </div>
            <div class="result-card-subtitle">{escape(route_explanation)}</div>
            {flow_html}
            {render_tool_arguments_html(getattr(inference_result, "tool_arguments", {}))}
            {output_html}
            {agent_model_line_html(model=model, backend=backend)}
        </div>
    """)


def render_direct_answer_card(
    inference_result: object,
    *,
    backend: object,
    model: object,
) -> str:
    """Render a direct-answer Agent Result card without exposing the internal no_tool label."""
    answer = (
        getattr(inference_result, "cleaned_direct_answer", "")
        or getattr(inference_result, "direct_answer", "")
        or getattr(inference_result, "agent_response", "")
        or getattr(inference_result, "assistant_answer", "")
        or getattr(inference_result, "final_answer", "")
        or getattr(inference_result, "raw_response", "")
        or ""
    )
    return flatten_markdown_html(f"""
        <div class="result-card">
            <div class="result-card-header">💬 Agent answered directly</div>
            <div class="result-card-subtitle">No external tool was called.</div>
            <div class="agent-flow-line">User request &rarr; Direct answer</div>
            <div class="agent-response-block">{escape(str(answer))}</div>
            {agent_model_line_html(model=model, backend=backend)}
        </div>
    """)


def render_parse_failure_card(parse_error: object) -> str:
    """Render native parser failures as explicit untrusted results."""
    return flatten_markdown_html(f"""
        <div class="error-card">
            <h3>⚠️ Native tool-call parsing failed</h3>
            <p>The model produced a tool-call structure, but it could not be parsed safely.
            The result was not treated as a direct answer.</p>
            <p>{escape(str(parse_error))}</p>
        </div>
    """)


def render_agent_result_card(
    inference_result: object,
    *,
    backend: object,
    model: object,
) -> str:
    """Render the main Agent Result card without developer-only routing details."""
    parse_error = getattr(inference_result, "parse_error", None)
    if parse_error:
        return render_parse_failure_card(parse_error)
    selected_tool = getattr(inference_result, "selected_tool", None)
    if selected_tool == "no_tool":
        return render_direct_answer_card(inference_result, backend=backend, model=model)
    return render_tool_decision_card(inference_result, backend=backend, model=model)


def build_player_chip_html(segment: ToolUseSegment, *, max_chars: int = 30) -> str:
    """Render one player chip: a small colored player-id badge plus a text preview.

    Presentation-only: truncates the segment's own text for display (via
    ``truncate_label``) and carries the full text in a ``title`` tooltip attribute,
    so long segments degrade gracefully without hiding the underlying content.
    """
    preview = truncate_label(segment.text, max_length=max_chars)
    return (
        f'<span class="player-chip" title="{escape(segment.text)}">'
        f'<span class="player-badge">{escape(segment.label)}</span>'
        f'<span class="player-chip-text">{escape(preview)}</span>'
        "</span>"
    )


def build_sv_mini_table_html(rows: pd.DataFrame) -> str:
    """Render a compact Segment/Text/SV mini-table as HTML rows.

    Presentation-only: the caller passes an already-sliced top-N frame (e.g.
    ``attribution_frame.head(3)``); this only formats it, it does not rank or select.
    """
    header = (
        "<div class='mini-table-row sv-row header'>"
        "<span>Segment</span><span>Text</span><span>SV</span></div>"
    )
    body_rows = "".join(
        "<div class='mini-table-row sv-row'>"
        f"<span class='cell-segment'>{escape(str(row['segment']))}</span>"
        f"<span class='cell-text' title='{escape(str(row['text']))}'>"
        f"{escape(truncate_label(str(row['text']), max_length=42))}</span>"
        f"<span class='cell-value'>{direction_arrow_html(float(row['attribution']))} "
        f"{format_attribution(float(row['attribution']))}</span>"
        "</div>"
        for _, row in rows.iterrows()
    )
    return f"<div class='mini-table'>{header}{body_rows}</div>"


def build_ksii_mini_table_html(rows: list[dict[str, object]]) -> str:
    """Render a compact Pair/Text/k-SII/Type mini-table as HTML rows, with a colored type pill.

    Presentation-only: the caller passes an already-ranked list of pairwise-interaction
    rows (from ``top_pairwise_interactions``, sliced or not by the caller); this only
    formats it, it does not rank, select, or compute k-SII values.
    """
    header = (
        "<div class='mini-table-row ksii-row header'>"
        "<span>Pair</span><span>Text</span><span>k-SII</span><span>Type</span></div>"
    )
    body_rows = "".join(
        "<div class='mini-table-row ksii-row'>"
        f"<span class='cell-segment'>{escape(str(row['segment_i']))} + "
        f"{escape(str(row['segment_j']))}</span>"
        f"<span class='cell-text' title='{escape(str(row['text_i']))} + "
        f"{escape(str(row['text_j']))}'>"
        f"{escape(truncate_label(str(row['text_i']), max_length=18))} + "
        f"{escape(truncate_label(str(row['text_j']), max_length=18))}</span>"
        f"<span class='cell-value'>{format_attribution(float(row['value']))}</span>"
        f"<span class='type-pill {escape(str(row['type']))}'>{escape(str(row['type']))}</span>"
        "</div>"
        for row in rows
    )
    return f"<div class='mini-table'>{header}{body_rows}</div>"


def build_player_legend_html(segments: list[ToolUseSegment], *, max_chars: int = 34) -> str:
    """Render a compact 'U1 = text' legend mapping bare player ids to their segment text.

    Presentation-only: used alongside a heatmap plotted with bare U-labels, so the
    reader can still map each axis tick back to its underlying text.
    """
    rows = "".join(
        f"<span class='player-legend-row'><strong>{escape(segment.label)}</strong> = "
        f"{escape(truncate_label(segment.text, max_length=max_chars))}</span>"
        for segment in segments
    )
    return f"<div class='player-legend'>{rows}</div>"


def build_delta_metric_html(
    delta_label: str,
    explained_increase: float,
    *,
    show_tier: bool = False,
    note_html: str = "",
) -> str:
    """Render the Δ Support metric card: the summary card's largest, most prominent
    element regardless of value-function branch, with a direction arrow (not just
    color) so sign is never color-alone.

    ``show_tier`` adds a low/moderate/high magnitude tier badge underneath. This
    is only meaningful where the tier thresholds were calibrated (the native
    log-probability value functions) -- other value functions live on a different
    numeric scale, so callers for those branches should leave it ``False``.

    ``note_html`` is an optional pre-rendered supplementary line placed below the
    tier badge (e.g. the native H(&empty;)/H(N) recap from
    :func:`build_native_metric_summary_line_html`), so that recap lives inside
    this card instead of as a separate element elsewhere in the summary card.

    ``delta_label`` is a fixed, code-authored string that may itself contain HTML
    entities (e.g. ``"...h(&empty;)"``), matching the existing convention for this
    card's labels -- it is not escaped, the same as the pre-existing label markup.
    """
    sign_class = "negative" if explained_increase < 0 else ""
    arrow_html = direction_arrow_html(explained_increase)
    tier_html = ""
    if show_tier:
        tier_label, tier_css = delta_support_tier(explained_increase)
        direction_css = "support" if explained_increase >= 0 else "opposition"
        tier_html = (
            f"<span class='delta-tier-badge {tier_css} {direction_css}'>{escape(tier_label)}</span>"
        )
    return (
        f"<div class='xai-metric delta {sign_class}'>"
        f"<span class='xai-metric-label' title='{delta_label}'>&Delta; Support</span>"
        f"<span class='xai-metric-value'>{arrow_html} {explained_increase:+.3f}</span>"
        f"{tier_html}"
        f"{note_html}"
        "</div>"
    )


def build_native_big_metric_card_html(*, short_label: str, label: str, value: float) -> str:
    """Render one full-size H(&empty;)/H(N) metric card (Developer-mode-only detail
    view for the native tool-identity/direct-answer value functions), including the
    4-decimal near-zero-safe value and its plain-language confidence-% caption.

    ``short_label``/``label`` are fixed, code-authored strings that may contain HTML
    entities (e.g. ``"Native h(&empty;)"``) -- not escaped, matching the existing
    convention for this card's labels.
    """
    value_html, confidence_caption = build_log_score_display(value)
    confidence_html = (
        f"<span class='log-score-confidence'>{escape(confidence_caption)}</span>"
        if confidence_caption
        else ""
    )
    return (
        "<div class='xai-metric'>"
        f"<span class='xai-metric-label' title='{label}'>{short_label}</span>"
        f"<span class='xai-metric-value'>{value_html}</span>"
        f"{confidence_html}"
        "</div>"
    )


def build_native_metric_summary_line_html(
    *,
    empty_short_label: str,
    empty_score: float,
    full_short_label: str,
    full_score: float,
) -> str:
    """Render the compact H(&empty;)/H(N) recap for the native value functions, as
    a supplementary note nested inside the Δ Support card (below its tier badge)
    via :func:`build_delta_metric_html`'s ``note_html`` -- not a standalone
    element elsewhere in the summary card.

    Does not repeat the Δ value itself: the card's own ``xai-metric-value``
    directly above already shows it, so this note only adds the H(&empty;)/H(N)
    detail behind that Δ.

    ``empty_short_label``/``full_short_label`` are fixed, code-authored strings
    that may contain HTML entities -- not escaped, matching the existing
    convention for this card's labels.
    """
    empty_value_html, empty_confidence = build_log_score_display(empty_score)
    full_value_html, full_confidence = build_log_score_display(full_score)
    return (
        "<div class='xai-score-flow delta-native-note'>"
        f"{empty_short_label} <strong>{empty_value_html}</strong>"
        f"<span class='native-metric-note'>{escape(empty_confidence)}</span>"
        f" &rarr; {full_short_label} <strong>{full_value_html}</strong>"
        f"<span class='native-metric-note'>{escape(full_confidence)}</span>"
        "</div>"
    )


def build_interaction_interpretation(
    *,
    pair_label: str,
    pair_value: float,
    top_pairs: list[dict[str, object]],
) -> str:
    """Generate the final plain-language interpretation of the strongest k-SII interaction.

    Uses only the already-computed strongest pair (``pair_label``/``pair_value``) and its
    text preview from ``top_pairs`` (the same ranked list used for the k-SII mini-table) --
    it does not run any separate free-form generation step or recompute any interaction.

    Returns HTML (segment text is escaped) meant for embedding via
    ``st.markdown(..., unsafe_allow_html=True)``, not markdown source.
    """
    if not top_pairs or pair_label == "No pair":
        return (
            "Only one segment was found for this request, so no pairwise interaction "
            "could be evaluated."
        )
    strongest = top_pairs[0]
    pair_text = f"“{escape(str(strongest['text_i']))}” + “{escape(str(strongest['text_j']))}”"
    pair_label_html = escape(pair_label)
    if abs(pair_value) < 0.03:
        return (
            f"<strong>No dominant pairwise interaction was detected</strong> (strongest pair "
            f"<code>{pair_label_html}</code>, k-SII = {pair_value:+.3f}). The decision is "
            "explained mostly by individual segment effects rather than by how segments combine."
        )
    if pair_value > 0:
        return (
            f"<strong>Strongest complementary interaction: {pair_label_html}</strong> -- "
            f"{pair_text} contribute more support together (k-SII = {pair_value:+.3f}) than "
            "the sum of their individual effects would predict: the two segments reinforce "
            "each other."
        )
    return (
        f"<strong>Strongest redundant interaction: {pair_label_html}</strong> -- "
        f"{pair_text} contribute less support together (k-SII = {pair_value:+.3f}) than "
        "the sum of their individual effects would predict: the segments carry overlapping "
        "evidence."
    )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    # ------------------------------------------------------------------
    # Header: logo + title + subtitle (left), developer mode toggle (right).
    # The toggle is a real Streamlit widget, so it cannot live inside the
    # hand-authored HTML on the left -- st.columns keeps them on one visual
    # row instead.
    # ------------------------------------------------------------------
    header_left, header_right = st.columns([5, 1], vertical_alignment="center")
    with header_left:
        st.markdown(
            """
            <div class="app-header">
                <div class="app-logo">T</div>
                <div>
                    <h1>Agentic Tool-Use Explanation</h1>
                    <p>Explain why an agent selected a tool by attributing the decision to
                    user-request segments.</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with header_right:
        developer_mode = st.toggle(
            "Developer mode",
            value=False,
            key="agentic_developer_mode",
            help="Show segmentation, scorer, and raw-diagnostic controls for debugging.",
        )
    if ROBOT_IMAGE_PATH.exists():
        robot_base64 = _image_to_base64(ROBOT_IMAGE_PATH)
        st.markdown(
            f"""
            <div class="floating-robot-wrap">
                <img
                    src="data:image/png;base64,{robot_base64}"
                    class="floating-robot"
                    title="Image from Canva"
                    alt="Decorative robot illustration"
                />
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
    # Mode selector: HF Local vs. API Agent, styled as two pill/cards.
    # st.container(key=...) is the only way to get a real, stable DOM
    # wrapper around a widget (a hand-written <div> in st.markdown would
    # not actually nest the radio inside it -- each Streamlit call renders
    # as its own sibling element), so the pill/card CSS targets the
    # `st-key-agentic_mode_card_row` class Streamlit derives from `key=`.
    # ------------------------------------------------------------------
    with st.container(key="agentic_mode_card_row"):
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

    inference_backend = "HF local" if explanation_mode == "HF Local" else "Groq"

    if inference_backend == "HF local":
        inference_model_name = st.selectbox(
            "HF model",
            HF_LOCAL_MODEL_OPTIONS,
            key=HF_MODEL_ID_SESSION_KEY,
            help=(
                "Local HF model used for both native tool-call inference and explanation scorer."
            ),
        )
        selected_hf_config = selected_hf_model_config(inference_model_name)
    else:
        selected_hf_config = None
        inference_model_name = st.text_input(
            "Groq model",
            value="llama-3.1-8b-instant",
            key="agentic_groq_inference_model",
        )
        if not os.getenv("GROQ_API_KEY"):
            st.warning("GROQ_API_KEY is not set. Add it to run Groq inference.")

    # ------------------------------------------------------------------
    # Shared input bar: request + example + run button, grouped into one
    # bordered/shadowed container (see the mode-selector comment above for
    # why st.container(key=...) is what actually makes this one visual bar).
    # ------------------------------------------------------------------
    with st.container(key="agentic_input_bar"):
        input_col, example_col, button_col = st.columns([3, 1.3, 1.4], vertical_alignment="bottom")
        with input_col:
            st.markdown('<div class="input-bar-label">User request</div>', unsafe_allow_html=True)
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
        with example_col:
            st.markdown('<div class="input-bar-label">Try example</div>', unsafe_allow_html=True)
            st.selectbox(
                "Try example",
                example_options,
                format_func=lambda name: (
                    name if name == example_placeholder else scenario_prompt_label(name)
                ),
                key="agentic_try_example_select",
                on_change=apply_selected_example,
                label_visibility="collapsed",
            )
        with button_col:
            st.markdown('<div class="input-bar-label">&nbsp;</div>', unsafe_allow_html=True)
            run_full_pipeline_clicked = st.button(
                "▶ Run full pipeline",
                type="primary",
                key="agentic_run_full_pipeline",
                use_container_width=True,
                help="Runs the agent, then prepares the shapiq explanation for the selected tool.",
            )
    if st.session_state.get("agentic_pending_example_request") is not None:
        st.rerun()

    # Includes inference_backend/inference_model_name/explanation_mode so that switching
    # the HF model (e.g. 1.5B -> 3B) or the mode itself invalidates the previous agent
    # result exactly like changing the request text does -- otherwise a stale
    # `agentic_inferred_tool` from the old model can survive into the next run.
    current_inference_signature = (
        user_request,
        system_prompt,
        tool_context,
        inference_backend,
        inference_model_name,
        selected_hf_config.cache_key(scorer_mode="inference")
        if selected_hf_config is not None
        else None,
        explanation_mode,
    )
    if st.session_state.get("agentic_inference_signature") != current_inference_signature:
        st.session_state["agentic_inference_signature"] = current_inference_signature
        st.session_state["agentic_inferred_tool"] = None
        st.session_state["agentic_inference_result"] = None
        st.session_state["agentic_inference_backend"] = None
        st.session_state["agentic_inference_model"] = None
        st.session_state.has_run = False
        st.session_state.result = None
        st.session_state.result_signature = None

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
    # Always the current HF model selection, regardless of Developer mode: this is what
    # the LOGPROB_SCORER_LABEL explanation scorer loads, and it must never silently stay
    # on a stale default (e.g. after the user switches HF models with Developer mode
    # off) -- otherwise the explanation step's classifier can disagree with the agent
    # step's, which used `inference_model_name` directly.
    logprob_model_id = (
        selected_hf_config.model_id
        if selected_hf_config is not None
        else st.session_state.get(HF_MODEL_ID_SESSION_KEY, DEFAULT_LOGPROB_MODEL_ID)
    )
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
        NATIVE_HF_SCORER_LABEL
        if inference_backend == "HF local"
        else ("Groq soft-vote scorer" if has_groq_inference_result else NATIVE_HF_SCORER_LABEL)
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
            scorer_options = [NATIVE_HF_SCORER_LABEL, FINAL_ANSWER_SIMILARITY_LABEL]
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
                scorer_options.extend([LOGPROB_SCORER_LABEL, "Keyword scorer", "Mock model scorer"])
            if scorer_backend_key not in st.session_state:
                st.session_state[scorer_backend_key] = forced_default_scorer
            if st.session_state[scorer_backend_key] not in scorer_options:
                st.session_state[scorer_backend_key] = forced_default_scorer
            scorer_backend = st.selectbox(
                "Scoring method",
                scorer_options,
                key=scorer_backend_key,
            )
            if scorer_backend == NATIVE_HF_SCORER_LABEL:
                # `logprob_model_id` is already set above from HF_MODEL_ID_SESSION_KEY,
                # unconditional on Developer mode -- just display it here.
                st.caption(f"Inherited from HF model selection: `{logprob_model_id}`")
                st.caption(SAME_HF_MODEL_EXPLANATION)
                max_pairs_per_batch = st.number_input(
                    "HF continuation batch size",
                    min_value=1,
                    max_value=16,
                    value=1,
                    step=1,
                    key="agentic_logprob_pair_batch_size",
                    help=(
                        "Number of prompt/continuation pairs scored per local-model forward pass. "
                        "Use 1 on Colab T4 to avoid CUDA out-of-memory errors."
                    ),
                )
                st.caption(NATIVE_HF_SCORER_HELP)
            elif scorer_backend == LOGPROB_SCORER_LABEL:
                # A/B/C/D ablation. LocalHFClassificationRouter may be used only by
                # explicit developer-mode inference, never by the primary native scorer.
                st.caption(f"Inherited from HF model selection: `{logprob_model_id}`")
                max_pairs_per_batch = st.number_input(
                    "HF A/B/C/D pair batch size",
                    min_value=1,
                    max_value=16,
                    value=1,
                    step=1,
                    key="agentic_logprob_pair_batch_size",
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
            hf_model_config=selected_hf_config,
            hf_max_new_tokens=DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
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
        # Hard-reset any previous explanation before computing the new one: a fresh
        # agent result must never be paired with a stale XAI result computed for a
        # different backend/model/tool (e.g. after switching the HF model).
        st.session_state.has_run = False
        st.session_state.result = None
        st.session_state.result_signature = None
        st.session_state.pending_run = not bool(getattr(inference_result, "parse_error", None))
        st.rerun()

    # ------------------------------------------------------------------
    # 3. Two tabs, feeling like two stages of one pipeline
    # ------------------------------------------------------------------
    agent_tab, xai_tab = st.tabs(["1. Agent Result", "2. XAI Explanation"])

    # ------------------------------------------------------------------
    # 4. Tab 1: Agent Result
    # ------------------------------------------------------------------
    with agent_tab:
        inference_result = st.session_state.get("agentic_inference_result")

        if inference_result is None:
            st.markdown(
                """
                <div class="empty-state-card">
                    <h3>Run the agent first</h3>
                    <p>Click <strong>Run full pipeline</strong> to select a tool and prepare
                    the explanation.</p>
                    <div class="pipeline-hint">User request &rarr; tool decision &rarr;
                    XAI explanation</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            inference_error = getattr(inference_result, "error", None)
            parse_error = getattr(inference_result, "parse_error", None)
            selected_tool = getattr(inference_result, "selected_tool", None)
            result_backend = getattr(inference_result, "backend", inference_backend)
            result_model = getattr(inference_result, "model", inference_model_name)

            if inference_error:
                st.markdown(
                    f"""
                    <div class="error-card">
                        <h3>Agent run failed</h3>
                        <p>{escape(str(inference_error))}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    render_agent_result_card(
                        inference_result,
                        backend=result_backend,
                        model=result_model,
                    ),
                    unsafe_allow_html=True,
                )

            if developer_mode:
                with st.expander("Developer diagnostics", expanded=False):
                    st.write(f"Backend: `{result_backend}`")
                    st.write(f"Model: `{result_model}`")
                    st.write(f"Internal route label: `{selected_tool}`")
                    st.write(f"Parser status: `{'error' if parse_error else 'ok'}`")
                    st.markdown("**Raw tool arguments**")
                    st.json(getattr(inference_result, "tool_arguments", {}))
                    raw_trace = getattr(inference_result, "raw_trace", None)
                    if raw_trace is not None:
                        st.markdown("**Raw trace**")
                        st.json(raw_trace)
                    debug_prompt = getattr(inference_result, "debug_prompt", None)
                    if debug_prompt is not None:
                        st.markdown("**Debug prompt**")
                        st.code(str(debug_prompt), language="text")
                    raw_response = getattr(inference_result, "raw_response", None)
                    if raw_response:
                        st.markdown("**Raw response**")
                        st.code(str(raw_response), language="text")
                    normalized_response = getattr(
                        inference_result,
                        "normalized_response_used_for_parsing",
                        None,
                    )
                    if normalized_response:
                        st.markdown("**Normalized response used for parsing**")
                        st.code(str(normalized_response), language="text")
                    extracted_tool_call_json = getattr(
                        inference_result,
                        "extracted_tool_call_json",
                        None,
                    )
                    if extracted_tool_call_json:
                        st.markdown("**Extracted tool-call JSON**")
                        st.code(str(extracted_tool_call_json), language="json")
                    cleaned_direct_answer = getattr(
                        inference_result,
                        "cleaned_direct_answer",
                        None,
                    )
                    if cleaned_direct_answer:
                        st.markdown("**Cleaned direct answer**")
                        st.code(str(cleaned_direct_answer), language="text")
                    st.write(
                        "Removed direct-answer sentinel: "
                        f"`{getattr(inference_result, 'removed_direct_answer_sentinel', False)}`"
                    )
                    generation_parameters = getattr(
                        inference_result,
                        "generation_parameters",
                        None,
                    )
                    if generation_parameters:
                        st.markdown("**Generation parameters**")
                        st.json(generation_parameters)

    # ------------------------------------------------------------------
    # 5. Tab 2: XAI Explanation
    # ------------------------------------------------------------------
    with xai_tab:
        current_inference_result = st.session_state.get("agentic_inference_result")
        if has_untrusted_native_parse_result(current_inference_result):
            st.error(
                "Model output contained a tool-call structure, but it could not be "
                "parsed safely. The result is not treated as no_tool, and XAI is "
                "blocked because there is no trustworthy target tool."
            )
            return
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
            inference_backend,
            inference_model_name,
            selected_hf_config.cache_key(scorer_mode=str(scorer_backend))
            if selected_hf_config is not None
            else None,
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
                    inference_backend,
                    inference_model_name,
                    selected_hf_config.cache_key(scorer_mode=str(scorer_backend))
                    if selected_hf_config is not None
                    else None,
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

        # ---- Run-gating + compute happens here, ahead of every visual section below
        # (including the executive summary), so the whole layout has final results
        # available on first paint instead of computing mid-scroll. ----

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
            effective_scorer_backend = (
                hf_value_function_label_for_target(target_tool, scorer_backend)
                if inference_backend == "HF local"
                else scorer_backend
            )
            frozen_direct_answer_target = None
            if effective_scorer_backend == "Keyword scorer":
                primary_scorer = lexical_scorer
                primary_label = "Keyword scorer"
                native_hf_consistency = None
            elif effective_scorer_backend == NATIVE_HF_SCORER_LABEL:
                try:
                    selected_native_hf_config = require_selected_hf_config(selected_hf_config)
                    hf_router = load_local_hf_router(
                        selected_native_hf_config.model_id,
                        DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
                        trust_remote_code=False,
                        quantization_mode=selected_native_hf_config.quantization_mode,
                        device=selected_native_hf_config.device,
                        dtype=selected_native_hf_config.dtype,
                    )
                    primary_scorer = build_native_hf_scorer_from_router(
                        hf_router,
                        max_pairs_per_batch=int(max_pairs_per_batch),
                        selected_config=selected_native_hf_config,
                    )
                    native_hf_consistency = validate_hf_inference_xai_consistency(
                        selected_config=selected_native_hf_config,
                        router=hf_router,
                        scorer=primary_scorer,
                    )
                except Exception as error:  # noqa: BLE001
                    st.error(
                        "Could not prepare the native HF tool-call scorer. Install/check "
                        "`transformers` and `torch`, try a smaller causal language model, "
                        f"or check your environment. Details: {error}"
                    )
                    return
                primary_label = NATIVE_HF_SCORER_LABEL
                target_source = "Agent Result"
            elif effective_scorer_backend == NATIVE_DIRECT_ANSWER_SCORER_LABEL:
                try:
                    selected_native_hf_config = require_selected_hf_config(selected_hf_config)
                    direct_answer_target_text = require_trusted_direct_answer_target(
                        st.session_state.get("agentic_inference_result"),
                        inference_backend=inference_backend,
                    )
                    hf_router = load_local_hf_router(
                        selected_native_hf_config.model_id,
                        DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
                        trust_remote_code=False,
                        quantization_mode=selected_native_hf_config.quantization_mode,
                        device=selected_native_hf_config.device,
                        dtype=selected_native_hf_config.dtype,
                    )
                    # Frozen once, here -- not rebuilt per coalition below.
                    frozen_direct_answer_target = build_canonical_direct_answer_target(
                        direct_answer_target_text,
                        tokenizer=hf_router.tokenizer,
                    )
                    primary_scorer = build_native_direct_answer_scorer_from_router(
                        hf_router,
                        direct_answer_target=frozen_direct_answer_target,
                        max_pairs_per_batch=int(max_pairs_per_batch),
                        selected_config=selected_native_hf_config,
                    )
                    native_hf_consistency = validate_hf_inference_xai_consistency(
                        selected_config=selected_native_hf_config,
                        router=hf_router,
                        scorer=primary_scorer,
                    )
                except RuntimeError as error:
                    st.error(str(error))
                    return
                except Exception as error:  # noqa: BLE001
                    st.error(
                        "Could not prepare the native direct-answer continuation scorer. "
                        "Install/check `transformers` and `torch`, try a smaller causal "
                        f"language model, or check your environment. Details: {error}"
                    )
                    return
                primary_label = NATIVE_DIRECT_ANSWER_SCORER_LABEL
                target_source = "Agent Result"
            elif effective_scorer_backend in {LOGPROB_SCORER_LABEL, NO_TOOL_SURROGATE_SCORER_LABEL}:
                # No explicit st.spinner() wrapper here: load_logprob_scorer's own
                # @st.cache_resource(show_spinner="Preparing local HF scorer...") already
                # shows a friendly progress message on a real (non-cached) load.
                try:
                    primary_scorer = load_logprob_scorer(
                        logprob_model_id,
                        max_pairs_per_batch=int(max_pairs_per_batch),
                        device=selected_hf_config.device if selected_hf_config else None,
                        dtype=selected_hf_config.dtype if selected_hf_config else "auto",
                        quantization_mode=(
                            selected_hf_config.quantization_mode if selected_hf_config else "none"
                        ),
                        tokenizer_id=selected_hf_config.tokenizer_id
                        if selected_hf_config
                        else None,
                    )
                except Exception as error:  # noqa: BLE001
                    st.error(
                        "Could not load the calibrated log-odds scorer. Install/check "
                        "`transformers` and `torch`, try a smaller causal language model, "
                        f"or check your environment. Details: {error}"
                    )
                    return
                primary_label = effective_scorer_backend
                # Developer ablation only: LocalHFClassificationRouter remains available
                # through build_complete_agent_callable(calibrated_hf_mode=True), but this
                # XAI stage must not reroute and overwrite the Agent Result target.
                if using_inferred_tool:
                    target_source = "Agent Result"
                native_hf_consistency = None
            elif effective_scorer_backend == FINAL_ANSWER_SIMILARITY_LABEL:
                # No explicit st.spinner() wrapper here: load_final_answer_embedder's own
                # @st.cache_resource(show_spinner="Loading embedding model...") already
                # shows a friendly progress message on a real (non-cached) load.
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
                    hf_model_config=selected_hf_config,
                    hf_max_new_tokens=DEFAULT_NATIVE_HF_MAX_NEW_TOKENS,
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
                native_hf_consistency = None
            elif effective_scorer_backend == "Groq deterministic router":
                if not os.getenv("GROQ_API_KEY"):
                    st.error(
                        "GROQ_API_KEY is not set. Add it to the environment to use the Groq "
                        "deterministic router scorer."
                    )
                    return
                primary_scorer = GroqDeterministicRouterScorer(model_name=router_model_id)
                primary_label = "Groq deterministic router"
                native_hf_consistency = None
            elif effective_scorer_backend == "Groq soft-vote scorer":
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
                native_hf_consistency = None
            elif effective_scorer_backend == "Trajectory match: tool + normalized args":
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
                native_hf_consistency = None
            else:
                primary_scorer = LLMToolScorer(llm=MockLLM())
                primary_label = "Mock model scorer"
                native_hf_consistency = None

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
                if primary_label in {LOGPROB_SCORER_LABEL, NO_TOOL_SURROGATE_SCORER_LABEL}
                and primary_scorer.last_debug_outputs
                else None
            )
            empty_score = primary_scorer.score_batch(
                [empty_prompt],
                target_tool=target_tool,
                tool_descriptions=TOOLS,
            )[0]

            with st.spinner("Computing SV and k-SII..."):
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
                # Uses the full (unfiltered) ksii_explanation, not pairwise_ksii above --
                # the combined matrix needs its order-1 singleton entries for the
                # diagonal, which get_n_order(order=2) strips out.
                pairwise_matrix = pairwise_matrix_from_explanation(ksii_explanation, game.n_players)
                pair_label, pair_value = strongest_pair(pairwise_matrix, labels)
                notes = build_interpretation_notes(
                    attribution_frame,
                    pair_label,
                    pair_value,
                    full_score,
                    empty_score,
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
                        # Only used for strongest_pair's off-diagonal ranking below, never
                        # displayed -- the diagonal-free variant is enough here.
                        lexical_matrix = pairwise_only_matrix_from_explanation(
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
            value_function_type, value_function_fidelity = value_function_metadata(primary_label)
            st.session_state.has_run = True
            result_signature = (
                trace_name,
                inference_backend,
                inference_model_name,
                selected_hf_config.cache_key(scorer_mode=str(primary_label))
                if selected_hf_config is not None
                else None,
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
                "scorer_label": primary_label,
                "value_function_type": value_function_type,
                "value_function_fidelity": value_function_fidelity,
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
                "native_hf_consistency": native_hf_consistency,
                "hf_selected_model_config": selected_hf_config,
                "direct_answer_target": frozen_direct_answer_target,
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
        scorer_label = result.get("scorer_label", primary_label)
        value_function_type = result.get("value_function_type", "")
        value_function_fidelity = result.get("value_function_fidelity", "")
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
            # Full (unfiltered) ksii_explanation, not pairwise_ksii -- see the
            # matching comment at the primary computation call site above.
            pairwise_matrix = pairwise_matrix_from_explanation(ksii_explanation, len(user_segments))
        pair_label = result["pair_label"]
        pair_value = result["pair_value"]
        compare_with_lexical = result["compare_with_lexical"]
        llm_debug_outputs = result["llm_debug_outputs"]
        lexical_result = result["lexical_result"]
        final_answer_scorer_meta = result.get("final_answer_scorer_meta")
        logprob_full_diagnostics = result.get("logprob_full_diagnostics")
        native_hf_consistency = result.get("native_hf_consistency")
        hf_selected_model_config = result.get("hf_selected_model_config")
        is_final_answer_result = primary_label == FINAL_ANSWER_SIMILARITY_LABEL
        target_tool = result["target_tool"]
        target_source = result["target_source"]

        # ---- Defensive consistency check ----
        # The cached `result` dict must never explain a different tool than the current
        # Agent Result (e.g. a stale result computed for the previous HF model before a
        # 1.5B -> 3B switch). This should never trigger given the signature invalidation
        # above; it exists as a last-resort guard so a mismatch is caught and surfaced
        # instead of silently rendering the wrong explanation. Only checked when there is
        # a genuine current agent-selected tool to compare against -- the fallback
        # explanation scorer path legitimately has no `agentic_inferred_tool` to match.
        current_agentic_inferred_tool = st.session_state.get("agentic_inferred_tool")
        if current_agentic_inferred_tool in TOOLS and target_tool != current_agentic_inferred_tool:
            st.session_state.has_run = False
            st.session_state.result = None
            st.session_state.result_signature = None
            st.warning(
                "The previous explanation no longer matches the current Agent Result "
                f"(`{current_agentic_inferred_tool}`). Click **Run full pipeline** above "
                "to recompute the explanation."
            )
            return

        # ================================================================
        # Visual layout: a readable explanation first (0), then progressively
        # more detail (1-4). Every value below is read from the `result` dict
        # already unpacked above -- no new computation is introduced here.
        # ================================================================

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
        is_native_branch = primary_label in {
            NATIVE_HF_SCORER_LABEL,
            NATIVE_DIRECT_ANSWER_SCORER_LABEL,
        }
        if raw_log_odds_summary is not None:
            raw_full_score, raw_empty_score = raw_log_odds_summary
            explained_increase = raw_full_score - raw_empty_score
            empty_label, empty_display = "Raw empty score h(&empty;)", f"{raw_empty_score:.3f}"
            full_label, full_display = "Raw full score h(N)", f"{raw_full_score:.3f}"
            delta_label = "Support change h(N)-h(&empty;)"
            empty_short_label, full_short_label = "Baseline H(&empty;)", "Full H(N)"
        elif is_native_branch:
            explained_increase = float(full_score) - float(empty_score)
            empty_label, full_label = "Empty native score h(&empty;)", "Full native score h(N)"
            delta_label = "Native value v(N)=h(N)-h(&empty;)"
            empty_short_label, full_short_label = "Native h(&empty;)", "Native h(N)"
        else:
            explained_increase = float(full_score) - float(empty_score)
            empty_label, empty_display = "Baseline V(&empty;)", f"{empty_score:.3f}"
            full_label, full_display = "Full V(N)", f"{full_score:.3f}"
            delta_label = "Support change V(N)-V(&empty;)"
            empty_short_label, full_short_label = "Baseline V(&empty;)", "Full V(N)"

        # Computed once here and reused below by the explanation card, the k-SII mini-table,
        # and the interpretation card, instead of being recomputed per section.
        top_row = attribution_frame.iloc[0] if not attribution_frame.empty else None
        top_pairs = (
            top_pairwise_interactions(pairwise_matrix, user_segments)
            if pairwise_matrix.shape[0] >= 2
            else []
        )
        short_labels = [short_player_label(segment) for segment in user_segments]

        if top_row is not None:
            top_attribution = float(top_row["attribution"])
            main_evidence_label = (
                "Strongest supporting segment"
                if top_attribution >= 0
                else "Strongest opposing segment"
            )
            main_evidence_chip_html = (
                f"<span class='evidence-chip'>{escape(str(top_row['segment']))}: "
                f"“{escape(truncate_label(str(top_row['text']), max_length=48))}”</span>"
            )
        else:
            main_evidence_label = "Segment evidence"
            main_evidence_chip_html = "<span class='evidence-chip'>No segment evidence</span>"

        if top_pairs:
            strongest_pair_row = top_pairs[0]
            interaction_evidence_chip_html = (
                f"<span class='evidence-chip'>"
                f"{escape(str(strongest_pair_row['segment_i']))} + "
                f"{escape(str(strongest_pair_row['segment_j']))}: "
                f"“{escape(truncate_label(str(strongest_pair_row['text_i']), max_length=24))}” + "
                f"“{escape(truncate_label(str(strongest_pair_row['text_j']), max_length=24))}”"
                "</span>"
            )
        else:
            interaction_evidence_chip_html = "<span class='evidence-chip'>No pair available</span>"

        # Suggested interpretation logic, reusing the already-computed strongest-pair
        # value/label (result["pair_value"]/result["pair_label"]) -- no recomputation.
        if not top_pairs or pair_label == "No pair":
            xai_summary_interpretation = "Only one segment was found for this request."
        elif abs(pair_value) < 0.03:
            xai_summary_interpretation = "No dominant pairwise interaction was detected."
        elif pair_value > 0:
            if explained_increase >= 0:
                xai_summary_interpretation = (
                    "These segments add complementary evidence for the selected tool."
                )
            else:
                xai_summary_interpretation = (
                    "These segments interact positively, but the full request lowers "
                    "selected-tool support overall."
                )
        else:
            xai_summary_interpretation = "These segments carry redundant or counteracting evidence."

        # ---- 0. Decision explained: compact "why <tool>?" summary card ----
        xai_summary_icon = TOOL_ICONS.get(target_tool, "?")
        summary_title = (
            f"Why did the model call {target_tool}?"
            if target_tool in EXECUTABLE_TOOL_NAMES
            else "What supports this direct-answer continuation?"
        )

        # Δ Support is always the largest, most prominent metric. For the native
        # log-probability value functions specifically, H(&empty;)/H(N) are demoted
        # to a small-text recap nested inside the Δ Support card itself, below its
        # tier badge (Developer mode additionally reveals the full detail cards) --
        # other value functions keep their existing 3-card layout and separate
        # full-width recap line unchanged, since the 4-decimal/confidence-%/tier
        # treatment below is only calibrated for native log-probabilities.
        if is_native_branch:
            native_note_html = build_native_metric_summary_line_html(
                empty_short_label=empty_short_label,
                empty_score=float(empty_score),
                full_short_label=full_short_label,
                full_score=float(full_score),
            )
            score_flow_html = ""
            if developer_mode:
                metrics_section_html = (
                    "<div class='xai-summary-metrics'>"
                    + build_native_big_metric_card_html(
                        short_label=empty_short_label,
                        label=empty_label,
                        value=float(empty_score),
                    )
                    + build_native_big_metric_card_html(
                        short_label=full_short_label, label=full_label, value=float(full_score)
                    )
                    + build_delta_metric_html(
                        delta_label,
                        explained_increase,
                        show_tier=True,
                        note_html=native_note_html,
                    )
                    + "</div>"
                )
            else:
                metrics_section_html = (
                    "<div class='xai-summary-metrics'>"
                    + build_delta_metric_html(
                        delta_label,
                        explained_increase,
                        show_tier=True,
                        note_html=native_note_html,
                    )
                    + "</div>"
                )
        else:
            metrics_section_html = (
                "<div class='xai-summary-metrics'>"
                "<div class='xai-metric'>"
                f"<span class='xai-metric-label' title='{empty_label}'>{empty_short_label}</span>"
                f"<span class='xai-metric-value'>{empty_display}</span>"
                "</div>"
                "<div class='xai-metric'>"
                f"<span class='xai-metric-label' title='{full_label}'>{full_short_label}</span>"
                f"<span class='xai-metric-value'>{full_display}</span>"
                "</div>"
                + build_delta_metric_html(delta_label, explained_increase, show_tier=False)
                + "</div>"
            )
            score_flow_sign_class = " negative" if explained_increase < 0 else ""
            score_flow_html = (
                "<div class='xai-score-flow'>"
                f"Baseline <strong>{empty_display}</strong>"
                f" &rarr; Full <strong>{full_display}</strong>"
                f"<span class='delta-highlight{score_flow_sign_class}'>"
                f"{direction_arrow_html(explained_increase)} "
                f"&Delta; {explained_increase:+.3f}</span>"
                "</div>"
            )

        st.markdown(
            flatten_markdown_html(f"""
            <div class="xai-summary-card">
                <div class="xai-summary-left">
                    <span class="xai-summary-icon">{xai_summary_icon}</span>
                    <span class="xai-summary-title">
                        <span class="target-highlight">{escape(summary_title)}</span>
                    </span>
                </div>
                <div class="xai-summary-main">
                    <div class="evidence-row">
                        <span class="evidence-row-label">{escape(main_evidence_label)}</span>
                        {main_evidence_chip_html}
                    </div>
                    <div class="evidence-row">
                        <span class="evidence-row-label">Interaction evidence</span>
                        {interaction_evidence_chip_html}
                    </div>
                    <p class="xai-summary-interpretation">{escape(xai_summary_interpretation)}</p>
                </div>
                {metrics_section_html}
                {score_flow_html}
            </div>
            """),
            unsafe_allow_html=True,
        )

        # ---- Setup chips: compact, user-facing run metadata only ----
        if using_exact_computation:
            coalition_chip_label = "Coalitions"
            coalition_chip_value = f"{2 ** len(user_segments)} exact"
        else:
            coalition_chip_label = "Budget"
            coalition_chip_value = f"{budget} (approx.)"
        st.markdown(
            f"""
            <div class="setup-chip-row">
                <span class="setup-chip">
                    <strong>Players:</strong> {len(user_segments)} user-request segments
                </span>
                <span class="setup-chip"><strong>Indices:</strong> SV + k-SII</span>
                <span class="setup-chip">
                    <strong>{coalition_chip_label}:</strong> {coalition_chip_value}
                </span>
                <span class="setup-chip">
                    <strong>VF:</strong>
                    {escape(scorer_short_label(scorer_label))}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if (
            developer_mode
            and target_source == "fallback explanation scorer"
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

        # ---- 1. Player segmentation card, with fixed context nested inside it ----
        with st.container(key="agentic_player_card"):
            player_chip_html = "".join(build_player_chip_html(segment) for segment in user_segments)
            st.markdown(
                f"""
                <div class="player-card-header">Player segmentation</div>
                <div class="player-chip-grid">{player_chip_html}</div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander("Fixed context — system prompt + tool definitions", expanded=False):
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
                    # Duck-typed rather than isinstance(...): Streamlit's local file watcher
                    # can hot-reload a same-directory module while an older cached segmenter
                    # instance from st.cache_resource survives the rerun, which breaks
                    # isinstance checks against the freshly reloaded class.
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

        # ---- 2 & 3. SV | k-SII evidence, two balanced cards side by side ----
        st.caption(
            "SV (left) is a standalone attribution: one value per segment. k-SII (right) "
            "is a separate explanation with its own main effects (diagonal) and pairwise "
            "interactions (off-diagonal) -- SV and k-SII values are not directly comparable "
            "or interchangeable."
        )
        token_attribution_bar_plot, sentence_interaction_heatmap, plot_import_error = (
            load_text_plotters()
        )
        bar_xlabel = (
            "Final-answer semantic-fidelity attribution"
            if is_final_answer_result
            else "Target-tool attribution"
        )
        sv_col, ksii_col = st.columns(2, gap="medium")
        with sv_col, st.container(key="agentic_sv_card"):
            st.markdown(
                """
                <div class="evidence-card-header">
                    <div class="evidence-card-title">Individual effects — SV</div>
                    <div class="evidence-card-caption">Positive bars push toward the
                    selected tool; negative bars push away.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if token_attribution_bar_plot is None:
                st.warning(
                    "The shapiq text attribution plot is unavailable in this environment. "
                    f"Showing a simple fallback chart instead. Details: {plot_import_error}"
                )
                show_fallback_attribution_chart(attribution_frame)
            else:
                try:
                    fig_ax = token_attribution_bar_plot(first_order_sv, short_labels, show=False)
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
            # Compact by default; for larger player sets, show the top rows inline and
            # keep the rest in an expander instead of growing the card unboundedly.
            # `attribution_frame` is already ranked by |attribution| descending.
            show_all_sv_inline = len(user_segments) <= 6
            sv_rows_to_show = attribution_frame if show_all_sv_inline else attribution_frame.head(5)
            st.markdown(build_sv_mini_table_html(sv_rows_to_show), unsafe_allow_html=True)
            if not show_all_sv_inline:
                with st.expander("Show full SV attribution table", expanded=False):
                    st.dataframe(attribution_frame, use_container_width=True, hide_index=True)

        with ksii_col, st.container(key="agentic_ksii_card"):
            st.markdown(
                """
                <div class="evidence-card-header">
                    <div class="evidence-card-title">Main and pairwise effects — k-SII</div>
                    <div class="evidence-card-caption">Diagonal: individual k-SII main
                    effects. Off-diagonal: pairwise k-SII interactions. Red indicates
                    positive effects; blue indicates negative effects. For off-diagonal
                    cells, positive values indicate reinforcement, while negative values
                    indicate redundancy or suppression.</div>
                </div>
                """,
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
            st.markdown(build_player_legend_html(user_segments), unsafe_allow_html=True)

            # `top_pairwise_interactions` already ranks by |k-SII| descending; requesting
            # every pair (instead of the default top-5) here is the same ranking, just
            # unsliced, so the compact table below can show them all when there are few
            # enough players.
            all_pairs = (
                top_pairwise_interactions(
                    pairwise_matrix,
                    user_segments,
                    n=max(len(user_segments) * (len(user_segments) - 1) // 2, 1),
                )
                if pairwise_matrix.shape[0] >= 2
                else []
            )
            show_all_ksii_inline = len(user_segments) <= 6
            ksii_rows_to_show = all_pairs if show_all_ksii_inline else all_pairs[:5]

            if ksii_rows_to_show:
                st.markdown(build_ksii_mini_table_html(ksii_rows_to_show), unsafe_allow_html=True)
            else:
                st.caption("Only one player -- no pairwise interactions to show.")

            if not show_all_ksii_inline and all_pairs:
                with st.expander("Show all pairwise interactions", expanded=False):
                    st.markdown(build_ksii_mini_table_html(all_pairs), unsafe_allow_html=True)

            with st.expander("Show full interaction matrix", expanded=False):
                show_fallback_interaction_table(pairwise_matrix, labels)

        # ---- Detailed interpretation (collapsed: already summarized in the top card) ----
        with st.expander("Detailed interpretation", expanded=False):
            interpretation_html = build_interaction_interpretation(
                pair_label=pair_label,
                pair_value=pair_value,
                top_pairs=top_pairs,
            )
            st.markdown(
                f'<div class="interpretation-card">{interpretation_html}</div>',
                unsafe_allow_html=True,
            )

        # ---- Developer mode: computation diagnostics + raw backend diagnostics ----
        if developer_mode:
            with st.expander("Show computation diagnostics", expanded=False):
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
                sv_sum = float(sum(getattr(first_order_sv, "dict_values", {}).values()))
                sv_residual = abs(sv_sum - explained_increase)
                sv_efficiency_status = (
                    "passes" if sv_residual <= SV_EFFICIENCY_TOLERANCE else "exceeds tolerance"
                )
                st.caption(
                    f"SV efficiency check: sum(SV)={sv_sum:.3f}, "
                    f"support change={explained_increase:.3f}, "
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
                    st.write(f"Value function type: `{value_function_type}`")
                    st.write(f"Value function fidelity: `{value_function_fidelity}`")
                    if not using_exact_computation:
                        st.write(f"Budget: `{budget}`")
                    st.write("Full coalition prompt:")
                    st.code(full_prompt, language="text")
                    st.write("Empty coalition prompt:")
                    st.code(empty_prompt, language="text")

            with st.expander("Raw backend diagnostics", expanded=False):
                if native_hf_consistency is not None:
                    st.markdown("**HF native inference/XAI consistency**")
                    inference_identity = native_hf_consistency["inference"]
                    scorer_identity = native_hf_consistency["scorer"]
                    st.write(
                        "Inference/XAI model match: "
                        f"`{'yes' if native_hf_consistency['match'] else 'no'}`"
                    )
                    st.write(f"Inference model ID: `{inference_identity['model_id']}`")
                    st.write(f"Scorer model ID: `{scorer_identity['model_id']}`")
                    st.write(f"Inference tokenizer ID: `{inference_identity['tokenizer_id']}`")
                    st.write(f"Scorer tokenizer ID: `{scorer_identity['tokenizer_id']}`")
                    st.write(f"Model family: `{native_hf_consistency['model_family']}`")
                    st.write(
                        f"Requested quantization: `{inference_identity['requested_quantization']}`"
                    )
                    st.write(f"Actual quantization: `{inference_identity['actual_quantization']}`")
                    st.write(f"Device: `{inference_identity['device']}`")
                    st.write(f"Dtype / compute dtype: `{inference_identity['dtype']}`")
                    st.write(
                        "Native tool template source: `selected tokenizer.apply_chat_template`"
                    )
                    continuation_diagnostics = (
                        NATIVE_DIRECT_ANSWER_CONTINUATION_DIAGNOSTICS
                        if primary_label == NATIVE_DIRECT_ANSWER_SCORER_LABEL
                        else NATIVE_HF_CONTINUATION_DIAGNOSTICS
                    )
                    for (
                        diagnostic_name,
                        diagnostic_value,
                    ) in continuation_diagnostics.items():
                        st.write(f"{diagnostic_name}: `{diagnostic_value}`")
                    st.write(f"Scorer mode: `{primary_label}`")
                    if isinstance(hf_selected_model_config, SelectedHFModelConfig):
                        st.write(
                            "Cache key: "
                            f"`{hf_selected_model_config.cache_key(scorer_mode=str(primary_label))}`"
                        )
                    st.write(f"Target tool: `{target_tool}`")
                    if llm_debug_outputs:
                        token_count = llm_debug_outputs[0].get("continuation_token_count")
                        st.write(f"Continuation token count: `{token_count}`")

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
                        "target_source",
                        "target_label",
                        "continuation_type",
                        "continuation_scope",
                        "arguments_included",
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
                        "model_id",
                        "tokenizer_id",
                        "requested_quantization",
                        "actual_quantization",
                        "device",
                        "dtype",
                        "continuation_token_count",
                    ]
                    st.dataframe(
                        debug_frame[[column for column in debug_columns if column in debug_frame]],
                        use_container_width=True,
                        hide_index=True,
                    )

                if (
                    primary_label in {LOGPROB_SCORER_LABEL, NO_TOOL_SURROGATE_SCORER_LABEL}
                    and logprob_full_diagnostics is not None
                ):
                    st.markdown("**Legacy A/B/C/D probe diagnostics (full user request)**")
                    argmax_tool = logprob_full_diagnostics.get("argmax_tool")
                    is_router_argmax = argmax_tool == target_tool
                    diagnostic_softmax = logprob_full_diagnostics["calibrated_probabilities"].get(
                        target_tool
                    )
                    st.caption(
                        NO_TOOL_SURROGATE_HELP
                        if primary_label == NO_TOOL_SURROGATE_SCORER_LABEL
                        else LOGPROB_SCORER_HELP
                    )
                    logprob_metric_left, logprob_metric_mid, logprob_metric_right = st.columns(3)
                    logprob_metric_left.metric(
                        "Diagnostic softmax score",
                        f"{diagnostic_softmax:.3f}" if diagnostic_softmax is not None else "n/a",
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

        st.caption(f"Demo path: `{display_demo_path()}`")


if __name__ == "__main__":
    main()
