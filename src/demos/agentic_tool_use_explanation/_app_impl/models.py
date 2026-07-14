"""Streamlit demo for explaining agentic tool-use decisions with shapiq."""

# ruff: noqa: F401

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
from persistence import write_config_snapshot_safely
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
    from collections.abc import Callable, MutableMapping

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
DEFAULT_HF_LOCAL_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
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


def initialize_hf_model_session_state(session_state: MutableMapping[str, object]) -> None:
    """Set the HF-local default without replacing an existing user selection."""
    if HF_MODEL_ID_SESSION_KEY not in session_state:
        session_state[HF_MODEL_ID_SESSION_KEY] = DEFAULT_HF_LOCAL_MODEL_ID


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
ASSET_DIR = Path(__file__).parent.parent / "assets"
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
        with router.tokenizer_lock:
            first_param = next(router.model.parameters())
        _hf_lifecycle_log(
            "[HF MODEL]",
            f"device={first_param.device}",
            f"dtype={first_param.dtype}",
            f"model_id={model_name}",
        )
    write_config_snapshot_safely(
        hf_model_id=model_name,
        device=getattr(router, "device", device),
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
        tokenizer_lock=threading.RLock(),
    )
    _hf_lifecycle_log("[HF LOAD] CalibratedToolLogOddsScorer loaded", f"model_id={model_id}")
    if DEBUG_HF_LIFECYCLE:
        with scorer.tokenizer_lock:
            first_param = next(scorer.model.parameters())
        _hf_lifecycle_log(
            "[HF MODEL]",
            f"device={first_param.device}",
            f"dtype={first_param.dtype}",
            f"model_id={model_id}",
        )
    write_config_snapshot_safely(
        hf_model_id=model_id,
        device=getattr(scorer, "device", device),
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
        tokenizer_lock=getattr(router, "tokenizer_lock", None) or threading.RLock(),
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
        tokenizer_lock=getattr(router, "tokenizer_lock", None) or threading.RLock(),
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


__all__ = [name for name in globals() if not name.startswith("__")]
