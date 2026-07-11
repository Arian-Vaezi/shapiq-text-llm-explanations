"""Tests for the local HuggingFace router helper."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

DEMO_DIR = Path(__file__).parents[3] / "src" / "demos" / "agentic_tool_use_explanation"
sys.path.insert(0, str(DEMO_DIR))

from hf_router import (  # noqa: E402
    HFArgumentExtractor,
    LocalHFClassificationRouter,
    LocalHFRouter,
    select_tool_from_scores,
)
from scorers import (  # noqa: E402
    CALIBRATION_USER_REQUESTS,
    ROUTING_LABELS,
    CalibratedToolLogOddsScorer,
    build_routing_classification_prompt,
)
from tool_game import ToolUseGame  # noqa: E402
from tool_schemas import TOOL_DESCRIPTIONS  # noqa: E402


def test_parse_valid_json_web_search_with_query_argument() -> None:
    result = LocalHFRouter.parse_response(
        """
        Here is the decision:
        {"agent_response":"I would search for current F1 results.",
         "selected_tool":"web_search_tool",
         "tool_arguments":{"query":"latest Formula 1 race winner this weekend"}}
        """
    )

    assert result.selected_tool == "web_search_tool"
    assert result.tool_arguments == {
        "query": "latest Formula 1 race winner this weekend",
    }
    assert result.agent_response == "I would search for current F1 results."


def test_parse_valid_json_calculator_with_expression_argument() -> None:
    result = LocalHFRouter.parse_response(
        '{"agent_response":"I would calculate it.",'
        '"selected_tool":"calculator_tool",'
        '"tool_arguments":{"expression":"2 + 3 * 4"}}'
    )

    assert result.selected_tool == "calculator_tool"
    assert result.tool_arguments == {"expression": "2 + 3 * 4"}
    assert result.agent_response == "I would calculate it."


def test_fallback_parser_detects_web_search_from_non_json_output() -> None:
    result = LocalHFRouter.parse_response("This needs web search because it asks latest news.")

    assert result.selected_tool == "web_search_tool"
    assert result.tool_arguments == {}
    assert result.agent_response == "I would use web_search_tool for this request."


def test_fallback_parser_detects_calculator_from_non_json_output() -> None:
    result = LocalHFRouter.parse_response("Use the calculator for the arithmetic.")

    assert result.selected_tool == "calculator_tool"
    assert result.tool_arguments == {}


def test_invalid_selected_tool_falls_back_to_no_tool() -> None:
    result = LocalHFRouter.parse_response(
        '{"selected_tool":"calendar_tool","tool_arguments":{"date":"tomorrow"}}'
    )

    assert result.selected_tool == "no_tool"
    assert result.tool_arguments == {"date": "tomorrow"}
    assert result.agent_response == "I would use no_tool for this request."


def test_choose_tool_can_use_monkeypatched_generation_without_loading_model() -> None:
    class FakeRouter(LocalHFRouter):
        def __init__(self) -> None:
            self.seen_prompt = ""

        def _generate_raw_response(self, router_prompt: str) -> tuple[str, str]:
            self.seen_prompt = router_prompt
            return (
                f"formatted::{router_prompt}",
                '{"agent_response":"I would use weather.",'
                '"selected_tool":"weather_tool",'
                '"tool_arguments":{"location":"Berlin","date":"tomorrow"}}',
            )

    router = FakeRouter()
    result = router.choose_tool("Will it rain in Berlin tomorrow?", TOOL_DESCRIPTIONS)

    assert result.selected_tool == "weather_tool"
    assert result.tool_arguments == {"location": "Berlin", "date": "tomorrow"}
    assert result.debug_prompt.startswith("formatted::")
    assert "weather_tool" in router.seen_prompt
    assert "Return only valid JSON" in router.seen_prompt


def test_local_hf_classification_router_picks_argmax_via_scorer_labels() -> None:
    """Required test 2/5: target-tool selection is produced by the same A/B/C/D protocol.

    ``LocalHFClassificationRouter`` must be a thin argmax wrapper around
    ``CalibratedToolLogOddsScorer.score_full_request_calibrated_labels`` -- the
    scorer's own calibrated routing-label scoring method -- not a separate
    structured JSON decision, so the tool it selects and the tool the scorer's
    coalition values explain are always produced under the identical protocol.
    """
    captured: dict[str, object] = {}

    class FakeScorer:
        def __init__(self) -> None:
            self.routing_labels = dict(ROUTING_LABELS)

        def score_full_request_calibrated_labels(
            self,
            user_request: str,
            *,
            system_prompt: str,
            tool_descriptions: dict[str, str],
        ) -> dict[str, float]:
            captured["user_request"] = user_request
            captured["system_prompt"] = system_prompt
            captured["tool_descriptions"] = tool_descriptions
            return {
                "weather_tool": -0.5,
                "calculator_tool": -2.0,
                "web_search_tool": -3.0,
                "no_tool": -4.0,
            }

    router = LocalHFClassificationRouter(FakeScorer())

    choice = router.choose_tool(
        "Will it rain in Berlin tomorrow?",
        system_prompt="Route to the correct tool.",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert choice.tool == "weather_tool"
    assert choice.score == -0.5
    assert choice.scores == {
        "weather_tool": -0.5,
        "calculator_tool": -2.0,
        "web_search_tool": -3.0,
        "no_tool": -3.25,
    }
    assert "A" in choice.reason
    assert captured["user_request"] == "Will it rain in Berlin tomorrow?"
    assert captured["system_prompt"] == "Route to the correct tool."
    assert captured["tool_descriptions"] == TOOL_DESCRIPTIONS


def test_local_hf_classification_router_uses_calibrated_argmax_not_raw_argmax() -> None:
    """Required test 3: the router must pick argmax_t r_t(N), not argmax_t g_t(N).

    Constructs a fake scorer where the raw (uncalibrated) argmax and the
    calibrated argmax disagree -- e.g. a tool with a strong fixed label prior
    dominates the raw likelihood but not the calibrated evidence -- and checks
    the router follows the calibrated evidence.
    """

    class FakeScorer:
        def __init__(self) -> None:
            self.routing_labels = dict(ROUTING_LABELS)

        def score_full_request_labels(self, user_request, *, system_prompt, tool_descriptions):
            del user_request, system_prompt, tool_descriptions
            # Raw argmax is calculator_tool (highest raw log-likelihood).
            return {
                "weather_tool": -2.0,
                "calculator_tool": -0.1,
                "web_search_tool": -3.0,
                "no_tool": -4.0,
            }

        def score_full_request_calibrated_labels(
            self, user_request, *, system_prompt, tool_descriptions
        ):
            del user_request, system_prompt, tool_descriptions
            # Calibrated argmax is weather_tool once each label's fixed prior
            # (calculator_tool's strong -0.1 prior in particular) is removed.
            return {
                "weather_tool": 1.0,
                "calculator_tool": 0.2,
                "web_search_tool": -1.0,
                "no_tool": -2.0,
            }

    router = LocalHFClassificationRouter(FakeScorer())

    choice = router.choose_tool(
        "Will it rain in Berlin tomorrow?",
        system_prompt="Route to the correct tool.",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert choice.tool == "weather_tool"
    assert choice.scores == {
        "weather_tool": 1.0,
        "calculator_tool": 0.2,
        "web_search_tool": -1.0,
        "no_tool": -1.25,
    }
    assert "no_tool_boost" in choice.reason.lower()
    assert "raw" not in choice.reason.lower()


def test_confidence_aware_raw_margin_gate_is_optional_and_thresholded() -> None:
    raw_scores = {
        "weather_tool": -0.4,
        "calculator_tool": -2.0,
        "web_search_tool": -3.0,
        "no_tool": 0.0,
    }
    calibrated_scores = {
        "weather_tool": 0.2,
        "calculator_tool": -1.0,
        "web_search_tool": -2.0,
        "no_tool": 0.0,
    }

    default_tool, default_scores = select_tool_from_scores(raw_scores, calibrated_scores)
    gated_tool, gated_scores = select_tool_from_scores(
        raw_scores,
        calibrated_scores,
        mode="raw_margin_gate",
        raw_margin_threshold=0.3,
    )
    ungated_tool, ungated_scores = select_tool_from_scores(
        raw_scores,
        calibrated_scores,
        mode="raw_margin_gate",
        raw_margin_threshold=0.5,
    )

    assert default_tool == "no_tool"
    assert default_scores["no_tool"] == 0.75
    assert gated_tool == "no_tool"
    assert gated_scores == raw_scores
    assert ungated_tool == "weather_tool"
    assert ungated_scores == calibrated_scores


def test_scaled_calibration_interpolates_raw_and_fully_calibrated_scores() -> None:
    raw_scores = {
        "weather_tool": -0.4,
        "calculator_tool": -2.0,
        "web_search_tool": -3.0,
        "no_tool": 0.0,
    }
    calibrated_scores = {
        "weather_tool": 0.2,
        "calculator_tool": -1.0,
        "web_search_tool": -2.0,
        "no_tool": 0.0,
    }

    raw_tool, raw_endpoint = select_tool_from_scores(
        raw_scores,
        calibrated_scores,
        mode="scaled_calibration",
        calibration_strength=0.0,
    )
    partial_tool, partial_scores = select_tool_from_scores(
        raw_scores,
        calibrated_scores,
        mode="scaled_calibration",
        calibration_strength=0.5,
    )
    calibrated_tool, calibrated_endpoint = select_tool_from_scores(
        raw_scores,
        calibrated_scores,
        mode="scaled_calibration",
        calibration_strength=1.0,
    )

    assert raw_tool == "no_tool"
    assert raw_endpoint == raw_scores
    assert partial_tool == "no_tool"
    assert partial_scores["weather_tool"] == pytest.approx(-0.1)
    assert calibrated_tool == "weather_tool"
    assert calibrated_endpoint == calibrated_scores


def test_no_tool_boost_adjusts_only_no_tool_and_is_the_default() -> None:
    raw_scores = {
        "weather_tool": -1.0,
        "calculator_tool": -2.0,
        "web_search_tool": -3.0,
        "no_tool": -4.0,
    }
    calibrated_scores = {
        "weather_tool": 0.5,
        "calculator_tool": 0.1,
        "web_search_tool": 0.0,
        "no_tool": 0.0,
    }

    calibrated_tool, unadjusted_scores = select_tool_from_scores(
        raw_scores,
        calibrated_scores,
        mode="calibrated",
    )
    boosted_tool, boosted_scores = select_tool_from_scores(raw_scores, calibrated_scores)

    assert calibrated_tool == "weather_tool"
    assert unadjusted_scores == calibrated_scores
    assert boosted_tool == "no_tool"
    assert boosted_scores == {
        "weather_tool": 0.5,
        "calculator_tool": 0.1,
        "web_search_tool": 0.0,
        "no_tool": 0.75,
    }
    assert calibrated_scores["no_tool"] == 0.0


def test_local_hf_classification_router_scores_the_exact_canonical_prompt() -> None:
    """LocalHFClassificationRouter must query the same prompts CalibratedToolLogOddsScorer builds.

    Calibrated evidence r_t(N) = g_t(N) - b_t requires two model calls: the
    real routing prompt (for g_t) and the content-free calibration probe
    prompt (for b_t) -- both built through the same canonical
    ``build_routing_classification_prompt`` helper.
    """
    scorer = CalibratedToolLogOddsScorer.__new__(CalibratedToolLogOddsScorer)
    scorer.routing_labels = dict(ROUTING_LABELS)
    scorer.calibration_user_requests = CALIBRATION_USER_REQUESTS
    scorer.model_id = "fake-model-id"
    scorer.routing_label_separator = " "
    scorer._calibration_cache = {}
    captured_prompts: list[str] = []

    def fake_compute_label_log_scores_batched(
        prompts: list[str],
        candidate_tools: list[str],
    ) -> list[dict[str, float]]:
        captured_prompts.extend(prompts)
        return [dict.fromkeys(candidate_tools, -1.0) for _ in prompts]

    scorer._raw_score_row_cache = {}
    scorer._compute_label_log_scores_batched = fake_compute_label_log_scores_batched

    router = LocalHFClassificationRouter(scorer)
    router.choose_tool(
        "Will it rain in Berlin tomorrow?",
        system_prompt="Route to the correct tool.",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert captured_prompts == [
        build_routing_classification_prompt(
            system_prompt="Route to the correct tool.",
            user_request="Will it rain in Berlin tomorrow?",
            tool_descriptions=TOOL_DESCRIPTIONS,
            routing_labels=ROUTING_LABELS,
        ),
        *[
            build_routing_classification_prompt(
                system_prompt="Route to the correct tool.",
                user_request=calibration_user_request,
                tool_descriptions=TOOL_DESCRIPTIONS,
                routing_labels=ROUTING_LABELS,
            )
            for calibration_user_request in CALIBRATION_USER_REQUESTS
        ],
    ]


def test_local_hf_router_loads_with_dtype_kwarg_and_no_device_map(monkeypatch) -> None:
    """Regression test for the macOS/MPS crash investigation.

    ``LocalHFRouter`` must resolve a single concrete device itself (matching
    ``CalibratedToolLogOddsScorer``'s pattern and the verified-working
    standalone load script) and call ``from_pretrained`` with ``dtype=`` and
    ``low_cpu_mem_usage=True`` -- never the deprecated ``torch_dtype=`` alias,
    and never accelerate's CUDA/multi-GPU-oriented ``device_map="auto"``,
    which is fragile on an MPS-only Mac.
    """
    transformers = pytest.importorskip("transformers")

    captured: dict[str, object] = {}

    class FakeTokenizer:
        pad_token = "eos"
        eos_token = "eos"

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: object) -> FakeTokenizer:
            del model_name, kwargs
            return cls()

    class FakeModel:
        def to(self, device: str) -> FakeModel:
            captured["to_device"] = device
            return self

        def eval(self) -> FakeModel:
            return self

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs: object) -> FakeModel:
            del model_name
            captured["kwargs"] = dict(kwargs)
            return cls()

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", FakeModel)

    LocalHFRouter(model_name="fake-model")

    kwargs = captured["kwargs"]
    assert "dtype" in kwargs
    assert "torch_dtype" not in kwargs
    assert "device_map" not in kwargs
    assert kwargs["low_cpu_mem_usage"] is True
    assert captured["to_device"] in {"cuda", "mps", "cpu"}


def test_local_hf_classification_router_owns_no_model_or_tokenizer() -> None:
    """Required test 2: LocalHFClassificationRouter must not own a model/tokenizer.

    Constructing it must not create anything beyond a reference to the
    injected scorer -- proving (behaviorally, not by reading source) that it
    is a thin wrapper, never an independent model owner.
    """
    sentinel_scorer = object()

    router = LocalHFClassificationRouter(sentinel_scorer)

    assert vars(router) == {
        "scorer": sentinel_scorer,
        "selection_mode": "no_tool_boost",
        "raw_margin_threshold": 1.0,
        "calibration_strength": 1.0,
        "no_tool_boost_delta": 0.75,
    }
    assert not hasattr(router, "model")
    assert not hasattr(router, "tokenizer")
    assert not hasattr(router, "model")
    assert not hasattr(router, "tokenizer")


def test_local_hf_classification_router_never_touches_old_router_or_from_pretrained(
    monkeypatch,
) -> None:
    """Required test 1: the calibrated multiclass HF path never loads a second model.

    Monkeypatches both the old structured-JSON ``LocalHFRouter.__init__`` and
    ``transformers.AutoModelForCausalLM.from_pretrained`` to raise if called,
    then exercises ``LocalHFClassificationRouter.choose_tool`` end to end with
    a fake (non-model-owning) scorer. If the classification router path ever
    reached either of those, this test would fail with the raised error
    instead of returning a normal result.
    """
    transformers = pytest.importorskip("transformers")

    def exploding_router_init(self, *args, **kwargs) -> None:
        del self, args, kwargs
        msg = "LocalHFRouter must not be constructed in calibrated log-odds mode"
        raise AssertionError(msg)

    def exploding_from_pretrained(*args, **kwargs) -> None:
        del args, kwargs
        msg = "from_pretrained must not be called in calibrated log-odds mode"
        raise AssertionError(msg)

    monkeypatch.setattr(LocalHFRouter, "__init__", exploding_router_init)
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", exploding_from_pretrained
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", exploding_from_pretrained)

    class FakeScorer:
        def __init__(self) -> None:
            self.routing_labels = dict(ROUTING_LABELS)

        def score_full_request_calibrated_labels(
            self, user_request, *, system_prompt, tool_descriptions
        ):
            del user_request, system_prompt, tool_descriptions
            return {
                "weather_tool": -0.1,
                "calculator_tool": -1.0,
                "web_search_tool": -2.0,
                "no_tool": -3.0,
            }

    router = LocalHFClassificationRouter(FakeScorer())

    choice = router.choose_tool(
        "Will it rain in Berlin tomorrow?",
        system_prompt="Route to the correct tool.",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert choice.tool == "weather_tool"


def test_hf_argument_extractor_reuses_injected_model_and_tokenizer() -> None:
    """Required test 5: HFArgumentExtractor reuses an injected model/tokenizer/device."""
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, text: str, return_tensors: str = "pt"):
            del text, return_tensors
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
            del token_ids, skip_special_tokens
            return '{"tool_arguments": {"location": "Berlin", "date": "tomorrow"}}'

    class FakeModel:
        def generate(self, input_ids, **generation_kwargs):
            del generation_kwargs
            batch_size = input_ids.shape[0]
            generated = torch.tensor([[9, 9]] * batch_size)
            return torch.cat([input_ids, generated], dim=-1)

    extractor = HFArgumentExtractor(model=FakeModel(), tokenizer=FakeTokenizer(), device="cpu")

    arguments = extractor.extract_arguments(
        user_request="Will it rain in Berlin tomorrow?",
        selected_tool="weather_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert arguments == {"location": "Berlin", "date": "tomorrow"}


def test_hf_argument_extractor_skips_generation_for_no_tool() -> None:
    """no_tool has no arguments to extract, so generation must not be attempted."""

    class ExplodingModel:
        def generate(self, *args, **kwargs) -> None:
            del args, kwargs
            msg = "must not generate for no_tool"
            raise AssertionError(msg)

    extractor = HFArgumentExtractor(model=ExplodingModel(), tokenizer=object(), device="cpu")

    arguments = extractor.extract_arguments(
        user_request="Explain photosynthesis.",
        selected_tool="no_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert arguments == {}


def test_hf_argument_extractor_never_calls_from_pretrained(monkeypatch) -> None:
    """Required test 5: HFArgumentExtractor must never load its own model or tokenizer."""
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")

    def exploding_from_pretrained(*args, **kwargs) -> None:
        del args, kwargs
        msg = "HFArgumentExtractor must not call from_pretrained"
        raise AssertionError(msg)

    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, "from_pretrained", exploding_from_pretrained
    )
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", exploding_from_pretrained)

    class FakeTokenizer:
        eos_token_id = 0

        def __call__(self, text: str, return_tensors: str = "pt"):
            del text, return_tensors
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
            del token_ids, skip_special_tokens
            return '{"tool_arguments": {"expression": "238 * 47"}}'

    class FakeModel:
        def generate(self, input_ids, **generation_kwargs):
            del generation_kwargs
            return torch.cat([input_ids, torch.tensor([[9, 9]] * input_ids.shape[0])], dim=-1)

    extractor = HFArgumentExtractor(model=FakeModel(), tokenizer=FakeTokenizer(), device="cpu")

    arguments = extractor.extract_arguments(
        user_request="Calculate 238 times 47",
        selected_tool="calculator_tool",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    assert arguments == {"expression": "238 * 47"}


def test_classification_router_and_coalition_game_share_one_scorer_instance() -> None:
    """Required test 4: coalition scoring reuses the exact same primary_scorer object.

    Mirrors the app's calibrated-log-odds wiring: build one scorer, use it for
    target-tool selection via ``LocalHFClassificationRouter``, then pass that
    *same* object into ``ToolUseGame`` for Shapley/k-SII coalition scoring --
    proving there is exactly one scorer (and therefore one model) instance
    behind both roles, by identity, not just by equal configuration.
    """

    class FakeScorer:
        def __init__(self) -> None:
            self.routing_labels = dict(ROUTING_LABELS)

        def score_full_request_calibrated_labels(
            self, user_request, *, system_prompt, tool_descriptions
        ):
            del user_request, system_prompt, tool_descriptions
            return {
                "weather_tool": -0.1,
                "calculator_tool": -1.0,
                "web_search_tool": -2.0,
                "no_tool": -3.0,
            }

        def score_batch(self, prompts, *, target_tool, tool_descriptions):
            del target_tool, tool_descriptions
            return [0.0 for _ in prompts]

    primary_scorer = FakeScorer()

    classification_router = LocalHFClassificationRouter(primary_scorer)
    target_choice = classification_router.choose_tool(
        "Will it rain in Berlin tomorrow?",
        system_prompt="Route to the correct tool.",
        tool_descriptions=TOOL_DESCRIPTIONS,
    )

    game = ToolUseGame(
        target_tool=target_choice.tool,
        user_segments=["Will it rain in Berlin tomorrow?"],
        system_prompt="Route to the correct tool.",
        scorer=primary_scorer,
        tool_descriptions=TOOL_DESCRIPTIONS,
        normalize=False,
    )

    assert classification_router.scorer is primary_scorer
    assert game.scorer is primary_scorer
