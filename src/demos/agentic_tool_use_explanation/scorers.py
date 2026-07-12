"""Scorers for agentic tool-use coalition value functions."""

from __future__ import annotations

import math
import re
import sys
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    from demos.agentic_tool_use_explanation.tool_schemas import (
        EXECUTABLE_TOOL_SCHEMAS,
        NO_TOOL_NAME,
        get_executable_tool_schemas,
        render_tool_schemas_text,
    )
except ModuleNotFoundError:
    from tool_schemas import (
        EXECUTABLE_TOOL_SCHEMAS,
        NO_TOOL_NAME,
        get_executable_tool_schemas,
        render_tool_schemas_text,
    )

DEFAULT_HF_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_LOGPROB_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
NATIVE_TOOL_ROUTING_INSTRUCTION = (
    "You are a tool-routing assistant.\n\n"
    "Use weather_tool only for weather, rain, temperature, forecast, or city-and-date "
    "weather requests.\n"
    "Use calculator_tool only for exact arithmetic, totals, percentages, numeric "
    "expressions, or deterministic calculations.\n"
    "Use web_search_tool only when the request requires current, latest, recent, "
    "live, changing, or externally retrieved information.\n\n"
    "For stable conceptual questions, definitions, explanations, educational "
    "questions, and general knowledge: do not call any tool; answer the user "
    "directly in natural language.\n"
    "Do not call web_search_tool for definitions, explanations, or stable "
    "conceptual questions.\n\n"
    "Never output the literal strings no_tool, No_tool, NoTool, or no tool. "
    "These are internal application labels and are not valid user-facing responses.\n\n"
    "If an external tool is required, emit exactly one valid tool call using the "
    "provided tool schema. Otherwise, provide a complete natural-language answer "
    "without emitting a <tool_call> block."
)

# Fixed routing-label mapping for the constrained classification protocol used by
# the HF local router and the calibrated logprob scorer. Single-letter decision
# codes avoid the tool-name tokenization, candidate-length, and English-template
# priors that a natural-language continuation such as "The correct tool is
# weather_tool." would introduce into the score.
ROUTING_LABELS: dict[str, str] = {
    "weather_tool": "A",
    "calculator_tool": "B",
    "web_search_tool": "C",
    "no_tool": "D",
}

# Separator inserted between the fixed "Decision:" prompt suffix and the label
# token(s). An explicit separator keeps the prompt/label token boundary stable
# across tokenizers that would otherwise merge the final prompt character with
# the label's first character into a single (different) token.
ROUTING_LABEL_SEPARATOR = " "

# Diverse neutral calibration requests. Distinct from the empty Shapley
# coalition (which uses an empty user request "") -- these fixed probes are
# used only to estimate each label's per-prompt/per-label prior under the
# routing protocol.
CALIBRATION_USER_REQUESTS: tuple[str, ...] = (
    "Hello there.",
    "I have a quick question for you.",
    "Let's talk about something for a moment.",
    "Can you help me with this?",
    "Just checking in.",
)

TOOL_KEYWORDS = {
    "weather_tool": {
        "weather",
        "rain",
        "forecast",
        "temperature",
        "snow",
        "wind",
        "berlin",
        "tomorrow",
        "morning",
    },
    "calculator_tool": {
        "calculate",
        "compute",
        "times",
        "multiply",
        "plus",
        "minus",
        "divide",
        "percent",
        "number",
        "final",
    },
    "web_search_tool": {
        "latest",
        "newest",
        "current",
        "recent",
        "today",
        "weekend",
        "won",
        "race",
        "product",
        "search",
        "web",
    },
    "no_tool": {
        "explain",
        "what",
        "simple",
        "terms",
        "conceptual",
        "stable",
        "knowledge",
        "directly",
    },
}


@dataclass(frozen=True)
class ToolChoice:
    """A lightweight tool-router decision."""

    tool: str
    score: float
    reason: str
    scores: dict[str, float]


class ToolScorerProtocol(Protocol):
    """Common interface for coalition value-function scorers."""

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        """Score how strongly each coalition prompt supports the target tool."""


class TextGeneratorProtocol(Protocol):
    """Minimal interface expected by LLMToolScorer."""

    def generate(self, prompt: str) -> str:
        """Generate a text response for one prompt."""


@dataclass
class HuggingFaceTextGenerator:
    """Adapt the shared HuggingFace wrapper to TextGeneratorProtocol."""

    model_id: str = DEFAULT_HF_MODEL_ID
    device: str = "auto"
    hf_token: str | None = None
    max_new_tokens: int = 8
    use_chat_template: bool = True

    def __post_init__(self) -> None:
        wrapper_device = "cuda" if self.device == "auto" else self.device
        try:
            from demos.shared.hf_model import HFModelWrapper
        except ModuleNotFoundError:
            src_dir = Path(__file__).resolve().parents[2]
            if str(src_dir) not in sys.path:
                sys.path.insert(0, str(src_dir))
            from demos.shared.hf_model import HFModelWrapper

        self._model = HFModelWrapper(
            model_name=self.model_id,
            device=wrapper_device,
            hf_token=self.hf_token or None,
        )

    def generate(self, prompt: str) -> str:
        """Generate one scoring response for an LLM-as-a-judge prompt."""
        return self._model.generate_text(
            prompt,
            max_new_tokens=self.max_new_tokens,
            chat=self.use_chat_template,
        ).strip()


def normalize_tokens(text: str) -> set[str]:
    """Return lowercase alphanumeric tokens."""
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def clamp_score(score: float) -> float:
    """Clamp a numeric score to the value-function range."""
    if not math.isfinite(score):
        msg = "Score must be finite."
        raise ValueError(msg)
    return float(min(1.0, max(0.0, score)))


def build_tool_calling_prompt(
    tokenizer: object,
    *,
    system_prompt: str,
    user_request: str,
    tool_schemas: Sequence[Mapping[str, object]] = EXECUTABLE_TOOL_SCHEMAS,
) -> str:
    """Build model input with native structured tools when available.

    The input context uses structured tool schemas through ``tools=`` whenever
    the tokenizer supports it. Older non-native scorers may still use the text
    fallback; strict native scoring uses :func:`build_native_tool_call_prompt`.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_request},
    ]
    schemas = get_executable_tool_schemas()
    if tool_schemas is not EXECUTABLE_TOOL_SCHEMAS:
        schemas = tuple(_copy_schema(schema) for schema in tool_schemas)
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=schemas,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError as error:
        if not _is_unsupported_tools_argument_error(error):
            raise
    return _build_tool_calling_prompt_fallback(
        system_prompt=system_prompt,
        user_request=user_request,
        tool_schemas=schemas,
    )


def build_native_tool_call_messages(
    *,
    system_prompt: str,
    user_request: str,
) -> list[dict[str, object]]:
    """Return the canonical messages shared by native HF inference and scoring."""
    system_text = system_prompt.strip()
    if system_text:
        system_text = f"{system_text}\n\n{NATIVE_TOOL_ROUTING_INSTRUCTION}"
    else:
        system_text = NATIVE_TOOL_ROUTING_INSTRUCTION
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_request.strip()},
    ]


def build_native_tool_call_prompt(
    tokenizer: object,
    *,
    system_prompt: str,
    user_request: str,
    tool_schemas: Sequence[Mapping[str, object]] = EXECUTABLE_TOOL_SCHEMAS,
) -> str:
    """Build the strict native HF tool-calling prompt with structured schemas."""
    schemas = tuple(_copy_schema(schema) for schema in tool_schemas)
    return str(
        tokenizer.apply_chat_template(
            build_native_tool_call_messages(
                system_prompt=system_prompt,
                user_request=user_request,
            ),
            tools=schemas,
            tokenize=False,
            add_generation_prompt=True,
        )
    )


def build_native_tool_call_continuation(
    tokenizer: object,
    *,
    system_prompt: str,
    user_request: str,
    target_tool: str,
    tool_schemas: Sequence[Mapping[str, object]] = EXECUTABLE_TOOL_SCHEMAS,
) -> str:
    """Render the canonical native tool-identity continuation through the tool name.

    The returned continuation is computed by asking the tokenizer chat template
    to render the same system/user messages plus an assistant ``tool_calls``
    message for ``target_tool``. The scorer keeps only the assistant
    continuation prefix through the rendered tool name and deliberately excludes
    free-form argument text, isolating tool-identity evidence from argument
    generation variability.
    """
    executable_names = {
        str(schema["function"]["name"])
        for schema in tool_schemas
        if isinstance(schema.get("function"), Mapping)
    }
    if target_tool == NO_TOOL_NAME:
        msg = "NativeToolCallScorer is valid only for executable tools, not no_tool."
        raise ValueError(msg)
    if target_tool not in executable_names:
        msg = f"Target tool {target_tool!r} is not an executable native tool."
        raise ValueError(msg)

    schemas = tuple(_copy_schema(schema) for schema in tool_schemas)
    messages = build_native_tool_call_messages(
        system_prompt=system_prompt,
        user_request=user_request,
    )
    prompt = str(
        tokenizer.apply_chat_template(
            messages,
            tools=schemas,
            tokenize=False,
            add_generation_prompt=True,
        )
    )
    full = str(
        tokenizer.apply_chat_template(
            [
                *messages,
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": target_tool,
                                "arguments": {},
                            },
                        }
                    ],
                },
            ],
            tools=schemas,
            tokenize=False,
            add_generation_prompt=False,
        )
    )
    if not full.startswith(prompt):
        msg = "Tokenizer chat template did not preserve the native prompt as a prefix."
        raise ValueError(msg)
    continuation = full[len(prompt) :]
    name_end = continuation.find(target_tool)
    if name_end < 0:
        msg = f"Rendered native tool-call continuation did not contain {target_tool!r}."
        raise ValueError(msg)
    return continuation[: name_end + len(target_tool)]


def build_native_direct_answer_prompt(
    tokenizer: object,
    *,
    system_prompt: str,
    user_request: str,
    tool_schemas: Sequence[Mapping[str, object]] = EXECUTABLE_TOOL_SCHEMAS,
) -> str:
    """Build the native assistant-continuation boundary used for direct-answer scoring.

    Reuses :func:`build_native_tool_call_prompt` -- the identical system/user
    messages, structured tool schemas, and chat-template assistant-generation
    boundary used by full-context native inference and by
    :class:`NativeToolCallScorer` -- so a direct-answer continuation is
    teacher-forced at exactly the same assistant boundary as a tool-identity
    continuation, never scored as part of the user message.
    """
    return build_native_tool_call_prompt(
        tokenizer,
        system_prompt=system_prompt,
        user_request=user_request,
        tool_schemas=tool_schemas,
    )


# Fixed, deterministic set of leading discourse preambles stripped from a direct
# answer before target-fragment extraction. Not an NLP heuristic -- a literal
# prefix match evaluated only once, only at the very start of the (already
# artifact-stripped) answer text.
DIRECT_ANSWER_DISCOURSE_PREAMBLES: tuple[str, ...] = (
    "Sure,",
    "Certainly,",
    "Of course,",
    "Here is the answer:",
    "Here's the answer:",
    "I can help with that.",
)

# Known trailing native-generation artifacts stripped before extraction, mirroring
# hf_router.normalize_native_tool_output's handling for the same tokens. Duplicated
# here (rather than imported) because hf_router imports from this module already;
# importing back would create a circular import.
_DIRECT_ANSWER_TRAILING_ARTIFACTS: tuple[str, ...] = ("<|im_end|>",)

_DIRECT_ANSWER_SENTINEL_ONLY_RE = re.compile(
    r"^\s*(?:no_tool|notool|no tool)\s*$",
    flags=re.IGNORECASE,
)

# Small fixed set of common abbreviations that must not be treated as a sentence
# boundary when immediately followed by a period. Deterministic lookup only --
# no NLP/grammar model.
_DIRECT_ANSWER_SENTENCE_ABBREVIATIONS: frozenset[str] = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "vs",
        "etc",
        "e.g",
        "i.e",
        "u.s",
        "u.k",
        "st",
        "no",
        "approx",
        "fig",
        "vol",
    }
)


def _strip_trailing_direct_answer_artifacts(text: str) -> str:
    """Strip known trailing model-generation artifacts, repeatedly, from the end."""
    stripped = text.strip()
    changed = True
    while changed:
        changed = False
        for artifact in _DIRECT_ANSWER_TRAILING_ARTIFACTS:
            if stripped.endswith(artifact):
                stripped = stripped[: -len(artifact)].rstrip()
                changed = True
    return stripped


def _strip_leading_direct_answer_preamble(text: str) -> str:
    """Strip one leading discourse preamble, only if it occurs at the very start."""
    for preamble in DIRECT_ANSWER_DISCOURSE_PREAMBLES:
        if text.startswith(preamble):
            return text[len(preamble) :].lstrip()
    return text


def _find_direct_answer_sentence_end(text: str, start: int = 0) -> int | None:
    """Return the index just after the first qualifying sentence-ending punctuation.

    Treats ``.``, ``!``, and ``?`` as sentence-enders, but a lone ``.`` is
    skipped (not treated as a boundary) when it separates two digits (a decimal
    number such as ``3.14``) or immediately follows a known abbreviation (such
    as ``Dr.`` or ``e.g.``). A qualifying boundary must be followed by
    whitespace or by the end of the text, so punctuation glued to further
    non-space characters (e.g. inside a URL) is not treated as a boundary.
    """
    length = len(text)
    for match in re.finditer(r"[.!?]+", text):
        if match.start() < start:
            continue
        end = match.end()
        punctuation = match.group(0)
        if punctuation == ".":
            before = text[: match.start()]
            after = text[end:]
            if before[-1:].isdigit() and after[:1].isdigit():
                continue
            trailing_word_match = re.search(r"(\S+)$", before)
            if (
                trailing_word_match is not None
                and trailing_word_match.group(1).lower().rstrip(".")
                in _DIRECT_ANSWER_SENTENCE_ABBREVIATIONS
            ):
                continue
        if end == length or text[end].isspace():
            return end
    return None


def build_canonical_direct_answer_target(
    answer: str,
    *,
    tokenizer: object,
    max_target_tokens: int = 24,
    min_target_tokens: int = 4,
) -> str:
    """Return a deterministic, bounded first-sentence fragment of a direct answer.

    This is a pure function -- no model calls, no nondeterministic or semantic
    heuristics. It strips known trailing native-generation artifacts and a
    small fixed set of leading discourse preambles, then prefers the first
    sentence (respecting common abbreviations and decimal numbers), extending
    into subsequent text only if the first sentence would tokenize under
    ``min_target_tokens``, and finally truncates to ``max_target_tokens`` by
    re-decoding the truncated token ids. The result is the *canonical
    deterministic answer fragment* -- calling this twice with the same inputs
    always returns the same string.
    """
    if max_target_tokens < min_target_tokens:
        msg = "max_target_tokens must be greater than or equal to min_target_tokens."
        raise ValueError(msg)

    text = _strip_trailing_direct_answer_artifacts(answer)
    text = _strip_leading_direct_answer_preamble(text).strip()
    if not text:
        msg = "No usable canonical direct-answer target remains after cleanup."
        raise ValueError(msg)
    if _DIRECT_ANSWER_SENTINEL_ONLY_RE.fullmatch(text):
        msg = "Direct-answer target text is only an internal sentinel, not a usable answer."
        raise ValueError(msg)

    end = _find_direct_answer_sentence_end(text)
    candidate = text[:end].strip() if end is not None else text
    candidate_ids = _tokenize_to_ids(tokenizer, candidate)

    # Extend into subsequent sentences only if the first sentence under-shoots
    # min_target_tokens, stopping once the minimum is reached or the input ends.
    while len(candidate_ids) < min_target_tokens and end is not None and end < len(text):
        next_end = _find_direct_answer_sentence_end(text, end)
        if next_end is None:
            candidate = text.strip()
            end = len(text)
        else:
            candidate = text[:next_end].strip()
            end = next_end
        candidate_ids = _tokenize_to_ids(tokenizer, candidate)

    if len(candidate_ids) > max_target_tokens:
        candidate_ids = candidate_ids[:max_target_tokens]
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            msg = "Tokenizer must support decode() to truncate the direct-answer target."
            raise ValueError(msg)
        candidate = str(decode(candidate_ids)).strip()

    if not candidate:
        msg = "Canonical direct-answer target extraction produced an empty fragment."
        raise ValueError(msg)
    return candidate


def _build_tool_calling_prompt_fallback(
    *,
    system_prompt: str,
    user_request: str,
    tool_schemas: Sequence[Mapping[str, object]],
) -> str:
    """Build a deterministic text fallback from canonical tool schemas."""
    return (
        f"System:\n{system_prompt}\n\n"
        f"Available tools:\n{render_tool_schemas_text(tool_schemas)}\n\n"
        f"User:\n{user_request}\n\n"
        "Assistant:"
    )


def _copy_schema(schema: Mapping[str, object]) -> dict[str, object]:
    import copy

    return copy.deepcopy(dict(schema))


def _is_unsupported_tools_argument_error(error: TypeError) -> bool:
    message = str(error)
    return "tools" in message and (
        "unexpected keyword" in message
        or "unsupported" in message.lower()
        or "not supported" in message.lower()
    )


def split_coalition_prompt(prompt: str) -> tuple[str, str]:
    """Extract fixed system prompt and coalition user request from demo prompts."""
    system_prompt, separator, after_system = prompt.partition("\n\nAvailable tools:\n")
    if not separator:
        return "", prompt
    _, user_separator, after_tools = after_system.partition("\n\nUser request:\n")
    if not user_separator:
        return system_prompt.strip(), prompt
    if after_tools.startswith("\nAssistant:"):
        return system_prompt.strip(), ""
    user_request, _, _ = after_tools.partition("\n\nAssistant:")
    return system_prompt.strip(), user_request.strip()


def join_user_request_segments(selected_segments: Sequence[object]) -> str:
    """Join coalition-selected user-request segments into one canonical request string.

    Accepts either bare strings or objects with a ``.text`` attribute (e.g. the
    demo's ``ToolUseSegment``), stripping each segment and dropping ones that are
    empty after stripping. This is the single place that decides how a coalition's
    selected segments become the "user request" text; every prompt builder in this
    demo (``app.py`` and ``tool_game.ToolUseGame``) must call this instead of
    reimplementing the joining logic, so the same coalition always produces the
    same user request string regardless of call site.
    """
    texts = []
    for segment in selected_segments:
        text = str(segment.text).strip() if hasattr(segment, "text") else str(segment).strip()
        if text:
            texts.append(text)
    return " ".join(texts)


def build_coalition_prompt(user_request: str, *, system_prompt: str, tool_context: str) -> str:
    """Build the one canonical coalition prompt format used everywhere in this demo.

    This is the single source of truth for embedding a (possibly empty) masked
    user request into the fixed system prompt and tool context. Every prompt
    builder in this demo (the Inference/Explanation tabs in ``app.py`` and
    ``tool_game.ToolUseGame.build_prompt``) must call this function instead of
    reimplementing the formatting -- otherwise the empty-coalition prompt used as
    the Shapley normalization baseline can end up textually different across call
    sites (extra/missing blank line), which causes duplicate, independently
    re-run agent calls for what should be a single cached "empty coalition" value
    and can break Shapley efficiency under any backend that is not perfectly
    deterministic. This is the structural counterpart of
    :func:`split_coalition_prompt`, which recovers ``user_request`` from this
    exact format.
    """
    user_request_block = f"User request:\n{user_request}" if user_request else "User request:"
    return (
        f"{system_prompt.strip()}\n\n"
        f"Available tools:\n{tool_context.strip()}\n\n"
        f"{user_request_block}\n\n"
        "Assistant:"
    )


def build_routing_classification_prompt(
    *,
    system_prompt: str,
    user_request: str,
    tool_descriptions: Mapping[str, str],
    routing_labels: Mapping[str, str] = ROUTING_LABELS,
) -> str:
    """Build the one canonical constrained-classification routing prompt.

    This is the single source of truth for the routing protocol required by HF
    local routing inference, HF local logprob scoring, calibration scoring, and
    coalition scoring: a fixed decision-code legend -- rendered from the same
    canonical ``tool_descriptions`` used everywhere else in the demo, not a
    second hard-coded description dictionary -- followed by an instruction to
    return exactly one code. Every call site must build this prompt through
    this function instead of reimplementing the formatting, so token
    boundaries and any residual template prior stay identical across the
    coalition prompt, the calibration prompt, and the empty-coalition prompt.
    Changing a tool's canonical description changes this prompt (and therefore
    the calibration cache key), since the description is part of what the
    model conditions on.
    """
    legend_entries = []
    for tool_name, label in routing_labels.items():
        description = tool_descriptions.get(tool_name, "").strip()
        entry = f"{label} = {tool_name}"
        if description:
            entry = f"{entry}\nDescription: {description}"
        legend_entries.append(entry)
    decision_legend = "\n\n".join(legend_entries)
    system_block = system_prompt.strip()
    system_section = f"{system_block}\n\n" if system_block else ""
    user_request_block = f"User request:\n{user_request}" if user_request else "User request:"
    return (
        "System:\n"
        "You are a tool-routing classifier.\n\n"
        f"{system_section}"
        "Available routing decisions:\n"
        f"{decision_legend}\n\n"
        "Return exactly one decision code and nothing else.\n\n"
        f"{user_request_block}\n\n"
        "Decision:"
    )


@dataclass(frozen=True)
class RoutingLabelTokenization:
    """Tokenizer diagnostics for one routing label under one prompt prefix."""

    tool_name: str
    label: str
    token_length: int


def validate_routing_label_tokens(
    tokenizer: object,
    prompt_prefix: str,
    *,
    routing_labels: Mapping[str, str] = ROUTING_LABELS,
    separator: str = ROUTING_LABEL_SEPARATOR,
) -> dict[str, RoutingLabelTokenization]:
    """Validate routing-label tokenizer prefix stability and record token lengths.

    For every routing label, verifies that tokenizing ``prompt_prefix`` alone
    produces a token sequence that is an exact prefix of tokenizing
    ``prompt_prefix + separator + label``, and that the label adds at least one
    token. Raises ``ValueError`` naming the offending label when the tokenizer
    breaks prefix stability (e.g. retokenizes the boundary once the label is
    appended), since that would make it impossible to reliably locate where the
    label's continuation begins for teacher-forced scoring.
    """
    prefix_ids = _tokenize_to_ids(tokenizer, prompt_prefix)
    results: dict[str, RoutingLabelTokenization] = {}
    for tool_name, label in routing_labels.items():
        full_ids = _tokenize_to_ids(tokenizer, prompt_prefix + separator + label)
        if full_ids[: len(prefix_ids)] != prefix_ids:
            msg = (
                f"Tokenizer is not prefix-stable for routing label {label!r} "
                f"({tool_name!r}): tokenizing prompt_prefix + separator + label does not "
                "keep the tokenized prompt_prefix as a strict prefix of the full token "
                "sequence. The prompt/label token boundary is invalid for this tokenizer."
            )
            raise ValueError(msg)
        added_tokens = full_ids[len(prefix_ids) :]
        if len(added_tokens) < 1:
            msg = f"Routing label {label!r} for {tool_name!r} must add at least one token."
            raise ValueError(msg)
        results[tool_name] = RoutingLabelTokenization(
            tool_name=tool_name,
            label=label,
            token_length=len(added_tokens),
        )
    return results


def _tokenize_to_ids(tokenizer: object, text: str) -> list[int]:
    """Return a plain list of token ids for one text, without special tokens."""
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
    return list(input_ids)


def stable_logsumexp(values: Sequence[float]) -> float:
    """Return a numerically stable log-sum-exp of one non-empty sequence of floats."""
    if not values:
        msg = "stable_logsumexp requires at least one value."
        raise ValueError(msg)
    max_value = max(values)
    if math.isinf(max_value) and max_value < 0:
        return max_value
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


@dataclass
class LexicalToolScorer:
    """Fast keyword baseline for target-tool support."""

    tool_keywords: dict[str, set[str]] = field(default_factory=lambda: TOOL_KEYWORDS)

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        """Score prompts using lightweight keyword evidence."""
        return [
            self.score_prompt(
                prompt,
                target_tool=target_tool,
                tool_descriptions=tool_descriptions,
            )
            for prompt in prompts
        ]

    def score_prompt(
        self,
        prompt: str,
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> float:
        """Score one prompt with lexical evidence for the target tool."""
        del tool_descriptions
        if not prompt.strip():
            return 0.0

        target_keywords = self.tool_keywords[target_tool]
        tokens = normalize_tokens(prompt)
        target_hits = len(tokens & target_keywords)
        explicit_tool_name = target_tool.lower() in prompt.lower()

        competing_hits = 0.0
        for tool, keywords in self.tool_keywords.items():
            if tool == target_tool:
                continue
            competing_hits += len(tokens & keywords) * 0.35

        raw_score = 0.85 * target_hits + (1.25 if explicit_tool_name else 0.0)
        raw_score -= competing_hits
        return float(1 / (1 + math.exp(-(raw_score - 1.3))))


@dataclass
class LexicalToolRouter:
    """Small local router used when no real LLM backend is loaded."""

    scorer: LexicalToolScorer = field(default_factory=LexicalToolScorer)

    def choose_tool(self, prompt: str, tool_descriptions: dict[str, str]) -> ToolChoice:
        """Choose the most supported tool for a user prompt."""
        scores = {
            tool_name: self.scorer.score_prompt(
                prompt,
                target_tool=tool_name,
                tool_descriptions=tool_descriptions,
            )
            for tool_name in tool_descriptions
        }
        selected_tool = max(scores, key=scores.get)
        return ToolChoice(
            tool=selected_tool,
            score=scores[selected_tool],
            reason=self._build_reason(prompt, selected_tool),
            scores=scores,
        )

    def _build_reason(self, prompt: str, selected_tool: str) -> str:
        tokens = normalize_tokens(prompt)
        hits = sorted(tokens & self.scorer.tool_keywords[selected_tool])
        if selected_tool == "weather_tool":
            purpose = "the question asks about weather, forecast, place, or time."
        elif selected_tool == "calculator_tool":
            purpose = "the question asks for exact arithmetic or a numeric result."
        elif selected_tool == "web_search_tool":
            purpose = "the question depends on current, recent, latest, or external facts."
        else:
            purpose = "the question can be answered directly from stable knowledge."

        if hits:
            return f"Matched {', '.join(hits[:5])}; {purpose}"
        return purpose


@dataclass
class MockLLM:
    """Fake LLM for tests and local wiring."""

    response: str | None = None

    def generate(self, prompt: str) -> str:
        """Return a fixed response or a deterministic prompt-aware score."""
        if self.response is not None:
            return self.response
        target_tool = self._extract_block(prompt, "Target tool:", "Available tools:").strip()
        coalition_prompt = self._extract_block(prompt, "Prompt:", "Return only one number")
        return f"{self._score_prompt(coalition_prompt, target_tool):.3f}"

    def _score_prompt(self, prompt: str, target_tool: str) -> float:
        if not prompt.strip():
            return 0.0
        target_keywords = TOOL_KEYWORDS[target_tool]
        tokens = normalize_tokens(prompt)
        target_hits = len(tokens & target_keywords)
        explicit_tool_name = target_tool.lower() in prompt.lower()
        raw_score = 0.95 * target_hits + (1.5 if explicit_tool_name else 0.0)
        return float(1 / (1 + math.exp(-(raw_score - 1.1))))

    @staticmethod
    def _extract_block(text: str, start_marker: str, end_marker: str) -> str:
        _, _, after_start = text.partition(start_marker)
        block, _, _ = after_start.partition(end_marker)
        return block.strip()


@dataclass
class LLMToolScorer:
    """Experimental LLM-as-a-judge scorer for target-tool support."""

    llm: TextGeneratorProtocol
    fallback_scorer: ToolScorerProtocol | None = None
    last_debug_outputs: list[dict[str, object]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.fallback_scorer is None:
            self.fallback_scorer = LexicalToolScorer()

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        """Score prompts with the LLM, falling back per prompt when needed."""
        self.last_debug_outputs = []
        scores = []
        for prompt in prompts:
            scoring_prompt = self.build_scoring_prompt(
                prompt,
                target_tool=target_tool,
                tool_descriptions=tool_descriptions,
            )
            raw_output = None
            parsed_score = None
            fallback_score = None
            used_fallback = False
            try:
                raw_output = self.llm.generate(scoring_prompt)
                parsed_score = self.parse_score(raw_output)
                final_score = parsed_score
            except (RuntimeError, TypeError, ValueError):
                used_fallback = True
                fallback_score = self._fallback_score(
                    prompt,
                    target_tool=target_tool,
                    tool_descriptions=tool_descriptions,
                )
                final_score = fallback_score
            scores.append(final_score)
            self.last_debug_outputs.append(
                {
                    "target_tool": target_tool,
                    "raw_output": raw_output,
                    "parsed_score": parsed_score,
                    "used_fallback": used_fallback,
                    "fallback_score": fallback_score,
                    "final_score": final_score,
                }
            )
        return scores

    def build_scoring_prompt(
        self,
        prompt: str,
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> str:
        """Build the LLM-as-a-judge prompt for one coalition prompt."""
        tool_lines = "\n".join(
            f"- {tool_name}: {description}" for tool_name, description in tool_descriptions.items()
        )
        return (
            "You are evaluating whether an assistant should call a specific tool.\n\n"
            f"Target tool:\n{target_tool}\n\n"
            f"Available tools:\n{tool_lines}\n\n"
            f"Prompt:\n{prompt}\n\n"
            "Return only one number between 0 and 1."
        )

    def parse_score(self, output: str) -> float:
        """Parse and validate one LLM score."""
        match = re.search(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", output)
        if match is None:
            msg = "LLM output did not contain a numeric score."
            raise ValueError(msg)
        score = float(match.group(0))
        if not math.isfinite(score):
            msg = "LLM score must be finite."
            raise ValueError(msg)
        if not 0.0 <= score <= 1.0:
            msg = "LLM score must be between 0 and 1."
            raise ValueError(msg)
        return score

    def _fallback_score(
        self,
        prompt: str,
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> float:
        """Return a safe fallback score for one prompt."""
        if self.fallback_scorer is None:
            return 0.0
        scores = self.fallback_scorer.score_batch(
            [prompt],
            target_tool=target_tool,
            tool_descriptions=tool_descriptions,
        )
        if len(scores) != 1:
            msg = "Fallback scorer must return one score per prompt."
            raise ValueError(msg)
        return clamp_score(float(scores[0]))


class CalibratedToolLogOddsScorer:
    """Score tool decisions with calibrated multiclass target-vs-all log-odds.

    For each coalition prompt and each candidate decision ``t``, this scorer
    teacher-forces the constrained routing-label continuation (see
    :func:`build_routing_classification_prompt`) and computes:

    ``g_t(S) = log p_theta(label_t | coalition_prompt(S))``
    ``b_t    = log p_theta(label_t | calibration_prompt)``
    ``r_t(S) = g_t(S) - b_t``

    The returned value is the calibrated target-vs-all multiclass log-odds,
    normalized against the true empty coalition:

    ``h(S; t*) = r_t*(S) - logsumexp({r_u(S) : u in candidates, u != t*})``
    ``V(S; t*) = h(S; t*) - h(empty_coalition; t*)``

    ``no_tool`` is one alternative among the candidates, not a fixed reference:
    unlike a target-vs-``no_tool`` contrast, ``V`` cannot mistake a
    non-``no_tool`` candidate outscoring ``no_tool`` for genuine support when a
    third candidate is actually strongest.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_LOGPROB_MODEL_ID,
        device: str | None = None,
        dtype: str = "auto",
        *,  # future-proof: force all following args to be keyword-only
        routing_labels: Mapping[str, str] = ROUTING_LABELS,
        routing_label_separator: str = ROUTING_LABEL_SEPARATOR,
        calibration_user_requests: tuple[str, ...] = CALIBRATION_USER_REQUESTS,
        max_pairs_per_batch: int | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.routing_labels = dict(routing_labels)
        self.routing_label_separator = routing_label_separator
        self.calibration_user_requests = tuple(calibration_user_requests)
        self.last_debug_outputs: list[dict[str, object]] = []
        self.tokenizer_label_diagnostics: dict[str, RoutingLabelTokenization] | None = None
        # Keyed by the full protocol identity (see _protocol_key): model id,
        # system prompt, tool descriptions, routing-label mapping (order
        # matters), separator, calibration probe text, and candidate-tool set.
        # Calibration never depends on the varying per-coalition user request,
        # so one calibration probe covers every coalition scored with this
        # exact configuration -- but a changed tool description, system
        # prompt, or label order is a different configuration and must miss.
        self._calibration_cache: dict[tuple[object, ...], dict[str, float]] = {}
        # Keyed by (protocol_key, target_tool) -- the empty coalition is scored
        # at most once per configuration and target tool, and cached, rather
        # than being re-evaluated by the model for every coalition batch.
        self._empty_log_odds_cache: dict[tuple[object, ...], float] = {}
        # Keyed by the exact routing-prompt text -- dedupes real model calls
        # whenever the identical (system_prompt, tool_descriptions, user_request)
        # combination is scored more than once, e.g. when the literal empty
        # coalition is scored both as an ordinary coalition and, separately, as
        # the empty-coalition normalization baseline.
        self._raw_score_row_cache: dict[str, dict[str, float]] = {}
        if max_pairs_per_batch is not None and max_pairs_per_batch < 1:
            msg = "max_pairs_per_batch must be positive or None."
            raise ValueError(msg)

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        if self.device is None or self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        if max_pairs_per_batch is None:
            max_pairs_per_batch = 1 if self.device in {"cuda", "mps"} else 4
        self.max_pairs_per_batch = max_pairs_per_batch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # ``dtype=`` (not the deprecated ``torch_dtype=``) matches both the
        # current transformers API and the verified-working load pattern;
        # ``low_cpu_mem_usage=True`` avoids materializing a full fp32 copy on
        # CPU before casting/moving to the accelerator. Half precision is used
        # on both CUDA and MPS (not just CUDA) since MPS is a first-class
        # accelerated backend here, not a CPU fallback.
        model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
        if dtype != "auto":
            model_kwargs["dtype"] = getattr(torch, dtype)
        elif self.device in {"cuda", "mps"}:
            model_kwargs["dtype"] = torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        self.model.to(self.device)
        self.model.eval()

        self._validate_tokenizer_labels()

    def _validate_tokenizer_labels(self) -> None:
        """Validate routing-label tokenization once and warn if labels are multi-token."""
        diagnostics_by_probe = []
        for calibration_user_request in self.calibration_user_requests:
            probe_prompt = build_routing_classification_prompt(
                system_prompt="",
                user_request=calibration_user_request,
                tool_descriptions={},
                routing_labels=self.routing_labels,
            )
            diagnostics_by_probe.append(
                validate_routing_label_tokens(
                    self.tokenizer,
                    probe_prompt,
                    routing_labels=self.routing_labels,
                    separator=self.routing_label_separator,
                )
            )
        diagnostics = {
            tool_name: max(
                (probe_diagnostics[tool_name] for probe_diagnostics in diagnostics_by_probe),
                key=lambda diagnostic: diagnostic.token_length,
            )
            for tool_name in self.routing_labels
        }
        self.tokenizer_label_diagnostics = diagnostics
        multi_token_labels = {
            tool_name: diag.token_length
            for tool_name, diag in diagnostics.items()
            if diag.token_length != 1
        }
        if multi_token_labels:
            warnings.warn(
                "One or more routing labels are multi-token under this tokenizer: "
                f"{multi_token_labels!r}. Falling back to full teacher-forced sequence "
                "log-likelihood scoring for those labels (no length averaging).",
                stacklevel=2,
            )

    def _candidate_continuation(self, tool_name: str) -> str:
        """Return the teacher-forced continuation used to score one candidate tool."""
        return f"{self.routing_label_separator}{self.routing_labels[tool_name]}"

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        """Return the calibrated target-vs-all log-odds ``V(S; target_tool)`` per prompt."""
        candidate_tools = self._validate_candidate_tools(target_tool, tool_descriptions)
        self.last_debug_outputs = []

        system_prompts = [split_coalition_prompt(prompt)[0] for prompt in prompts]
        routing_prompts = [
            self._routing_prompt(
                prompt, system_prompt=system_prompt, tool_descriptions=tool_descriptions
            )
            for prompt, system_prompt in zip(prompts, system_prompts, strict=True)
        ]
        raw_score_rows = self._label_log_scores_batched(routing_prompts, candidate_tools)
        if len(raw_score_rows) != len(prompts):
            msg = "Candidate scoring must return one score dictionary per prompt."
            raise ValueError(msg)

        target_label = self.routing_labels[target_tool]
        candidate_labels = [self.routing_labels[tool_name] for tool_name in candidate_tools]

        scores: list[float] = []
        for prompt, system_prompt, routing_prompt, raw_scores in zip(
            prompts, system_prompts, routing_prompts, raw_score_rows, strict=True
        ):
            protocol_key = self._protocol_key(system_prompt, tool_descriptions, candidate_tools)
            calibration_scores = self._get_calibration_scores(
                protocol_key, system_prompt, tool_descriptions, candidate_tools
            )
            calibrated_scores = {
                tool_name: self._validate_log_score(
                    tool_name, raw_scores[tool_name] - calibration_scores[tool_name]
                )
                for tool_name in candidate_tools
            }
            target_vs_all_log_odds = self._target_vs_all_log_odds(
                calibrated_scores, target_tool=target_tool, candidate_tools=candidate_tools
            )
            empty_coalition_log_odds = self._get_empty_coalition_log_odds(
                protocol_key, system_prompt, tool_descriptions, candidate_tools, target_tool
            )
            final_game_value = target_vs_all_log_odds - empty_coalition_log_odds
            if not math.isfinite(final_game_value):
                msg = "Final game value must be finite."
                raise ValueError(msg)
            calibrated_probabilities = _softmax_probabilities(calibrated_scores, candidate_tools)

            scores.append(final_game_value)
            self.last_debug_outputs.append(
                {
                    "target_tool": target_tool,
                    "target_label": target_label,
                    "candidate_tools": list(candidate_tools),
                    "candidate_labels": candidate_labels,
                    "raw_log_scores": dict(raw_scores),
                    "calibration_log_scores": dict(calibration_scores),
                    "calibrated_scores": dict(calibrated_scores),
                    "target_vs_all_log_odds": target_vs_all_log_odds,
                    "empty_coalition_log_odds": empty_coalition_log_odds,
                    "final_game_value": final_game_value,
                    "calibrated_probabilities": calibrated_probabilities,
                    "argmax_tool": max(raw_scores, key=raw_scores.get),
                    "final_score": final_game_value,
                    "prompt_preview": prompt[:240],
                    "routing_prompt_preview": routing_prompt[:240],
                }
            )
        return scores

    def _validate_candidate_tools(
        self,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[str]:
        """Return available decision candidates after validating required tools."""
        candidate_tools = list(tool_descriptions)
        if len(candidate_tools) < 2:
            msg = "CalibratedToolLogOddsScorer requires at least two decision candidates."
            raise ValueError(msg)
        if target_tool not in candidate_tools:
            msg = f"Target tool {target_tool!r} is not available."
            raise ValueError(msg)
        missing_labels = [
            tool_name for tool_name in candidate_tools if tool_name not in self.routing_labels
        ]
        if missing_labels:
            msg = f"No routing label configured for candidate tool(s): {missing_labels!r}."
            raise ValueError(msg)
        return candidate_tools

    def _routing_prompt(
        self,
        prompt: str,
        *,
        tool_descriptions: Mapping[str, str],
        system_prompt: str | None = None,
    ) -> str:
        """Build the canonical routing-classification prompt for one coalition prompt."""
        extracted_system_prompt, user_request = split_coalition_prompt(prompt)
        return build_routing_classification_prompt(
            system_prompt=system_prompt if system_prompt is not None else extracted_system_prompt,
            user_request=user_request,
            tool_descriptions=tool_descriptions,
            routing_labels=self.routing_labels,
        )

    def build_scoring_prompt(
        self,
        prompt: str,
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> str:
        """Return the model-formatted routing prompt used for log-probability scoring."""
        del target_tool
        return self._routing_prompt(prompt, tool_descriptions=tool_descriptions)

    def score_full_request_labels(
        self,
        user_request: str,
        *,
        system_prompt: str,
        tool_descriptions: Mapping[str, str],
    ) -> dict[str, float]:
        """Return raw (uncalibrated) per-candidate routing-label log scores for one request.

        Exposes the same teacher-forced label scoring used by :meth:`score_batch`
        under the identical canonical routing protocol, for callers (e.g.
        ``hf_router.LocalHFClassificationRouter``) that need the model's raw
        preference for a full (unmasked) request -- not a Shapley coalition
        value -- without calibration or empty-coalition normalization.
        """
        candidate_tools = list(tool_descriptions)
        routing_prompt = build_routing_classification_prompt(
            system_prompt=system_prompt,
            user_request=user_request,
            tool_descriptions=tool_descriptions,
            routing_labels=self.routing_labels,
        )
        return self._label_log_scores_batched([routing_prompt], candidate_tools)[0]

    def score_full_request_calibrated_labels(
        self,
        user_request: str,
        *,
        system_prompt: str,
        tool_descriptions: Mapping[str, str],
    ) -> dict[str, float]:
        """Return calibrated routing evidence ``r_t(N)`` for one full request.

        ``r_t(N) = g_t(N) - b_t``: the same per-label calibration baseline
        subtraction used inside :meth:`score_batch` for every coalition, so
        ``argmax_t`` of this dict is the calibrated full-request routing
        decision consistent with the coalition value function -- not the raw,
        uncalibrated ``argmax_t g_t(N)`` that :meth:`score_full_request_labels`
        returns. This does not apply target-vs-all or empty-coalition
        normalization (those are payoff-specific and require a fixed target
        tool); it is exactly the ``r_t`` evidence used to *choose* that target
        tool in the first place. Uses the same calibration cache as
        :meth:`score_batch`, so calling this before scoring coalitions for the
        same configuration does not trigger a second calibration probe.
        """
        candidate_tools = list(tool_descriptions)
        raw_scores = self.score_full_request_labels(
            user_request,
            system_prompt=system_prompt,
            tool_descriptions=tool_descriptions,
        )
        protocol_key = self._protocol_key(system_prompt, tool_descriptions, candidate_tools)
        calibration_scores = self._get_calibration_scores(
            protocol_key, system_prompt, tool_descriptions, candidate_tools
        )
        return {
            tool_name: raw_scores[tool_name] - calibration_scores[tool_name]
            for tool_name in candidate_tools
        }

    def _protocol_key(
        self,
        system_prompt: str,
        tool_descriptions: Mapping[str, str],
        candidate_tools: list[str],
    ) -> tuple[object, ...]:
        """Return the full protocol-identity key used for calibration/empty-coalition caching.

        Includes the model id, system prompt, tool descriptions, routing-label
        mapping (order-sensitive), separator, calibration probe text, and
        candidate-tool set, so a change to any of these -- e.g. a tool
        description edit, or a different routing-label order -- is a cache
        miss rather than silently reusing a stale calibration/baseline.
        """
        return (
            self.model_id,
            system_prompt,
            tuple(tool_descriptions.items()),
            tuple(self.routing_labels.items()),
            self.routing_label_separator,
            tuple(self.calibration_user_requests),
            tuple(candidate_tools),
        )

    def _get_calibration_scores(
        self,
        protocol_key: tuple[object, ...],
        system_prompt: str,
        tool_descriptions: Mapping[str, str],
        candidate_tools: list[str],
    ) -> dict[str, float]:
        """Return the cached (or freshly computed) calibration label scores ``b_t``."""
        if protocol_key in self._calibration_cache:
            return self._calibration_cache[protocol_key]
        calibration_prompts = [
            build_routing_classification_prompt(
                system_prompt=system_prompt,
                user_request=calibration_user_request,
                tool_descriptions=tool_descriptions,
                routing_labels=self.routing_labels,
            )
            for calibration_user_request in self.calibration_user_requests
        ]
        calibration_score_rows = self._label_log_scores_batched(
            calibration_prompts, candidate_tools
        )
        calibration_scores = {
            tool_name: sum(row[tool_name] for row in calibration_score_rows)
            / len(calibration_score_rows)
            for tool_name in candidate_tools
        }
        self._calibration_cache[protocol_key] = calibration_scores
        return calibration_scores

    def _get_empty_coalition_log_odds(
        self,
        protocol_key: tuple[object, ...],
        system_prompt: str,
        tool_descriptions: Mapping[str, str],
        candidate_tools: list[str],
        target_tool: str,
    ) -> float:
        """Return the cached (or freshly computed) ``h(empty_coalition; target_tool)``."""
        cache_key = (protocol_key, target_tool)
        if cache_key in self._empty_log_odds_cache:
            return self._empty_log_odds_cache[cache_key]
        empty_routing_prompt = build_routing_classification_prompt(
            system_prompt=system_prompt,
            user_request="",
            tool_descriptions=tool_descriptions,
            routing_labels=self.routing_labels,
        )
        raw_scores = self._label_log_scores_batched([empty_routing_prompt], candidate_tools)[0]
        calibration_scores = self._get_calibration_scores(
            protocol_key, system_prompt, tool_descriptions, candidate_tools
        )
        calibrated_scores = {
            tool_name: raw_scores[tool_name] - calibration_scores[tool_name]
            for tool_name in candidate_tools
        }
        value = self._target_vs_all_log_odds(
            calibrated_scores, target_tool=target_tool, candidate_tools=candidate_tools
        )
        self._empty_log_odds_cache[cache_key] = value
        return value

    @staticmethod
    def _target_vs_all_log_odds(
        calibrated_scores: dict[str, float],
        *,
        target_tool: str,
        candidate_tools: list[str],
    ) -> float:
        """Return ``r_target - logsumexp({r_u : u != target})`` for one coalition."""
        alternatives = [
            calibrated_scores[tool_name]
            for tool_name in candidate_tools
            if tool_name != target_tool
        ]
        if not alternatives:
            msg = "Target-vs-all log-odds requires at least one alternative candidate tool."
            raise ValueError(msg)
        return calibrated_scores[target_tool] - stable_logsumexp(alternatives)

    def _label_log_scores_batched(
        self,
        prompts: list[str],
        candidate_tools: list[str],
    ) -> list[dict[str, float]]:
        """Return raw per-candidate label scores per prompt, reusing cached rows.

        Caches by the exact routing-prompt text, so identical prompts requested
        from different call sites (e.g. the same literal empty-coalition prompt
        scored once as an ordinary coalition and again for empty-coalition
        normalization) trigger at most one real model call, not one per call site.
        """
        rows: list[dict[str, float]] = [{} for _ in prompts]
        to_compute_prompts: list[str] = []
        to_compute_indices: list[int] = []
        for index, prompt in enumerate(prompts):
            cached_row = self._raw_score_row_cache.get(prompt)
            if cached_row is not None and set(candidate_tools) <= set(cached_row):
                rows[index] = {tool_name: cached_row[tool_name] for tool_name in candidate_tools}
            else:
                to_compute_prompts.append(prompt)
                to_compute_indices.append(index)

        if to_compute_prompts:
            computed_rows = self._compute_label_log_scores_batched(
                to_compute_prompts, candidate_tools
            )
            if len(computed_rows) != len(to_compute_prompts):
                msg = "Candidate scoring must return one score dictionary per prompt."
                raise ValueError(msg)
            for index, prompt, computed_row in zip(
                to_compute_indices, to_compute_prompts, computed_rows, strict=True
            ):
                rows[index] = computed_row
                self._raw_score_row_cache.setdefault(prompt, {}).update(computed_row)

        return rows

    def _compute_label_log_scores_batched(
        self,
        prompts: list[str],
        candidate_tools: list[str],
    ) -> list[dict[str, float]]:
        """Score all prompt/label continuations in batched teacher-forced model calls."""
        pair_prompts = []
        pair_continuations = []
        pair_tools = []
        for prompt in prompts:
            for tool_name in candidate_tools:
                pair_prompts.append(prompt)
                pair_continuations.append(self._candidate_continuation(tool_name))
                pair_tools.append(tool_name)

        pair_scores = self._sequence_logprobs_batched(pair_prompts, pair_continuations)
        if len(pair_scores) != len(pair_tools):
            msg = "Candidate scoring must return one log score per prompt/candidate pair."
            raise ValueError(msg)

        rows = [dict.fromkeys(candidate_tools, math.nan) for _ in prompts]
        for pair_index, (tool_name, score) in enumerate(zip(pair_tools, pair_scores, strict=True)):
            prompt_index = pair_index // len(candidate_tools)
            rows[prompt_index][tool_name] = self._validate_log_score(tool_name, score)
        return rows

    def _validate_log_score(self, tool_name: str, score: float) -> float:
        """Return a finite log score as float."""
        score = float(score)
        if not math.isfinite(score):
            msg = f"Candidate score for {tool_name!r} must be finite."
            raise ValueError(msg)
        return score

    @staticmethod
    def _next_token_log_probs(logits: object, token_ids: object) -> object:
        """Return log probabilities for the observed next-token labels only."""
        shifted_logits = logits[:, :-1, :]
        shifted_token_ids = token_ids[:, 1:]
        target_logits = shifted_logits.gather(
            dim=-1,
            index=shifted_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        return target_logits - shifted_logits.logsumexp(dim=-1)

    def _sequence_logprobs_batched(
        self,
        prompts: list[str],
        continuations: list[str],
    ) -> list[float]:
        """Score continuation likelihoods for prompt/candidate pairs in batches."""
        if len(prompts) != len(continuations):
            msg = "Prompts and continuations must have the same length."
            raise ValueError(msg)
        if not prompts:
            return []

        max_pairs_per_batch = getattr(self, "max_pairs_per_batch", None)
        if max_pairs_per_batch is None:
            return self._sequence_logprobs_batch(prompts, continuations)

        scores: list[float] = []
        for start in range(0, len(prompts), max_pairs_per_batch):
            stop = start + max_pairs_per_batch
            scores.extend(
                self._sequence_logprobs_batch(
                    prompts[start:stop],
                    continuations[start:stop],
                )
            )
            self._release_device_cache()
        return scores

    def _sequence_logprobs_batch(
        self,
        prompts: list[str],
        continuations: list[str],
    ) -> list[float]:
        """Score continuation token summed log-likelihood under a causal LM in one batch.

        Always sums per-token log-probabilities over the full continuation
        (never divides by continuation length), so multi-token routing labels
        are scored by their exact teacher-forced sequence log-likelihood.
        """
        torch = self._torch
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        full_inputs = self.tokenizer(
            [
                prompt + continuation
                for prompt, continuation in zip(prompts, continuations, strict=True)
            ],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=-1).tolist()
        full_lengths = full_inputs["attention_mask"].sum(dim=-1).tolist()
        for row_index, prompt in enumerate(prompts):
            prompt_only_ids = prompt_inputs["input_ids"][row_index][
                : int(prompt_inputs["attention_mask"][row_index].sum())
            ].tolist()
            full_ids = full_inputs["input_ids"][row_index][
                : int(full_inputs["attention_mask"][row_index].sum())
            ].tolist()
            prefix_len = min(len(prompt_only_ids), len(full_ids))
            if prompt_only_ids[:prefix_len] != full_ids[:prefix_len]:
                msg = (
                    f"Tokenizer boundary mismatch at row {row_index}: tokenizing "
                    "prompt_prefix alone is not a strict prefix of tokenizing "
                    "prompt_prefix + continuation, so the scored span cannot be "
                    f"located safely for prompt ending {prompt[-40:]!r}."
                )
                raise ValueError(msg)
        continuation_lengths = [
            int(full_len - prompt_len)
            for prompt_len, full_len in zip(prompt_lengths, full_lengths, strict=True)
        ]
        if any(length <= 0 for length in continuation_lengths):
            msg = "Continuation must add at least one token."
            raise ValueError(msg)

        input_ids = full_inputs["input_ids"].to(self.device)
        attention_mask = full_inputs["attention_mask"].to(self.device)
        self.model.eval()
        try:
            with torch.inference_mode():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                # Compute logprobs in float32 even when the model runs in fp16, so
                # the log-sum-exp used by the routing payoff stays numerically stable.
                logits = logits.to(torch.float32)
                token_log_probs = self._next_token_log_probs(logits, input_ids)

                scores = []
                for row_index, (prompt_len, continuation_len) in enumerate(
                    zip(prompt_lengths, continuation_lengths, strict=True)
                ):
                    start = int(prompt_len - 1)
                    stop = start + int(continuation_len)
                    score = float(token_log_probs[row_index, start:stop].sum().item())
                    scores.append(score)
                return scores
        finally:
            del input_ids, attention_mask, full_inputs, prompt_inputs
            if "logits" in locals():
                del logits
            if "token_log_probs" in locals():
                del token_log_probs
            self._release_device_cache()

    def _release_device_cache(self) -> None:
        """Release cached accelerator memory between HF micro-batches."""
        torch = self._torch
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()


@dataclass(frozen=True)
class FrozenContinuationTarget:
    """A frozen teacher-forced continuation target shared by both scorer families.

    ``kind`` distinguishes a tool-identity continuation (truncated at the tool
    name, varies with the chosen executable target tool) from a direct-answer
    continuation (a fixed deterministic answer fragment, identical for every
    coalition). ``token_ids`` is the tokenizer identity of ``text`` computed
    once, so callers can use it as a stable cache/identity key without
    retokenizing on every coalition.
    """

    kind: Literal["tool_identity", "direct_answer"]
    text: str
    token_ids: tuple[int, ...] | None
    source: str


class _NativeContinuationScorer:
    """Shared teacher-forced continuation scoring machinery.

    Owns everything that is expensive and identical between scoring a
    tool-identity continuation and scoring a direct-answer continuation:
    tokenizer/model loading, batched forward passes, next-token
    log-probability extraction, continuation-mean scoring, device-cache
    cleanup, and finite-score validation. Subclasses
    (:class:`NativeToolCallScorer`, :class:`NativeDirectAnswerScorer`) only
    decide which continuation text is scored and how debug metadata for it is
    described -- neither one reimplements the model-scoring loop.
    """

    def __init__(
        self,
        *,
        model: object | None,
        tokenizer: object | None,
        device: str | None,
        model_id: str,
        tokenizer_id: str | None,
        quantization_mode: str,
        actual_quantization_mode: str | None,
        dtype: str,
        max_pairs_per_batch: int | None,
    ) -> None:
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id or model_id
        self.requested_quantization_mode = quantization_mode
        self.actual_quantization_mode = actual_quantization_mode or quantization_mode
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype

        import torch

        self._torch = torch
        if quantization_mode != "none":
            msg = (
                f"{type(self).__name__} does not currently support quantization mode "
                f"{quantization_mode!r}. Use 'none' so teacher-forced logits are computed "
                "on the same supported model configuration as inference."
            )
            raise RuntimeError(msg)
        if self.model is None or self.tokenizer is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if self.device is None or self.device == "auto":
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            model_kwargs: dict[str, Any] = {"low_cpu_mem_usage": True}
            if dtype != "auto":
                model_kwargs["dtype"] = getattr(torch, dtype)
            elif self.device in {"cuda", "mps"}:
                model_kwargs["dtype"] = torch.float16
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_id, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
            self.model.to(self.device)
        elif self.device is None:
            self.device = str(next(self.model.parameters()).device)

        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.model.eval()
        if max_pairs_per_batch is None:
            max_pairs_per_batch = 1 if self.device in {"cuda", "mps"} else 4
        if max_pairs_per_batch < 1:
            msg = "max_pairs_per_batch must be positive."
            raise ValueError(msg)
        self.max_pairs_per_batch = max_pairs_per_batch

    def _continuation_token_count(self, prompt: str, continuation: str) -> int | None:
        try:
            prompt_ids = _tokenize_to_ids(self.tokenizer, prompt)
            full_ids = _tokenize_to_ids(self.tokenizer, prompt + continuation)
        except (AttributeError, TypeError):
            return None
        return max(0, len(full_ids) - len(prompt_ids))

    def _sequence_mean_logprobs_batched(
        self,
        prompts: list[str],
        continuations: list[str],
    ) -> list[float]:
        if len(prompts) != len(continuations):
            msg = "Prompts and continuations must have the same length."
            raise ValueError(msg)
        scores: list[float] = []
        for start in range(0, len(prompts), self.max_pairs_per_batch):
            stop = start + self.max_pairs_per_batch
            scores.extend(
                self._sequence_mean_logprobs_batch(
                    prompts[start:stop],
                    continuations[start:stop],
                )
            )
            self._release_device_cache()
        return scores

    def _sequence_mean_logprobs_batch(
        self,
        prompts: list[str],
        continuations: list[str],
    ) -> list[float]:
        torch = self._torch
        prompt_inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        full_inputs = self.tokenizer(
            [
                prompt + continuation
                for prompt, continuation in zip(prompts, continuations, strict=True)
            ],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        prompt_lengths = prompt_inputs["attention_mask"].sum(dim=-1).tolist()
        full_lengths = full_inputs["attention_mask"].sum(dim=-1).tolist()
        # Continuation length is derived from the robust full-sequence boundary
        # (attention-mask length delta), not from independently
        # tokenize(prompt) + tokenize(continuation) token counts, which are not
        # always identical to tokenize(prompt + continuation) once tokenizer
        # boundary merging is taken into account. A non-positive delta here
        # means the boundary could not be located safely, and is treated as a
        # hard failure rather than silently scoring the wrong span.
        continuation_lengths = [
            int(full_len - prompt_len)
            for prompt_len, full_len in zip(prompt_lengths, full_lengths, strict=True)
        ]
        if any(length <= 0 for length in continuation_lengths):
            msg = "Native target continuation must add at least one token."
            raise ValueError(msg)

        input_ids = full_inputs["input_ids"].to(self.device)
        attention_mask = full_inputs["attention_mask"].to(self.device)
        try:
            with torch.inference_mode():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
                logits = logits.to(torch.float32)
                token_log_probs = self._next_token_log_probs(
                    logits,
                    input_ids,
                )
                scores = []
                for row_index, (prompt_len, continuation_len) in enumerate(
                    zip(prompt_lengths, continuation_lengths, strict=True)
                ):
                    start = int(prompt_len - 1)
                    stop = start + int(continuation_len)
                    score = float(token_log_probs[row_index, start:stop].mean().item())
                    if not math.isfinite(score):
                        msg = "Native continuation score must be finite."
                        raise ValueError(msg)
                    scores.append(score)
                return scores
        finally:
            del input_ids, attention_mask, full_inputs, prompt_inputs
            if "logits" in locals():
                del logits
            if "token_log_probs" in locals():
                del token_log_probs
            self._release_device_cache()

    def _release_device_cache(self) -> None:
        torch = self._torch
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif self.device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    @staticmethod
    def _next_token_log_probs(logits: object, token_ids: object) -> object:
        """Return log probabilities for observed next-token labels."""
        shifted_logits = logits[:, :-1, :]
        shifted_token_ids = token_ids[:, 1:]
        target_logits = shifted_logits.gather(
            dim=-1,
            index=shifted_token_ids.unsqueeze(-1),
        ).squeeze(-1)
        return target_logits - shifted_logits.logsumexp(dim=-1)


class NativeToolCallScorer(_NativeContinuationScorer):
    """Score canonical native tool-identity continuation support for one frozen tool.

    The model input always uses the tokenizer's structured tool template. For
    executable tools, full-context native inference first freezes the target
    tool identity. The scorer then constructs a canonical native-format
    assistant continuation with that same tokenizer/template and keeps only the
    prefix through the target tool name, intentionally excluding free-form
    arguments. ``no_tool`` is intentionally rejected here; direct-answer cases
    use :class:`NativeDirectAnswerScorer` instead. Scores are mean
    continuation-token log-probabilities, so tool names with different token
    lengths are less distorted by sequence length.
    """

    def __init__(
        self,
        *,
        model: object | None = None,
        tokenizer: object | None = None,
        device: str | None = None,
        model_id: str = DEFAULT_LOGPROB_MODEL_ID,
        tokenizer_id: str | None = None,
        quantization_mode: str = "none",
        actual_quantization_mode: str | None = None,
        dtype: str = "auto",
        max_pairs_per_batch: int | None = None,
        tool_schemas: Sequence[Mapping[str, object]] = EXECUTABLE_TOOL_SCHEMAS,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            quantization_mode=quantization_mode,
            actual_quantization_mode=actual_quantization_mode,
            dtype=dtype,
            max_pairs_per_batch=max_pairs_per_batch,
        )
        self.tool_schemas = tuple(_copy_schema(schema) for schema in tool_schemas)
        self.last_debug_outputs: list[dict[str, object]] = []
        self._score_cache: dict[tuple[str, str], float] = {}

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        """Return canonical native tool-identity support for each coalition prompt."""
        if target_tool == NO_TOOL_NAME:
            msg = "NativeToolCallScorer is valid only for executable tools, not no_tool."
            raise ValueError(msg)
        if target_tool not in tool_descriptions:
            msg = f"Target tool {target_tool!r} is not available."
            raise ValueError(msg)

        self.last_debug_outputs = []
        model_prompts = []
        continuations = []
        for prompt in prompts:
            system_prompt, user_request = split_coalition_prompt(prompt)
            model_prompt = build_native_tool_call_prompt(
                self.tokenizer,
                system_prompt=system_prompt,
                user_request=user_request,
                tool_schemas=self.tool_schemas,
            )
            continuation = self._target_continuation(
                system_prompt=system_prompt,
                user_request=user_request,
                target_tool=target_tool,
            )
            model_prompts.append(model_prompt)
            continuations.append(continuation)

        continuation_token_counts = [
            self._continuation_token_count(model_prompt, continuation)
            for model_prompt, continuation in zip(model_prompts, continuations, strict=True)
        ]
        scores = self._sequence_mean_logprobs_batched(model_prompts, continuations)
        for prompt, continuation, token_count, score in zip(
            prompts,
            continuations,
            continuation_token_counts,
            scores,
            strict=True,
        ):
            self.last_debug_outputs.append(
                {
                    "score_kind": "native_tool_call_mean_logprob",
                    "score_description": (
                        "Mean log-probability of the canonical native tool-identity "
                        "continuation; free-form arguments are excluded."
                    ),
                    "model_id": self.model_id,
                    "tokenizer_id": self.tokenizer_id,
                    "requested_quantization": self.requested_quantization_mode,
                    "actual_quantization": self.actual_quantization_mode,
                    "device": self.device,
                    "dtype": self.dtype,
                    "target_tool": target_tool,
                    "target_source": "full-context native inference",
                    "continuation_type": "canonical native template",
                    "continuation_scope": "tool identity only",
                    "arguments_included": "no",
                    "continuation_token_count": token_count,
                    "candidate_continuations": {target_tool: continuation},
                    "final_score": score,
                    "prompt_preview": prompt[:240],
                }
            )
        return scores

    def build_scoring_prompt(
        self,
        prompt: str,
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> str:
        """Return the native chat-template input used before the scored continuation."""
        del tool_descriptions
        system_prompt, user_request = split_coalition_prompt(prompt)
        continuation = self._target_continuation(
            system_prompt=system_prompt,
            user_request=user_request,
            target_tool=target_tool,
        )
        model_prompt = build_native_tool_call_prompt(
            self.tokenizer,
            system_prompt=system_prompt,
            user_request=user_request,
            tool_schemas=self.tool_schemas,
        )
        return f"{model_prompt}{continuation}"

    def _target_continuation(
        self,
        *,
        system_prompt: str,
        user_request: str,
        target_tool: str,
    ) -> str:
        return build_native_tool_call_continuation(
            self.tokenizer,
            system_prompt=system_prompt,
            user_request=user_request,
            target_tool=target_tool,
            tool_schemas=self.tool_schemas,
        )


class NativeDirectAnswerScorer(_NativeContinuationScorer):
    """Score canonical direct-answer continuation support for one frozen answer fragment.

    Scores the teacher-forced mean log-probability of a single frozen
    deterministic answer fragment (see :func:`build_canonical_direct_answer_target`),
    extracted once from the full-context HF-native direct answer and reused
    unchanged for every coalition -- never regenerated, never re-run through
    native routing, and never produced by calling ``generate()`` during
    coalition scoring. This explains support for that specific answer content;
    it is not a probability of choosing not to call a tool. Uses the identical
    teacher-forced sequence-scoring machinery as :class:`NativeToolCallScorer`
    (via :class:`_NativeContinuationScorer`), so both scorer families share
    tokenization, batching, and scoring semantics and differ only in which
    continuation text is frozen and scored.
    """

    def __init__(
        self,
        *,
        model: object | None = None,
        tokenizer: object | None = None,
        device: str | None = None,
        model_id: str = DEFAULT_LOGPROB_MODEL_ID,
        tokenizer_id: str | None = None,
        quantization_mode: str = "none",
        actual_quantization_mode: str | None = None,
        dtype: str = "auto",
        max_pairs_per_batch: int | None = None,
        direct_answer_target: str,
        tool_schemas: Sequence[Mapping[str, object]] = EXECUTABLE_TOOL_SCHEMAS,
    ) -> None:
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            device=device,
            model_id=model_id,
            tokenizer_id=tokenizer_id,
            quantization_mode=quantization_mode,
            actual_quantization_mode=actual_quantization_mode,
            dtype=dtype,
            max_pairs_per_batch=max_pairs_per_batch,
        )
        if not direct_answer_target or not direct_answer_target.strip():
            msg = "NativeDirectAnswerScorer requires a non-empty direct_answer_target."
            raise ValueError(msg)
        self.tool_schemas = tuple(_copy_schema(schema) for schema in tool_schemas)
        target_token_ids = tuple(_tokenize_to_ids(self.tokenizer, direct_answer_target))
        if not target_token_ids:
            msg = "Frozen direct-answer target tokenized to zero continuation tokens."
            raise ValueError(msg)
        # Frozen once, here, at construction -- score_batch below reuses
        # self.frozen_target.text unchanged for every coalition it is asked to
        # score, and never rebuilds it from coalition-specific text.
        self.frozen_target = FrozenContinuationTarget(
            kind="direct_answer",
            text=direct_answer_target,
            token_ids=target_token_ids,
            source="full-context native direct answer",
        )
        self.last_debug_outputs: list[dict[str, object]] = []
        # Keyed by (coalition prompt, frozen target token identity, model id,
        # tokenizer id) so repeated coalition prompts (e.g. the empty
        # coalition scored both as an ordinary coalition and as the
        # normalization baseline) reuse one real model call, and two
        # differently-constructed scorers (different frozen targets and/or
        # runtimes) can never collide on the same cache entry.
        self._score_cache: dict[tuple[object, ...], tuple[float, dict[str, object]]] = {}

    @property
    def direct_answer_target(self) -> str:
        """Return the frozen direct-answer continuation text."""
        return self.frozen_target.text

    def score_batch(
        self,
        prompts: list[str],
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> list[float]:
        """Return frozen direct-answer continuation support for each coalition prompt."""
        del tool_descriptions
        if target_tool != NO_TOOL_NAME:
            msg = (
                "NativeDirectAnswerScorer is valid only for target_tool='no_tool', "
                f"got {target_tool!r}."
            )
            raise ValueError(msg)

        cache_key_suffix = (self.frozen_target.token_ids, self.model_id, self.tokenizer_id)
        results: list[float | None] = [None] * len(prompts)
        debug_entries: list[dict[str, object] | None] = [None] * len(prompts)

        pending_prompts: list[str] = []
        pending_model_prompts: list[str] = []
        pending_indices: list[int] = []
        for index, prompt in enumerate(prompts):
            cache_key = (prompt, *cache_key_suffix)
            cached = self._score_cache.get(cache_key)
            if cached is not None:
                results[index], debug_entries[index] = cached
                continue
            system_prompt, user_request = split_coalition_prompt(prompt)
            model_prompt = build_native_direct_answer_prompt(
                self.tokenizer,
                system_prompt=system_prompt,
                user_request=user_request,
                tool_schemas=self.tool_schemas,
            )
            pending_prompts.append(prompt)
            pending_model_prompts.append(model_prompt)
            pending_indices.append(index)

        if pending_model_prompts:
            continuations = [self.frozen_target.text] * len(pending_model_prompts)
            token_counts = [
                self._continuation_token_count(model_prompt, continuation)
                for model_prompt, continuation in zip(
                    pending_model_prompts, continuations, strict=True
                )
            ]
            computed_scores = self._sequence_mean_logprobs_batched(
                pending_model_prompts, continuations
            )
            for original_prompt, index, model_prompt, token_count, score in zip(
                pending_prompts,
                pending_indices,
                pending_model_prompts,
                token_counts,
                computed_scores,
                strict=True,
            ):
                debug_entry: dict[str, object] = {
                    "score_kind": "native_direct_answer_mean_logprob",
                    "score_description": (
                        "Mean log-probability of the frozen canonical deterministic "
                        "direct-answer fragment; not a probability of choosing no tool."
                    ),
                    "model_id": self.model_id,
                    "tokenizer_id": self.tokenizer_id,
                    "requested_quantization": self.requested_quantization_mode,
                    "actual_quantization": self.actual_quantization_mode,
                    "device": self.device,
                    "dtype": self.dtype,
                    "target_tool": target_tool,
                    "target_source": "full-context native direct answer",
                    "continuation_type": "canonical deterministic answer fragment",
                    "continuation_scope": "bounded direct-answer text",
                    "target_regenerated_per_coalition": "no",
                    "continuation_token_count": token_count,
                    "frozen_target_text": self.frozen_target.text,
                    "final_score": score,
                    "prompt_preview": model_prompt[:240],
                }
                results[index] = score
                debug_entries[index] = debug_entry
                self._score_cache[(original_prompt, *cache_key_suffix)] = (score, debug_entry)

        self.last_debug_outputs = [entry for entry in debug_entries if entry is not None]
        final_scores: list[float] = []
        for score in results:
            if score is None:
                msg = "Internal error: direct-answer score not computed for a coalition prompt."
                raise RuntimeError(msg)
            final_scores.append(score)
        return final_scores

    def build_scoring_prompt(
        self,
        prompt: str,
        *,
        target_tool: str,
        tool_descriptions: dict[str, str],
    ) -> str:
        """Return the native chat-template input used before the frozen continuation."""
        del tool_descriptions
        if target_tool != NO_TOOL_NAME:
            msg = "NativeDirectAnswerScorer is valid only for target_tool='no_tool'."
            raise ValueError(msg)
        system_prompt, user_request = split_coalition_prompt(prompt)
        model_prompt = build_native_direct_answer_prompt(
            self.tokenizer,
            system_prompt=system_prompt,
            user_request=user_request,
            tool_schemas=self.tool_schemas,
        )
        return f"{model_prompt}{self.frozen_target.text}"


def _softmax_probabilities(
    calibrated_scores: Mapping[str, float],
    candidate_tools: Sequence[str],
) -> dict[str, float]:
    """Return diagnostic-only ``softmax(calibrated_scores)`` over all candidates.

    This is never used as the Shapley/k-SII game value -- it is exposed purely
    for UI display of a calibrated multiclass probability distribution.
    """
    values = [calibrated_scores[tool_name] for tool_name in candidate_tools]
    log_normalizer = stable_logsumexp(values)
    return {
        tool_name: math.exp(calibrated_scores[tool_name] - log_normalizer)
        for tool_name in candidate_tools
    }
