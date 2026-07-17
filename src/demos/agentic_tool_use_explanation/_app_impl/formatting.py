"""Text, prompt, and tabular formatting helpers."""

# ruff: noqa: F405, I001

from __future__ import annotations

# Mechanical re-export chain preserves the monolith's shared global namespace.
from .styles import *  # noqa: F403


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


__all__ = [name for name in globals() if not name.startswith("__")]
