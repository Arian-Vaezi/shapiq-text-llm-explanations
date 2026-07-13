"""Agent-result and explanation-card rendering helpers."""

# ruff: noqa: F405, RUF059

from __future__ import annotations

# Mechanical re-export chain preserves the monolith's shared global namespace.
from .game import *  # noqa: F403

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


def render_tool_output_html(tool_output: object | None) -> str:
    """Render a non-empty tool result as escaped Agent Result content."""
    if tool_output is None:
        return ""
    output_text = str(tool_output).strip()
    if not output_text:
        return ""
    return (
        "<p class='result-meta-row' style='margin-top:0.7rem;'><strong>Tool output</strong></p>"
        f"<div class='agent-response-block agent-tool-output'>{escape(output_text)}</div>"
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
            {render_tool_output_html(tool_output)}
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


__all__ = [name for name in globals() if not name.startswith("__")]
