"""Shapley and interaction computation helpers."""

# ruff: noqa: F405, I001, PLC0415

from __future__ import annotations

# Mechanical re-export chain preserves the monolith's shared global namespace.
from .formatting import *  # noqa: F403


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
    effective_max_order = min(max_order, game.n_players)
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
            max_order=effective_max_order,
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
    approximator = make_approximator(index, game.n_players, effective_max_order)
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
    if n_players != explanation.n_players:
        msg = (
            "n_players must match the explanation's player count: "
            f"{n_players} != {explanation.n_players}."
        )
        raise ValueError(msg)
    if n_players == 1:
        return pd.DataFrame([[float(explanation[(0,)])]])
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


__all__ = [name for name in globals() if not name.startswith("__")]
