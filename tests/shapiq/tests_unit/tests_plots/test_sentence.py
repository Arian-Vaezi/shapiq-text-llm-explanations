"""This test module contains all tests for the sentence plot."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.plot import (
    interaction_matrix_from_explanation,
    sentence_interaction_heatmap,
    sentence_plot,
    token_attribution_bar_plot,
)


def _text_values() -> tuple[list[str], InteractionValues]:
    words = ["I", "really", "enjoy", "working", "with", "Shapley", "values", "in", "Python", "!"]
    values = [0.45, 0.01, 0.67, -0.2, -0.05, 0.7, 0.1, -0.04, 0.56, 0.7]
    iv = InteractionValues(
        n_players=10,
        values=np.array(values),
        index="SV",
        min_order=1,
        max_order=1,
        estimated=False,
        baseline_value=0.0,
    )
    return words, iv


def test_sentence_plot():
    """Test the sentence plot function."""
    words, iv = _text_values()

    fig, ax = sentence_plot(iv, words, show=False)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

    fig, ax = iv.plot_sentence(
        words,
        show=False,
        connected_words=[("Shapley", "values")],
        max_score=0.5,  # max_score is intentionally lower than the attributions here
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)

    fig, ax = sentence_plot(
        iv,
        words,
        chars_per_line=100,
        show=False,
        connected_words=[("Shapley", "values")],
        max_score=1.0,
    )
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_token_attribution_bar_plot():
    """Test that the token attribution bar plot returns a figure and axis."""
    words, iv = _text_values()

    fig, ax = token_attribution_bar_plot(iv, words, show=False)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(ax.patches) == len(words)
    assert ax.get_title() == "Token attribution bar plot"

    plt.close(fig)


def test_token_attribution_bar_plot_word_count_mismatch():
    """Test that the token attribution bar plot raises if words and players do not match."""
    words, iv = _text_values()

    with pytest.raises(ValueError, match="Number of words must match number of players"):
        token_attribution_bar_plot(iv, words[:-1], show=False)


def test_token_attribution_bar_plot_show(monkeypatch):
    """Test the show=True path of the token attribution bar plot."""
    words, iv = _text_values()

    show_called = []

    def fake_show():
        show_called.append(True)

    monkeypatch.setattr(plt, "show", fake_show)

    result = token_attribution_bar_plot(iv, words, show=True)

    assert result is None
    assert show_called == [True]

    plt.close("all")


def test_sentence_interaction_heatmap_empty_figure():
    """Test that the sentence interaction heatmap returns a figure and axis."""

    words, iv = _text_values()

    fig, ax = sentence_interaction_heatmap(iv, words, show=False)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert len(ax.images) == 1  # draw heatmap images in achxis
    plt.close(fig)


def test_sentence_interaction_heatmap_word_count_mismatch():
    """Test that the heatmap raises if words and players do not match."""

    words, iv = _text_values()

    with pytest.raises(ValueError, match="Number of words must match number of players"):
        sentence_interaction_heatmap(iv, words[:-1], show=False)


def test_sentence_interaction_heatmap_zero_values_show(monkeypatch):
    """Test zero-valued heatmap and show=True path."""
    words, _ = _text_values()

    iv = InteractionValues(
        n_players=len(words),
        values=np.zeros(len(words)),
        index="SV",
        min_order=1,
        max_order=1,
        estimated=False,
        baseline_value=0.0,
    )

    show_called = []

    def fake_show():
        show_called.append(True)

    monkeypatch.setattr(plt, "show", fake_show)

    result = sentence_interaction_heatmap(iv, words, show=True)

    assert result is None
    assert show_called == [True]

    plt.close("all")


def test_token_attribution_bar_plot_zero_values():
    """Test zero-valued token attribution bar plot."""
    words, _ = _text_values()

    iv = InteractionValues(
        n_players=len(words),
        values=np.zeros(len(words)),
        index="SV",
        min_order=1,
        max_order=1,
        estimated=False,
        baseline_value=0.0,
    )

    fig, ax = token_attribution_bar_plot(iv, words, show=False)

    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    assert ax.get_xlim() == (-1.0, 1.0)

    plt.close(fig)


def _ksii_values() -> InteractionValues:
    """A small 3-player k-SII explanation with distinct singleton and pairwise values."""
    return InteractionValues(
        n_players=3,
        values=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        index="k-SII",
        min_order=1,
        max_order=2,
        estimated=False,
        baseline_value=0.0,
        interaction_lookup={(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5},
    )


def test_interaction_matrix_from_explanation_diagonal_holds_main_effects():
    """The diagonal must hold each player's order-1 main effect, by default."""
    iv = _ksii_values()

    matrix = interaction_matrix_from_explanation(iv, 3)

    assert list(np.diag(matrix)) == [1.0, 2.0, 3.0]


def test_interaction_matrix_from_explanation_pairwise_values_are_symmetric():
    iv = _ksii_values()

    matrix = interaction_matrix_from_explanation(iv, 3)

    assert matrix[0, 1] == matrix[1, 0] == 4.0
    assert matrix[0, 2] == matrix[2, 0] == 5.0
    assert matrix[1, 2] == matrix[2, 1] == 6.0


def test_interaction_matrix_from_explanation_missing_interactions_are_zero():
    """Interactions absent from the lookup must read as 0.0 instead of raising."""
    iv = InteractionValues(
        n_players=3,
        values=np.array([1.0]),
        index="k-SII",
        min_order=1,
        max_order=2,
        estimated=False,
        baseline_value=0.0,
        interaction_lookup={(0,): 0},
    )

    matrix = interaction_matrix_from_explanation(iv, 3)

    assert matrix[1, 1] == 0.0
    assert matrix[2, 2] == 0.0
    assert matrix[0, 1] == matrix[1, 0] == 0.0
    assert matrix[0, 2] == matrix[2, 0] == 0.0
    assert matrix[1, 2] == matrix[2, 1] == 0.0


def test_interaction_matrix_from_explanation_include_main_effects_false():
    """With ``include_main_effects=False``, the diagonal stays zero regardless of the
    explanation's own singleton values.
    """
    iv = _ksii_values()

    matrix = interaction_matrix_from_explanation(iv, 3, include_main_effects=False)

    assert list(np.diag(matrix)) == [0.0, 0.0, 0.0]
    # off-diagonal values are unaffected
    assert matrix[0, 1] == matrix[1, 0] == 4.0


def test_interaction_matrix_from_explanation_player_count_mismatch_raises():
    iv = _ksii_values()

    with pytest.raises(ValueError, match="n_players must match"):
        interaction_matrix_from_explanation(iv, 4)


def test_sentence_interaction_heatmap_uses_shared_matrix_helper():
    """The heatmap's drawn image must be pixel-identical to the shared matrix helper's
    output, so any other consumer building the same matrix agrees with what's plotted.
    """
    iv = _ksii_values()
    words = ["a", "b", "c"]

    fig, ax = sentence_interaction_heatmap(iv, words, show=False)

    drawn_matrix = ax.images[0].get_array()
    expected_matrix = interaction_matrix_from_explanation(iv, 3)
    assert np.allclose(drawn_matrix, expected_matrix)

    plt.close(fig)


def test_sentence_interaction_heatmap_annotates_every_cell():
    """Every matrix cell must get exactly one text annotation, n * n total."""
    iv = _ksii_values()
    words = ["a", "b", "c"]

    fig, ax = sentence_interaction_heatmap(iv, words, show=False)

    assert len(ax.texts) == len(words) * len(words)

    plt.close(fig)


def test_sentence_interaction_heatmap_annotation_text_matches_matrix():
    """Annotation text at each (row, col) position must match the matrix value there,
    covering both the diagonal (main effects) and off-diagonal (pairwise interactions).
    """
    iv = _ksii_values()
    words = ["a", "b", "c"]
    n = len(words)

    fig, ax = sentence_interaction_heatmap(iv, words, show=False)

    expected_matrix = interaction_matrix_from_explanation(iv, n)
    texts_by_position = {tuple(text.get_position()): text.get_text() for text in ax.texts}

    for i in range(n):
        for j in range(n):
            expected_value = expected_matrix[i, j]
            expected_text = f"{expected_value:+.3f}"
            assert texts_by_position[(j, i)] == expected_text

    # diagonal holds main effects
    assert texts_by_position[(0, 0)] == "+1.000"
    assert texts_by_position[(1, 1)] == "+2.000"
    assert texts_by_position[(2, 2)] == "+3.000"

    # off-diagonal holds pairwise interactions, symmetric across the diagonal
    assert texts_by_position[(1, 0)] == texts_by_position[(0, 1)] == "+4.000"
    assert texts_by_position[(2, 0)] == texts_by_position[(0, 2)] == "+5.000"
    assert texts_by_position[(2, 1)] == texts_by_position[(1, 2)] == "+6.000"

    plt.close(fig)


def test_sentence_interaction_heatmap_near_zero_negative_normalized():
    """A tiny negative value must render as '+0.000', not '-0.000'."""
    words = ["a", "b", "c"]
    iv = InteractionValues(
        n_players=3,
        values=np.array([-0.00001, 2.0, 3.0, 4.0, 5.0, 6.0]),
        index="k-SII",
        min_order=1,
        max_order=2,
        estimated=False,
        baseline_value=0.0,
        interaction_lookup={(0,): 0, (1,): 1, (2,): 2, (0, 1): 3, (0, 2): 4, (1, 2): 5},
    )

    fig, ax = sentence_interaction_heatmap(iv, words, show=False)

    texts_by_position = {tuple(text.get_position()): text.get_text() for text in ax.texts}
    assert texts_by_position[(0, 0)] == "+0.000"

    plt.close(fig)


def test_sentence_interaction_heatmap_annotation_text_colors():
    """Cells whose |value| is a large fraction of max_abs_score get white text (dark
    cells); small-magnitude cells get black text (light cells).
    """
    iv = _ksii_values()
    words = ["a", "b", "c"]

    fig, ax = sentence_interaction_heatmap(iv, words, show=False)

    texts_by_position = {tuple(text.get_position()): text for text in ax.texts}

    # max_abs_score is 6.0 (the largest magnitude in the matrix); the (2, 1)/(1, 2)
    # cell holding 6.0 is fully saturated and must use white text.
    assert texts_by_position[(1, 2)].get_color() == "white"

    # the (0, 0) cell holds 1.0, well under the 0.55 * max_abs_score threshold, and
    # must use black text.
    assert texts_by_position[(0, 0)].get_color() == "black"

    plt.close(fig)


def test_sentence_interaction_heatmap_matrix_unchanged_by_annotations():
    """Adding text annotations must not alter the underlying heatmap matrix values."""
    iv = _ksii_values()
    words = ["a", "b", "c"]

    fig, ax = sentence_interaction_heatmap(iv, words, show=False)

    drawn_matrix = ax.images[0].get_array()
    expected_matrix = interaction_matrix_from_explanation(iv, 3)
    assert np.allclose(drawn_matrix, expected_matrix)
    assert len(ax.images) == 1

    plt.close(fig)
