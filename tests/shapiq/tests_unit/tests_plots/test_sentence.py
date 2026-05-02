"""This test module contains all tests for the sentence plot."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from shapiq.interaction_values import InteractionValues
from shapiq.plot import sentence_interaction_heatmap, sentence_plot, token_attribution_bar_plot


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
