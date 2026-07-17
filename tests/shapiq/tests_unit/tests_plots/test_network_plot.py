"""This module contains all tests for the network plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp

from shapiq.interaction_values import InteractionValues
from shapiq.plot import network_plot


def test_network_plot():
    """Tests whether the network plot can be created."""
    n_players = 5
    n_values = n_players + int(sp.special.binom(n_players, 2))
    iv = InteractionValues(
        values=np.linspace(0.1, 1.0, n_values),
        index="k-SII",
        n_players=n_players,
        min_order=1,
        max_order=2,
        baseline_value=0.0,
    )
    fig, axes = network_plot(interaction_values=iv)
    assert fig is not None
    assert axes is not None
    plt.close(fig)
