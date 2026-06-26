"""Adapter computing official shapiq interaction indices exactly via ``ExactComputer``.

Exact coalition enumeration costs ``2**n_players`` game evaluations, so this module only
supports small games (see ``MAX_EXACT_DEMO_PLAYERS``). The demo must only label a result
with an official shapiq interaction index (``k-SII``, ``STII``, ``FSII``, ``SII``, ...)
when it was actually produced by ``shapiq.game_theory.exact.ExactComputer``, never by a
hand-rolled finite-difference heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from shapiq.game_theory.exact import ExactComputer

if TYPE_CHECKING:
    from shapiq.game import Game
    from shapiq.interaction_values import InteractionValues
    from shapiq.typing import IndexType

MAX_EXACT_DEMO_PLAYERS = 10
"""Largest player count this demo will explain exactly.

Exact computation evaluates the game on all ``2**n_players`` coalitions. Above this
limit the demo refuses to silently substitute an approximation under an official
shapiq index label.
"""


class ExactComputationLimitError(ValueError):
    """Raised when exact demo computation would exceed the configured player limit."""


class UnsupportedExactIndexError(ValueError):
    """Raised when the requested index is unsupported by ExactComputer."""


@dataclass(frozen=True)
class ExactComputationMetadata:
    """Describes how an exact interaction result was computed."""

    algorithm: str
    index: str
    n_players: int
    max_order: int
    coalition_count: int


def compute_exact_interactions(
    *,
    game: Game,
    index: IndexType,
    max_order: int,
) -> tuple[InteractionValues, ExactComputationMetadata]:
    """Compute an official shapiq interaction index exactly with ``shapiq.ExactComputer``.

    Args:
        game: A ``shapiq.Game`` instance whose ``value_function`` is evaluated on all
            ``2**game.n_players`` coalitions.
        index: The shapiq interaction index to compute, validated against
            ``ExactComputer.valid_indices``.
        max_order: The maximum interaction order to compute.

    Returns:
        A tuple of the native ``shapiq.InteractionValues`` result and metadata describing
        the exact computation.

    Raises:
        ExactComputationLimitError: If ``game.n_players`` is outside
            ``(0, MAX_EXACT_DEMO_PLAYERS]``.
        UnsupportedExactIndexError: If ``index`` is not supported by ``ExactComputer``.
    """
    n_players = game.n_players
    if not (0 < n_players <= MAX_EXACT_DEMO_PLAYERS):
        msg = (
            f"Exact computation supports 1 to {MAX_EXACT_DEMO_PLAYERS} players "
            f"(cost is 2**n_players coalition evaluations); got {n_players} players."
        )
        raise ExactComputationLimitError(msg)

    if index not in ExactComputer.valid_indices:
        msg = (
            f"Index {index!r} is not supported by shapiq.ExactComputer. "
            f"Supported indices: {ExactComputer.valid_indices}."
        )
        raise UnsupportedExactIndexError(msg)

    exact_computer = ExactComputer(game=game, n_players=n_players)
    interaction_values = exact_computer(index=index, order=max_order)

    metadata = ExactComputationMetadata(
        algorithm="ExactComputer",
        index=index,
        n_players=n_players,
        max_order=max_order,
        coalition_count=2**n_players,
    )
    return interaction_values, metadata
