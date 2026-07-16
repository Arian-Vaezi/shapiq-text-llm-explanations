"""Evaluation utilities for retrieval attribution runs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .rag_game import RAGRetrievalGame


@dataclass(frozen=True)
class FaithfulnessSummary:
    """Compact deletion-style faithfulness summary."""

    full_score: float
    score_without_top: float
    score_with_top_only: float
    deletion_drop: float
    sufficiency_gap: float


def attribution_order(attribution_frame: pd.DataFrame) -> list[int]:
    """Return zero-based player ids sorted by attribution magnitude."""
    if attribution_frame.empty:
        return []
    return [int(player) - 1 for player in attribution_frame["player"].tolist()]


def score_mask(game: RAGRetrievalGame, selected_indices: list[int]) -> float:
    """Score one coalition identified by selected player indices."""
    coalition = np.zeros((1, game.n_players), dtype=bool)
    coalition[0, selected_indices] = True
    return float(game(coalition)[0])


def player_title(game: RAGRetrievalGame, player_id: int) -> str:
    """Return the display title for one game player."""
    return game.chunks[player_id].title


def deletion_evaluation(game: RAGRetrievalGame, attribution_frame: pd.DataFrame) -> pd.DataFrame:
    """Remove chunks in attribution order and record the support-score curve."""
    ordered_players = attribution_order(attribution_frame)
    return deletion_evaluation_from_order(game, ordered_players)


def deletion_evaluation_from_order(
    game: RAGRetrievalGame,
    ordered_players: list[int],
) -> pd.DataFrame:
    """Remove chunks in a supplied order and record the support-score curve."""
    remaining = list(range(game.n_players))
    rows = [
        {
            "step": 0,
            "removed_chunk": "none",
            "remaining_chunks": game.n_players,
            "score": score_mask(game, remaining),
        }
    ]

    for step, player_id in enumerate(ordered_players, start=1):
        if player_id in remaining:
            remaining.remove(player_id)
        rows.append(
            {
                "step": step,
                "removed_chunk": player_title(game, player_id),
                "remaining_chunks": len(remaining),
                "score": score_mask(game, remaining),
            }
        )

    return pd.DataFrame(rows)


def retrieval_score_order(retrieval_scores: list[float], n_players: int) -> list[int]:
    """Return zero-based player ids sorted by retrieval score descending."""
    if len(retrieval_scores) != n_players:
        return []
    return sorted(range(n_players), key=lambda idx: retrieval_scores[idx], reverse=True)


def retrieval_score_deletion_baseline(
    game: RAGRetrievalGame,
    retrieval_scores: list[float],
) -> np.ndarray | None:
    """Deletion curve after removing chunks by retriever relevance score."""
    ordered_players = retrieval_score_order(retrieval_scores, game.n_players)
    if not ordered_players:
        return None
    frame = deletion_evaluation_from_order(game, ordered_players)
    return frame["score"].to_numpy(dtype=float)


def leave_one_out_order(game: RAGRetrievalGame) -> list[int]:
    """Rank players by the full-score drop caused by removing each one."""
    full = score_mask(game, list(range(game.n_players)))
    drops = []
    for player_id in range(game.n_players):
        remaining = [idx for idx in range(game.n_players) if idx != player_id]
        drops.append((player_id, full - score_mask(game, remaining)))
    return [player_id for player_id, _drop in sorted(drops, key=lambda item: item[1], reverse=True)]


def insertion_evaluation(game: RAGRetrievalGame, attribution_frame: pd.DataFrame) -> pd.DataFrame:
    """Add chunks in attribution order and record the support-score curve."""
    ordered_players = attribution_order(attribution_frame)
    selected: list[int] = []
    rows = [
        {
            "step": 0,
            "added_chunk": "none",
            "selected_chunks": 0,
            "score": score_mask(game, selected),
        }
    ]

    for step, player_id in enumerate(ordered_players, start=1):
        selected.append(player_id)
        rows.append(
            {
                "step": step,
                "added_chunk": player_title(game, player_id),
                "selected_chunks": len(selected),
                "score": score_mask(game, selected),
            }
        )

    return pd.DataFrame(rows)


def random_deletion_baseline(game: RAGRetrievalGame, max_perms: int = 60) -> np.ndarray:
    """Average deletion curve over all (or sampled) random orderings."""
    n = game.n_players
    all_perms = list(permutations(range(n)))
    if len(all_perms) > max_perms:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_perms), max_perms, replace=False)
        selected_perms = [all_perms[i] for i in idx]
    else:
        selected_perms = all_perms

    total = np.zeros(n + 1)
    for perm in selected_perms:
        remaining = list(range(n))
        scores = [score_mask(game, remaining)]
        for player_id in perm:
            remaining = [p for p in remaining if p != player_id]
            scores.append(score_mask(game, remaining))
        total += np.array(scores)
    return total / len(selected_perms)


def random_insertion_baseline(game: RAGRetrievalGame, max_perms: int = 60) -> np.ndarray:
    """Average insertion curve over all (or sampled) random orderings."""
    n = game.n_players
    all_perms = list(permutations(range(n)))
    if len(all_perms) > max_perms:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(all_perms), max_perms, replace=False)
        selected_perms = [all_perms[i] for i in idx]
    else:
        selected_perms = all_perms

    total = np.zeros(n + 1)
    for perm in selected_perms:
        selected: list[int] = []
        scores = [score_mask(game, selected)]
        for player_id in perm:
            selected.append(player_id)
            scores.append(score_mask(game, selected))
        total += np.array(scores)
    return total / len(selected_perms)


def summarize_faithfulness(
    game: RAGRetrievalGame,
    attribution_frame: pd.DataFrame,
) -> FaithfulnessSummary:
    """Summarize whether the top chunk is necessary and sufficient."""
    full_score = score_mask(game, list(range(game.n_players)))
    ordered_players = attribution_order(attribution_frame)
    if not ordered_players:
        return FaithfulnessSummary(
            full_score=full_score,
            score_without_top=full_score,
            score_with_top_only=0.0,
            deletion_drop=0.0,
            sufficiency_gap=full_score,
        )

    top_player = ordered_players[0]
    without_top = [idx for idx in range(game.n_players) if idx != top_player]
    score_without_top = score_mask(game, without_top)
    score_with_top_only = score_mask(game, [top_player])
    return FaithfulnessSummary(
        full_score=full_score,
        score_without_top=score_without_top,
        score_with_top_only=score_with_top_only,
        deletion_drop=full_score - score_without_top,
        sufficiency_gap=full_score - score_with_top_only,
    )
