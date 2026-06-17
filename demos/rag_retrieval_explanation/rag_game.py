"""Backward-compatible shim — re-exports from core.rag_game."""

from __future__ import annotations

from core.rag_game import (  # noqa: F401
    STOPWORDS,
    RAGRetrievalGame,
    RetrievedChunk,
    ScoreCallable,
    budget_for_exactish_demo,
    lexical_grounding_score,
    normalize_tokens,
)
