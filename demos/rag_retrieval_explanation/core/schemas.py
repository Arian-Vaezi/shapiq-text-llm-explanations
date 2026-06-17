"""Canonical domain dataclasses for the RAG retrieval explanation demo.

All other core modules import from here rather than from each other, so
there are no circular dependencies between rag_game, rag_pipeline, and
value_functions.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Chunk types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved context chunk visible to the RAG model and the Shapley game."""

    title: str
    text: str
    page_number: int | None = None
    chunk_type: str = "body_text"
    section_title: str = ""
    text_length: int = 0
    retrieval_score: float | None = None
    dense_score: float | None = None
    keyword_score: float | None = None
    rerank_score: float | None = None
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class CandidateChunk:
    """A raw chunk produced by the chunker, before retrieval ranking."""

    title: str
    text: str
    page_number: int
    chunk_number: int
    section_title: str = ""
    chunk_type: str = "body_text"
    text_length: int = 0
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankedChunk:
    """A candidate chunk after retrieval scoring and reranking."""

    chunk: RetrievedChunk
    score: float
    dense_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0
    reasons: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Pipeline output types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PDFPage:
    """Text extracted from one PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True)
class RetrievalDebugInfo:
    """Transparent retrieval diagnostics shown in the Streamlit UI."""

    query_intent: str
    original_query: str
    expanded_queries: list[str]
    raw_dense_results: list[dict[str, object]]
    raw_keyword_results: list[dict[str, object]]
    raw_rerank_order: list[dict[str, object]]
    reranked_results: list[dict[str, object]]
    selected_context: list[dict[str, object]]
    coverage_summary: dict[str, object]
