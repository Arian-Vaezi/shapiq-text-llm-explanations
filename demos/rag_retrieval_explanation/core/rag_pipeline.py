"""Backward-compatible re-export shim for the split RAG pipeline modules.

Real implementations live in:
  core/chunking.py   — PDF extraction and chunking
  core/retrieval.py  — scoring, reranking, context selection
  core/generation.py — answer generation

This module re-exports every public name so that existing callers of
`from core.rag_pipeline import X` continue to work without changes.
"""

from __future__ import annotations

from .chunking import (  # noqa: F401
    DOI_RE,
    FIGURE_QUERY_RE,
    REFERENCE_HEADING_RE,
    YEAR_RE,
    chunk_pdf_pages,
    classify_chunk,
    extract_pdf_pages,
    has_reference_section_heading,
    infer_section_title,
    looks_like_reference_page,
    normalize_whitespace,
    query_asks_for_figures,
    query_asks_for_references,
)
from .generation import format_rag_context, generate_answer_from_chunks  # noqa: F401
from .retrieval import (  # noqa: F401
    BROAD_OVERVIEW_TERMS,
    QUERY_STOPWORDS,
    TECHNICAL_TERMS,
    cached_embedding_backend,
    chunk_query_hits,
    classify_query_intent,
    dense_embed_texts,
    expand_retrieval_queries,
    is_broad_explanatory_question,
    is_overview_chunk,
    query_concept_terms,
    retrieve_relevant_chunks_with_debug,
)
from .schemas import (  # noqa: F401
    CandidateChunk,
    PDFPage,
    RankedChunk,
    RetrievalDebugInfo,
    RetrievedChunk,
)
