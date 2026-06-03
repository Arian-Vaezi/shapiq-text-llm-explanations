"""Backward-compatible shim — re-exports from core.rag_pipeline.

This file keeps the original import paths working for app.py and tests while
the real implementation lives in core/rag_pipeline.py.
"""

from __future__ import annotations

from core.rag_pipeline import (  # noqa: F401
    BROAD_OVERVIEW_TERMS,
    DOI_RE,
    FIGURE_QUERY_RE,
    QUERY_STOPWORDS,
    REFERENCE_HEADING_RE,
    TECHNICAL_TERMS,
    YEAR_RE,
    CandidateChunk,
    PDFPage,
    RankedChunk,
    RetrievalDebugInfo,
    cached_embedding_backend,
    chunk_pdf_pages,
    chunk_query_hits,
    classify_chunk,
    classify_query_intent,
    dense_embed_texts,
    expand_retrieval_queries,
    extract_pdf_pages,
    format_rag_context,
    generate_answer_from_chunks,
    has_reference_section_heading,
    is_broad_explanatory_question,
    is_overview_chunk,
    looks_like_reference_page,
    normalize_whitespace,
    query_asks_for_figures,
    query_asks_for_references,
    query_concept_terms,
    retrieve_relevant_chunks_with_debug,
)
