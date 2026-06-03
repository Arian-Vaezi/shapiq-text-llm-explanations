"""Answer generation interface for the RAG pipeline.

This module provides:
- format_rag_context: format retrieved chunks into a grounded prompt context
- generate_answer_from_chunks: high-level entry point for HF model generation

The HuggingFaceCausalLMBackend implementation lives in core/model_backends.py
and is imported here to give callers a single generation-oriented import path.
Future backends (LM Studio, extractive) will be added here without changing
the model_backends module.
"""

from __future__ import annotations

from .model_backends import GenerationResult, HuggingFaceCausalLMBackend
from .schemas import RetrievedChunk


def format_rag_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks for a grounded generation prompt."""
    if not chunks:
        return "(no retrieved context)"
    blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        metadata = []
        if chunk.page_number:
            metadata.append(f"page {chunk.page_number}")
        if chunk.chunk_type:
            metadata.append(f"type: {chunk.chunk_type}")
        if chunk.section_title:
            metadata.append(f"section: {chunk.section_title}")
        meta_line = f" ({'; '.join(metadata)})" if metadata else ""
        blocks.append(f"[{idx}] {chunk.title}{meta_line}\n{chunk.text}")
    return "\n\n".join(blocks)


def generate_answer_from_chunks(
    *,
    question: str,
    chunks: list[RetrievedChunk],
    model_id: str,
    device_map: str,
    torch_dtype: str,
    max_new_tokens: int,
) -> str:
    """Use the configured HF causal LM to answer from retrieved chunks."""
    import re

    backend = HuggingFaceCausalLMBackend(
        model_id=model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
    )
    context = format_rag_context(chunks)
    result = backend.generate(question, context)
    answer = re.sub(r"\s+", " ", result.answer).strip()
    if not answer:
        return "I could not generate an answer from the retrieved context."
    return answer


__all__ = [
    "GenerationResult",
    "HuggingFaceCausalLMBackend",
    "format_rag_context",
    "generate_answer_from_chunks",
]
