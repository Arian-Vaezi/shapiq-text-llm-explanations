"""Answer generation interface for the local GGUF RAG pipeline."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from .model_backends import GenerationResult, LlamaCppBackend

if TYPE_CHECKING:
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
    model_path: str,
    n_ctx: int,
    n_gpu_layers: int,
    n_threads: int,
    max_new_tokens: int,
) -> str:
    """Use the configured local GGUF model to answer from retrieved chunks."""
    backend = LlamaCppBackend(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
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
    "LlamaCppBackend",
    "format_rag_context",
    "generate_answer_from_chunks",
]
