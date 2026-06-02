"""Small generic retrieval smoke check for the PDF RAG demo.

Usage:
    uv run python demos/rag_retrieval_explanation/regression_check.py path/to/file.pdf \
        --question "What is the main contribution?" \
        --expected-evidence-term "Shapley"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from rag_pipeline import (  # noqa: E402
    chunk_pdf_pages,
    extract_pdf_pages,
    retrieve_relevant_chunks_with_debug,
)


def _contains_term(text: str, term: str) -> bool:
    """Return True when text contains a case-insensitive literal term."""
    return re.search(re.escape(term), text, re.IGNORECASE) is not None


def _missing_terms(text: str, terms: list[str]) -> list[str]:
    """Return expected terms not found in text."""
    return [term for term in terms if not _contains_term(text, term)]


def main() -> int:
    """Run retrieval only and fail if generic smoke-check expectations are unmet."""
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--words-per-chunk", type=int, default=180)
    parser.add_argument("--overlap-words", type=int, default=35)
    parser.add_argument(
        "--method", choices=["Dense embeddings", "TF-IDF"], default="Dense embeddings"
    )
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument(
        "--expected-evidence-term",
        action="append",
        default=[],
        help="Literal term expected somewhere in the retrieved context. Repeatable.",
    )
    parser.add_argument(
        "--answer-file",
        type=Path,
        help="Optional generated-answer text file to check answer-level expectations.",
    )
    parser.add_argument(
        "--expected-answer-term",
        action="append",
        default=[],
        help="Literal term expected in --answer-file. Repeatable.",
    )
    parser.add_argument(
        "--forbidden-answer-term",
        action="append",
        default=[],
        help="Literal term that should not appear in --answer-file. Repeatable.",
    )
    args = parser.parse_args()

    pages = extract_pdf_pages(args.pdf.read_bytes())
    candidates = chunk_pdf_pages(
        pages,
        words_per_chunk=args.words_per_chunk,
        overlap_words=args.overlap_words,
    )
    ranked, debug = retrieve_relevant_chunks_with_debug(
        args.question,
        candidates,
        top_k=args.top_k,
        method=args.method,
        embedding_model_id=args.embedding_model,
        embedding_device=args.embedding_device,
    )

    print(f"Question: {args.question}")  # noqa: T201
    print(f"Expanded queries: {debug.expanded_queries}")  # noqa: T201
    for idx, item in enumerate(ranked, start=1):
        print(  # noqa: T201
            f"{idx}. {item.chunk.title} | page={item.chunk.page_number} "
            f"type={item.chunk.chunk_type} rerank={item.rerank_score:.3f} "
            f"dense={item.dense_score:.3f} keyword={item.keyword_score:.3f} "
            f"flags={item.chunk.flags}"
        )
    print(f"Coverage summary: {debug.coverage_summary}")  # noqa: T201

    retrieved_text = "\n\n".join(item.chunk.text for item in ranked)
    failures = []
    missing_evidence_terms = _missing_terms(retrieved_text, args.expected_evidence_term)
    if missing_evidence_terms:
        failures.append("retrieved context missing: " + ", ".join(missing_evidence_terms))

    if args.answer_file:
        answer = args.answer_file.read_text(encoding="utf-8")
        missing_answer_terms = _missing_terms(answer, args.expected_answer_term)
        if missing_answer_terms:
            failures.append("generated answer missing: " + ", ".join(missing_answer_terms))
        forbidden_hits = [
            term for term in args.forbidden_answer_term if _contains_term(answer, term)
        ]
        if forbidden_hits:
            failures.append(
                "generated answer contains forbidden terms: " + ", ".join(forbidden_hits)
            )

    if failures:
        print("FAIL: " + "; ".join(failures))  # noqa: T201
        return 1
    print("PASS: retrieval smoke check passed.")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
