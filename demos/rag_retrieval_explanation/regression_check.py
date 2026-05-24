"""Small retrieval regression check for the PDF RAG demo.

Usage:
    uv run python demos/rag_retrieval_explanation/regression_check.py path/to/report.pdf
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
    detect_aspects,
    dominant_aspect,
    extract_pdf_pages,
    retrieve_relevant_chunks_with_debug,
)

QUESTION = "Why is the lunar south pole scientifically interesting?"


def main() -> int:
    """Run retrieval only and fail if the selected context regresses."""
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--words-per-chunk", type=int, default=180)
    parser.add_argument("--overlap-words", type=int, default=35)
    parser.add_argument("--method", choices=["Dense embeddings", "TF-IDF"], default="Dense embeddings")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--embedding-device", default="auto")
    parser.add_argument(
        "--answer-file",
        type=Path,
        help="Optional generated-answer text file to check answer-level coverage.",
    )
    args = parser.parse_args()

    pages = extract_pdf_pages(args.pdf.read_bytes())
    candidates = chunk_pdf_pages(
        pages,
        words_per_chunk=args.words_per_chunk,
        overlap_words=args.overlap_words,
    )
    ranked, debug = retrieve_relevant_chunks_with_debug(
        QUESTION,
        candidates,
        top_k=args.top_k,
        method=args.method,
        embedding_model_id=args.embedding_model,
        embedding_device=args.embedding_device,
    )

    print(f"Question: {QUESTION}")  # noqa: T201
    print(f"Expanded queries: {debug.expanded_queries}")  # noqa: T201
    for idx, item in enumerate(ranked, start=1):
        aspects = detect_aspects(item.chunk.text)
        print(  # noqa: T201
            f"{idx}. {item.chunk.title} | page={item.chunk.page_number} "
            f"type={item.chunk.chunk_type} rerank={item.rerank_score:.3f} "
            f"dense={item.dense_score:.3f} keyword={item.keyword_score:.3f} "
            f"dominant={dominant_aspect(item.chunk.text)} aspects={aspects} flags={item.chunk.flags}"
        )
    print(f"Aspect coverage: {debug.aspect_coverage}")  # noqa: T201

    selected_aspects = {aspect for item in ranked for aspect in detect_aspects(item.chunk.text)}
    selected_types = [item.chunk.chunk_type for item in ranked]
    dominant_aspects = [dominant_aspect(item.chunk.text) for item in ranked]
    failures = []
    if "volatiles_psr" not in selected_aspects:
        failures.append("missing volatile/PSR/cold-trap context")
    if "illumination_access" not in selected_aspects and "south_polar_overview" not in selected_aspects:
        failures.append("missing illumination/access or south polar overview context")
    if "artemis_objectives" not in selected_aspects:
        failures.append("missing Artemis/science-objectives context")
    if "impact_chronology" not in selected_aspects:
        failures.append("missing impact chronology context")
    if dominant_aspects and dominant_aspects[0] == "dust_electrodynamics":
        failures.append("dust/electrodynamics is the top selected context aspect")
    low_value_count = sum(chunk_type in {"reference", "caption"} for chunk_type in selected_types)
    if low_value_count >= max(2, len(selected_types)):
        failures.append("reference/caption chunks dominate final context")

    if args.answer_file:
        answer = args.answer_file.read_text(encoding="utf-8").lower()
        answer_checks = {
            "permanently shadowed regions or cold traps": r"permanently shadowed|cold trap|cold traps|psr",
            "water ice or polar volatiles": r"water ice|ice|volatile|volatiles",
            "near-persistent illumination or long-duration sunlight": (
                r"near[- ]persistent illumination|persistent illumination|long[- ]duration sunlight|sunlight"
            ),
            "impact basins/craters or cratering chronology": (
                r"impact basin|impact basins|crater|craters|cratering chronology|impact chronology"
            ),
        }
        for label, pattern in answer_checks.items():
            if not re.search(pattern, answer):
                failures.append(f"generated answer missing {label}")
        dust_hits = len(re.findall(r"\b(dust|plasma|electrodynamic|electrodynamics|charging)\b", answer))
        core_hits = sum(1 for pattern in answer_checks.values() if re.search(pattern, answer))
        if dust_hits >= 2 and core_hits < 3:
            failures.append("generated answer is dominated by dust/electrodynamics")

    if failures:
        print("FAIL: " + "; ".join(failures))  # noqa: T201
        return 1
    print("PASS: selected context covers volatile/PSR and impact chronology evidence.")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
