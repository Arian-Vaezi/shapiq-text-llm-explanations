"""PDF extraction and overlapping-word chunking for the RAG pipeline.

Responsibilities:
- Extract text from PDF bytes page-by-page
- Classify pages and chunks (body, reference, caption, heading, table)
- Split pages into overlapping CandidateChunk objects
"""

from __future__ import annotations

import re
from io import BytesIO

from .schemas import CandidateChunk, PDFPage

# ---------------------------------------------------------------------------
# Regex constants
# ---------------------------------------------------------------------------

REFERENCE_HEADING_RE = re.compile(
    r"(^|\s)(references|bibliography|works cited)(\s|$)",
    re.IGNORECASE,
)
FIGURE_QUERY_RE = re.compile(
    r"\b(figure|fig\.?|image|caption|map|table|chart|diagram)\b",
    re.IGNORECASE,
)
YEAR_RE = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
DOI_RE = re.compile(
    r"\b(doi|https?://|et al\.|journal|proceedings|conference)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace while preserving readable sentence spacing."""
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# PDF extraction
# ---------------------------------------------------------------------------


def extract_pdf_pages(pdf_bytes: bytes) -> list[PDFPage]:
    """Extract text from uploaded PDF bytes."""
    try:
        from pypdf import PdfReader
    except ImportError as error:  # pragma: no cover - depends on optional extra
        try:
            from PyPDF2 import PdfReader  # type: ignore[no-redef]
        except ImportError as fallback_error:  # pragma: no cover - depends on optional extra
            msg = (
                "PDF upload requires `pypdf` or `PyPDF2`. Install the demo requirements "
                "with `uv sync --group rag_demo`."
            )
            raise RuntimeError(msg) from fallback_error
        _ = error

    reader = PdfReader(BytesIO(pdf_bytes))
    pages = []
    for page_idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = normalize_whitespace(text)
        if text:
            pages.append(PDFPage(page_number=page_idx, text=text))
    if not pages:
        msg = "No selectable text was found in this PDF. OCR-scanned PDFs are not supported yet."
        raise ValueError(msg)
    return pages


# ---------------------------------------------------------------------------
# Page / chunk classification
# ---------------------------------------------------------------------------


def looks_like_reference_page(text: str) -> bool:
    """Detect pages that are probably in the bibliography/reference section."""
    low = text.lower()
    if has_reference_section_heading(text):
        return True
    words = max(1, len(low.split()))
    year_density = len(YEAR_RE.findall(low)) / words
    return year_density > 0.055 and len(DOI_RE.findall(low)) >= 2


def has_reference_section_heading(text: str) -> bool:
    """Detect a real bibliography heading near the top of a page."""
    first_words = " ".join(text.lower().split()[:16])
    if not REFERENCE_HEADING_RE.search(first_words):
        return False
    # Avoid treating running text like "for references to prior work..." as the
    # start of a persistent reference section.
    return bool(re.search(r"^(references|bibliography|works cited)\b", first_words))


def infer_section_title(text: str, *, fallback: str = "") -> str:
    """Best-effort section label from the beginning of a PDF page."""
    cleaned = normalize_whitespace(text)
    candidates = re.split(r"(?<=[.!?])\s+", cleaned[:350])
    for candidate in candidates[:3]:
        words = candidate.split()
        if (
            2 <= len(words) <= 14
            and not candidate.endswith(",")
            and (candidate.isupper() or re.match(r"^\d+(\.\d+)*\s+[A-Z]", candidate))
        ):
            return candidate[:120]
    return fallback


def classify_chunk(text: str, *, page_is_reference: bool) -> tuple[str, tuple[str, ...]]:
    """Classify a PDF chunk with conservative, inspectable heuristics."""
    low = text.lower()
    words = text.split()
    word_count = len(words)
    flags: list[str] = []

    year_density = len(YEAR_RE.findall(text)) / max(1, word_count)
    citation_density = year_density + (0.035 * len(DOI_RE.findall(text)))
    if page_is_reference or citation_density > 0.12:
        flags.append("reference section")
        if citation_density > 0.12:
            flags.append("citation-dense")
        return "reference", tuple(flags)

    starts_like_caption = re.match(r"^(figure|fig\.|table|map|image)\s+\d+", low) is not None
    caption_terms = len(re.findall(r"\b(figure|fig\.|caption|image|map|table)\b", low))
    if starts_like_caption or (caption_terms >= 2 and word_count <= 140):
        flags.append("caption-only")
        return "caption", tuple(flags)

    if word_count <= 35:
        flags.append("very short")
        if not re.search(r"[.!?]", text):
            flags.append("heading-like")
            return "heading_only", tuple(flags)

    number_tokens = sum(1 for token in words if re.search(r"\d", token))
    if word_count >= 20 and number_tokens / max(1, word_count) > 0.38:
        flags.append("numeric/table-like")
        return "table", tuple(flags)

    if citation_density > 0.055:
        flags.append("citation-heavy")
    return "body_text", tuple(flags)


def query_asks_for_references(question: str) -> bool:
    """Return True for bibliographic/reference-list questions."""
    return bool(
        re.search(
            r"\b(reference|references|bibliography|citation|cite|source list)\b",
            question,
            re.IGNORECASE,
        )
    )


def query_asks_for_figures(question: str) -> bool:
    """Return True when captions/figures are likely directly relevant."""
    return bool(FIGURE_QUERY_RE.search(question))


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


def chunk_pdf_pages(
    pages: list[PDFPage],
    *,
    words_per_chunk: int,
    overlap_words: int,
) -> list[CandidateChunk]:
    """Split extracted PDF pages into overlapping word chunks."""
    if words_per_chunk <= 0:
        msg = "words_per_chunk must be positive."
        raise ValueError(msg)
    if overlap_words < 0 or overlap_words >= words_per_chunk:
        msg = "overlap_words must be non-negative and smaller than words_per_chunk."
        raise ValueError(msg)

    chunks: list[CandidateChunk] = []
    in_reference_section = False
    current_section = ""
    for page in pages:
        if has_reference_section_heading(page.text):
            in_reference_section = True
        page_is_reference = in_reference_section or looks_like_reference_page(page.text)
        inferred_section = infer_section_title(page.text, fallback=current_section)
        if inferred_section:
            current_section = inferred_section
        words = page.text.split()
        if not words:
            continue
        step = words_per_chunk - overlap_words
        for start in range(0, len(words), step):
            window = words[start : start + words_per_chunk]
            if len(window) < max(25, words_per_chunk // 5) and chunks:
                break
            chunk_number = len(chunks) + 1
            title = f"Page {page.page_number}, chunk {chunk_number}"
            chunk_text = " ".join(window)
            chunk_type, flags = classify_chunk(
                chunk_text,
                page_is_reference=page_is_reference,
            )
            chunks.append(
                CandidateChunk(
                    title=title,
                    text=chunk_text,
                    page_number=page.page_number,
                    chunk_number=chunk_number,
                    section_title=current_section,
                    chunk_type=chunk_type,
                    text_length=len(window),
                    flags=flags,
                )
            )
            if start + words_per_chunk >= len(words):
                break
    if not chunks:
        msg = "The PDF text was too short to build retrieval chunks."
        raise ValueError(msg)
    return chunks
