"""Minimal PDF-to-RAG pipeline for the retrieval attribution demo."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from model_backends import HuggingFaceCausalLMBackend  # noqa: E402
from rag_game import RetrievedChunk  # noqa: E402


@dataclass(frozen=True)
class PDFPage:
    """Text extracted from one PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True)
class CandidateChunk:
    """A chunk before retrieval ranking."""

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
    """A retrieved chunk with its retrieval score."""

    chunk: RetrievedChunk
    score: float
    dense_score: float = 0.0
    keyword_score: float = 0.0
    rerank_score: float = 0.0
    reasons: tuple[str, ...] = ()


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
    aspect_coverage: dict[str, object]


def extract_pdf_pages(pdf_bytes: bytes) -> list[PDFPage]:
    """Extract text from uploaded PDF bytes."""
    try:
        from pypdf import PdfReader  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on optional extra
        try:
            from PyPDF2 import PdfReader  # type: ignore[no-redef] # noqa: PLC0415
        except ImportError as fallback_error:  # pragma: no cover - depends on optional extra
            msg = (
                "PDF upload requires `pypdf` or `PyPDF2`. Install the demo requirements "
                "with `uv pip install -r demos/rag_retrieval_explanation/requirements.txt`."
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


def normalize_whitespace(text: str) -> str:
    """Collapse whitespace while preserving readable sentence spacing."""
    return re.sub(r"\s+", " ", text).strip()


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


def _tokenize_for_retrieval(text: str) -> list[str]:
    """Return lowercase word tokens for transparent retrieval heuristics."""
    return re.findall(r"[a-z0-9][a-z0-9-]*", text.lower())


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


def classify_query_intent(question: str) -> str:
    """Classify whether final context should favor precision or synthesis."""
    low = question.lower().strip()
    broad_patterns = [
        r"\bwhy\b.*\b(important|interesting|scientifically interesting|valuable|matter|matters)\b",
        r"\b(summarize|overview|synthesis|main|major|broad|overall)\b",
        r"\b(motivations?|themes?|objectives?|reasons?|science goals?|scientific value)\b",
        r"\bwhat makes\b.*\b(important|interesting|valuable)\b",
    ]
    if any(re.search(pattern, low) for pattern in broad_patterns):
        return "broad_synthesis"

    narrow_patterns = [
        r"^(who|what|when|where|which|how many|how much)\b",
        r"\b(specific|exact|name|list|prioritize|prioritizes|define|definition)\b",
    ]
    if any(re.search(pattern, low) for pattern in narrow_patterns):
        return "narrow_factual"
    return "narrow_factual"


def is_broad_explanatory_question(question: str) -> bool:
    """Detect broad why/value/importance questions that need balanced evidence."""
    return classify_query_intent(question) == "broad_synthesis"


def query_asks_for_dust_electrodynamics(question: str) -> bool:
    """Return True when dust/plasma/electrodynamics is explicitly requested."""
    return bool(
        re.search(
            r"\b(dust|plasma|electrodynamic|electrodynamics|charging|terminator|surface potential|wake|wakes)\b",
            question,
            re.IGNORECASE,
        )
    )


def expand_retrieval_queries(question: str) -> list[str]:
    """Expand broad explanatory questions with recall-oriented search phrasings."""
    q = normalize_whitespace(question)
    low = q.lower()
    expansions = [q]

    if is_broad_explanatory_question(question):
        topic = re.sub(
            r"\b(why|is|are|the|a|an|scientifically|interesting|important|what|makes)\b",
            " ",
            low,
        )
        topic = re.sub(r"[^a-z0-9\s-]", " ", topic)
        topic = normalize_whitespace(topic)
        if topic:
            expansions.extend(
                [
                    f"{topic} overview",
                    f"{topic} main reasons",
                    f"{topic} major themes",
                    f"{topic} objectives motivations",
                    f"{topic} scientific value",
                ]
            )

    deduped: list[str] = []
    seen: set[str] = set()
    for item in expansions:
        key = item.lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)
    return deduped


ASPECT_PRIORITY = {
    "south_polar_overview": 0,
    "volatiles_psr": 1,
    "artemis_objectives": 2,
    "impact_chronology": 3,
    "illumination_access": 4,
    "dust_electrodynamics": 5,
}


def detect_aspects(text: str) -> tuple[str, ...]:
    """Label science aspects covered by a chunk for diverse context selection."""
    low = text.lower()
    tokens = set(_tokenize_for_retrieval(text))
    aspects: list[str] = []

    south_pole = "south pole" in low or "south polar" in low or "polar region" in low
    overview_terms = {"region", "site", "sites", "terrain", "environment", "access", "science"}
    if south_pole and len(tokens & overview_terms) >= 2:
        aspects.append("south_polar_overview")

    volatile_terms = {
        "volatile",
        "volatiles",
        "water",
        "ice",
        "deposit",
        "deposits",
        "hydrogen",
        "oxygen",
    }
    psr_phrase = bool(re.search(r"\b(permanently shadowed region|permanently shadowed regions|psr|psrs)\b", low))
    cold_trap_phrase = bool(re.search(r"\b(cold trap|cold traps)\b", low))
    volatile_hits = tokens & volatile_terms
    if cold_trap_phrase or len(volatile_hits) >= 2 or (psr_phrase and volatile_hits):
        aspects.append("volatiles_psr")

    illumination_phrase = bool(
        re.search(
            r"\b(persistent illumination|near[- ]persistent illumination|near permanent illumination|"
            r"long[- ]duration sunlight|sunlight|solar power|lunar night|low sun angle|"
            r"illuminated for over 200 days|operational access)\b",
            low,
        )
    )
    if illumination_phrase and not query_asks_for_dust_electrodynamics(text):
        aspects.append("illumination_access")

    impact_phrase = bool(
        re.search(
            r"\b(impact history|cratering chronology|basin age|basin ages|impact flux|"
            r"ejecta|sample age|sample ages|south pole-aitken|spa basin|spa|"
            r"impact basin|impact basins)\b",
            low,
        )
    )
    if impact_phrase:
        aspects.append("impact_chronology")

    artemis_terms = {"artemis", "objective", "objectives", "goal", "goals", "science"}
    core_objective_terms = {"volatiles", "volatile", "sample", "samples", "impact", "history", "origin"}
    if "artemis" in tokens or (
        len(tokens & artemis_terms) >= 2 and len(tokens & core_objective_terms) >= 1
    ):
        aspects.append("artemis_objectives")

    dust_terms = {
        "dust",
        "electrodynamic",
        "electrodynamics",
        "plasma",
        "terminator",
        "charging",
        "charged",
        "wake",
        "wakes",
    }
    if "surface potential" in low or len(tokens & dust_terms) >= 2:
        aspects.append("dust_electrodynamics")

    aspects = sorted(dict.fromkeys(aspects), key=lambda name: ASPECT_PRIORITY.get(name, 99))
    return tuple(aspects)


def dominant_aspect(text: str) -> str:
    """Return the primary aspect label used for balanced context selection."""
    aspects = detect_aspects(text)
    return aspects[0] if aspects else "general"


ASPECT_KEYWORDS = {
    "south_polar_overview": {"south", "polar", "pole", "region", "site", "terrain"},
    "volatiles_psr": {"volatile", "volatiles", "water", "ice", "deposit", "deposits", "hydrogen", "oxygen", "cold", "trap", "traps", "psr", "psrs"},
    "artemis_objectives": {"artemis", "objective", "objectives", "goal", "goals", "science", "sample", "samples"},
    "impact_chronology": {"impact", "cratering", "chronology", "basin", "basins", "ejecta", "flux", "age", "ages", "sample", "samples", "spa"},
    "illumination_access": {"illumination", "illuminated", "sunlight", "solar", "power", "access", "persistent", "duration", "night"},
    "dust_electrodynamics": {"dust", "plasma", "electrodynamic", "electrodynamics", "charging", "charged", "terminator", "potential", "wake", "wakes"},
}

BROAD_OVERVIEW_TERMS = {
    "overview",
    "summary",
    "summarize",
    "introduction",
    "background",
    "motivation",
    "motivations",
    "objective",
    "objectives",
    "theme",
    "themes",
    "reason",
    "reasons",
    "important",
    "importance",
    "major",
    "main",
    "overall",
    "broad",
    "science",
    "scientific",
}

TECHNICAL_TERMS = {
    "electrodynamics",
    "plasma",
    "terminator",
    "charging",
    "charged",
    "potential",
    "wakes",
    "spectrometer",
    "isotope",
    "mineralogy",
    "granulometry",
    "topography",
    "photometry",
    "regolith",
    "volatile",
    "volatiles",
    "chronology",
    "geochemistry",
}


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


@lru_cache(maxsize=2)
def cached_embedding_backend(
    model_id: str,
    device_name: str,
) -> tuple[object, object, object]:
    """Load and cache a Hugging Face embedding model."""
    try:
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - depends on optional extra
        msg = "Dense retrieval requires `torch` and `transformers`."
        raise RuntimeError(msg) from error

    if device_name == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    return tokenizer, model, device


def dense_embed_texts(
    texts: list[str],
    *,
    model_id: str,
    device_name: str,
    batch_size: int = 16,
) -> np.ndarray:
    """Embed texts with mean pooled Transformer hidden states."""
    import torch  # noqa: PLC0415

    tokenizer, model, device = cached_embedding_backend(model_id, device_name)
    embeddings = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())
        masked_embeddings = token_embeddings * attention_mask
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1)
        batch_embeddings = summed / counts
        batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
        embeddings.append(batch_embeddings.detach().cpu().numpy())
    return np.vstack(embeddings)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Scale scores into [0, 1] while keeping all-zero arrays stable."""
    scores = np.nan_to_num(scores.astype(float))
    if scores.size == 0:
        return scores
    low = float(np.min(scores))
    high = float(np.max(scores))
    if high - low < 1e-9:
        return np.zeros_like(scores, dtype=float)
    return (scores - low) / (high - low)


def _keyword_scores(question: str, chunks: list[CandidateChunk]) -> np.ndarray:
    """Compute sparse keyword scores with expanded queries."""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415
        from sklearn.metrics.pairwise import linear_kernel  # noqa: PLC0415
    except ImportError as error:  # pragma: no cover - project dependency
        msg = "PDF retrieval requires scikit-learn, which is part of the project dependencies."
        raise RuntimeError(msg) from error

    documents = [chunk.text for chunk in chunks]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    matrix = vectorizer.fit_transform(documents)
    query = vectorizer.transform([" ".join(expand_retrieval_queries(question))])
    scores = linear_kernel(query, matrix).ravel()
    return np.nan_to_num(scores)


def _dense_scores(
    question: str,
    chunks: list[CandidateChunk],
    *,
    embedding_model_id: str,
    embedding_device: str,
) -> np.ndarray:
    """Compute dense cosine scores, taking the best score over expanded queries."""
    documents = [chunk.text for chunk in chunks]
    document_embeddings = dense_embed_texts(
        documents,
        model_id=embedding_model_id,
        device_name=embedding_device,
    )
    query_embeddings = dense_embed_texts(
        expand_retrieval_queries(question),
        model_id=embedding_model_id,
        device_name=embedding_device,
    )
    scores = document_embeddings @ query_embeddings.T
    best_scores = np.max(scores, axis=1)
    return np.nan_to_num(best_scores)


def _keyword_overlap_bonus(question: str, chunk: CandidateChunk) -> float:
    """Small interpretable bonus for query/expansion term overlap."""
    query_tokens = set(_tokenize_for_retrieval(" ".join(expand_retrieval_queries(question))))
    chunk_tokens = set(_tokenize_for_retrieval(chunk.text))
    if not query_tokens:
        return 0.0
    return min(1.0, len(query_tokens & chunk_tokens) / max(4, len(query_tokens) * 0.22))


def _metadata_adjustment(question: str, chunk: CandidateChunk) -> tuple[float, tuple[str, ...]]:
    """Return transparent metadata bonus/penalty and human-readable reasons."""
    reasons: list[str] = list(chunk.flags)
    adjustment = 0.0
    asks_refs = query_asks_for_references(question)
    asks_figures = query_asks_for_figures(question)

    if chunk.chunk_type == "reference" and not asks_refs:
        adjustment -= 0.55
        reasons.append("downweighted: reference section")
    elif chunk.chunk_type == "reference":
        adjustment += 0.10
        reasons.append("kept: reference query")

    if chunk.chunk_type == "caption" and not asks_figures:
        adjustment -= 0.24
        reasons.append("downweighted: caption-only")
    elif chunk.chunk_type == "caption":
        adjustment += 0.08
        reasons.append("kept: figure/caption query")

    if chunk.chunk_type in {"body_text", "table"}:
        adjustment += 0.08
        reasons.append("body/table evidence")
    if chunk.text_length < 60:
        adjustment -= 0.10
        reasons.append("downweighted: short chunk")
    if "citation-heavy" in chunk.flags:
        adjustment -= 0.12
        reasons.append("downweighted: citation-heavy")

    aspects = detect_aspects(chunk.text)
    if aspects:
        adjustment += 0.04 * min(3, len(aspects))
        reasons.append("science aspect: " + ", ".join(aspects))
    return adjustment, tuple(dict.fromkeys(reasons))


def _raw_result_rows(
    chunks: list[CandidateChunk],
    scores: np.ndarray,
    *,
    label: str,
    limit: int = 12,
) -> list[dict[str, object]]:
    """Format raw retrieval rows for debug display."""
    rows = []
    for idx in np.argsort(scores)[::-1][: min(limit, len(chunks))]:
        chunk = chunks[int(idx)]
        rows.append(
            {
                "rank": len(rows) + 1,
                "source": chunk.title,
                label: round(float(scores[int(idx)]), 4),
                "page": chunk.page_number,
                "type": chunk.chunk_type,
                "dominant_aspect": dominant_aspect(chunk.text),
                "aspects": ", ".join(detect_aspects(chunk.text)),
                "flags": ", ".join(chunk.flags),
                "preview": chunk.text[:220],
            }
        )
    return rows


def _selection_reason(question: str, chunk: CandidateChunk) -> str:
    """Explain why a selected chunk survived final context balancing."""
    if classify_query_intent(question) == "broad_synthesis":
        aspects = detect_aspects(chunk.text)
        if aspects:
            return "included for relevance plus new aspect coverage: " + ", ".join(aspects)
        if is_overview_chunk(chunk):
            return "included as overview/summary-quality context"
    return "included by relevance score"


def _aspect_coverage_summary(selected_chunks: list[CandidateChunk]) -> dict[str, object]:
    """Summarize final context aspect coverage for debug output."""
    coverage: dict[str, list[str]] = {}
    for chunk in selected_chunks:
        for aspect in detect_aspects(chunk.text):
            coverage.setdefault(aspect, []).append(chunk.title)
    return {
        "covered_aspects": sorted(coverage),
        "chunks_by_aspect": coverage,
        "section_titles": sorted({chunk.section_title for chunk in selected_chunks if chunk.section_title}),
    }


def _jaccard_similarity(left: str, right: str) -> float:
    """Cheap semantic redundancy proxy for context selection."""
    left_tokens = set(_tokenize_for_retrieval(left))
    right_tokens = set(_tokenize_for_retrieval(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def is_overview_chunk(chunk: CandidateChunk) -> bool:
    """Detect section-intro/overview/summary style chunks with broad framing."""
    text_tokens = set(_tokenize_for_retrieval(chunk.text))
    section_tokens = set(_tokenize_for_retrieval(chunk.section_title))
    return bool((text_tokens | section_tokens) & BROAD_OVERVIEW_TERMS)


def _query_mentions_aspect(question: str, aspect: str) -> bool:
    """Return True when the user explicitly asks for a detected aspect."""
    query_tokens = set(_tokenize_for_retrieval(question))
    return bool(query_tokens & ASPECT_KEYWORDS.get(aspect, set()))


def _query_concept_coverage(question: str, chunk: CandidateChunk) -> float:
    """Measure how many non-stopword query concepts appear in a chunk."""
    query_tokens = set(_tokenize_for_retrieval(" ".join(expand_retrieval_queries(question))))
    chunk_tokens = set(_tokenize_for_retrieval(chunk.text))
    if not query_tokens:
        return 0.0
    return min(1.0, len(query_tokens & chunk_tokens) / max(3, len(query_tokens) * 0.18))


def _quality_bonus(chunk: CandidateChunk) -> float:
    """Reward chunks that look useful for answer synthesis."""
    bonus = 0.0
    if chunk.chunk_type == "body_text":
        bonus += 0.12
    elif chunk.chunk_type == "table":
        bonus += 0.04
    elif chunk.chunk_type in {"reference", "caption", "heading_only"}:
        bonus -= 0.18
    if is_overview_chunk(chunk):
        bonus += 0.16
    if 70 <= chunk.text_length <= 260:
        bonus += 0.08
    elif chunk.text_length < 50:
        bonus -= 0.10
    if "citation-heavy" in chunk.flags or "reference section" in chunk.flags:
        bonus -= 0.14
    return bonus


def _narrow_subsection_penalty(question: str, chunk: CandidateChunk, intent: str) -> float:
    """Penalize highly specific subsection chunks for broad synthesis queries."""
    if intent != "broad_synthesis":
        return 0.0
    low = chunk.text.lower()
    tokens = set(_tokenize_for_retrieval(chunk.text))
    overview_hits = len(tokens & BROAD_OVERVIEW_TERMS)
    technical_hits = len(tokens & TECHNICAL_TERMS)
    penalty = 0.0
    if re.search(r"\b(goal|objective|task|requirement)\s+\d+[a-z]?\b", low):
        penalty += 0.16
    if technical_hits >= 3 and overview_hits == 0:
        penalty += 0.18
    if technical_hits >= 5 and overview_hits <= 1:
        penalty += 0.12
    dominant = dominant_aspect(chunk.text)
    if dominant != "general" and not _query_mentions_aspect(question, dominant) and overview_hits == 0:
        penalty += 0.12
    return penalty


def _novelty_score(chunk: CandidateChunk, selected_chunks: list[CandidateChunk]) -> float:
    """Reward new sections/aspects and low redundancy with selected context."""
    if not selected_chunks:
        return 1.0
    selected_sections = {selected.section_title for selected in selected_chunks if selected.section_title}
    selected_aspects = {aspect for selected in selected_chunks for aspect in detect_aspects(selected.text)}
    chunk_aspects = set(detect_aspects(chunk.text))
    section_bonus = 0.25 if chunk.section_title and chunk.section_title not in selected_sections else 0.0
    aspect_bonus = 0.35 * min(1.0, len(chunk_aspects - selected_aspects) / max(1, len(chunk_aspects)))
    max_similarity = max(_jaccard_similarity(chunk.text, selected.text) for selected in selected_chunks)
    anti_redundancy = 1.0 - max_similarity
    return float(max(0.0, min(1.0, 0.40 * anti_redundancy + section_bonus + aspect_bonus)))


def _retrieved_chunk_from_candidate(
    source: CandidateChunk,
    *,
    rank: int,
    dense_score: float,
    keyword_score: float,
    rerank_score: float,
    reasons: tuple[str, ...],
) -> RetrievedChunk:
    """Attach retrieval metadata to the chunk consumed by generation and shapiq."""
    title = f"Retrieved {rank}: {source.title}"
    return RetrievedChunk(
        title=title,
        text=source.text,
        page_number=source.page_number,
        chunk_type=source.chunk_type,
        section_title=source.section_title,
        text_length=source.text_length,
        retrieval_score=rerank_score,
        dense_score=dense_score,
        keyword_score=keyword_score,
        rerank_score=rerank_score,
        flags=tuple(dict.fromkeys((*source.flags, *reasons))),
    )


def _select_diverse_context(
    question: str,
    chunks: list[CandidateChunk],
    rerank_scores: np.ndarray,
    *,
    top_k: int,
) -> tuple[list[int], dict[int, dict[str, object]]]:
    """Select final context with intent-aware MMR-style scoring."""
    intent = classify_query_intent(question)
    asks_refs = query_asks_for_references(question)
    asks_figures = query_asks_for_figures(question)
    relevance_scores = _normalize_scores(rerank_scores)
    lambda_weight = 0.55 if intent == "broad_synthesis" else 0.86
    selected: list[int] = []
    components_by_idx: dict[int, dict[str, object]] = {}
    caption_count = 0
    candidate_indices = [int(idx) for idx in np.argsort(rerank_scores)[::-1]]

    def is_allowed(idx: int) -> bool:
        chunk = chunks[idx]
        if chunk.chunk_type == "reference" and not asks_refs:
            return False
        return not (chunk.chunk_type == "caption" and not asks_figures and caption_count >= 1)

    while len(selected) < min(top_k, len(chunks)) and candidate_indices:
        best_idx = None
        best_score = -1e9
        selected_chunks = [chunks[idx] for idx in selected]
        for idx in candidate_indices:
            chunk = chunks[idx]
            if not is_allowed(idx):
                continue
            relevance = float(relevance_scores[idx])
            novelty = _novelty_score(chunk, selected_chunks)
            quality = _quality_bonus(chunk)
            penalty = _narrow_subsection_penalty(question, chunk, intent)
            concept_coverage = _query_concept_coverage(question, chunk)
            score = (
                lambda_weight * relevance
                + (1.0 - lambda_weight) * novelty
                + 0.12 * concept_coverage
                + quality
                - penalty
            )
            components_by_idx[idx] = {
                "query_intent": intent,
                "lambda": round(lambda_weight, 3),
                "relevance": round(relevance, 4),
                "novelty": round(novelty, 4),
                "concept_coverage": round(concept_coverage, 4),
                "quality_bonus": round(quality, 4),
                "narrow_subsection_penalty": round(penalty, 4),
                "final_selection_score": round(float(score), 4),
            }
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            best_idx = candidate_indices[0]
            components_by_idx.setdefault(
                best_idx,
                {
                    "query_intent": intent,
                    "lambda": round(lambda_weight, 3),
                    "relevance": round(float(relevance_scores[best_idx]), 4),
                    "novelty": 0.0,
                    "concept_coverage": 0.0,
                    "quality_bonus": round(_quality_bonus(chunks[best_idx]), 4),
                    "narrow_subsection_penalty": round(
                        _narrow_subsection_penalty(question, chunks[best_idx], intent),
                        4,
                    ),
                    "final_selection_score": round(float(relevance_scores[best_idx]), 4),
                },
            )
        selected.append(best_idx)
        candidate_indices.remove(best_idx)
        if chunks[best_idx].chunk_type == "caption":
            caption_count += 1

    return list(dict.fromkeys(selected)), components_by_idx


def retrieve_relevant_chunks_with_debug(
    question: str,
    chunks: list[CandidateChunk],
    *,
    top_k: int,
    method: str,
    embedding_model_id: str,
    embedding_device: str,
) -> tuple[list[RankedChunk], RetrievalDebugInfo]:
    """Retrieve chunks with query expansion, hybrid scoring, reranking, and debug logs."""
    if not question.strip():
        msg = "Enter a question before running PDF RAG."
        raise ValueError(msg)
    if top_k <= 0:
        msg = "top_k must be positive."
        raise ValueError(msg)

    keyword_scores = _keyword_scores(question, chunks)
    if method == "Dense embeddings":
        dense_scores = _dense_scores(
            question,
            chunks,
            embedding_model_id=embedding_model_id,
            embedding_device=embedding_device,
        )
    else:
        dense_scores = np.zeros(len(chunks), dtype=float)

    dense_norm = _normalize_scores(dense_scores)
    keyword_norm = _normalize_scores(keyword_scores)
    rerank_scores = np.zeros(len(chunks), dtype=float)
    reasons_by_idx: dict[int, tuple[str, ...]] = {}
    for idx, chunk in enumerate(chunks):
        metadata_adjustment, reasons = _metadata_adjustment(question, chunk)
        overlap_bonus = _keyword_overlap_bonus(question, chunk)
        if method == "Dense embeddings":
            base_score = 0.55 * dense_norm[idx] + 0.30 * keyword_norm[idx]
        else:
            base_score = 0.78 * keyword_norm[idx]
        rerank_scores[idx] = base_score + 0.12 * overlap_bonus + metadata_adjustment
        reasons_by_idx[idx] = reasons

    query_intent = classify_query_intent(question)
    selected_indices, selection_components = _select_diverse_context(
        question,
        chunks,
        rerank_scores,
        top_k=top_k,
    )
    ranked: list[RankedChunk] = []
    for rank, idx in enumerate(selected_indices, start=1):
        source = chunks[idx]
        reasons = reasons_by_idx[idx]
        chunk = _retrieved_chunk_from_candidate(
            source,
            rank=rank,
            dense_score=float(dense_scores[idx]),
            keyword_score=float(keyword_scores[idx]),
            rerank_score=float(rerank_scores[idx]),
            reasons=reasons,
        )
        ranked.append(
            RankedChunk(
                chunk=chunk,
                score=float(rerank_scores[idx]),
                dense_score=float(dense_scores[idx]),
                keyword_score=float(keyword_scores[idx]),
                rerank_score=float(rerank_scores[idx]),
                reasons=reasons,
            )
        )

    reranked_rows = []
    raw_rerank_rows = []
    raw_order = [int(idx) for idx in np.argsort(rerank_scores)[::-1]]
    for row_rank, idx in enumerate(raw_order[: min(24, len(chunks))], start=1):
        chunk = chunks[int(idx)]
        row = {
            "rank": row_rank,
            "source": chunk.title,
            "page": chunk.page_number,
            "type": chunk.chunk_type,
            "dense": round(float(dense_scores[int(idx)]), 4),
            "keyword": round(float(keyword_scores[int(idx)]), 4),
            "rerank": round(float(rerank_scores[int(idx)]), 4),
            "dominant_aspect": dominant_aspect(chunk.text),
            "aspects": ", ".join(detect_aspects(chunk.text)),
            "reasons": "; ".join(reasons_by_idx[int(idx)]),
            "preview": chunk.text[:220],
        }
        raw_rerank_rows.append(row)
        reranked_rows.append(row)

    selected_rows = [
        {
            "rank": rank,
            "source": chunks[idx].title,
            "page": chunks[idx].page_number,
            "type": chunks[idx].chunk_type,
            "dense": round(float(dense_scores[idx]), 4),
            "keyword": round(float(keyword_scores[idx]), 4),
            "rerank": round(float(rerank_scores[idx]), 4),
            "dominant_aspect": dominant_aspect(chunks[idx].text),
            "aspects": ", ".join(detect_aspects(chunks[idx].text)),
            "selection_reason": _selection_reason(question, chunks[idx]),
            **selection_components.get(idx, {}),
            "reasons": "; ".join(reasons_by_idx[idx]),
            "preview": chunks[idx].text[:220],
        }
        for rank, idx in enumerate(selected_indices, start=1)
    ]
    selected_chunks = [chunks[idx] for idx in selected_indices]
    debug = RetrievalDebugInfo(
        query_intent=query_intent,
        original_query=question,
        expanded_queries=expand_retrieval_queries(question),
        raw_dense_results=_raw_result_rows(chunks, dense_scores, label="dense"),
        raw_keyword_results=_raw_result_rows(chunks, keyword_scores, label="keyword"),
        raw_rerank_order=raw_rerank_rows,
        reranked_results=reranked_rows,
        selected_context=selected_rows,
        aspect_coverage=_aspect_coverage_summary(selected_chunks),
    )
    return ranked, debug


def retrieve_relevant_chunks_tfidf(
    question: str,
    chunks: list[CandidateChunk],
    *,
    top_k: int,
) -> list[RankedChunk]:
    """Retrieve the top-k chunks with a local sparse TF-IDF retriever."""
    ranked, _debug = retrieve_relevant_chunks_with_debug(
        question,
        chunks,
        top_k=top_k,
        method="TF-IDF",
        embedding_model_id="",
        embedding_device="cpu",
    )
    return ranked


def retrieve_relevant_chunks_dense(
    question: str,
    chunks: list[CandidateChunk],
    *,
    top_k: int,
    embedding_model_id: str,
    embedding_device: str,
) -> list[RankedChunk]:
    """Retrieve top-k chunks with dense embedding cosine similarity."""
    ranked, _debug = retrieve_relevant_chunks_with_debug(
        question,
        chunks,
        top_k=top_k,
        method="Dense embeddings",
        embedding_model_id=embedding_model_id,
        embedding_device=embedding_device,
    )
    return ranked


def retrieve_relevant_chunks(
    question: str,
    chunks: list[CandidateChunk],
    *,
    top_k: int,
    method: str,
    embedding_model_id: str,
    embedding_device: str,
) -> list[RankedChunk]:
    """Retrieve top-k chunks with the configured retriever."""
    ranked, _debug = retrieve_relevant_chunks_with_debug(
        question,
        chunks,
        top_k=top_k,
        method=method,
        embedding_model_id=embedding_model_id,
        embedding_device=embedding_device,
    )
    return ranked


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
    """Use the configured Gemma/HF model to answer from retrieved chunks."""
    backend = HuggingFaceCausalLMBackend(
        model_id=model_id,
        device_map=device_map,
        torch_dtype=torch_dtype,
        max_new_tokens=max_new_tokens,
    )
    context = format_rag_context(chunks)
    result = backend.generate(question, context)
    answer = normalize_whitespace(result.answer)
    if not answer:
        return "I could not generate an answer from the retrieved context."
    return answer
