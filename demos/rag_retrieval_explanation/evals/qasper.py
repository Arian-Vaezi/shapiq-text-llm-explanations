"""Download, select, and adapt QASPER cases for retrieval evaluation."""

from __future__ import annotations

import json
import re
import unicodedata
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from demos.rag_retrieval_explanation.core.schemas import CandidateChunk, RetrievedChunk

from .benchmark import PASSAGE_ID_SEPARATOR

DATASET_ID = "allenai/qasper"
DATASET_CONFIG = "qasper"
DEFAULT_SPLIT = "validation"
DEFAULT_CASE_COUNT = 30
DEFAULT_CACHE_PATH = Path(__file__).parent / "cache" / "qasper_validation.json"
DEFAULT_SELECTION_PATH = Path(__file__).parent / "cases" / "qasper_validation_30.json"
DATASET_ROWS_URL = "https://datasets-server.huggingface.co/rows"
DATASET_INFO_URL = "https://huggingface.co/api/datasets/allenai/qasper"


@dataclass(frozen=True)
class QasperPassage:
    """One paragraph-level retrieval candidate from a QASPER paper."""

    passage_id: str
    section_title: str
    text: str


@dataclass(frozen=True)
class QasperCase:
    """One frozen QASPER question with one answer annotation."""

    case_id: str
    paper_id: str
    paper_title: str
    question_id: str
    annotation_id: str
    question: str
    gold_answer: str
    answer_type: str
    gold_evidence_ids: tuple[str, ...]
    passages: tuple[QasperPassage, ...]
    top_k: int = 6


def _normalize(text: str) -> str:
    """Normalize whitespace and Unicode for stable evidence matching."""
    normalized = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", normalized).strip()


def _fetch_json(url: str) -> dict[str, Any]:
    if not url.startswith(("https://datasets-server.huggingface.co/", "https://huggingface.co/")):
        msg = f"Refusing unexpected QASPER data URL: {url}"
        raise ValueError(msg)
    request = urllib.request.Request(  # noqa: S310
        url,
        headers={"User-Agent": "rag-shapley-qasper-evaluation/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return json.load(response)


def download_qasper_validation(
    cache_path: Path = DEFAULT_CACHE_PATH,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Download the official QASPER validation split through Dataset Viewer."""
    if cache_path.is_file() and not force:
        return json.loads(cache_path.read_text(encoding="utf-8"))

    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 100
    total = None
    while total is None or offset < total:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET_ID,
                "config": DATASET_CONFIG,
                "split": DEFAULT_SPLIT,
                "offset": offset,
                "length": page_size,
            }
        )
        payload = _fetch_json(f"{DATASET_ROWS_URL}?{query}")
        page_rows = [item["row"] for item in payload["rows"]]
        rows.extend(page_rows)
        total = int(payload["num_rows_total"])
        offset += len(page_rows)
        if not page_rows:
            break

    dataset_info = _fetch_json(DATASET_INFO_URL)
    cache_payload = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "split": DEFAULT_SPLIT,
        "source_revision": dataset_info.get("sha"),
        "downloaded_at": datetime.now(tz=UTC).isoformat(),
        "license": "CC BY 4.0",
        "rows": rows,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache_payload), encoding="utf-8")
    return cache_payload


def _paper_passages(row: dict[str, Any]) -> tuple[QasperPassage, ...]:
    paper_id = str(row["id"])
    passages: list[QasperPassage] = []
    abstract = row.get("abstract", [])
    abstract_parts = abstract if isinstance(abstract, list) else [abstract]
    for index, text in enumerate(abstract_parts):
        normalized = _normalize(str(text))
        if normalized:
            passages.append(
                QasperPassage(
                    passage_id=f"{paper_id}:abstract:{index:03d}",
                    section_title="Abstract",
                    text=normalized,
                )
            )

    full_text = row["full_text"]
    chunk_number = 0
    for section, paragraphs in zip(
        full_text["section_name"],
        full_text["paragraphs"],
        strict=True,
    ):
        for paragraph in paragraphs:
            normalized = _normalize(str(paragraph))
            if not normalized:
                continue
            passages.append(
                QasperPassage(
                    passage_id=f"{paper_id}:paragraph:{chunk_number:04d}",
                    section_title=_normalize(str(section)),
                    text=normalized,
                )
            )
            chunk_number += 1
    return tuple(passages)


def _map_evidence(
    evidence_texts: list[str],
    passages: tuple[QasperPassage, ...],
) -> tuple[str, ...]:
    """Map QASPER evidence strings back to paragraph-level passage IDs."""
    normalized_passages = {passage.passage_id: _normalize(passage.text) for passage in passages}
    matches: list[str] = []
    for raw_evidence in evidence_texts:
        evidence = _normalize(str(raw_evidence))
        if not evidence:
            continue
        exact = [
            passage_id
            for passage_id, paragraph in normalized_passages.items()
            if paragraph == evidence
        ]
        if exact:
            matches.extend(exact)
            continue
        containment = [
            passage_id
            for passage_id, paragraph in normalized_passages.items()
            if len(evidence) >= 80
            and len(paragraph) >= 80
            and (evidence in paragraph or paragraph in evidence)
        ]
        matches.extend(containment)
    return tuple(dict.fromkeys(matches))


def _evidence_mapping_detail(
    evidence_texts: list[str],
    passages: tuple[QasperPassage, ...],
) -> dict[str, object]:
    """Return mapping counts and short unmapped evidence examples."""
    mapped = _map_evidence(evidence_texts, passages)
    normalized_passages = {passage.passage_id: _normalize(passage.text) for passage in passages}
    unmapped = []
    for raw_evidence in evidence_texts:
        evidence = _normalize(str(raw_evidence))
        if not evidence:
            continue
        evidence_matches = [
            passage_id
            for passage_id, paragraph in normalized_passages.items()
            if paragraph == evidence
            or (
                len(evidence) >= 80
                and len(paragraph) >= 80
                and (evidence in paragraph or paragraph in evidence)
            )
        ]
        if not evidence_matches:
            unmapped.append(evidence[:240])
    return {
        "mapped_passage_ids": mapped,
        "mapped_evidence_count": len(mapped),
        "unmapped_evidence_count": len(unmapped),
        "unmapped_examples": unmapped[:3],
    }


def _answer_text(annotation: dict[str, Any]) -> tuple[str, str] | None:
    if bool(annotation.get("unanswerable")) or annotation.get("yes_no") is not None:
        return None
    free_form = _normalize(str(annotation.get("free_form_answer", "")))
    if free_form:
        answer = free_form
        answer_type = "free_form"
    else:
        spans = [_normalize(str(span)) for span in annotation.get("extractive_spans", [])]
        spans = [span for span in spans if span]
        if not spans:
            return None
        answer = "; ".join(dict.fromkeys(spans))
        answer_type = "extractive"
    word_count = len(answer.split())
    if not 2 <= word_count <= 40:
        return None
    answer_without_reference_markers = re.sub(
        r"\b(?:BIB|TAB|FIG)REF\d+\b",
        "",
        answer,
        flags=re.IGNORECASE,
    )
    if len(re.findall(r"[A-Za-z]{2,}", answer_without_reference_markers)) < 2:
        return None
    return answer, answer_type


def _case_for_question(
    row: dict[str, Any],
    question_index: int,
    passages: tuple[QasperPassage, ...],
    *,
    required_annotation_id: str | None = None,
) -> QasperCase | None:
    qas = row["qas"]
    answers = qas["answers"][question_index]
    for annotation, raw_annotation_id in zip(
        answers["answer"],
        answers["annotation_id"],
        strict=True,
    ):
        annotation_id = str(raw_annotation_id)
        if required_annotation_id is not None and annotation_id != required_annotation_id:
            continue
        answer = _answer_text(annotation)
        evidence_ids = _map_evidence(annotation.get("evidence", []), passages)
        if answer is None or not evidence_ids:
            continue
        answer_text, answer_type = answer
        paper_id = str(row["id"])
        question_id = str(qas["question_id"][question_index])
        return QasperCase(
            case_id=f"qasper_{question_id}",
            paper_id=paper_id,
            paper_title=_normalize(str(row["title"])),
            question_id=question_id,
            annotation_id=annotation_id,
            question=_normalize(str(qas["question"][question_index])),
            gold_answer=answer_text,
            answer_type=answer_type,
            gold_evidence_ids=evidence_ids,
            passages=passages,
        )
    return None


def select_qasper_cases(
    cache_payload: dict[str, Any],
    *,
    limit: int = DEFAULT_CASE_COUNT,
) -> list[QasperCase]:
    """Select one deterministic, answerable, evidence-mappable case per paper."""
    selected: list[QasperCase] = []
    for row in cache_payload["rows"]:
        passages = _paper_passages(row)
        for question_index in range(len(row["qas"]["question"])):
            case = _case_for_question(row, question_index, passages)
            if case is not None:
                selected.append(case)
                break
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        msg = f"Only {len(selected)} eligible QASPER cases were found; requested {limit}."
        raise RuntimeError(msg)
    return selected


def qasper_selection_diagnostics(cache_payload: dict[str, Any]) -> dict[str, Any]:
    """Summarize why QASPER annotations are eligible or excluded."""
    diagnostics: dict[str, Any] = {
        "papers_seen": 0,
        "questions_seen": 0,
        "annotations_seen": 0,
        "eligible_annotations": 0,
        "excluded_unanswerable_or_yes_no": 0,
        "excluded_answer_shape": 0,
        "excluded_unmapped_evidence": 0,
        "unmapped_evidence_examples": [],
    }
    for row in cache_payload["rows"]:
        diagnostics["papers_seen"] += 1
        passages = _paper_passages(row)
        qas = row["qas"]
        diagnostics["questions_seen"] += len(qas["question"])
        for answers in qas["answers"]:
            for annotation in answers["answer"]:
                diagnostics["annotations_seen"] += 1
                if bool(annotation.get("unanswerable")) or annotation.get("yes_no") is not None:
                    diagnostics["excluded_unanswerable_or_yes_no"] += 1
                    continue
                if _answer_text(annotation) is None:
                    diagnostics["excluded_answer_shape"] += 1
                    continue
                detail = _evidence_mapping_detail(annotation.get("evidence", []), passages)
                if not detail["mapped_passage_ids"]:
                    diagnostics["excluded_unmapped_evidence"] += 1
                    examples = diagnostics["unmapped_evidence_examples"]
                    if len(examples) < 10:
                        examples.append(
                            {
                                "paper_id": row["id"],
                                "unmapped_examples": detail["unmapped_examples"],
                            }
                        )
                    continue
                diagnostics["eligible_annotations"] += 1
    return diagnostics


def write_selection_manifest(
    cache_payload: dict[str, Any],
    cases: list[QasperCase],
    path: Path = DEFAULT_SELECTION_PATH,
) -> None:
    """Freeze case and annotation IDs without copying QASPER text into git."""
    payload = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "split": DEFAULT_SPLIT,
        "source_revision": cache_payload.get("source_revision"),
        "license": cache_payload.get("license"),
        "selection_rules": {
            "order": "Dataset Viewer validation row order, then question and annotation order",
            "paper_limit": "At most one question per paper",
            "answers": "Answerable free-form or extractive annotations with 2-40 words",
            "evidence": "At least one evidence string maps to a paper paragraph",
            "retrieval_unit": "QASPER abstract or full-text paragraph",
            "top_k": 6,
        },
        "selection_diagnostics": qasper_selection_diagnostics(cache_payload),
        "cases": [
            {
                "case_id": case.case_id,
                "paper_id": case.paper_id,
                "question_id": case.question_id,
                "annotation_id": case.annotation_id,
            }
            for case in cases
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_frozen_qasper_cases(
    cache_path: Path = DEFAULT_CACHE_PATH,
    selection_path: Path = DEFAULT_SELECTION_PATH,
) -> tuple[dict[str, Any], list[QasperCase]]:
    """Load the frozen case IDs and rebuild their text from the local cache."""
    cache_payload = download_qasper_validation(cache_path)
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    rows_by_id = {str(row["id"]): row for row in cache_payload["rows"]}
    cases: list[QasperCase] = []
    for item in selection["cases"]:
        row = rows_by_id[str(item["paper_id"])]
        passages = _paper_passages(row)
        question_ids = [str(value) for value in row["qas"]["question_id"]]
        question_index = question_ids.index(str(item["question_id"]))
        case = _case_for_question(
            row,
            question_index,
            passages,
            required_annotation_id=str(item["annotation_id"]),
        )
        if case is None:
            msg = f"Frozen QASPER case can no longer be reconstructed: {item}"
            raise RuntimeError(msg)
        cases.append(case)
    return selection, cases


def qasper_candidates(case: QasperCase) -> list[CandidateChunk]:
    """Convert one paper's paragraphs into candidates for the real retriever."""
    return [
        CandidateChunk(
            title=(
                f"{passage.passage_id}{PASSAGE_ID_SEPARATOR}"
                f"{case.paper_title} / {passage.section_title}"
            ),
            text=passage.text,
            page_number=1,
            chunk_number=index,
            section_title=passage.section_title,
            text_length=len(passage.text.split()),
        )
        for index, passage in enumerate(case.passages, start=1)
    ]


def qasper_gold_chunks(case: QasperCase) -> list[RetrievedChunk]:
    """Return the independently annotated evidence paragraphs."""
    by_id = {passage.passage_id: passage for passage in case.passages}
    return [
        RetrievedChunk(
            title=(
                f"{passage_id}{PASSAGE_ID_SEPARATOR}"
                f"{case.paper_title} / {by_id[passage_id].section_title}"
            ),
            text=by_id[passage_id].text,
            section_title=by_id[passage_id].section_title,
            text_length=len(by_id[passage_id].text.split()),
        )
        for passage_id in case.gold_evidence_ids
    ]


def case_metadata(case: QasperCase) -> dict[str, Any]:
    """Return compact serializable case metadata for run artifacts."""
    payload = asdict(case)
    payload.pop("passages")
    return payload
