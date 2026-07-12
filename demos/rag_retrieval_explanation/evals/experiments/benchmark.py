"""Load and validate the controlled retrieval benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from demos.rag_retrieval_explanation.core.schemas import CandidateChunk, RetrievedChunk

DEFAULT_BENCHMARK_PATH = Path(__file__).parents[1] / "cases" / "controlled_benchmark.json"
PASSAGE_ID_SEPARATOR = " | "


@dataclass(frozen=True)
class BenchmarkPassage:
    """One passage in the corpus searched by the retriever."""

    passage_id: str
    document_id: str
    title: str
    text: str


@dataclass(frozen=True)
class InteractionLabel:
    """One expected pairwise interaction between controlled passages."""

    left: str
    right: str
    relation: str
    expected_sign: str


@dataclass(frozen=True)
class BenchmarkCase:
    """One question with independently declared answer and evidence labels."""

    case_id: str
    pattern: str
    question: str
    gold_answer: str
    gold_evidence_ids: tuple[str, ...]
    expected_interactions: tuple[InteractionLabel, ...]
    top_k: int


@dataclass(frozen=True)
class ControlledBenchmark:
    """Validated controlled corpus and its question-answer cases."""

    passages: tuple[BenchmarkPassage, ...]
    cases: tuple[BenchmarkCase, ...]


def candidate_title(passage: BenchmarkPassage) -> str:
    """Encode the stable passage ID in a human-readable retrieval title."""
    return f"{passage.passage_id}{PASSAGE_ID_SEPARATOR}{passage.title}"


def passage_id_from_title(title: str) -> str:
    """Recover a stable passage ID from candidate or retrieved titles."""
    unprefixed = title.split(": ", maxsplit=1)[-1] if title.startswith("Retrieved ") else title
    return unprefixed.split(PASSAGE_ID_SEPARATOR, maxsplit=1)[0]


def _expected_sign_for_relation(relation: str) -> str:
    """Return the default expected interaction sign for a relation label."""
    if relation == "complementary":
        return "positive"
    if relation in {"redundant", "conflicting"}:
        return "negative"
    msg = f"Unknown interaction relation: {relation}"
    raise ValueError(msg)


def _parse_interactions(raw: list[dict[str, str]]) -> tuple[InteractionLabel, ...]:
    interactions = []
    for item in raw:
        relation = str(item["relation"])
        expected_sign = str(item.get("expected_sign", _expected_sign_for_relation(relation)))
        if expected_sign not in {"positive", "negative"}:
            msg = f"Invalid expected interaction sign: {expected_sign}"
            raise ValueError(msg)
        interactions.append(
            InteractionLabel(
                left=str(item["left"]),
                right=str(item["right"]),
                relation=relation,
                expected_sign=expected_sign,
            )
        )
    return tuple(interactions)


def load_benchmark(path: Path = DEFAULT_BENCHMARK_PATH) -> ControlledBenchmark:
    """Load a benchmark JSON file and reject inconsistent evidence labels."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    passages = tuple(
        BenchmarkPassage(
            passage_id=str(item["id"]),
            document_id=str(item["document_id"]),
            title=str(item["title"]),
            text=str(item["text"]),
        )
        for item in payload["passages"]
    )
    cases = tuple(
        BenchmarkCase(
            case_id=str(item["id"]),
            pattern=str(item["pattern"]),
            question=str(item["question"]),
            gold_answer=str(item["gold_answer"]),
            gold_evidence_ids=tuple(str(value) for value in item["gold_evidence_ids"]),
            expected_interactions=_parse_interactions(item.get("expected_interactions", [])),
            top_k=int(item.get("top_k", 6)),
        )
        for item in payload["cases"]
    )

    passage_ids = [passage.passage_id for passage in passages]
    if len(passage_ids) != len(set(passage_ids)):
        msg = "Benchmark passage IDs must be unique."
        raise ValueError(msg)
    case_ids = [case.case_id for case in cases]
    if len(case_ids) != len(set(case_ids)):
        msg = "Benchmark case IDs must be unique."
        raise ValueError(msg)

    known_ids = set(passage_ids)
    for case in cases:
        unknown = set(case.gold_evidence_ids) - known_ids
        for interaction in case.expected_interactions:
            unknown.update({interaction.left, interaction.right} - known_ids)
        if unknown:
            msg = f"Case {case.case_id} references unknown passages: {sorted(unknown)}"
            raise ValueError(msg)
    return ControlledBenchmark(passages=passages, cases=cases)


def benchmark_candidates(benchmark: ControlledBenchmark) -> list[CandidateChunk]:
    """Convert every corpus passage into a candidate searched by the real retriever."""
    document_numbers: dict[str, int] = {}
    candidates = []
    for chunk_number, passage in enumerate(benchmark.passages, start=1):
        page_number = document_numbers.setdefault(passage.document_id, len(document_numbers) + 1)
        candidates.append(
            CandidateChunk(
                title=candidate_title(passage),
                text=passage.text,
                page_number=page_number,
                chunk_number=chunk_number,
                section_title=passage.document_id,
                text_length=len(passage.text.split()),
            )
        )
    return candidates


def gold_chunks(
    benchmark: ControlledBenchmark,
    case: BenchmarkCase,
) -> list[RetrievedChunk]:
    """Return independently labelled evidence, without running retrieval."""
    by_id = {passage.passage_id: passage for passage in benchmark.passages}
    return [
        RetrievedChunk(
            title=candidate_title(by_id[passage_id]),
            text=by_id[passage_id].text,
            section_title=by_id[passage_id].document_id,
            text_length=len(by_id[passage_id].text.split()),
        )
        for passage_id in case.gold_evidence_ids
    ]
