"""Linguistic segmentation for agentic tool-use prompts."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

try:
    from demos.agentic_tool_use_explanation.semantic_segmenter import validate_partition
except ModuleNotFoundError:
    from semantic_segmenter import validate_partition

SegmentRule = Literal[
    "PREP_PHRASE",
    "DATE_TIME",
    "NOUN_CHUNK",
    "FUNCTION_SHELL",
    "STRAY_MERGE",
    "FALLBACK",
]
StrayMerge = Literal["auto", "left", "right"]

SHELL_POS = {"AUX", "PART", "PRON", "DET", "SCONJ", "CCONJ"}
WH_TAGS = {"WDT", "WP", "WP$", "WRB"}


class _SpacyToken(Protocol):
    idx: int
    text: str
    dep_: str
    pos_: str
    tag_: str
    subtree: Iterable[_SpacyToken]


class _SpacySpan(Protocol):
    start_char: int
    end_char: int
    label_: str


class _SpacyDoc(Protocol):
    ents: Iterable[_SpacySpan]
    noun_chunks: Iterable[_SpacySpan]

    def __iter__(self) -> Iterator[_SpacyToken]: ...


@dataclass(frozen=True)
class WhitespaceToken:
    """A whitespace-delimited token and its character offsets."""

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class CandidateSpan:
    """A candidate or accepted inclusive whitespace-token interval."""

    word_start: int
    word_end: int
    priority: int
    rule: SegmentRule


@dataclass(frozen=True)
class CharacterCandidate:
    """A linguistic candidate expressed as a half-open character range."""

    start: int
    end: int
    priority: int
    rule: SegmentRule


def whitespace_tokens(text: str) -> list[WhitespaceToken]:
    """Return the exact non-whitespace tokens and their character offsets."""
    return [
        WhitespaceToken(match.group(), match.start(), match.end())
        for match in re.finditer(r"\S+", text)
    ]


def snap_candidates_to_words(
    tokens: list[WhitespaceToken],
    candidates: list[CharacterCandidate],
) -> list[CandidateSpan]:
    """Expand character candidates to every overlapping whitespace token."""
    snapped: list[CandidateSpan] = []
    for candidate in candidates:
        touched = [
            index
            for index, token in enumerate(tokens)
            if token.start < candidate.end and token.end > candidate.start
        ]
        if touched:
            snapped.append(
                CandidateSpan(
                    word_start=min(touched),
                    word_end=max(touched),
                    priority=candidate.priority,
                    rule=candidate.rule,
                )
            )
    return snapped


def resolve_overlaps(candidates: list[CandidateSpan]) -> list[CandidateSpan]:
    """Choose non-overlapping candidates by priority, then longest span."""
    accepted: list[CandidateSpan] = []
    occupied: set[int] = set()
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.priority,
            -(item.word_end - item.word_start + 1),
            item.word_start,
            item.word_end,
        ),
    ):
        candidate_words = set(range(candidate.word_start, candidate.word_end + 1))
        if candidate_words.isdisjoint(occupied):
            accepted.append(candidate)
            occupied.update(candidate_words)
    return sorted(accepted, key=lambda item: item.word_start)


def fallback_partition(
    text: str,
    tokens: list[WhitespaceToken] | None = None,
) -> tuple[list[str], list[dict[str, object]]]:
    """Return one segment per whitespace token with fallback diagnostics."""
    tokens = whitespace_tokens(text) if tokens is None else tokens
    segments = [token.text for token in tokens]
    debug_rows = [
        {
            "segment": token.text,
            "rule": "FALLBACK",
            "word_start": index,
            "word_end": index,
        }
        for index, token in enumerate(tokens)
    ]
    validate_partition(text, segments)
    return segments, debug_rows


def assemble_segments(
    text: str,
    tokens: list[WhitespaceToken],
    accepted: list[CandidateSpan],
    shell_words: list[bool],
    *,
    stray_merge: StrayMerge = "auto",
) -> tuple[list[str], list[dict[str, object]]]:
    """Fill stray spans, merge content strays, and assemble a valid partition."""
    if stray_merge not in {"auto", "left", "right"}:
        msg = "stray_merge must be one of: auto, left, right."
        raise ValueError(msg)
    if len(shell_words) != len(tokens):
        msg = "shell_words must contain one classification per whitespace token."
        raise ValueError(msg)
    if not accepted:
        return fallback_partition(text, tokens)

    accepted = [
        CandidateSpan(
            candidate.word_start,
            candidate.word_end,
            candidate.priority,
            (
                "FUNCTION_SHELL"
                if all(shell_words[candidate.word_start : candidate.word_end + 1])
                else candidate.rule
            ),
        )
        for candidate in accepted
    ]
    final_spans: list[CandidateSpan] = []
    cursor = 0
    for accepted_index, candidate in enumerate(accepted):
        current_candidate = candidate
        if cursor < candidate.word_start:
            stray_start = cursor
            stray_end = candidate.word_start - 1
            is_shell = all(shell_words[stray_start : stray_end + 1])
            if is_shell:
                final_spans.append(CandidateSpan(stray_start, stray_end, 0, "FUNCTION_SHELL"))
            elif accepted_index == 0 or stray_merge == "right":
                current_candidate = CandidateSpan(
                    stray_start,
                    candidate.word_end,
                    candidate.priority,
                    "STRAY_MERGE",
                )
            else:
                previous = final_spans[-1]
                final_spans[-1] = CandidateSpan(
                    previous.word_start,
                    stray_end,
                    previous.priority,
                    "STRAY_MERGE",
                )
        final_spans.append(current_candidate)
        cursor = current_candidate.word_end + 1

    if cursor < len(tokens):
        stray_start = cursor
        stray_end = len(tokens) - 1
        is_shell = all(shell_words[stray_start : stray_end + 1])
        if is_shell:
            final_spans.append(CandidateSpan(stray_start, stray_end, 0, "FUNCTION_SHELL"))
        else:
            previous = final_spans[-1]
            final_spans[-1] = CandidateSpan(
                previous.word_start,
                stray_end,
                previous.priority,
                "STRAY_MERGE",
            )

    coalesced_spans: list[CandidateSpan] = []
    for span in final_spans:
        if coalesced_spans and coalesced_spans[-1].rule == span.rule == "FUNCTION_SHELL":
            previous = coalesced_spans[-1]
            coalesced_spans[-1] = CandidateSpan(
                previous.word_start,
                span.word_end,
                min(previous.priority, span.priority),
                "FUNCTION_SHELL",
            )
        else:
            coalesced_spans.append(span)
    final_spans = coalesced_spans

    segments = [
        " ".join(token.text for token in tokens[span.word_start : span.word_end + 1])
        for span in final_spans
    ]
    debug_rows = [
        {
            "segment": segment,
            "rule": span.rule,
            "word_start": span.word_start,
            "word_end": span.word_end,
        }
        for segment, span in zip(segments, final_spans, strict=True)
    ]
    validate_partition(text, segments)
    return segments, debug_rows


def _token_char_range(token: _SpacyToken) -> tuple[int, int]:
    start = int(token.idx)
    return start, start + len(str(token.text))


def _collect_character_candidates(doc: object) -> list[CharacterCandidate]:
    parsed_doc = cast("_SpacyDoc", doc)
    candidates: list[CharacterCandidate] = []
    for token in parsed_doc:
        if token.dep_ != "prep":
            continue
        subtree_ranges = [_token_char_range(child) for child in token.subtree]
        candidates.append(
            CharacterCandidate(
                min(start for start, _ in subtree_ranges),
                max(end for _, end in subtree_ranges),
                1,
                "PREP_PHRASE",
            )
        )

    candidates.extend(
        CharacterCandidate(
            int(entity.start_char),
            int(entity.end_char),
            2,
            "DATE_TIME",
        )
        for entity in parsed_doc.ents
        if entity.label_ in {"DATE", "TIME"}
    )

    candidates.extend(
        CharacterCandidate(
            int(chunk.start_char),
            int(chunk.end_char),
            3,
            "NOUN_CHUNK",
        )
        for chunk in parsed_doc.noun_chunks
    )
    return candidates


def _classify_shell_words(doc: object, tokens: list[WhitespaceToken]) -> list[bool]:
    doc_tokens = list(cast("_SpacyDoc", doc))
    classifications: list[bool] = []
    for token in tokens:
        overlapping = [
            doc_token
            for doc_token in doc_tokens
            if _token_char_range(doc_token)[0] < token.end
            and _token_char_range(doc_token)[1] > token.start
        ]
        classifications.append(
            bool(overlapping)
            and all(
                doc_token.pos_ in SHELL_POS or doc_token.tag_ in WH_TAGS
                for doc_token in overlapping
            )
        )
    return classifications


@dataclass
class LinguisticSegmenter:
    """Split text into linguistic signal units using spaCy chunking."""

    model_id: str = "en_core_web_sm"
    stray_merge: StrayMerge = "auto"

    def __post_init__(self) -> None:
        if self.stray_merge not in {"auto", "left", "right"}:
            msg = "stray_merge must be one of: auto, left, right."
            raise ValueError(msg)
        try:
            spacy = importlib.import_module("spacy")
            self.nlp = spacy.load(self.model_id)
        except Exception as error:
            msg = (
                f"Could not load spaCy model {self.model_id!r}. Install it with "
                "`python -m spacy download en_core_web_sm`."
            )
            raise RuntimeError(msg) from error

    def segment(self, text: str) -> list[str]:
        """Return linguistic segments preserving every whitespace token exactly once."""
        segments, _ = self.segment_with_debug(text)
        return segments

    def segment_with_debug(self, text: str) -> tuple[list[str], list[dict[str, object]]]:
        """Return linguistic segments plus the rule used for every final segment."""
        tokens = whitespace_tokens(text)
        try:
            doc = self.nlp(text)
            candidates = snap_candidates_to_words(
                tokens,
                _collect_character_candidates(doc),
            )
            accepted = resolve_overlaps(candidates)
            if not accepted:
                return fallback_partition(text, tokens)
            shell_words = _classify_shell_words(doc, tokens)
        except Exception:  # noqa: BLE001 - any parse failure must use the safe fallback
            return fallback_partition(text, tokens)

        return assemble_segments(
            text,
            tokens,
            accepted,
            shell_words,
            stray_merge=self.stray_merge,
        )
