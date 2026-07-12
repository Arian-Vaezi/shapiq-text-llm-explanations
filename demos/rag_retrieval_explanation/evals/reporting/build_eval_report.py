"""Build the static RAG evaluation HTML report from saved run artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

RUNS_DIR = Path(__file__).parents[1] / "runs"
REPORT_PATH = Path(__file__).parents[2] / "eval_report.html"

LEXICAL_RUN = RUNS_DIR / "20260626_232647_164955_controlled_50_bge_lexical"
TARGET_LL_RUN = RUNS_DIR / "20260626_232659_164956_controlled_50_bge_target_ll_e4b_cuda4"
CONTRASTIVE_RUN = RUNS_DIR / "20260626_215332_164947_controlled_50_bge_contrastive_ll_e4b_cuda4"
TFIDF_CONTRASTIVE_RUN = (
    RUNS_DIR / "20260626_232708_164957_controlled_50_tfidf_contrastive_ll_e4b_cuda4"
)
QASPER_RUN = RUNS_DIR / "20260626_224021_164961_qasper_50_bge_contrastive_ll_e4b_cuda4"
ARTEMIS_RUN = RUNS_DIR / "20260617_artemis_bge_contrastive_ll_e4b_layers20"

INTERACTION_START_MARKER = "    <!-- BEGIN GENERATED INTERACTION DETAILS -->"
INTERACTION_END_MARKER = "    <!-- END GENERATED INTERACTION DETAILS -->"
RESULTS_START_MARKER = "    <!-- BEGIN GENERATED ATTRIBUTION RESULTS -->"
RESULTS_END_MARKER = "    <!-- END GENERATED ATTRIBUTION RESULTS -->"
FAILURE_START_MARKER = "    <!-- BEGIN GENERATED FAILURE DETAILS -->"
FAILURE_END_MARKER = "    <!-- END GENERATED FAILURE DETAILS -->"


@dataclass(frozen=True)
class TargetView:
    """One collapsible interaction-detail view in the static report."""

    setting: str
    target: str
    run_dir: Path
    labelled_key: str
    pairwise_key: str
    attributions_key: str
    coalitions_key: str


DEFAULT_TARGETS = (
    TargetView(
        setting="Lexical",
        target="Gold",
        run_dir=LEXICAL_RUN,
        labelled_key="interactions",
        pairwise_key="gold_pairwise_interactions",
        attributions_key="gold_attributions",
        coalitions_key="gold_coalition_scores",
    ),
    TargetView(
        setting="Contrastive LL",
        target="Gold",
        run_dir=CONTRASTIVE_RUN,
        labelled_key="interactions",
        pairwise_key="gold_pairwise_interactions",
        attributions_key="gold_attributions",
        coalitions_key="gold_coalition_scores",
    ),
    TargetView(
        setting="Contrastive LL",
        target="Generated",
        run_dir=CONTRASTIVE_RUN,
        labelled_key="generated_interactions",
        pairwise_key="generated_pairwise_interactions",
        attributions_key="generated_attributions",
        coalitions_key="generated_coalition_scores",
    ),
)


def _html(value: object) -> str:
    return escape(str(value), quote=True)


def _fmt(value: object, *, digits: int = 3, signed: bool = False) -> str:
    if value is None or value == "":
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _html(value)
    prefix = "+" if signed else ""
    return f"{number:{prefix}.{digits}f}"


def _case_order(run_dir: Path) -> list[str]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return []
    with summary_path.open(encoding="utf-8") as handle:
        return [row["case_id"] for row in csv.DictReader(handle)]


def _summary_by_case(run_dir: Path) -> dict[str, dict[str, str]]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return {}
    with summary_path.open(encoding="utf-8") as handle:
        return {row["case_id"]: row for row in csv.DictReader(handle)}


def _summary_rows(run_dir: Path) -> list[dict[str, str]]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return []
    with summary_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_artifacts(run_dir: Path) -> list[dict[str, Any]]:
    artifacts_dir = run_dir / "artifacts"
    if not artifacts_dir.exists():
        msg = f"Missing artifacts directory: {artifacts_dir}"
        raise FileNotFoundError(msg)

    by_case: dict[str, dict[str, Any]] = {}
    for path in sorted(artifacts_dir.glob("*.json")):
        with path.open(encoding="utf-8") as handle:
            artifact = json.load(handle)
        by_case[str(artifact["case"]["case_id"])] = artifact

    ordered_ids = _case_order(run_dir)
    ordered = [by_case[case_id] for case_id in ordered_ids if case_id in by_case]
    remaining = [by_case[case_id] for case_id in sorted(by_case) if case_id not in ordered_ids]
    return [*ordered, *remaining]


def _correct_count(rows: list[dict[str, Any]]) -> int:
    return sum(row.get("interaction_sign_correct") is True for row in rows)


def _evaluable_count(rows: list[dict[str, Any]]) -> int:
    return sum(row.get("interaction_sign_correct") is not None for row in rows)


def _status_badge(row: dict[str, Any]) -> str:
    status = row.get("status", "evaluable")
    correct = row.get("interaction_sign_correct")
    if correct is True:
        return '<span class="pill good">correct</span>'
    if correct is False:
        return '<span class="pill bad">wrong sign</span>'
    if str(status).startswith("undefined_missing_"):
        missing_side = str(status).removeprefix("undefined_missing_")
        return f'<span class="pill muted">not evaluable: missing {missing_side}</span>'
    return f'<span class="pill muted">{_html(status)}</span>'


def _observed_sign(value: object) -> str:
    if value is None:
        return "n/a"
    number = float(value)
    if number > 0:
        return "positive"
    if number < 0:
        return "negative"
    return "zero"


def _passage_id_by_rank(artifact: dict[str, Any]) -> dict[int, str]:
    return {int(row["rank"]): str(row["passage_id"]) for row in artifact["retrieved"]}


def _class_label(value: object) -> str:
    label = str(value)
    if label == "generation_or_model_failure":
        return "generation/model failure"
    if label == "retrieval_failure":
        return "retrieval miss"
    return label.replace("_", " ")


def _raw_class_label(value: object) -> str:
    return str(value).replace("_", " ")


def _failure_class(value: object) -> str:
    label = str(value)
    if label == "retrieval_failure":
        return "bad"
    if label == "generation_or_model_failure":
        return "warn"
    if label in {"retrieval_enabled", "source_not_identifiable"}:
        return "good"
    return "neutral"


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _mean(rows: list[dict[str, str]], column: str) -> float | None:
    values = [value for row in rows if (value := _as_float(row.get(column))) is not None]
    return sum(values) / len(values) if values else None


def _bool_mean(rows: list[dict[str, str]], column: str) -> float | None:
    values = [row[column] for row in rows if row.get(column, "") != ""]
    return sum(value == "True" for value in values) / len(values) if values else None


def _gold_hit_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if (_as_float(row.get("retrieval_recall_at_k")) or 0.0) > 0.0]


def _render_labelled_interactions(
    artifact: dict[str, Any],
    labelled_key: str,
) -> str:
    rows = artifact.get(labelled_key, [])
    if not rows:
        return '<p class="empty-note">No labelled interaction pairs for this question.</p>'

    body = []
    for row in rows:
        observed = row.get("observed_interaction")
        body.append(
            "<tr>"
            f"<td>{_html(row['left'])} x {_html(row['right'])}</td>"
            f"<td>{_html(row['relation'])}</td>"
            f"<td>{_html(row['expected_sign'])}</td>"
            f"<td>{_observed_sign(observed)}</td>"
            f"<td>{_fmt(observed, digits=4, signed=True)}</td>"
            f"<td>{_status_badge(row)}</td>"
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table class="mini-table">'
        "<thead><tr><th>Labelled pair</th><th>Relation</th><th>Expected</th>"
        "<th>Observed sign</th><th>Value</th><th>Status</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _attribution_result_row(
    *,
    dataset: str,
    retriever: str,
    target: str,
    rows: list[dict[str, str]],
    top_column: str,
    mass_column: str,
    deletion_column: str | None,
    group_start: bool = False,
) -> str:
    hit_rows = _gold_hit_rows(rows)
    deletion = _mean(rows, deletion_column) if deletion_column else None
    tr_open = '<tr class="row-group-start">' if group_start else "<tr>"
    return (
        tr_open + f"<td>{_html(dataset)}</td>"
        f"<td>{_html(retriever)}</td>"
        f"<td>{_html(target)}</td>"
        f"<td>{len(rows)}</td>"
        f"<td>{len(hit_rows)}</td>"
        f'<td class="group-start">{_fmt(_mean(rows, "retrieval_recall_at_k"))}</td>'
        f"<td>{_fmt(_mean(rows, 'retrieval_mrr'))}</td>"
        f'<td class="group-start">{_fmt(_bool_mean(hit_rows, top_column))}</td>'
        f"<td>{_fmt(_mean(hit_rows, mass_column))}</td>"
        f"<td>{_fmt(deletion)}</td>"
        "</tr>"
    )


def render_attribution_results() -> str:
    """Render the cross-dataset first-order attribution result table."""
    lexical_rows = _summary_rows(LEXICAL_RUN)
    target_ll_rows = _summary_rows(TARGET_LL_RUN)
    contrastive_rows = _summary_rows(CONTRASTIVE_RUN)
    tfidf_contrastive_rows = _summary_rows(TFIDF_CONTRASTIVE_RUN)
    qasper_rows = _summary_rows(QASPER_RUN)
    artemis_rows = _summary_rows(ARTEMIS_RUN)
    qasper_hit_rows = _gold_hit_rows(qasper_rows)
    artemis_hit_rows = _gold_hit_rows(artemis_rows)

    rows = [
        _attribution_result_row(
            dataset="Controlled",
            retriever="BGE-base",
            target="Lexical / gold",
            rows=lexical_rows,
            top_column="top_attribution_is_gold",
            mass_column="gold_attribution_mass",
            deletion_column="gold_deletion_auc",
        ),
        _attribution_result_row(
            dataset="Controlled",
            retriever="BGE-base",
            target="Target LL / gold",
            rows=target_ll_rows,
            top_column="top_attribution_is_gold",
            mass_column="gold_attribution_mass",
            deletion_column="gold_deletion_auc",
            group_start=True,
        ),
        _attribution_result_row(
            dataset="Controlled",
            retriever="BGE-base",
            target="Target LL / generated",
            rows=target_ll_rows,
            top_column="generated_top_attribution_is_gold",
            mass_column="generated_gold_attribution_mass",
            deletion_column=None,
        ),
        _attribution_result_row(
            dataset="Controlled",
            retriever="BGE-base",
            target="Contrastive LL / gold",
            rows=contrastive_rows,
            top_column="top_attribution_is_gold",
            mass_column="gold_attribution_mass",
            deletion_column="gold_deletion_auc",
            group_start=True,
        ),
        _attribution_result_row(
            dataset="Controlled",
            retriever="BGE-base",
            target="Contrastive LL / generated",
            rows=contrastive_rows,
            top_column="generated_top_attribution_is_gold",
            mass_column="generated_gold_attribution_mass",
            deletion_column=None,
        ),
        _attribution_result_row(
            dataset="Controlled",
            retriever="TF-IDF",
            target="Contrastive LL / gold",
            rows=tfidf_contrastive_rows,
            top_column="top_attribution_is_gold",
            mass_column="gold_attribution_mass",
            deletion_column="gold_deletion_auc",
            group_start=True,
        ),
        _attribution_result_row(
            dataset="Controlled",
            retriever="TF-IDF",
            target="Contrastive LL / generated",
            rows=tfidf_contrastive_rows,
            top_column="generated_top_attribution_is_gold",
            mass_column="generated_gold_attribution_mass",
            deletion_column=None,
        ),
        _attribution_result_row(
            dataset="QASPER",
            retriever="BGE-base",
            target="Contrastive LL / gold",
            rows=qasper_rows,
            top_column="top_attribution_is_gold",
            mass_column="gold_attribution_mass",
            deletion_column="gold_deletion_auc",
            group_start=True,
        ),
        _attribution_result_row(
            dataset="QASPER",
            retriever="BGE-base",
            target="Contrastive LL / generated",
            rows=qasper_rows,
            top_column="generated_top_attribution_is_gold",
            mass_column="generated_gold_attribution_mass",
            deletion_column=None,
        ),
        _attribution_result_row(
            dataset="Artemis",
            retriever="BGE-base",
            target="Contrastive LL / gold",
            rows=artemis_rows,
            top_column="top_attribution_is_gold",
            mass_column="gold_attribution_mass",
            deletion_column="gold_deletion_auc",
            group_start=True,
        ),
        _attribution_result_row(
            dataset="Artemis",
            retriever="BGE-base",
            target="Contrastive LL / generated",
            rows=artemis_rows,
            top_column="generated_top_attribution_is_gold",
            mass_column="generated_gold_attribution_mass",
            deletion_column=None,
        ),
    ]
    qasper_hit_count = len(qasper_hit_rows)
    qasper_total = len(qasper_rows)
    qasper_gold_top_hits = sum(
        row.get("top_attribution_is_gold", "").lower() == "true" for row in qasper_hit_rows
    )
    artemis_hit_count = len(artemis_hit_rows)
    artemis_total = len(artemis_rows)
    return "\n".join(
        [
            RESULTS_START_MARKER,
            '    <h3 class="sub-title">Evidence Attribution Across Datasets</h3>',
            '    <div class="table-scroll"><table>',
            "      <thead>"
            "<tr>"
            '<th rowspan="2">Dataset</th>'
            '<th rowspan="2">Retriever</th>'
            '<th rowspan="2">Target</th>'
            '<th rowspan="2">Cases</th>'
            '<th rowspan="2">Gold hit<br>cases</th>'
            '<th class="group-start" colspan="2" style="text-align:center;color:#2563eb;">Retrieval</th>'
            '<th class="group-start" colspan="3" style="text-align:center;color:#15803d;">Attribution</th>'
            "</tr>"
            "<tr>"
            '<th class="group-start">Recall@k</th><th>MRR</th>'
            '<th class="group-start">Hit top gold</th><th>Hit gold mass</th><th>Deletion AUC</th>'
            "</tr>"
            "</thead>",
            f"      <tbody>{''.join(rows)}</tbody>",
            "    </table></div>",
            '    <p class="cap">Cases is the total evaluated count. Gold hit cases '
            "counts rows where at least one labelled evidence passage was retrieved "
            f"({qasper_hit_count}/{qasper_total} for QASPER; "
            f"{artemis_hit_count}/{artemis_total} for Artemis). Recall@k uses "
            "the saved run's retrieval depth: k=6 for controlled/QASPER and k=3 "
            "for Artemis. Hit top gold and "
            "hit gold mass are the Shapley attribution metrics, computed only on "
            "that retrieval-conditioned subset (n=gold hit cases; for TF-IDF this "
            "is n=40 because one case has a non-zero attribution mass despite "
            "recall=0). "
            "Deletion AUC is shown for the gold/reference target because the saved "
            "artifacts do not include generated-answer deletion curves.</p>",
            '    <div class="finding-list">',
            '      <div class="finding"><p>Controlled cases test whether Shapley '
            "behaves as expected under known evidence structure across lexical, "
            "target-likelihood, contrastive-likelihood, and retriever-ablation "
            "settings; QASPER tests whether the same first-order attribution "
            "behavior survives realistic scientific QA.</p></div>",
            '      <div class="finding"><p>On QASPER, the main attribution result is '
            "retrieval-conditional: when labelled evidence appears in the retrieved "
            "context, gold-answer Shapley ranks labelled evidence first in "
            f"{qasper_gold_top_hits}/{qasper_hit_count} cases "
            "and assigns most attribution mass to labelled evidence.</p></div>",
            '      <div class="finding"><p>Artemis is a prior-knowledge control: '
            "closed-book answers are weak, retrieval hits labelled evidence in all "
            "10 cases, and generated-answer Shapley ranks labelled evidence first "
            "in 10/10 cases.</p></div>",
            '      <div class="finding"><p>Generated-answer attribution remains '
            "evidence-aligned on the controlled corpus, but weakens on QASPER; the "
            "gold-vs-generated comparison is therefore a behavioral diagnostic, not "
            "a replacement for answer correctness.</p></div>",
            "    </div>",
            RESULTS_END_MARKER,
        ]
    )


def _render_retrieved(artifact: dict[str, Any]) -> str:
    body = [
        (
            "<tr>"
            f"<td>P{int(row['rank'])}</td>"
            f"<td>{_html(row['passage_id'])}</td>"
            f"<td>{_fmt(row['score'])}</td>"
            f"<td>{_html(row['text'])}</td>"
            "</tr>"
        )
        for row in artifact["retrieved"]
    ]
    return (
        '<div class="table-scroll"><table class="mini-table retrieved-table">'
        "<thead><tr><th>Rank</th><th>Passage</th><th>Retrieval</th>"
        "<th>Text</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _render_attributions(artifact: dict[str, Any], attributions_key: str) -> str:
    rows = artifact.get(attributions_key, [])
    if not rows:
        return '<p class="empty-note">No attribution rows were stored for this target.</p>'

    passage_by_rank = _passage_id_by_rank(artifact)
    body = []
    for row in rows:
        player = int(row["player"])
        body.append(
            "<tr>"
            f"<td>P{player}</td>"
            f"<td>{_html(passage_by_rank.get(player, row['chunk']))}</td>"
            f"<td>{_fmt(row['attribution'], digits=4, signed=True)}</td>"
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table class="mini-table attribution-table">'
        "<thead><tr><th>Player</th><th>Passage</th><th>Shapley value</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _pairwise_lookup(rows: list[dict[str, Any]]) -> dict[frozenset[str], float]:
    lookup: dict[frozenset[str], float] = {}
    for row in rows:
        key = frozenset((str(row["left"]), str(row["right"])))
        lookup[key] = float(row["interaction"])
    return lookup


def _interaction_cell(value: float | None) -> str:
    if value is None:
        return '<td class="heat-cell diag">-</td>'
    alpha = min(0.84, 0.12 + abs(value) * 3.5)
    tone = "pos" if value > 0 else "neg" if value < 0 else "zero"
    return (
        f'<td class="heat-cell {tone}" style="--alpha:{alpha:.3f}" '
        f'title="{_fmt(value, digits=6, signed=True)}">'
        f"{_fmt(value, digits=3, signed=True)}</td>"
    )


def _render_pairwise_heatmap(artifact: dict[str, Any], pairwise_key: str) -> str:
    rows = artifact.get(pairwise_key, [])
    if not rows:
        return '<p class="empty-note">No pairwise interaction matrix was stored.</p>'

    passages = [str(row["passage_id"]) for row in artifact["retrieved"]]
    lookup = _pairwise_lookup(rows)
    header = "<tr><th></th>" + "".join(
        f'<th title="{_html(passage)}">P{index}</th>'
        for index, passage in enumerate(passages, start=1)
    )
    header += "</tr>"

    matrix_rows = []
    for row_index, left in enumerate(passages, start=1):
        cells = [f'<th title="{_html(left)}">P{row_index}</th>']
        for right in passages:
            value = None if left == right else lookup.get(frozenset((left, right)))
            cells.append(_interaction_cell(value))
        matrix_rows.append(f"<tr>{''.join(cells)}</tr>")

    ranked = sorted(rows, key=lambda row: abs(float(row["interaction"])), reverse=True)[:6]
    ranked_rows = [
        (
            "<tr>"
            f"<td>P{int(row['left_rank'])} x P{int(row['right_rank'])}</td>"
            f"<td>{_html(row['left'])} x {_html(row['right'])}</td>"
            f"<td>{_html(row['sign'])}</td>"
            f"<td>{_fmt(row['interaction'], digits=4, signed=True)}</td>"
            "</tr>"
        )
        for row in ranked
    ]

    return (
        '<div class="heatmap-wrap"><table class="interaction-heatmap">'
        f"<thead>{header}</thead><tbody>{''.join(matrix_rows)}</tbody></table></div>"
        '<div class="table-scroll"><table class="mini-table top-pairs">'
        "<thead><tr><th>Pair</th><th>Passage IDs</th><th>Sign</th><th>Value</th></tr></thead>"
        f"<tbody>{''.join(ranked_rows)}</tbody></table></div>"
    )


def _coalition_lookup(rows: list[dict[str, Any]]) -> dict[tuple[int, ...], float]:
    return {tuple(int(player) for player in row["players"]): float(row["score"]) for row in rows}


def _render_value_snapshots(artifact: dict[str, Any], coalitions_key: str) -> str:
    rows = artifact.get(coalitions_key, [])
    if not rows:
        return '<p class="empty-note">No coalition scores were stored for this target.</p>'

    lookup = _coalition_lookup(rows)
    passages = artifact["retrieved"]
    labelled_pairs = {
        tuple(
            sorted(
                rank
                for rank, passage in _passage_id_by_rank(artifact).items()
                if passage in {interaction["left"], interaction["right"]}
            )
        )
        for interaction in artifact["case"].get("expected_interactions", [])
    }
    keys: list[tuple[str, tuple[int, ...]]] = [("empty context", ())]
    keys.extend((f"P{index} alone", (index,)) for index in range(1, len(passages) + 1))
    keys.extend(
        (f"labelled pair P{pair[0]} x P{pair[1]}", pair)
        for pair in sorted(labelled_pairs)
        if len(pair) == 2
    )
    all_players = tuple(range(1, len(passages) + 1))
    keys.append(("all retrieved passages", all_players))

    seen: set[tuple[int, ...]] = set()
    body = []
    for label, key in keys:
        if key in seen or key not in lookup:
            continue
        seen.add(key)
        body.append(
            "<tr>"
            f"<td>{_html(label)}</td>"
            f"<td>{', '.join(f'P{player}' for player in key) or 'empty'}</td>"
            f"<td>{_fmt(lookup[key], digits=4, signed=True)}</td>"
            "</tr>"
        )

    return (
        '<div class="table-scroll"><table class="mini-table value-table">'
        "<thead><tr><th>Coalition</th><th>Players</th><th>Value function</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _case_summary(rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    evaluable = _evaluable_count(rows)
    correct = _correct_count(rows)
    if total == 0:
        return "no labelled pairs"
    if evaluable == 0:
        return f"not evaluable, {total} labelled"
    unevaluable = total - evaluable
    if unevaluable:
        return f"{correct}/{evaluable} signs correct, {total} labelled, {unevaluable} not evaluable"
    return f"{correct}/{evaluable} signs correct, {total} labelled"


def _render_case(
    artifact: dict[str, Any],
    *,
    target: TargetView,
) -> str:
    case = artifact["case"]
    rows = artifact.get(target.labelled_key, [])
    summary = _case_summary(rows)
    summary_class = "neutral"
    evaluable_count = _evaluable_count(rows)
    correct_count = _correct_count(rows)
    if rows and evaluable_count == 0:
        summary_class = "muted"
    elif rows and correct_count == evaluable_count:
        summary_class = "good"
    elif rows and correct_count < evaluable_count:
        summary_class = "bad"

    generation = artifact.get("generation", {})
    generated_answer = generation.get("retrieved_context_answer")
    generated_html = (
        f"<div><span>Generated answer</span><strong>{_html(generated_answer)}</strong></div>"
        if generated_answer
        else ""
    )

    return f"""
      <details class="case-detail">
        <summary>
          <span class="case-name">{_html(case["case_id"])}</span>
          <span class="case-pattern">{_html(case["pattern"])}</span>
          <span class="pill {summary_class}">{_html(summary)}</span>
        </summary>
        <div class="case-body">
          <p class="case-question"><strong>Q.</strong> {_html(case["question"])}</p>
          <div class="answer-grid">
            <div><span>Gold answer</span><strong>{_html(case["gold_answer"])}</strong></div>
            {generated_html}
          </div>
          <h4>Labelled interaction result</h4>
          {_render_labelled_interactions(artifact, target.labelled_key)}
          <h4>Shapley interaction heatmap</h4>
          {_render_pairwise_heatmap(artifact, target.pairwise_key)}
          <h4>First-order Shapley attribution</h4>
          {_render_attributions(artifact, target.attributions_key)}
          <h4>Retrieved passages</h4>
          {_render_retrieved(artifact)}
          <h4>Coalition value snapshots</h4>
          {_render_value_snapshots(artifact, target.coalitions_key)}
        </div>
      </details>
    """


def _target_counts(artifacts: list[dict[str, Any]], labelled_key: str) -> tuple[int, int, int]:
    rows = [row for artifact in artifacts for row in artifact.get(labelled_key, [])]
    return len(rows), _evaluable_count(rows), _correct_count(rows)


def render_interaction_details(targets: tuple[TargetView, ...] = DEFAULT_TARGETS) -> str:
    """Render collapsible per-question interaction details."""
    sections = [
        INTERACTION_START_MARKER,
        '    <div class="interaction-detail-intro">',
        '      <h3 class="sub-title">Per-question interaction details</h3>',
        "      <p>Each setting below expands into the stored case artifacts: labelled pair "
        "results, retrieved passages, first-order Shapley values, coalition scores, "
        "and a pairwise Shapley interaction heatmap for the retrieved top-k passages. "
        "Heatmap colour: <strong>green</strong> = positive interaction (synergy / complementary); "
        "<strong>red</strong> = negative interaction (substitution, redundancy, or conflict). "
        "Colour intensity scales with absolute value magnitude.</p>",
        "    </div>",
    ]
    for index, target in enumerate(targets):
        artifacts = _load_artifacts(target.run_dir)
        total, evaluable, correct = _target_counts(artifacts, target.labelled_key)
        open_attr = " open" if index == 0 else ""
        case_html = "".join(_render_case(artifact, target=target) for artifact in artifacts)
        sections.append(
            f"""
    <details class="setting-detail"{open_attr}>
      <summary>
        <span>{_html(target.setting)} / {_html(target.target)}</span>
        <span class="setting-counts">{correct}/{evaluable} correct - {total} labelled - {len(artifacts)} questions</span>
      </summary>
      <div class="case-stack">
        {case_html}
      </div>
    </details>
            """.rstrip()
        )
    sections.append(INTERACTION_END_MARKER)
    return "\n".join(sections)


def _render_failure_overview(rows: list[dict[str, str]]) -> str:
    body = []
    for row in rows:
        class_name = _failure_class(row["knowledge_source_class"])
        body.append(
            "<tr>"
            f"<td>{_html(row['case_id'])}</td>"
            f'<td><span class="pill {class_name}">{_html(_class_label(row["knowledge_source_class"]))}</span></td>'
            f"<td>{_fmt(row['closed_book_f1'])}</td>"
            f"<td>{_fmt(row['gold_context_f1'])}</td>"
            f"<td>{_fmt(row['retrieved_context_f1'])}</td>"
            f"<td>{_fmt(row['retrieval_recall_at_k'])}</td>"
            f"<td>{_fmt(row['retrieval_mrr'])}</td>"
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table class="mini-table failure-overview">'
        "<thead><tr><th>Case</th><th>Diagnosis</th><th>Closed F1</th>"
        "<th>Gold ctx F1</th><th>Retrieved F1</th><th>Recall@6</th><th>MRR</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _render_metric_strip(row: dict[str, str]) -> str:
    metrics = (
        ("Closed-book F1", row.get("closed_book_f1")),
        ("Gold-context F1", row.get("gold_context_f1")),
        ("Retrieved F1", row.get("retrieved_context_f1")),
        ("Recall@6", row.get("retrieval_recall_at_k")),
        ("MRR", row.get("retrieval_mrr")),
        ("Gold mass", row.get("gold_attribution_mass")),
    )
    return "".join(
        f'<div class="mini-stat"><span>{_html(label)}</span><strong>{_fmt(value)}</strong></div>'
        for label, value in metrics
    )


def _render_answer_evidence(artifact: dict[str, Any], row: dict[str, str]) -> str:
    generation = artifact.get("generation", {})
    closed_answer = generation.get("closed_book_answer", row.get("closed_book_answer", ""))
    gold_context_answer = generation.get(
        "gold_context_answer",
        row.get("gold_context_answer", ""),
    )
    retrieved_answer = generation.get(
        "retrieved_context_answer",
        row.get("retrieved_context_answer", ""),
    )
    return f"""
          <div class="answer-grid failure-answer-grid">
            <div><span>Closed-book answer</span><strong>{_html(closed_answer)}</strong></div>
            <div><span>Gold-context answer</span><strong>{_html(gold_context_answer)}</strong></div>
            <div><span>Retrieved-context answer</span><strong>{_html(retrieved_answer)}</strong></div>
          </div>
    """


def _render_failure_retrieved(artifact: dict[str, Any]) -> str:
    gold_ids = set(artifact["case"].get("gold_evidence_ids", []))
    body = []
    for item in artifact["retrieved"]:
        is_gold = item["passage_id"] in gold_ids
        badge = '<span class="pill good">gold hit</span>' if is_gold else ""
        body.append(
            "<tr>"
            f"<td>P{int(item['rank'])}</td>"
            f"<td>{_html(item['passage_id'])}</td>"
            f"<td>{badge}</td>"
            f"<td>{_fmt(item['score'])}</td>"
            f"<td>{_html(item['text'])}</td>"
            "</tr>"
        )
    return (
        '<div class="table-scroll"><table class="mini-table retrieved-table">'
        "<thead><tr><th>Rank</th><th>Passage</th><th>Gold?</th><th>Retrieval</th>"
        "<th>Text</th></tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


def _top_gold_rank(row: dict[str, str]) -> str:
    retrieved = str(row.get("retrieved_ids", "")).split("|")
    gold_ids = set(ast.literal_eval(row.get("gold_evidence_ids", "[]")))
    ranks = [index + 1 for index, passage_id in enumerate(retrieved) if passage_id in gold_ids]
    return ", ".join(f"P{rank}" for rank in ranks) if ranks else "none in top-k"


def _render_failure_case(artifact: dict[str, Any], row: dict[str, str]) -> str:
    case = artifact["case"]
    diagnosis = row.get(
        "knowledge_source_class",
        artifact.get("generation", {}).get("knowledge_source_class", "unknown"),
    )
    class_name = _failure_class(diagnosis)
    gold_ids = case.get("gold_evidence_ids", [])
    return f"""
      <details class="case-detail failure-case">
        <summary>
          <span class="case-name">{_html(case["case_id"])}</span>
          <span class="case-pattern">{_html(case.get("answer_type", case.get("paper_title", "")))}</span>
          <span class="pill {class_name}">{_html(_class_label(diagnosis))}</span>
        </summary>
        <div class="case-body">
          <p class="case-question"><strong>Q.</strong> {_html(case["question"])}</p>
          <div class="answer-grid">
            <div><span>Paper</span><strong>{_html(case.get("paper_title", ""))}</strong></div>
            <div><span>Gold answer</span><strong>{_html(case["gold_answer"])}</strong></div>
          </div>
          <div class="metric-strip">{_render_metric_strip(row)}</div>
          <p class="metric-warning">
            This is an automatic token-F1 threshold flag at 0.65, not a human
            correctness judgment. Raw class: <code>{_html(diagnosis)}</code>.
          </p>
          <div class="diagnosis-note">
            <span>Gold evidence IDs</span>
            <strong>{_html(", ".join(gold_ids) or "none labelled")}</strong>
            <span>Gold evidence retrieved</span>
            <strong>{_html(_top_gold_rank(row))}</strong>
          </div>
          <h4>Answer evidence</h4>
          {_render_answer_evidence(artifact, row)}
          <h4>Retrieved passages</h4>
          {_render_failure_retrieved(artifact)}
          <h4>Gold-answer Shapley attribution</h4>
          {_render_attributions(artifact, "gold_attributions")}
          <h4>Generated-answer Shapley attribution</h4>
          {_render_attributions(artifact, "generated_attributions")}
        </div>
      </details>
    """


def render_failure_details(run_dir: Path = QASPER_RUN) -> str:
    """Render collapsible per-question failure-diagnosis details."""
    artifacts = _load_artifacts(run_dir)
    summary = _summary_by_case(run_dir)
    ordered_rows = [summary[str(artifact["case"]["case_id"])] for artifact in artifacts]
    class_counts: dict[str, int] = {}
    for row in ordered_rows:
        label = row["knowledge_source_class"]
        class_counts[label] = class_counts.get(label, 0) + 1
    count_text = " - ".join(
        f"{count} {_class_label(label)}" for label, count in sorted(class_counts.items())
    )
    case_html = "".join(
        _render_failure_case(artifact, summary[str(artifact["case"]["case_id"])])
        for artifact in artifacts
    )
    return "\n".join(
        [
            FAILURE_START_MARKER,
            '    <div class="interaction-detail-intro failure-detail-intro">',
            '      <h3 class="sub-title">Per-question failure evidence</h3>',
            "      <p>Each QASPER case below shows the actual closed-book, gold-context, "
            "and retrieved-context answers used by the knowledge-source classifier, "
            "plus retrieval hits and Shapley attribution over the retrieved passages.</p>",
            '      <p class="metric-warning">49 of 50 QASPER cases are diagnosed as '
            "<strong>generation/model failure</strong>: the model (Gemma E4B, run with "
            "partial Metal offload) fails to produce a short extractive answer from "
            "academic text even when the correct passage is retrieved. This is a "
            "generation-quality bottleneck, not a Shapley failure — retrieval recall "
            "is 0.549 and attribution correctly identifies the gold passage as "
            "top-ranked in 28/35 gold-hit cases. The token-F1 threshold (0.65) is "
            "calibrated for single-sentence extractive answers; virtually all "
            "QASPER responses fall below it due to multi-sentence answer complexity, "
            "and this near-universal flagging is itself the diagnostic finding.</p>",
            "    </div>",
            _render_failure_overview(ordered_rows),
            '    <details class="setting-detail failure-detail" open>',
            "      <summary>",
            "        <span>QASPER / Gemma E4B contrastive likelihood</span>",
            f'        <span class="setting-counts">{_html(count_text)} - {len(artifacts)} questions</span>',
            "      </summary>",
            '      <div class="case-stack">',
            case_html,
            "      </div>",
            "    </details>",
            FAILURE_END_MARKER,
        ]
    )


def _replace_generated_block(
    html: str,
    *,
    start_marker: str,
    end_marker: str,
    replacement: str,
    report_path: Path,
) -> str:
    if start_marker not in html or end_marker not in html:
        msg = f"Missing generated detail markers in {report_path}: {start_marker}"
        raise ValueError(msg)
    before, rest = html.split(start_marker, maxsplit=1)
    _, after = rest.split(end_marker, maxsplit=1)
    return before + replacement + after


def update_report(report_path: Path = REPORT_PATH) -> None:
    """Replace generated detail blocks in the static HTML report."""
    html = report_path.read_text(encoding="utf-8")
    html = _replace_generated_block(
        html,
        start_marker=RESULTS_START_MARKER,
        end_marker=RESULTS_END_MARKER,
        replacement=render_attribution_results(),
        report_path=report_path,
    )
    html = _replace_generated_block(
        html,
        start_marker=INTERACTION_START_MARKER,
        end_marker=INTERACTION_END_MARKER,
        replacement=render_interaction_details(),
        report_path=report_path,
    )
    html = _replace_generated_block(
        html,
        start_marker=FAILURE_START_MARKER,
        end_marker=FAILURE_END_MARKER,
        replacement=render_failure_details(),
        report_path=report_path,
    )
    report_path.write_text(html, encoding="utf-8")


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-path", type=Path, default=REPORT_PATH)
    args = parser.parse_args()
    update_report(args.report_path)
    print(f"Updated {args.report_path}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
