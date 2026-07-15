import { useEffect, useState } from "react";
import { ChevronLeft, Network, ScanSearch, Target } from "lucide-react";
import type { AnswerTargetResult, ExplanationResult, RunResult } from "../types";
import { formatScore } from "../utils";
import { EvidencePanel } from "./EvidencePanel";
import { InteractionPanel } from "./InteractionPanel";
import { AuditPanel } from "./AuditPanel";
import { Diagnostics } from "./Diagnostics";

interface Props {
  question: string;
  result: RunResult;
  modelDisplayName?: string;
  resultIndex: number;
  onResultIndexChange: (i: number) => void;
  backLabel: string;
  onBack: () => void;
}

function MetricCard({
  label,
  value,
  detail,
  primary,
}: {
  label: string;
  value: string;
  detail?: string;
  primary?: boolean;
}) {
  return (
    <div className={`metric-card${primary ? " metric-primary" : ""}`}>
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      {detail && <span className="metric-detail">{detail}</span>}
    </div>
  );
}

export function EvidenceDetailView({
  question,
  result,
  modelDisplayName,
  resultIndex,
  onResultIndexChange,
  backLabel,
  onBack,
}: Props) {
  const [targetIndex, setTargetIndex] = useState(0);
  useEffect(() => setTargetIndex(0), [result.run_name]);

  const fallbackTarget: AnswerTargetResult = {
    target_type: result.answer_target_type,
    label:
      result.answer_target_type === "generated_answer"
        ? "Generated answer"
        : "Gold / reference answer",
    answer: result.answer,
    results: result.results,
  };
  const targets = result.targets?.length ? result.targets : [fallbackTarget];
  const isLegacySingleTarget = !result.targets?.length;
  const activeTarget = targets[targetIndex] ?? targets[0];
  const active: ExplanationResult = activeTarget.results[resultIndex];
  const isGeneratedTarget = activeTarget.target_type === "generated_answer";
  const hasDualTargets = targets.length > 1;

  return (
    <div className="detail-view">
      <div className="detail-header">
        <button className="back-btn" onClick={onBack}>
          <ChevronLeft size={13} />
          {backLabel}
        </button>

        {modelDisplayName && (
          <p style={{ fontSize: 12, fontWeight: 600, color: "var(--muted)", marginBottom: 6 }}>
            {modelDisplayName}
          </p>
        )}

        <p className="detail-question">{question}</p>

        {activeTarget.results.length > 1 && (
          <div className="result-tabs">
            {activeTarget.results.map((r, i) => (
              <button
                key={i}
                className={i === resultIndex ? "active" : ""}
                onClick={() => onResultIndexChange(i)}
              >
                {r.value_name}
              </button>
            ))}
          </div>
        )}
      </div>

      <div className="detail-body">
        <div className="research-flow" aria-label="Explanation pipeline">
          <div className="research-step">
            <ScanSearch size={17} />
            <div>
              <strong>Retrieved context</strong>
              <span>
                {result.retrieval_method} selected {result.retrieval_top_k} coalition players
              </span>
            </div>
          </div>
          <div className="research-step">
            <Target size={17} />
            <div>
              <strong>{hasDualTargets ? "Compare answer targets" : "Answer target"}</strong>
              <span>
                {hasDualTargets
                  ? "Reference and generated answers are explained independently"
                  : isLegacySingleTarget
                    ? "This saved result contains one answer target"
                    : isGeneratedTarget
                      ? "Generated answer only; add a reference answer for comparison"
                      : "Reference answer only; no generated answer was recorded"}
              </span>
            </div>
          </div>
          <div className="research-step">
            <Network size={17} />
            <div>
              <strong>Explain evidence effects</strong>
              <span>Individual Shapley contributions and pairwise interactions</span>
            </div>
          </div>
        </div>

        {isLegacySingleTarget && (
          <div className="legacy-result-note">
            This result was created before dual-target support was enabled. Return to the
            chat and run the attribution again to compute both reference-answer and
            generated-answer support.
          </div>
        )}

        {/* Answer block */}
        <div className="detail-answer-card card">
          {targets.length > 1 && (
            <div className="target-switch" role="tablist" aria-label="Answer target">
              {targets.map((target, index) => (
                <button
                  key={target.target_type}
                  role="tab"
                  aria-selected={targetIndex === index}
                  className={targetIndex === index ? "active" : ""}
                  onClick={() => setTargetIndex(index)}
                >
                  <span>{target.label}</span>
                  <small>
                    {target.target_type === "reference_answer"
                      ? "Does the evidence support the correct answer?"
                      : "What evidence supports the model output?"}
                  </small>
                </button>
              ))}
            </div>
          )}
          <div style={{ padding: "20px" }}>
            <div className="answer-heading-row">
              <div className="detail-answer-eyebrow">{activeTarget.label}</div>
              <span className={`target-badge ${isGeneratedTarget ? "generated" : "reference"}`}>
                {isGeneratedTarget
                  ? "Generated-answer attribution"
                  : "Gold/reference-answer attribution"}
              </span>
            </div>
            <blockquote className="answer-text">{activeTarget.answer}</blockquote>
            <p className="answer-scope-note">
              The scores below explain how the retrieved chunks change support for this
              exact answer. They do not certify that the answer is factually correct.
            </p>
          </div>

          <div className="metrics-row">
            <MetricCard
              primary
              label="Evidence support"
              value={formatScore(active.full_score, false)}
              detail="full-context score"
            />
            <MetricCard
              label="Empty-context baseline"
              value={formatScore(active.empty_score, false)}
            />
            <MetricCard
              label="Top attribution"
              value={formatScore(active.top_score)}
              detail={active.top_chunk}
            />
            <MetricCard
              label="Chunks retrieved"
              value={String(result.chunks.length)}
              detail={`${result.retrieval_method} · coalition players`}
            />
            <MetricCard
              label="Value function"
              value={active.value_name}
            />
          </div>
        </div>

        <EvidencePanel result={active} chunks={result.chunks} />

        <div className="two-col">
          <InteractionPanel result={active} />
          <AuditPanel result={active} />
        </div>

        <Diagnostics result={result} active={active} />
      </div>
    </div>
  );
}
