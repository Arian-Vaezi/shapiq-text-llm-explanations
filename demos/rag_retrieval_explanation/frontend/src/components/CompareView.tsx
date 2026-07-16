import { useState, useEffect, useMemo, Fragment } from "react";
import type { CSSProperties } from "react";
import { ArrowUp, ChevronDown, ChevronRight, FlaskConical } from "lucide-react";
import type { CompareRun, ModelRunResult } from "../types";
import { formatScore } from "../utils";

// ─── Local types ──────────────────────────────────────────────────────────────

type CompareTab = "overview" | "answers" | "evidence" | "attribution";
type AgreementLevel = "agree" | "partial" | "differ";

interface FaithfulnessAudit {
  supportStrength: "High" | "Medium" | "Low";
  unsupportedRisk: "High" | "Medium" | "Low";
  sensitivity: "Strong" | "Moderate" | "Weak";
  positiveCount: number;
  negativeCount: number;
  agreement: AgreementLevel | null;
  warnings: string[];
}

// ─── Pure utilities ───────────────────────────────────────────────────────────

function jaccardWords(a: string, b: string): number {
  const words = (s: string) =>
    new Set(s.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(Boolean).slice(0, 60));
  const sa = words(a);
  const sb = words(b);
  let inter = 0;
  sa.forEach((w) => { if (sb.has(w)) inter++; });
  const union = new Set([...sa, ...sb]).size;
  return union === 0 ? 0 : inter / union;
}

function computeAgreement(results: ModelRunResult[]): AgreementLevel | null {
  const answers = results
    .filter((r) => r.status === "completed" && r.result?.answer)
    .map((r) => r.result!.answer);
  if (answers.length < 2) return null;
  let minSim = 1;
  for (let i = 0; i < answers.length; i++) {
    for (let j = i + 1; j < answers.length; j++) {
      minSim = Math.min(minSim, jaccardWords(answers[i], answers[j]));
    }
  }
  return minSim >= 0.55 ? "agree" : minSim >= 0.25 ? "partial" : "differ";
}

function computeModelAgreement(
  mr: ModelRunResult,
  allResults: ModelRunResult[]
): AgreementLevel | null {
  if (!mr.result?.answer) return null;
  const others = allResults.filter(
    (r) => r.model_id !== mr.model_id && r.status === "completed" && r.result?.answer
  );
  if (others.length === 0) return null;
  const sims = others.map((r) => jaccardWords(mr.result!.answer, r.result!.answer));
  const avg = sims.reduce((a, b) => a + b, 0) / sims.length;
  return avg >= 0.55 ? "agree" : avg >= 0.25 ? "partial" : "differ";
}

function getBestModelId(results: ModelRunResult[]): string | null {
  return (
    [...results]
      .filter((r) => r.status === "completed" && r.result)
      .sort(
        (a, b) =>
          (b.result!.results[0]?.full_score ?? -Infinity) -
          (a.result!.results[0]?.full_score ?? -Infinity)
      )[0]?.model_id ?? null
  );
}

interface Summary {
  bestModelName: string | null;
  agreement: AgreementLevel | null;
  completedCount: number;
  missingCount: number;
  hasNegativeTopAttr: boolean;
  sharedChunkCount: number | null;
}

function buildSummary(run: CompareRun): Summary {
  const completed = run.model_results.filter((r) => r.status === "completed" && r.result);
  const notCompleted = run.model_results.filter((r) => r.status !== "completed");
  const bestId = getBestModelId(run.model_results);
  const bestName =
    run.model_results.find((r) => r.model_id === bestId)?.model_display_name ?? null;
  const hasNeg = completed.some((r) => (r.result?.results[0]?.top_score ?? 0) < 0);
  const firstN = completed[0]?.result?.chunks.length ?? null;
  const sharedN =
    firstN !== null && completed.every((r) => r.result?.chunks.length === firstN) ? firstN : null;
  return {
    bestModelName: bestName,
    agreement: computeAgreement(run.model_results),
    completedCount: completed.length,
    missingCount: notCompleted.length,
    hasNegativeTopAttr: hasNeg,
    sharedChunkCount: sharedN,
  };
}

// Thresholds: support High>=0.50 Medium>=0.25 Low<0.25
//             risk    Low if full>=0.35 AND empty<full*0.8; High if full<0.15 OR empty>=full
//             delta   Strong>=0.25 Moderate>=0.08 Weak<0.08
function computeFaithfulnessAudit(
  mr: ModelRunResult,
  allResults: ModelRunResult[]
): FaithfulnessAudit | null {
  const active = mr.result?.results[0];
  if (!active) return null;

  const { full_score: full, empty_score: empty } = active;
  const delta = full - empty;
  const attrs = active.attributions;

  const supportStrength: "High" | "Medium" | "Low" =
    full >= 0.5 ? "High" : full >= 0.25 ? "Medium" : "Low";

  const unsupportedRisk: "High" | "Medium" | "Low" =
    full < 0.15 || empty >= full
      ? "High"
      : full >= 0.35 && empty < full * 0.8
      ? "Low"
      : "Medium";

  const sensitivity: "Strong" | "Moderate" | "Weak" =
    delta >= 0.25 ? "Strong" : delta >= 0.08 ? "Moderate" : "Weak";

  const positiveCount = attrs.filter((a) => a.attribution > 0.001).length;
  const negativeCount = attrs.filter((a) => a.attribution < -0.001).length;

  const agreement = computeModelAgreement(mr, allResults);

  const warnings: string[] = [];
  if (active.top_score < 0)
    warnings.push("Negative top attribution — evidence may work against this answer");
  if (delta < 0.02)
    warnings.push("Context barely improves score — model may ignore evidence");
  if (empty > full)
    warnings.push("Empty context outscores full context — unexpected");
  if (negativeCount > positiveCount)
    warnings.push("More negative than positive contributing chunks");

  return {
    supportStrength,
    unsupportedRisk,
    sensitivity,
    positiveCount,
    negativeCount,
    agreement,
    warnings,
  };
}

function matrixCellStyle(value: number, maxAbs: number): CSSProperties {
  if (maxAbs < 0.001 || Math.abs(value) < 0.001) return {};
  const intensity = Math.min(Math.abs(value) / maxAbs, 1);
  if (value > 0) {
    const alpha = intensity * 0.75 + 0.05;
    return {
      background: `rgba(15,15,15,${alpha.toFixed(2)})`,
      color: intensity > 0.5 ? "#fff" : "#0f0f0f",
    };
  }
  const alpha = intensity * 0.35 + 0.05;
  return { background: `rgba(130,130,130,${alpha.toFixed(2)})` };
}

// ─── Main component ───────────────────────────────────────────────────────────

interface Props {
  compareRun: CompareRun | null;
  source: "sample" | "pdf";
  selectedScenario: { name: string; question: string; takeaway: string } | undefined;
  compareModelIds: string[];
  question: string;
  onQuestionChange: (q: string) => void;
  canRun: boolean;
  busy: boolean;
  error: string;
  onRun: () => void;
  onViewDetail: (modelId: string) => void;
}

export function CompareView({
  compareRun,
  source,
  selectedScenario,
  compareModelIds,
  question,
  onQuestionChange,
  canRun,
  busy,
  error,
  onRun,
  onViewDetail,
}: Props) {
  const [activeTab, setActiveTab] = useState<CompareTab>("overview");
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [expandedChunks, setExpandedChunks] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (compareRun) {
      setActiveTab("overview");
      setSelectedModelId(null);
      setExpandedChunks(new Set());
    }
  }, [compareRun?.id]);

  const bestModelId = useMemo(
    () => getBestModelId(compareRun?.model_results ?? []),
    [compareRun]
  );
  const effectiveSelectedId = selectedModelId ?? bestModelId;

  function toggleChunk(key: string) {
    setExpandedChunks((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }

  const hasRun = !!compareRun && !busy;
  const displayQuestion =
    compareRun?.question ??
    (source === "sample" ? selectedScenario?.question : undefined) ??
    "";

  return (
    <div className="compare-layout">
      <CommandBar
        question={displayQuestion}
        compareModelIds={compareModelIds}
        run={compareRun}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        hasRun={hasRun}
      />

      <div className="compare-body">
        {!compareRun && !busy && !error && (
          <div className="empty-state">
            <div className="empty-rule" />
            <h2>Model comparison</h2>
            <p>
              {compareModelIds.length < 2
                ? "Select 2 or more models in the sidebar."
                : source === "sample"
                ? "Select a scenario and click Run comparison."
                : "Upload a PDF and type your question below."}
            </p>
          </div>
        )}

        {busy && (
          <div className="compare-loading">
            <FlaskConical size={16} className="spin" />
            Running across {compareModelIds.length} model
            {compareModelIds.length !== 1 ? "s" : ""}…
          </div>
        )}

        {error && <div className="error-box">{error}</div>}

        {hasRun && (
          <div className="cmp-content">
            {activeTab === "overview" && (
              <OverviewTab
                run={compareRun!}
                bestModelId={bestModelId}
                selectedModelId={effectiveSelectedId}
                onSelectModel={setSelectedModelId}
                onViewDetail={onViewDetail}
              />
            )}
            {activeTab === "answers" && (
              <AnswersTab run={compareRun!} onViewDetail={onViewDetail} />
            )}
            {activeTab === "evidence" && (
              <EvidenceTab
                run={compareRun!}
                expandedChunks={expandedChunks}
                onToggleChunk={toggleChunk}
              />
            )}
            {activeTab === "attribution" && <AttributionTab run={compareRun!} />}
          </div>
        )}
      </div>

      <div className="chat-input-area">
        <div className="chat-input-inner">
          {source === "sample" ? (
            <div className="chat-sample-input">
              {selectedScenario ? (
                <span className="chat-sample-question">{selectedScenario.question}</span>
              ) : (
                <span className="chat-no-scenario">No scenario selected</span>
              )}
              <button className="chat-run-btn" onClick={onRun} disabled={!canRun || busy}>
                {busy ? "Running…" : compareRun ? "Re-run comparison" : "Run comparison"}
              </button>
            </div>
          ) : (
            <div className="chat-free-input">
              <textarea
                value={question}
                onChange={(e) => onQuestionChange(e.target.value)}
                placeholder="Ask a question about the uploaded document…"
                rows={1}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    if (canRun && !busy) onRun();
                  }
                }}
              />
              <button
                className="chat-send-btn"
                onClick={onRun}
                disabled={!canRun || busy}
                aria-label="Run comparison"
              >
                <ArrowUp size={15} />
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ─── Command bar ──────────────────────────────────────────────────────────────

const TABS: { key: CompareTab; label: string }[] = [
  { key: "overview", label: "Overview" },
  { key: "answers", label: "Answers" },
  { key: "evidence", label: "Evidence" },
  { key: "attribution", label: "Attribution" },
];

function CommandBar({
  question,
  compareModelIds,
  run,
  activeTab,
  onTabChange,
  hasRun,
}: {
  question: string;
  compareModelIds: string[];
  run: CompareRun | null;
  activeTab: CompareTab;
  onTabChange: (t: CompareTab) => void;
  hasRun: boolean;
}) {
  const completedCount = run?.model_results.filter((r) => r.status === "completed").length ?? 0;
  const totalCount = run?.model_results.length ?? compareModelIds.length;
  const metaText = run
    ? `${completedCount} of ${totalCount} model${totalCount !== 1 ? "s" : ""}`
    : `${totalCount} model${totalCount !== 1 ? "s" : ""} selected`;

  return (
    <div className="cmp-bar">
      <div className="cmp-bar-left">
        {question && (
          <span className="cmp-bar-question" title={question}>
            {question}
          </span>
        )}
        <span className="cmp-bar-meta">{metaText}</span>
      </div>
      {hasRun && (
        <nav className="cmp-tabs" aria-label="Comparison views">
          {TABS.map(({ key, label }) => (
            <button
              key={key}
              className={`cmp-tab${activeTab === key ? " active" : ""}`}
              onClick={() => onTabChange(key)}
            >
              {label}
            </button>
          ))}
        </nav>
      )}
    </div>
  );
}

// ─── Overview tab — full stacked report ───────────────────────────────────────

function OverviewTab({
  run,
  bestModelId,
  selectedModelId,
  onSelectModel,
  onViewDetail,
}: {
  run: CompareRun;
  bestModelId: string | null;
  selectedModelId: string | null;
  onSelectModel: (id: string) => void;
  onViewDetail: (id: string) => void;
}) {
  const summary = useMemo(() => buildSummary(run), [run]);

  const sorted = useMemo(
    () =>
      [...run.model_results].sort((a, b) => {
        const sa = a.result?.results[0]?.full_score ?? -Infinity;
        const sb = b.result?.results[0]?.full_score ?? -Infinity;
        return sb - sa;
      }),
    [run]
  );

  return (
    <div className="cmp-overview">
      <SummaryBanner summary={summary} />
      <Scoreboard
        results={sorted}
        bestModelId={bestModelId}
        selectedModelId={selectedModelId}
        onSelectModel={onSelectModel}
      />
      <AnswerComparisonSection run={run} onViewDetail={onViewDetail} />
      <EvidenceAttributionSection run={run} />
      <InteractionMatrixSection run={run} />
      <FaithfulnessAuditSection run={run} />
    </div>
  );
}

// ─── Summary banner ───────────────────────────────────────────────────────────

function SummaryBanner({ summary }: { summary: Summary }) {
  if (summary.completedCount === 0) return null;

  const parts: string[] = [];
  if (summary.bestModelName && summary.completedCount > 1)
    parts.push(`${summary.bestModelName} has the highest evidence support`);
  const agreeLabel: Record<AgreementLevel, string> = {
    agree: "all models produce similar answers",
    partial: "answers partially agree across models",
    differ: "answers differ noticeably across models",
  };
  if (summary.agreement) parts.push(agreeLabel[summary.agreement]);
  if (summary.hasNegativeTopAttr) parts.push("at least one model shows negative top attribution");
  if (summary.sharedChunkCount !== null && summary.completedCount > 1)
    parts.push(`retrieval shared — ${summary.sharedChunkCount} chunks across all models`);
  if (summary.missingCount > 0)
    parts.push(
      `${summary.missingCount} model${summary.missingCount !== 1 ? "s" : ""} could not run`
    );

  if (parts.length === 0) return null;
  const text =
    parts
      .map((p, i) => (i === 0 ? p.charAt(0).toUpperCase() + p.slice(1) : p))
      .join(". ") + ".";

  return (
    <div className="cmp-summary">
      <p className="cmp-summary-text">{text}</p>
    </div>
  );
}

// ─── Scoreboard ───────────────────────────────────────────────────────────────

function Scoreboard({
  results,
  bestModelId,
  selectedModelId,
  onSelectModel,
}: {
  results: ModelRunResult[];
  bestModelId: string | null;
  selectedModelId: string | null;
  onSelectModel: (id: string) => void;
}) {
  return (
    <div className="cmp-scoreboard">
      <div className="cmp-score-header">
        <span>Model</span>
        <span className="cmp-preview-head">Answer preview</span>
        <span className="cmp-num-head">Ev. support</span>
        <span className="cmp-num-head">Baseline</span>
        <span className="cmp-num-head">Top attr.</span>
        <span className="cmp-num-head">#</span>
      </div>
      {results.map((mr) => (
        <ScoreRow
          key={mr.model_id}
          mr={mr}
          isBest={mr.model_id === bestModelId}
          isSelected={mr.model_id === selectedModelId}
          onSelect={() => onSelectModel(mr.model_id)}
        />
      ))}
    </div>
  );
}

function ScoreRow({
  mr,
  isBest,
  isSelected,
  onSelect,
}: {
  mr: ModelRunResult;
  isBest: boolean;
  isSelected: boolean;
  onSelect: () => void;
}) {
  const active = mr.result?.results[0];
  const isOk = mr.status === "completed" && active;

  let cls = "cmp-score-row";
  if (isSelected) cls += " selected";
  else if (isBest && isOk) cls += " best";
  if (!isOk) cls += " dim";

  return (
    <div
      className={cls}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") onSelect();
      }}
    >
      <span className="cmp-score-name">
        {mr.model_display_name}
        {mr.status === "missing_model" && <span className="cmp-pill miss">Missing</span>}
        {mr.status === "failed" && <span className="cmp-pill fail">Failed</span>}
        {mr.status === "running" && <span className="cmp-pill run">Running</span>}
      </span>
      <span className="cmp-score-preview">
        {isOk ? (
          mr.result!.answer
        ) : (
          <span className="cmp-score-err">
            {mr.status === "missing_model" ? "Model file not found" : mr.error ?? "Run failed"}
          </span>
        )}
      </span>
      <span className="cmp-score-num">{isOk ? formatScore(active.full_score, false) : "—"}</span>
      <span className="cmp-score-num">{isOk ? formatScore(active.empty_score, false) : "—"}</span>
      <span className={`cmp-score-num${isOk && active.top_score < 0 ? " neg" : ""}`}>
        {isOk ? formatScore(active.top_score) : "—"}
      </span>
      <span className="cmp-score-num">{isOk ? mr.result!.chunks.length : "—"}</span>
    </div>
  );
}

// ─── Shared section / column wrappers ─────────────────────────────────────────

function ReportSection({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children: React.ReactNode;
}) {
  return (
    <div className="cmp-report-section">
      <div className="cmp-report-section-head">
        <h3 className="cmp-report-section-title">{title}</h3>
        <p className="cmp-report-section-desc">{description}</p>
      </div>
      <div className="cmp-report-section-body">{children}</div>
    </div>
  );
}

function ModelColumns({
  results,
  renderCol,
}: {
  results: ModelRunResult[];
  renderCol: (mr: ModelRunResult) => React.ReactNode;
}) {
  const scrollable = results.length > 3;
  return (
    <div className={`cmp-model-cols${scrollable ? " scroll" : ""}`}>
      {results.map((mr) => (
        <div key={mr.model_id} className="cmp-model-col">
          <div className="cmp-model-col-head">
            {mr.model_display_name}
            {mr.status === "missing_model" && <span className="cmp-pill miss">Missing</span>}
            {mr.status === "failed" && <span className="cmp-pill fail">Failed</span>}
          </div>
          {renderCol(mr)}
        </div>
      ))}
    </div>
  );
}

// ─── 1. Answer comparison ─────────────────────────────────────────────────────

function AnswerComparisonSection({
  run,
  onViewDetail,
}: {
  run: CompareRun;
  onViewDetail: (id: string) => void;
}) {
  return (
    <ReportSection
      title="Answer comparison"
      description="Generated answers from each model, shown side-by-side without clicking."
    >
      <ModelColumns
        results={run.model_results}
        renderCol={(mr) => {
          const active = mr.result?.results[0];
          const isOk = mr.status === "completed" && active && mr.result;
          if (!isOk) {
            return (
              <p className="cmp-col-miss">
                {mr.status === "missing_model"
                  ? `Not found. Download: uv run python scripts/download_models.py --model ${mr.model_id}`
                  : (mr.error ?? "Run failed")}
              </p>
            );
          }
          return (
            <>
              <p className="cmp-col-answer-text">{mr.result!.answer}</p>
              <div className="cmp-col-meta">
                <span>{formatScore(active.full_score, false)}</span>
                <span className="cmp-dot">·</span>
                <span>{mr.result!.chunks.length} chunks</span>
                <span className="cmp-dot">·</span>
                <span>{active.value_name}</span>
              </div>
              <button className="cmp-col-link" onClick={() => onViewDetail(mr.model_id)}>
                View evidence details →
              </button>
            </>
          );
        }}
      />
    </ReportSection>
  );
}

// ─── 2. Evidence attribution ──────────────────────────────────────────────────

function EvidenceAttributionSection({ run }: { run: CompareRun }) {
  const completed = useMemo(
    () => run.model_results.filter((r) => r.status === "completed" && r.result),
    [run]
  );

  const globalMaxAttr = useMemo(
    () =>
      Math.max(
        ...completed.flatMap(
          (mr) => mr.result!.results[0]?.attributions.map((a) => Math.abs(a.attribution)) ?? [0]
        ),
        0.01
      ),
    [completed]
  );

  if (completed.length === 0) return null;

  return (
    <ReportSection
      title="Evidence attribution"
      description="How each model distributes credit across retrieved chunks. All models share the same retrieval set."
    >
      <ModelColumns
        results={run.model_results}
        renderCol={(mr) => {
          const active = mr.result?.results[0];
          if (!active) return <p className="cmp-col-miss">No data</p>;
          const sorted = [...active.attributions].sort((a, b) => b.attribution - a.attribution);
          return (
            <>
              <div className="cmp-rank-scores">
                <span>
                  Support <strong>{formatScore(active.full_score, false)}</strong>
                </span>
                <span>
                  Base <strong>{formatScore(active.empty_score, false)}</strong>
                </span>
              </div>
              <div className="cmp-rank-list">
                {sorted.map((attr, i) => {
                  const isNeg = attr.attribution < 0;
                  const pct = Math.min((Math.abs(attr.attribution) / globalMaxAttr) * 100, 100);
                  return (
                    <div key={i} className={`cmp-rank-item${isNeg ? " neg" : ""}`}>
                      <span className="cmp-rank-label" title={attr.chunk}>
                        {attr.chunk.length > 22 ? attr.chunk.slice(0, 20) + "…" : attr.chunk}
                      </span>
                      <span className="cmp-rank-value">
                        {attr.attribution >= 0 ? "+" : ""}
                        {formatScore(attr.attribution)}
                      </span>
                      <div className="cmp-rank-bar-wrap">
                        <div
                          className={`cmp-rank-bar-fill${isNeg ? " neg" : ""}`}
                          style={{ width: `${pct}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </>
          );
        }}
      />
    </ReportSection>
  );
}

// ─── 3. Interaction matrix ────────────────────────────────────────────────────

function InteractionMatrixSection({ run }: { run: CompareRun }) {
  const completed = useMemo(
    () => run.model_results.filter((r) => r.status === "completed" && r.result),
    [run]
  );

  const globalMaxAbs = useMemo(() => {
    let max = 0.001;
    for (const mr of completed) {
      for (const row of mr.result!.results[0]?.pairwise_matrix ?? []) {
        for (const v of row) {
          if (Math.abs(v) > max) max = Math.abs(v);
        }
      }
    }
    return max;
  }, [completed]);

  const hasMatrix = completed.some(
    (mr) => (mr.result!.results[0]?.pairwise_matrix?.length ?? 0) > 0
  );

  if (!hasMatrix) return null;

  return (
    <ReportSection
      title="Interaction matrix"
      description="Pairwise k-SII interaction scores between chunks. Dark = strong synergy, gray = redundancy/interference. Hover a cell for exact value."
    >
      <ModelColumns
        results={run.model_results}
        renderCol={(mr) => {
          const active = mr.result?.results[0];
          const matrix = active?.pairwise_matrix;
          if (!active || !matrix || matrix.length === 0) {
            return (
              <p className="cmp-col-miss">
                {mr.status === "completed" ? "No interaction data (SV mode)" : "No data"}
              </p>
            );
          }
          const n = matrix.length;
          const chunks = mr.result!.chunks;
          const colLabel = (i: number) => {
            const t = chunks[i]?.title;
            return t ? t.split(/\s+/)[0].slice(0, 7) : `C${i + 1}`;
          };
          return (
            <div className="cmp-matrix-wrap">
              <div
                className="cmp-matrix"
                style={{ gridTemplateColumns: `40px repeat(${n}, 26px)` }}
              >
                {/* header row */}
                <div className="cmp-matrix-label" />
                {Array.from({ length: n }, (_, j) => (
                  <div
                    key={`h${j}`}
                    className="cmp-matrix-label"
                    title={chunks[j]?.title ?? `C${j + 1}`}
                  >
                    {colLabel(j)}
                  </div>
                ))}
                {/* data rows */}
                {matrix.map((row, i) => (
                  <Fragment key={i}>
                    <div
                      className="cmp-matrix-label"
                      title={chunks[i]?.title ?? `C${i + 1}`}
                    >
                      {colLabel(i)}
                    </div>
                    {row.map((v, j) => (
                      <div
                        key={j}
                        className="cmp-matrix-cell"
                        style={matrixCellStyle(v, globalMaxAbs)}
                        title={`${chunks[i]?.title ?? `C${i + 1}`} × ${chunks[j]?.title ?? `C${j + 1}`}: ${v >= 0 ? "+" : ""}${v.toFixed(4)}`}
                      />
                    ))}
                  </Fragment>
                ))}
              </div>
            </div>
          );
        }}
      />
    </ReportSection>
  );
}

// ─── 4. Faithfulness audit ────────────────────────────────────────────────────

function FaithfulnessAuditSection({ run }: { run: CompareRun }) {
  const completed = run.model_results.filter((r) => r.status === "completed" && r.result);
  if (completed.length === 0) return null;

  const strengthCls = { High: "aud-hi", Medium: "aud-md", Low: "aud-lo" } as const;
  const riskCls = { Low: "aud-hi", Medium: "aud-md", High: "aud-lo" } as const;
  const sensCls = { Strong: "aud-hi", Moderate: "aud-md", Weak: "aud-lo" } as const;
  const agreeCls = { agree: "aud-hi", partial: "aud-md", differ: "aud-lo" } as const;
  const agreeLabel = { agree: "Agree", partial: "Partial", differ: "Differ" } as const;

  return (
    <ReportSection
      title="Faithfulness audit"
      description="Rule-based assessment of how well each answer is grounded in the retrieved evidence."
    >
      <ModelColumns
        results={run.model_results}
        renderCol={(mr) => {
          const audit = computeFaithfulnessAudit(mr, run.model_results);
          if (!audit) return <p className="cmp-col-miss">No data</p>;
          return (
            <div className="cmp-audit-rows">
              <AuditRow
                label="Support strength"
                value={audit.supportStrength}
                cls={strengthCls[audit.supportStrength]}
              />
              <AuditRow
                label="Ungrounded risk"
                value={audit.unsupportedRisk}
                cls={riskCls[audit.unsupportedRisk]}
              />
              <AuditRow
                label="Context sensitivity"
                value={audit.sensitivity}
                cls={sensCls[audit.sensitivity]}
              />
              <AuditRow
                label="Chunk coverage"
                value={`${audit.positiveCount}+ · ${audit.negativeCount}−`}
              />
              {audit.agreement && (
                <AuditRow
                  label="Answer agreement"
                  value={agreeLabel[audit.agreement]}
                  cls={agreeCls[audit.agreement]}
                />
              )}
              {audit.warnings.length > 0 && (
                <div className="cmp-audit-warns">
                  {audit.warnings.map((w, i) => (
                    <p key={i} className="cmp-audit-warn">{w}</p>
                  ))}
                </div>
              )}
            </div>
          );
        }}
      />
    </ReportSection>
  );
}

function AuditRow({
  label,
  value,
  cls,
}: {
  label: string;
  value: string;
  cls?: string;
}) {
  return (
    <div className="cmp-audit-row">
      <span className="cmp-audit-label">{label}</span>
      <span className={`cmp-audit-value${cls ? ` ${cls}` : ""}`}>{value}</span>
    </div>
  );
}

// ─── Answers tab (supplementary) ─────────────────────────────────────────────

function AnswersTab({
  run,
  onViewDetail,
}: {
  run: CompareRun;
  onViewDetail: (id: string) => void;
}) {
  const n = run.model_results.length;
  return (
    <div className="cmp-answers">
      <div className={`cmp-answer-grid${n >= 4 ? " scroll" : ""}`}>
        {run.model_results.map((mr) => {
          const active = mr.result?.results[0];
          const isOk = mr.status === "completed" && active;
          return (
            <div key={mr.model_id} className={`cmp-answer-card${!isOk ? " dim" : ""}`}>
              <div className="cmp-answer-card-header">
                <span className="cmp-answer-card-model">{mr.model_display_name}</span>
                {mr.status !== "completed" && (
                  <span className={`cmp-pill ${mr.status === "missing_model" ? "miss" : "fail"}`}>
                    {mr.status === "missing_model" ? "Missing" : "Failed"}
                  </span>
                )}
              </div>
              {isOk ? (
                <>
                  <p className="cmp-answer-card-text">{mr.result!.answer}</p>
                  <div className="cmp-answer-card-meta">
                    <span>{formatScore(active.full_score, false)}</span>
                    <span className="cmp-dot">·</span>
                    <span>{mr.result!.chunks.length} chunks</span>
                    <span className="cmp-dot">·</span>
                    <span>{active.value_name}</span>
                  </div>
                  <button
                    className="cmp-answer-card-link"
                    onClick={() => onViewDetail(mr.model_id)}
                  >
                    View evidence details →
                  </button>
                </>
              ) : (
                <p className="cmp-answer-card-missing">
                  {mr.status === "missing_model" ? "Model file not found" : mr.error ?? "Run failed"}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Evidence tab (supplementary) ────────────────────────────────────────────

function EvidenceTab({
  run,
  expandedChunks,
  onToggleChunk,
}: {
  run: CompareRun;
  expandedChunks: Set<string>;
  onToggleChunk: (key: string) => void;
}) {
  const completed = run.model_results.filter((r) => r.status === "completed" && r.result);

  if (completed.length === 0) {
    return (
      <div className="cmp-evidence">
        <p className="cmp-empty-note">No completed runs to display.</p>
      </div>
    );
  }

  const maxScore = Math.max(
    ...completed.map((r) => r.result!.results[0]?.full_score ?? 0),
    0.01
  );
  const sharedChunks = completed[0].result!.chunks;

  return (
    <div className="cmp-evidence">
      <div className="cmp-section">
        <h3 className="cmp-section-title">Evidence support</h3>
        <div className="cmp-ev-bars">
          {completed.map((mr) => {
            const score = mr.result!.results[0]?.full_score ?? 0;
            const pct = Math.max(0, Math.min(1, score / maxScore)) * 100;
            return (
              <div key={mr.model_id} className="cmp-ev-row">
                <span className="cmp-ev-label">{mr.model_display_name}</span>
                <div className="cmp-ev-track">
                  <div className="cmp-ev-fill" style={{ width: `${pct}%` }} />
                </div>
                <span className="cmp-ev-value">{formatScore(score, false)}</span>
              </div>
            );
          })}
        </div>
      </div>

      <div className="cmp-section">
        <h3 className="cmp-section-title">
          Retrieved chunks
          {completed.length > 1 && (
            <span className="cmp-section-sub"> — shared retrieval across all models</span>
          )}
        </h3>
        <div className="cmp-chunks">
          {sharedChunks.map((chunk, idx) => {
            const key = `c${idx}`;
            const expanded = expandedChunks.has(key);
            const attrPerModel = completed.map((mr) => ({
              name: mr.model_display_name,
              value:
                mr.result!.results[0]?.attributions.find((a) => a.player === idx)?.attribution ??
                null,
            }));

            return (
              <div key={idx} className="cmp-chunk">
                <div className="cmp-chunk-row">
                  <button
                    className="cmp-chunk-toggle"
                    onClick={() => onToggleChunk(key)}
                    aria-expanded={expanded}
                  >
                    {expanded ? <ChevronDown size={11} /> : <ChevronRight size={11} />}
                  </button>
                  <span className="cmp-chunk-title">{chunk.title || `Chunk ${idx + 1}`}</span>
                  {chunk.retrieval_score !== null && (
                    <span className="cmp-chunk-rscore">
                      ret. {(chunk.retrieval_score * 100).toFixed(0)}%
                    </span>
                  )}
                  <span className="cmp-chunk-attrs">
                    {attrPerModel.map(({ name, value }) => (
                      <span
                        key={name}
                        className={`cmp-chunk-attr${value !== null && value < 0 ? " neg" : ""}`}
                      >
                        {name}: {value !== null ? formatScore(value) : "—"}
                      </span>
                    ))}
                  </span>
                </div>
                {expanded && <p className="cmp-chunk-text">{chunk.text}</p>}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ─── Attribution tab (supplementary) ─────────────────────────────────────────

function AttributionTab({ run }: { run: CompareRun }) {
  const completed = run.model_results.filter((r) => r.status === "completed" && r.result);

  if (completed.length === 0) {
    return (
      <div className="cmp-attr">
        <p className="cmp-empty-note">No completed runs.</p>
      </div>
    );
  }

  const maxAbsAttr = Math.max(
    ...completed.flatMap((r) =>
      (r.result!.results[0]?.attributions ?? []).map((a) => Math.abs(a.attribution))
    ),
    0.01
  );

  const topPositive = completed.map((mr) => {
    const attrs = mr.result!.results[0]?.attributions ?? [];
    const top = [...attrs].sort((a, b) => b.attribution - a.attribution)[0];
    return { mr, attr: top };
  });

  const topNegative = completed
    .map((mr) => {
      const attrs = mr.result!.results[0]?.attributions ?? [];
      const neg = [...attrs].sort((a, b) => a.attribution - b.attribution)[0];
      return { mr, attr: neg };
    })
    .filter(({ attr }) => attr && attr.attribution < 0);

  const hasPairwise = completed.some((r) => r.result!.results[0]?.strongest_pair);

  return (
    <div className="cmp-attr">
      <div className="cmp-section">
        <h3 className="cmp-section-title">Top positive contributor</h3>
        <AttrTable rows={topPositive} maxAbs={maxAbsAttr} />
      </div>

      {topNegative.length > 0 && (
        <div className="cmp-section">
          <h3 className="cmp-section-title">Highest negative attribution</h3>
          <AttrTable rows={topNegative} maxAbs={maxAbsAttr} />
        </div>
      )}

      {hasPairwise && (
        <div className="cmp-section">
          <h3 className="cmp-section-title">Strongest pairwise interaction</h3>
          <div className="cmp-attr-table">
            <div className="cmp-attr-header">
              <span>Model</span>
              <span>Pair</span>
              <span className="cmp-attr-val-head">Value</span>
              <span />
            </div>
            {completed.map((mr) => {
              const active = mr.result!.results[0];
              if (!active?.strongest_pair) return null;
              const val = active.strongest_pair_value;
              const barPct = Math.min((Math.abs(val) / maxAbsAttr) * 100, 100);
              return (
                <div key={mr.model_id} className="cmp-attr-row">
                  <span className="cmp-attr-model">{mr.model_display_name}</span>
                  <span className="cmp-attr-chunk">{active.strongest_pair}</span>
                  <span className={`cmp-attr-val${val < 0 ? " neg" : ""}`}>
                    {val >= 0 ? "+" : ""}
                    {formatScore(val)}
                  </span>
                  <div className="cmp-attr-bar-wrap">
                    <div
                      className={`cmp-attr-bar-fill${val < 0 ? " neg" : ""}`}
                      style={{ width: `${barPct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

function AttrTable({
  rows,
  maxAbs,
}: {
  rows: { mr: ModelRunResult; attr: { chunk: string; attribution: number } | undefined }[];
  maxAbs: number;
}) {
  return (
    <div className="cmp-attr-table">
      <div className="cmp-attr-header">
        <span>Model</span>
        <span>Chunk</span>
        <span className="cmp-attr-val-head">Attribution</span>
        <span />
      </div>
      {rows.map(({ mr, attr }) => {
        if (!attr) return null;
        const val = attr.attribution;
        const barPct = Math.min((Math.abs(val) / maxAbs) * 100, 100);
        return (
          <div key={mr.model_id} className="cmp-attr-row">
            <span className="cmp-attr-model">{mr.model_display_name}</span>
            <span className="cmp-attr-chunk" title={attr.chunk}>
              {attr.chunk}
            </span>
            <span className={`cmp-attr-val${val < 0 ? " neg" : ""}`}>
              {val >= 0 ? "+" : ""}
              {formatScore(val)}
            </span>
            <div className="cmp-attr-bar-wrap">
              <div
                className={`cmp-attr-bar-fill${val < 0 ? " neg" : ""}`}
                style={{ width: `${barPct}%` }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
