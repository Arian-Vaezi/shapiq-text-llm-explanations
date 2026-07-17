import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { Attribution, ExplanationResult, RunResult } from "../types";
import { buildChunkPreview, formatScore } from "../utils";

type Chunk = RunResult["chunks"][number];

function EvidenceRow({
  rank,
  item,
  chunk,
  max,
}: {
  rank: number;
  item: Attribution;
  chunk: Chunk | undefined;
  max: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const isPositive = item.attribution >= 0;
  const barWidth = `${Math.max(2, (Math.abs(item.attribution) / max) * 100)}%`;
  const preview = chunk ? buildChunkPreview(chunk.text) : "";
  const hasMore = chunk ? chunk.text.length > preview.length : false;

  const metaParts = [
    `retrieval rank ${item.player}`,
    chunk?.chunk_type?.replace(/_/g, " "),
    chunk?.page_number != null ? `page ${chunk.page_number}` : undefined,
    item.loo != null ? `LOO ${formatScore(item.loo)}` : undefined,
    chunk?.retrieval_score != null ? `score ${chunk.retrieval_score.toFixed(3)}` : undefined,
  ].filter((p): p is string => p != null);

  return (
    <article className="evidence-card">
      <div className="evidence-header">
        <span className="evidence-rank" title="Ranked by absolute Shapley attribution">
          φ{String(rank).padStart(2, "0")}
        </span>
        <span className="evidence-title">{item.chunk}</span>
        <span className={`evidence-score ${isPositive ? "positive" : "negative"}`}>
          {formatScore(item.attribution)}
        </span>
      </div>
      <div className="evidence-bar-track">
        <div
          className={`evidence-bar ${isPositive ? "positive" : "negative"}`}
          style={{ width: barWidth }}
        />
      </div>
      <p className="evidence-preview">{preview}</p>
      <div className="evidence-footer">
        {metaParts.length > 0 && (
          <span className="evidence-meta">{metaParts.join(" · ")}</span>
        )}
        {hasMore && (
          <button className="expand-btn" onClick={() => setExpanded(!expanded)}>
            {expanded ? (
              <><ChevronDown size={10} /> Collapse</>
            ) : (
              <><ChevronRight size={10} /> Show full text</>
            )}
          </button>
        )}
      </div>
      {expanded && chunk && (
        <div className="evidence-full-text">{chunk.text}</div>
      )}
    </article>
  );
}

interface Props {
  result: ExplanationResult;
  chunks: RunResult["chunks"];
}

export function EvidencePanel({ result, chunks }: Props) {
  const max = Math.max(...result.attributions.map((a) => Math.abs(a.attribution)), 0.001);

  return (
    <div className="evidence-panel card">
      <div className="panel-header">
        <h3 className="panel-title">Evidence attribution</h3>
        <p className="panel-caption">
          Chunks are ranked by absolute Shapley contribution, not retrieval score.
          Positive values support the target answer; negative values work against it.
        </p>
      </div>
      <div className="evidence-list">
        {result.attributions.map((item, i) => (
          <EvidenceRow
            key={item.player}
            rank={i + 1}
            item={item}
            chunk={chunks[item.player - 1]}
            max={max}
          />
        ))}
      </div>
    </div>
  );
}
