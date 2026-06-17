import { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import type { ExplanationResult, RunResult } from "../types";

interface Props {
  result: RunResult;
  active: ExplanationResult;
}

export function Diagnostics({ result, active }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className="diagnostics card">
      <button className="diagnostics-toggle" onClick={() => setOpen(!open)}>
        {open ? <ChevronDown size={13} /> : <ChevronRight size={13} />}
        Diagnostics / Advanced
      </button>

      {open && (
        <div className="diagnostics-body">
          <div className="diag-section">
            <div className="diag-label">Value function</div>
            <div className="diag-row">
              <span className="diag-item">{active.value_name}</span>
              {active.value_description && (
                <span className="diag-desc">{active.value_description}</span>
              )}
            </div>
            {active.value_formula && (
              <code className="diag-code">{active.value_formula}</code>
            )}
          </div>

          {result.retrieval_debug && (
            <div className="diag-section">
              <div className="diag-label">Retrieval debug</div>
              <pre className="diag-pre">
                {JSON.stringify(result.retrieval_debug, null, 2)}
              </pre>
            </div>
          )}

          <div className="diag-section">
            <div className="diag-label">Raw chunks ({result.chunks.length})</div>
            {result.chunks.map((chunk, i) => (
              <div key={i} className="diag-chunk">
                <div className="diag-chunk-header">
                  <span>chunk {i + 1}</span>
                  <span>{chunk.chunk_type}</span>
                  {chunk.page_number != null && <span>p.{chunk.page_number}</span>}
                  {chunk.retrieval_score != null && (
                    <span>score {chunk.retrieval_score.toFixed(3)}</span>
                  )}
                </div>
                <p className="diag-chunk-text">{chunk.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
