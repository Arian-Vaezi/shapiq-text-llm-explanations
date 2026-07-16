import type { ExplanationResult } from "../types";
import { formatScore } from "../utils";

interface Props {
  result: ExplanationResult;
}

export function InteractionPanel({ result }: Props) {
  const n = result.pairwise_matrix.length;
  const max = Math.max(...result.pairwise_matrix.flat().map(Math.abs), 0.001);

  return (
    <div className="card interaction-panel">
      <h3 className="panel-title">Interaction matrix</h3>
      {result.strongest_pair && (
        <p className="panel-caption">
          Strongest pair:{" "}
          <span className={result.strongest_pair_value >= 0 ? "positive" : "negative"}>
            {formatScore(result.strongest_pair_value)}
          </span>
          {" — "}
          {result.strongest_pair}
        </p>
      )}
      <div className="interaction-legend">
        <span><i className="legend-swatch positive-bg" /> Positive: complementarity</span>
        <span><i className="legend-swatch negative-bg" /> Negative: redundancy or interference</span>
      </div>
      <div
        className="matrix"
        style={{ gridTemplateColumns: `28px repeat(${n}, 1fr)` }}
      >
        <span />
        {result.pairwise_matrix.map((_, i) => (
          <b key={`h${i}`}>{i + 1}</b>
        ))}
        {result.pairwise_matrix.map((row, i) => [
          <b key={`v${i}`}>{i + 1}</b>,
          ...row.map((val, j) => {
            const alpha = Math.abs(val) / max;
            const bg =
              val >= 0
                ? `rgba(22, 101, 52, ${alpha})`
                : `rgba(153, 27, 27, ${alpha})`;
            return (
              <span
                key={`${i}-${j}`}
                className="matrix-cell"
                style={{ background: bg }}
                title={`${i + 1}×${j + 1}: ${val.toFixed(3)}`}
              >
                {i === j ? "·" : val.toFixed(2)}
              </span>
            );
          }),
        ])}
      </div>
    </div>
  );
}
