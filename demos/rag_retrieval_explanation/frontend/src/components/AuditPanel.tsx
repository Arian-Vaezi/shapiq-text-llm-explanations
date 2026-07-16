import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { ExplanationResult } from "../types";
import { formatScore } from "../utils";

interface Props {
  result: ExplanationResult;
}

export function AuditPanel({ result }: Props) {
  const data = result.deletion_curve.map((pt, i) => ({
    step: i,
    attribution: pt.score,
    random: result.random_deletion_baseline[i],
    ...(result.retrieval_deletion_baseline
      ? { retrieval: result.retrieval_deletion_baseline[i] }
      : {}),
  }));

  return (
    <div className="card audit-panel">
      <h3 className="panel-title">Faithfulness audit</h3>
      <p className="panel-caption">
        Deletion behavior — lower faster means the attribution order is more faithful.
      </p>

      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={210}>
          <LineChart data={data}>
            <CartesianGrid stroke="#e5e5e5" vertical={false} />
            <XAxis
              dataKey="step"
              tickLine={false}
              axisLine={false}
              tick={{ fontSize: 11, fill: "#999999" }}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              domain={["auto", "auto"]}
              tick={{ fontSize: 11, fill: "#999999" }}
            />
            <Tooltip
              contentStyle={{
                fontSize: 12,
                border: "1px solid #e5e5e5",
                borderRadius: 6,
                background: "#ffffff",
                boxShadow: "0 2px 8px rgba(0,0,0,.08)",
              }}
            />
            <Legend wrapperStyle={{ fontSize: 12 }} />
            <Line
              type="monotone"
              dataKey="attribution"
              stroke="#111110"
              strokeWidth={2}
              dot={{ r: 3 }}
            />
            <Line
              type="monotone"
              dataKey="random"
              stroke="#aaaaaa"
              strokeDasharray="4 4"
              dot={false}
            />
            {result.retrieval_deletion_baseline && (
              <Line
                type="monotone"
                dataKey="retrieval"
                stroke="#777777"
                dot={false}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className="audit-metrics">
        <div className="audit-metric">
          <span className="metric-label">Deletion drop</span>
          <strong>{formatScore(result.faithfulness.deletion_drop, false)}</strong>
        </div>
        <div className="audit-metric">
          <span className="metric-label">Top-only score</span>
          <strong>{formatScore(result.faithfulness.score_with_top_only, false)}</strong>
        </div>
        <div className="audit-metric">
          <span className="metric-label">Sufficiency gap</span>
          <strong>{formatScore(result.faithfulness.sufficiency_gap, false)}</strong>
        </div>
      </div>
    </div>
  );
}
