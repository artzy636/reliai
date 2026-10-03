import { useState } from "react";
import type { RCAResult } from "../../api/types";

export default function RootCauseAnalysis({ rcaResult }: { rcaResult: RCAResult }) {
  const [expanded, setExpanded] = useState(false);
  const top = rcaResult.hypotheses[0];

  return (
    <section>
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">
        4. Root Cause Analysis
      </h2>
      {!top ? (
        <p className="text-secondary">No hypotheses produced.</p>
      ) : (
        <>
          <div className="surface rounded-lg p-4">
            <div className="text-sm font-semibold text-secondary">Top hypothesis (rank {top.rank})</div>
            <dl className="mt-2 grid grid-cols-1 gap-1 text-sm sm:grid-cols-3">
              <div>
                <dt className="text-muted">Node</dt>
                <dd className="font-mono">{top.node_id.slice(0, 12)}…</dd>
              </div>
              <div>
                <dt className="text-muted">Confidence</dt>
                <dd style={{ fontVariantNumeric: "tabular-nums" }}>{top.confidence.toFixed(2)}</dd>
              </div>
              <div>
                <dt className="text-muted">Failure type</dt>
                <dd>{top.failure_type}</dd>
              </div>
            </dl>
            <p className="mt-3 text-sm">{top.explanation}</p>
          </div>

          {rcaResult.hypotheses.length > 1 && (
            <div className="mt-2">
              <button
                type="button"
                onClick={() => setExpanded((e) => !e)}
                className="text-sm font-medium text-[var(--series-structured)]"
              >
                {expanded ? "Hide" : "Show"} all {rcaResult.hypotheses.length} hypotheses
              </button>
              {expanded && (
                <ol className="surface mt-2 divide-y divide-[var(--gridline)] rounded-lg">
                  {rcaResult.hypotheses.map((h) => (
                    <li key={h.node_id} className="px-4 py-3">
                      <div className="flex items-baseline gap-2 text-sm">
                        <span className="font-semibold">#{h.rank}</span>
                        <span className="font-mono text-xs text-muted">{h.node_id.slice(0, 12)}…</span>
                        <span className="text-muted" style={{ fontVariantNumeric: "tabular-nums" }}>
                          conf {h.confidence.toFixed(2)}
                        </span>
                        <span className="text-muted">{h.failure_type}</span>
                      </div>
                      <p className="mt-1 text-sm text-secondary">{h.explanation}</p>
                    </li>
                  ))}
                </ol>
              )}
            </div>
          )}
        </>
      )}
    </section>
  );
}
