import type { DetectionRecallRow } from "../../api/types";

export default function DetectionRecallTable({ rows }: { rows: DetectionRecallRow[] }) {
  const detected = rows.filter((r) => r.detected).length;
  return (
    <div className="surface overflow-x-auto rounded-lg">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[var(--gridline)] text-left text-xs uppercase tracking-wide text-muted">
            <th className="px-4 py-2">Injected fault</th>
            <th className="px-4 py-2">Detector</th>
            <th className="px-4 py-2">Detected</th>
            <th className="px-4 py-2">Detector confidence</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--gridline)]">
          {rows.map((r) => {
            const conf = r.confidence ?? 0;
            const weak = conf < 0.3;
            return (
              <tr key={r.name}>
                <td className="px-4 py-2 font-mono text-xs">{r.name}</td>
                <td className="px-4 py-2 text-secondary">{r.detection_methods.join(", ") || "—"}</td>
                <td className="px-4 py-2" style={{ color: r.detected ? "var(--status-good)" : "var(--status-critical)" }}>
                  {r.detected ? "✓ yes" : "✗ no"}
                </td>
                <td className="px-4 py-2">
                  <span className="inline-flex items-center gap-2" style={{ fontVariantNumeric: "tabular-nums" }}>
                    <span className="inline-block h-1.5 w-24 rounded-full" style={{ background: "var(--gridline)" }}>
                      <span
                        className="block h-1.5 rounded-full"
                        style={{ width: `${Math.max(2, conf * 100)}%`, background: weak ? "var(--status-warning)" : "var(--series-structured)" }}
                      />
                    </span>
                    {conf.toFixed(2)}
                    {weak && <span className="text-xs" style={{ color: "var(--status-warning)" }}>▲ weak signal</span>}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="border-t border-[var(--gridline)] px-4 py-2 text-xs text-muted">
        Detection recall: {detected}/{rows.length} faults detected at all
      </div>
    </div>
  );
}
