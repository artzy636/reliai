import type { BenchmarkResult } from "../../api/types";

// The "trap" incidents are the ones specifically built with a distractor
// that reads more confident/longer in raw text than the true (isolated or
// weak) root -- exactly what naive is prone to falling for. Highlighting
// them here (border ring, not color-alone) is where the demo story lives.
const TRAP_INCIDENTS = new Set([
  "benchmark-duplicates-003",
  "benchmark-temporal-trap-008",
  "benchmark-label-shift-trap-011",
  "benchmark-schema-duplicates-trap-012",
]);

function Cell({ hits, total }: { hits: number; total: number }) {
  const allCorrect = hits === total;
  const allWrong = hits === 0;
  const color = allCorrect ? "var(--status-good)" : allWrong ? "var(--status-critical)" : "var(--status-warning)";
  return (
    <span className="font-medium" style={{ color, fontVariantNumeric: "tabular-nums" }}>
      {hits}/{total}
    </span>
  );
}

export default function PerIncidentTable({ result }: { result: BenchmarkResult }) {
  const rows = Object.entries(result.per_incident_hit_rates);
  return (
    <div className="surface overflow-x-auto rounded-lg">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[var(--gridline)] text-left text-xs uppercase tracking-wide text-muted">
            <th className="px-4 py-2">Incident</th>
            <th className="px-4 py-2">Structured</th>
            <th className="px-4 py-2">Naive</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--gridline)]">
          {rows.map(([id, hit]) => (
            <tr key={id} className={TRAP_INCIDENTS.has(id) ? "bg-[var(--gridline)]/40" : undefined}>
              <td className="px-4 py-2">
                <span className="font-mono text-xs">{id}</span>
                {TRAP_INCIDENTS.has(id) && (
                  <span className="ml-2 rounded border px-1.5 py-0.5 text-[10px] font-medium text-muted" style={{ borderColor: "var(--border)" }}>
                    trap
                  </span>
                )}
              </td>
              <td className="px-4 py-2">
                <Cell hits={hit.structured_hits} total={hit.total} />
              </td>
              <td className="px-4 py-2">
                <Cell hits={hit.naive_hits} total={hit.total} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
