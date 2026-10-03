import type { AblationResult } from "../../api/types";

const pct = (v: number) => `${Math.round(v * 100)}%`;

export default function AblationTable({ result }: { result: AblationResult }) {
  const entries = Object.entries(result.configs);
  return (
    <div className="surface overflow-x-auto rounded-lg">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[var(--gridline)] text-left text-xs uppercase tracking-wide text-muted">
            <th className="px-4 py-2">Configuration</th>
            <th className="px-4 py-2">Cases scored</th>
            <th className="px-4 py-2">Structured Hit@1</th>
            <th className="px-4 py-2">Structured Hit@3</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--gridline)]">
          {entries.map(([name, row]) => (
            <tr key={name} className={name === "baseline" ? "bg-[var(--gridline)]/40" : undefined}>
              <td className="px-4 py-2 font-mono text-xs">
                {name === "baseline" ? "all detectors on" : name.replace("without_", "− ")}
              </td>
              <td className="px-4 py-2" style={{ fontVariantNumeric: "tabular-nums" }}>{row.n_incidents}/6</td>
              <td className="px-4 py-2" style={{ fontVariantNumeric: "tabular-nums" }}>{pct(row.structured_accuracy)}</td>
              <td className="px-4 py-2" style={{ fontVariantNumeric: "tabular-nums" }}>{pct(row.structured_hit3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
