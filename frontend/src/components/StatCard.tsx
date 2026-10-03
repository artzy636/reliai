import type { ReactNode } from "react";

export default function StatCard({
  label,
  value,
  sub,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
}) {
  return (
    <div className="surface rounded-lg px-4 py-3">
      <div className="text-xs font-medium uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1 text-2xl font-semibold" style={{ fontVariantNumeric: "tabular-nums" }}>
        {value}
      </div>
      {sub && <div className="mt-0.5 text-sm text-secondary">{sub}</div>}
    </div>
  );
}
