import type { VerificationResult as VerificationResultT } from "../../api/types";
import StatCard from "../StatCard";
import StatusBadge from "../StatusBadge";

export default function VerificationResult({ result }: { result: VerificationResultT }) {
  const delta = result.value_after - result.value_before;
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

  return (
    <section>
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">
        6. Verification Result
      </h2>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatCard label={`${result.metric_name} (before)`} value={pct(result.value_before)} />
        <StatCard label={`${result.metric_name} (after)`} value={pct(result.value_after)} />
        <StatCard
          label="Change"
          value={
            <span style={{ color: delta >= 0 ? "var(--status-good)" : "var(--status-critical)" }}>
              {delta >= 0 ? "+" : ""}
              {pct(delta)}
            </span>
          }
          sub={`replay n=${result.replay_sample_size}`}
        />
      </div>
      <div className="mt-3">
        <StatusBadge good={result.improved} label={result.improved ? "Improved" : "Did not improve"} />
      </div>
      {result.notes && <p className="mt-2 text-sm text-secondary">{result.notes}</p>}
    </section>
  );
}
