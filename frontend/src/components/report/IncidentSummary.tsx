import type { IncidentReport } from "../../api/types";
import StatCard from "../StatCard";
import StatusBadge from "../StatusBadge";

export default function IncidentSummary({ report }: { report: IncidentReport }) {
  return (
    <section>
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">
        1. Incident Summary
      </h2>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <StatCard label="Incident ID" value={<span className="text-base">{report.incident_id.slice(0, 12)}…</span>} />
        <StatCard
          label="Detected At"
          value={
            <span className="text-base">
              {new Date(report.detected_at).toLocaleString(undefined, {
                dateStyle: "medium",
                timeStyle: "short",
              })}
            </span>
          }
        />
        <StatCard
          label="Ground Truth"
          value={<span className="text-base">{report.ground_truth_label ?? "—"}</span>}
        />
      </div>
      {report.root_cause_correct !== null && (
        <div className="mt-3">
          <StatusBadge
            good={report.root_cause_correct}
            label={
              report.root_cause_correct
                ? "Root cause correctly identified against ground truth"
                : "Root cause NOT correctly identified against ground truth"
            }
          />
        </div>
      )}
    </section>
  );
}
