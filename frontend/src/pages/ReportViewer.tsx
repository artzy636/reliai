import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { IncidentReport, ReportSummary } from "../api/types";
import IncidentSummary from "../components/report/IncidentSummary";
import EvidenceTimeline from "../components/report/EvidenceTimeline";
import EvidenceGraph from "../components/report/EvidenceGraph";
import RootCauseAnalysis from "../components/report/RootCauseAnalysis";
import RemediationPlan from "../components/report/RemediationPlan";
import VerificationResult from "../components/report/VerificationResult";

const LIVE = "__live__";

export default function ReportViewer() {
  const [summaries, setSummaries] = useState<ReportSummary[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [report, setReport] = useState<IncidentReport | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .listReports()
      .then((list) => {
        setSummaries(list);
        if (list.length > 0) setSelected(list[0].name);
      })
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!selected || selected === LIVE) return;
    setLoading(true);
    setError(null);
    api
      .getReport(selected)
      .then(setReport)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [selected]);

  const runLive = () => {
    setLoading(true);
    setError(null);
    setSelected(LIVE);
    api
      .runLive()
      .then(setReport)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  };

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-center gap-3">
        <select
          className="surface rounded-md px-3 py-2 text-sm"
          value={selected === LIVE ? "" : selected}
          onChange={(e) => setSelected(e.target.value)}
        >
          <option value="" disabled>
            Choose a saved report…
          </option>
          {summaries.map((s) => (
            <option key={s.name} value={s.name}>
              {s.label}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={runLive}
          className="rounded-md px-3 py-2 text-sm font-semibold text-white"
          style={{ background: "var(--series-structured)" }}
        >
          Run live pipeline
        </button>
        {selected === LIVE && <span className="text-sm text-muted">Freshly generated (stub LLM)</span>}
      </div>

      {loading && <p className="text-secondary">Running pipeline…</p>}
      {error && <p style={{ color: "var(--status-critical)" }}>{error}</p>}

      {report && !loading && (
        <div className="space-y-8">
          <IncidentSummary report={report} />
          <EvidenceTimeline events={report.evidence_events} />
          <section>
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">
              3. Evidence Graph
            </h2>
            <EvidenceGraph
              nodes={report.evidence_nodes}
              edges={report.edges}
              highlightNodeId={report.top_node_id}
            />
          </section>
          <RootCauseAnalysis rcaResult={report.rca_result} />
          <RemediationPlan plan={report.remediation_plan} />
          <VerificationResult result={report.verification_result} />
        </div>
      )}
    </div>
  );
}
