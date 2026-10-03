import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { AblationResult, BenchmarkResult } from "../api/types";
import StatCard from "../components/StatCard";
import AccuracyBarChart from "../components/benchmark/AccuracyBarChart";
import PerIncidentTable from "../components/benchmark/PerIncidentTable";
import DetectionRecallTable from "../components/benchmark/DetectionRecallTable";
import AblationTable from "../components/benchmark/AblationTable";

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

export default function BenchmarkComparison() {
  const [synthetic, setSynthetic] = useState<BenchmarkResult | null>(null);
  const [realData, setRealData] = useState<BenchmarkResult | null>(null);
  const [ablation, setAblation] = useState<AblationResult | null>(null);
  const [tab, setTab] = useState<"synthetic" | "real">("synthetic");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.syntheticBenchmark(), api.realDataBenchmark()])
      .then(([s, r]) => {
        setSynthetic(s);
        setRealData(r);
      })
      .catch((e) => setError(String(e)));
    api.ablation().then(setAblation).catch(() => setAblation(null));
  }, []);

  if (error) {
    return <p style={{ color: "var(--status-critical)" }}>{error}</p>;
  }
  if (!synthetic || !realData) {
    return <p className="text-secondary">Loading benchmark results…</p>;
  }

  const active = tab === "synthetic" ? synthetic : realData;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-xl font-semibold">Structured vs. Naive RCA</h1>
        <p className="mt-1 text-sm text-secondary">
          Graph-structured root-cause analysis vs. a raw-events-straight-into-one-LLM-call baseline, run with the
          real Gemini model on both sides.
        </p>
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <StatCard
          label="Synthetic — structured"
          value={pct(synthetic.structured_accuracy_mean)}
          sub={`± ${pct(synthetic.structured_accuracy_stdev)} stdev`}
        />
        <StatCard
          label="Synthetic — naive"
          value={pct(synthetic.naive_accuracy_mean)}
          sub={`± ${pct(synthetic.naive_accuracy_stdev)} stdev`}
        />
        <StatCard
          label="Real data — structured"
          value={pct(realData.structured_accuracy_mean)}
          sub={`± ${pct(realData.structured_accuracy_stdev)} stdev`}
        />
        <StatCard
          label="Real data — naive"
          value={pct(realData.naive_accuracy_mean)}
          sub={`± ${pct(realData.naive_accuracy_stdev)} stdev`}
        />
      </div>

      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">Mean accuracy ± stdev</h2>
        <div className="surface rounded-lg p-4">
          <AccuracyBarChart synthetic={synthetic} realData={realData} />
        </div>
      </section>

      <section>
        <div className="mb-3 flex items-center gap-1">
          {(["synthetic", "real"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => setTab(t)}
              className="rounded-md px-3 py-1.5 text-sm font-medium"
              style={
                tab === t
                  ? { background: "var(--series-structured)", color: "white" }
                  : { color: "var(--text-secondary)" }
              }
            >
              {t === "synthetic" ? "Synthetic incidents (12)" : "Real Adult/Census data (6)"}
            </button>
          ))}
        </div>
        <PerIncidentTable result={active} />
      </section>

      {realData.structured_hit3_mean !== undefined && (
        <section>
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">
            Beyond top-1: Hit@3 and MRR (real data)
          </h2>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <StatCard label="Structured Hit@3" value={pct(realData.structured_hit3_mean ?? 0)} />
            <StatCard label="Naive Hit@3" value={pct(realData.naive_hit3_mean ?? 0)} />
            <StatCard label="Structured MRR" value={(realData.structured_mrr_mean ?? 0).toFixed(2)} />
            <StatCard label="Naive MRR" value={(realData.naive_mrr_mean ?? 0).toFixed(2)} />
          </div>
        </section>
      )}

      {realData.detection_recall && (
        <section>
          <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide text-muted">
            Detection quality (real data)
          </h2>
          <p className="mb-3 text-sm text-secondary">
            Measured before any RCA reasoning: did the detector see the injected fault, and how strongly?
          </p>
          <DetectionRecallTable rows={realData.detection_recall} />
        </section>
      )}

      {ablation && (
        <section>
          <h2 className="mb-1 text-sm font-semibold uppercase tracking-wide text-muted">
            Detector ablation (real data)
          </h2>
          <p className="mb-3 text-sm text-secondary">
            Each row disables one detector. A drop to 5/6 cases means that detector was the only one able to see its fault.
          </p>
          <AblationTable result={ablation} />
        </section>
      )}

      <section className="surface rounded-lg p-4 text-sm text-secondary">
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-muted">What this shows</h2>
        {tab === "synthetic" ? (
          <p>
            Structured RCA is perfectly stable across repeats (0% stdev) because it reasons over the evidence
            graph's actual structure rather than guessing from raw text. The four incidents marked{" "}
            <span className="rounded border px-1 text-xs" style={{ borderColor: "var(--border)" }}>
              trap
            </span>{" "}
            are built with a distractor that reads more confident or longer in plain text than the true root —
            exactly where naive is prone to being fooled, and structured isn't.
          </p>
        ) : (
          <p>
            On real Adult/Census data, structured matches or ties naive on every fault type except two —{" "}
            <span className="font-mono text-xs">corrupted_values_age_and_education_num</span> and{" "}
            <span className="font-mono text-xs">label_shift_income</span> — where both agents fail identically
            (0/3 for each). The detection table above shows why: the detector did fire on both (6/6 recall), but with a very weak signal (0.00 and 0.16), while unrelated sampling noise chained together at full confidence. Weak confidence alone isn't fatal (duplicate_rows succeeds at 0.09), so the failure comes from the weak signal combined with a stronger, better-connected distractor. That is a limitation of the shared detection layer, not a difference between the two RCA approaches.
          </p>
        )}
      </section>
    </div>
  );
}
