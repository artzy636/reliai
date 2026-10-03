import type { AblationResult, BenchmarkResult, IncidentReport, ReportSummary } from "./types";

// Vite dev server proxies /api to the FastAPI backend (see vite.config.ts),
// so this stays a relative path in both dev and a same-origin production
// build -- no env var to configure per environment.
const BASE = "/api";

async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response wasn't JSON -- keep statusText
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listReports: () => getJSON<ReportSummary[]>("/reports"),
  getReport: (name: string) => getJSON<IncidentReport>(`/reports/${name}`),
  runLive: () => getJSON<IncidentReport>("/reports/live", { method: "POST" }),
  syntheticBenchmark: () => getJSON<BenchmarkResult>("/benchmarks/synthetic"),
  realDataBenchmark: () => getJSON<BenchmarkResult>("/benchmarks/real-data"),
  ablation: () => getJSON<AblationResult>("/benchmarks/ablation"),
};
