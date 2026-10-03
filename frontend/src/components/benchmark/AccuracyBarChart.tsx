import {
  Bar,
  BarChart,
  CartesianGrid,
  ErrorBar,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { BenchmarkResult } from "../../api/types";

interface Row {
  name: string;
  structured: number;
  structuredErr: number;
  naive: number;
  naiveErr: number;
}

function pct(v: number) {
  return `${(v * 100).toFixed(1)}%`;
}

// Custom tooltip: text always stays in ink tokens, never the series color
// (dataviz skill's "text wears text tokens" rule) -- the little swatch
// carries the color identity instead.
function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="surface rounded-md px-3 py-2 text-sm shadow-sm">
      <div className="font-medium">{label}</div>
      {payload.map((p: any) => (
        <div key={p.dataKey} className="mt-1 flex items-center gap-1.5">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ background: p.dataKey === "structured" ? "var(--series-structured)" : "var(--series-naive)" }}
          />
          <span className="text-secondary">{p.dataKey === "structured" ? "Structured" : "Naive"}:</span>
          <span className="font-medium" style={{ fontVariantNumeric: "tabular-nums" }}>
            {pct(p.value)}
            {p.dataKey === "structured" ? ` ± ${pct(p.payload.structuredErr)}` : ` ± ${pct(p.payload.naiveErr)}`}
          </span>
        </div>
      ))}
    </div>
  );
}

export default function AccuracyBarChart({
  synthetic,
  realData,
}: {
  synthetic: BenchmarkResult;
  realData: BenchmarkResult;
}) {
  const data: Row[] = [
    {
      name: `Synthetic (${synthetic.n_repeats}× repeats)`,
      structured: synthetic.structured_accuracy_mean,
      structuredErr: synthetic.structured_accuracy_stdev,
      naive: synthetic.naive_accuracy_mean,
      naiveErr: synthetic.naive_accuracy_stdev,
    },
    {
      name: `Real data (${realData.n_repeats}× repeats)`,
      structured: realData.structured_accuracy_mean,
      structuredErr: realData.structured_accuracy_stdev,
      naive: realData.naive_accuracy_mean,
      naiveErr: realData.naive_accuracy_stdev,
    },
  ];

  return (
    <div style={{ height: 340 }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 24, right: 16, left: 0, bottom: 0 }} barGap={6}>
          <CartesianGrid vertical={false} stroke="var(--gridline)" />
          <XAxis
            dataKey="name"
            tick={{ fill: "var(--text-secondary)", fontSize: 13 }}
            axisLine={{ stroke: "var(--baseline)" }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(v) => `${Math.round(v * 100)}%`}
            tick={{ fill: "var(--text-muted)", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
            width={44}
          />
          <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--gridline)", opacity: 0.4 }} />
          <Legend
            formatter={(value) => (value === "structured" ? "Structured (graph RCA)" : "Naive (raw events → LLM)")}
            wrapperStyle={{ fontSize: 13, color: "var(--text-secondary)" }}
          />
          <Bar dataKey="structured" name="structured" fill="var(--series-structured)" radius={[4, 4, 0, 0]} maxBarSize={64}>
            <ErrorBar dataKey="structuredErr" stroke="var(--text-muted)" width={6} strokeWidth={1.5} />
          </Bar>
          <Bar dataKey="naive" name="naive" fill="var(--series-naive)" radius={[4, 4, 0, 0]} maxBarSize={64}>
            <ErrorBar dataKey="naiveErr" stroke="var(--text-muted)" width={6} strokeWidth={1.5} />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
