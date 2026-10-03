import type { EvidenceEvent } from "../../api/types";

export default function EvidenceTimeline({ events }: { events: EvidenceEvent[] }) {
  if (events.length === 0) {
    return (
      <section>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">
          2. Evidence Timeline
        </h2>
        <p className="text-secondary">No evidence events.</p>
      </section>
    );
  }

  const sorted = [...events].sort((a, b) => a.timestamp.localeCompare(b.timestamp));

  return (
    <section>
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">
        2. Evidence Timeline
      </h2>
      <ol className="surface divide-y divide-[var(--gridline)] rounded-lg">
        {sorted.map((event) => (
          <li key={event.event_id} className="flex flex-wrap items-baseline gap-x-3 gap-y-1 px-4 py-3">
            <span className="text-sm font-medium" style={{ fontVariantNumeric: "tabular-nums" }}>
              {new Date(event.timestamp).toLocaleTimeString(undefined, { hour12: false })}
            </span>
            <span className="rounded bg-[var(--gridline)] px-1.5 py-0.5 text-xs font-mono">
              {event.feature_name ?? "pipeline-level"}
            </span>
            <span className="text-sm text-secondary">{event.detection_method}</span>
            <span className="text-sm text-muted">confidence {event.confidence.toFixed(2)}</span>
            <p className="w-full text-sm text-secondary">{event.description}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}
