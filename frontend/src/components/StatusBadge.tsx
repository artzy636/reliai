// Status color is never carried by hue alone (dataviz skill's status-color
// rule) -- every badge pairs the color with an icon glyph and a text
// label, so it still reads correctly for colorblind viewers or in print.
export default function StatusBadge({ good, label }: { good: boolean; label: string }) {
  const color = good ? "var(--status-good)" : "var(--status-critical)";
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-sm font-medium"
      style={{ color, background: good ? "rgba(12,163,12,0.1)" : "rgba(208,59,59,0.1)" }}
    >
      <span aria-hidden="true">{good ? "✓" : "✗"}</span>
      {label}
    </span>
  );
}
