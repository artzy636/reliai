import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import clsx from "clsx";

const NAV_ITEMS = [
  { to: "/", label: "Report Viewer" },
  { to: "/benchmarks", label: "Benchmark Results" },
];

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen">
      <header className="surface sticky top-0 z-10">
        <div className="mx-auto flex max-w-6xl items-center gap-8 px-6 py-4">
          <span className="text-lg font-semibold tracking-tight">ReliAI</span>
          <nav className="flex gap-1">
            {NAV_ITEMS.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                className={({ isActive }) =>
                  clsx(
                    "rounded-md px-3 py-1.5 text-sm font-medium transition-colors",
                    isActive
                      ? "bg-[var(--series-structured)] text-white"
                      : "text-secondary hover:text-[var(--text-primary)]",
                  )
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
    </div>
  );
}
