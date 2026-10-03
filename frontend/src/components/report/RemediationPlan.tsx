import type { RemediationPlan as RemediationPlanT } from "../../api/types";

const CATEGORY_LABEL: Record<string, string> = {
  data_fix: "Data fix",
  retrain: "Retrain",
  rollback: "Rollback",
  config_change: "Config change",
};

export default function RemediationPlan({ plan }: { plan: RemediationPlanT }) {
  return (
    <section>
      <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-muted">
        5. Remediation Plan
      </h2>
      <div className="surface rounded-lg p-4">
        <span
          className="inline-block rounded-full px-2.5 py-1 text-xs font-semibold text-white"
          style={{ background: "var(--series-structured)" }}
        >
          {CATEGORY_LABEL[plan.category] ?? plan.category}
        </span>
        <dl className="mt-3 space-y-2 text-sm">
          <div>
            <dt className="text-muted">Action</dt>
            <dd>{plan.action_description}</dd>
          </div>
          <div>
            <dt className="text-muted">Expected outcome</dt>
            <dd>{plan.expected_outcome}</dd>
          </div>
        </dl>
      </div>
    </section>
  );
}
