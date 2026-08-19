# ReliAI — Team Data Contract

This pairs with `schemas.py`, which is the actual contract (code, not prose —
code can't silently drift out of sync the way a doc can).

## Why this exists

With 3 people each using AI coding assistants independently, the biggest risk
isn't code quality — it's that each person's AI-generated code quietly
assumes a different data shape. This file is the fix: agree on the shapes
*once*, commit `schemas.py`, and everyone imports from it instead of
redefining classes locally.

## Ownership map

| Layer | Owner | Produces | Consumes |
|---|---|---|---|
| Detection & Data Agent | Detection | `EvidenceEvent` | raw pipeline data |
| Evidence Graph + RCA Agent | Reasoning | `EvidenceNode`, `EvidenceEdge`, `RCAResult`; also assembles `IncidentReport` (`evaluation/` layer) | `EvidenceEvent` (from Detection) |
| Remediation + Verification + Frontend | Remediation | `RemediationPlan`, `VerificationResult` | `RCAResult` (from Reasoning); frontend consumes/displays the assembled `IncidentReport` |

## Rules

1. **Import, don't redefine.** `from schemas import EvidenceEvent` — nobody
   writes their own version of these classes, even a "temporary" one.
2. **Changing a field is a team decision.** If your component needs a field
   that isn't here, add it to `schemas.py` first and ping the other two
   before you build against it. A silent field change breaks whoever's
   downstream of you without them knowing why.
3. **Confidence is always 0–1, normalized.** Raw statistical values (KS
   D-statistic, PSI score, etc.) go in `metric_value`, not `confidence`.
   This matters for anyone later comparing confidence across different
   detection methods.
4. **`ground_truth_label` only exists during benchmark runs.** It's how you
   score root-cause accuracy against RCAEval / your injected faults. Don't
   let it leak into anything that resembles a "production" code path — the
   RCA Agent should never see it, or your accuracy numbers are meaningless.
5. **Run `python schemas.py` before you push** if you touch this file. It
   round-trips a full incident through every stage and will catch a broken
   contract before it breaks someone else's branch.

## Suggested repo layout

```
reliai/
├── schemas.py              # this file — shared contract, edited by consensus
├── detection/               # Detection
│   ├── anomaly_detectors.py    # KS test, PSI, Isolation Forest, etc.
│   ├── data_agent.py            # emits EvidenceEvent
│   └── fault_injection.py       # benchmark harness, ground truth labels
├── reasoning/                # Reasoning
│   ├── evidence_graph.py        # NetworkX graph builder, consumes EvidenceEvent
│   └── rca_agent.py             # LangGraph agent, emits RCAResult
├── remediation/               # Remediation
│   ├── remediation_agent.py     # emits RemediationPlan
│   ├── verification_agent.py    # replay + before/after metrics, emits VerificationResult
│   └── app.py                    # Streamlit frontend, displays a finished IncidentReport
└── evaluation/
    └── benchmark_runner.py     # runs fault injection -> full pipeline -> scores results;
                                  # also where Reasoning assembles the final IncidentReport
```

## What's deliberately NOT in scope

Per the reduced project scope: no Neo4j (NetworkX is enough), no PostgreSQL
(in-memory / SQLite), no FAISS, no OpenTelemetry, no Docker unless someone
wants the infra experience as a stretch goal. One failure domain (data
quality) fully implemented; Pipeline Agent and Model Agent documented as
"designed, not implemented" in the report. Keep it that way unless all three
of you agree to expand — scope creep here is the main risk to finishing.
