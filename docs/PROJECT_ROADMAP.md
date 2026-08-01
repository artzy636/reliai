# ReliAI — Roadmap

Update this after each week's standup. Keep it honest — "in progress" is a
better status than a checked box that isn't actually true, especially
because your viva defense depends on knowing exactly what's really working.

## Milestone 1: MVP (data-quality domain only)

| Week | Detection Layer | Reasoning Layer | Remediation Layer |
|---|---|---|---|
| 1 | Repo setup, agree on `schemas.py` | Repo setup, read `CONTRACT.md` | Repo setup, sketch Streamlit skeleton |
| 2 | Implement 2-3 detectors (KS test, PSI, Isolation Forest) | Set up NetworkX graph structure, define node/edge builder | Sketch RemediationPlan categories for data-quality fixes |
| 3 | Fault injection harness with ground truth labels | Wire graph builder to consume `EvidenceEvent` | Verification agent: before/after metric replay logic |
| 4 | Data Agent end-to-end on synthetic data | RCA Agent (LangGraph) producing ranked hypotheses | Wire Remediation + Verification together |
| 5 | Integrate with RCAEval dataset | Evaluate causal path accuracy on RCAEval | Streamlit frontend showing IncidentReport |
| 6 | Buffer / bug fixing | Buffer / bug fixing | Buffer / bug fixing |

## Milestone 2: benchmark + evaluation

- Run full pipeline end-to-end on RCAEval subset
- Score root-cause accuracy, causal-path accuracy against ground truth
- Compare against an LLM-only baseline (no evidence graph, no statistical
  detection — just raw telemetry into an LLM) — this comparison is your
  actual research contribution, don't skip it

## Explicitly postponed (not deleted — future work in the report)

- Pipeline Agent, Model Agent, Infrastructure Agent
- Neo4j (using NetworkX instead)
- PostgreSQL, FAISS, OpenTelemetry, Docker
- Multi-domain fault coverage beyond data quality

## Status key

- ✅ Done and tested
- 🔧 In progress
- ⏳ Not started
- ⚠️ Blocked — note what it's blocked on
