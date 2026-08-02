# ReliAI — Task Breakdown (Week 1)

Copy each section below into a GitHub Issue, assign it, done. Each task is
scoped to be finishable independently — nobody needs to wait on anyone
else's code to start, only on `schemas.py` and `CONTRACT.md` (already in
the repo).

---

## 🟦 Detection Layer

**Folder:** `detection/`
**Produces:** `schemas.EvidenceEvent`
**Reads:** `configs/settings.py` → `DetectionThresholds`

### Issue 1: Implement first anomaly detector
- Pick ONE detector to start: KS test (for feature distribution shift) is
  the easiest to get working end-to-end first.
- Function signature: takes a reference distribution + current distribution
  (numpy arrays / pandas Series), returns an `EvidenceEvent`.
- Threshold comes from `configs.settings.DETECTION.ks_test_pvalue` — do not
  hardcode it.
- **Acceptance criteria:** given two clearly different distributions, the
  detector returns an `EvidenceEvent` with `confidence` correctly reflecting
  how far the statistic is from threshold. Given two identical
  distributions, it should NOT fire (or should fire with low confidence).

### Issue 2: Fault injection harness
- Write a function that takes clean data and injects one failure type from
  `schemas.FailureType` (start with `FEATURE_DRIFT` — shift a column's
  distribution by some controllable amount).
- Every injected fault must set `EvidenceEvent.ground_truth_label` when the
  detector later flags it — this is what makes benchmark scoring possible.
- **Acceptance criteria:** can generate N synthetic incidents with known
  ground truth, each reproducible via a fixed random seed.

### Issue 3: Data Agent wrapper
- Wraps 2-3 detectors (KS test + PSI + Isolation Forest) into one
  `DataAgent` class with an `investigate(data) -> list[EvidenceEvent]` method.
- **Acceptance criteria:** runs against the fault-injection harness output
  and correctly flags the injected fault in at least 80% of synthetic cases
  (tune thresholds if it doesn't — write down what you tried).

---

## 🟩 Reasoning Layer (Evidence Graph + RCA)

**Folder:** `reasoning/`
**Consumes:** `schemas.EvidenceEvent`
**Produces:** `schemas.RCAResult`

### Issue 1: Evidence graph builder
- Build a NetworkX `DiGraph` from a list of `EvidenceEvent`s.
- Cluster related events into `schemas.EvidenceNode`s (same feature, close
  in time — use `configs.settings.GRAPH.time_window_minutes`).
- Add `schemas.EvidenceEdge`s between nodes that plausibly propagate
  (earlier node → later node, above `GRAPH.min_edge_confidence`).
- **Acceptance criteria:** given 3-4 synthetic `EvidenceEvent`s with known
  time ordering, the graph correctly links them in the right causal
  direction.

### Issue 2: RCA Agent (LangGraph)
- Takes the evidence graph, walks it, and produces ranked
  `RootCauseHypothesis` objects wrapped in an `RCAResult`.
- Start simple: LLM reasons over the graph structure (nodes + edges +
  confidence), not raw telemetry — this is the core research claim, don't
  let it slip into "just paste the CSV into the prompt."
- Cap hypotheses at `configs.settings.RCA.max_hypotheses`.
- **Acceptance criteria:** on a synthetic single-cause incident, the top
  hypothesis correctly identifies the injected fault type in its
  explanation text.

### Issue 3: Baseline comparison agent 
- A second, deliberately dumb RCA path: same LLM, but fed raw telemetry
  directly with no graph, no statistical pre-filtering.
- This is your control group for the research question — you need this to
  claim the evidence-graph approach is actually better, not just different.
- **Acceptance criteria:** produces an `RCAResult` in the same schema, so
  it can be scored with the same evaluation code.

---

## 🟨 Remediation Layer (Remediation + Verification + Frontend)

**Folder:** `remediation/`
**Consumes:** `schemas.RCAResult`
**Produces:** `schemas.RemediationPlan`, `schemas.VerificationResult`, `schemas.IncidentReport`

### Issue 1: Remediation Agent
- Takes the top `RootCauseHypothesis` from an `RCAResult`, maps it to a
  `RemediationCategory`, and produces a `RemediationPlan`.
- Start with a simple rule-based mapping (feature drift → RETRAIN, schema
  mismatch → DATA_FIX) before reaching for an LLM here — keep it
  deterministic where you can, per the "LLM reasons, stats/rules act" split.
- **Acceptance criteria:** given each `FailureType`, produces a sensible
  `RemediationCategory` and a concrete `action_description`.

### Issue 2: Verification Agent
- Given a `RemediationPlan`, replay a held-out sample of data
  (`configs.settings.VERIFICATION.replay_sample_size`) and compute a
  before/after metric.
- Produces a `VerificationResult` — `improved=True` only if the delta
  exceeds `VERIFICATION.improvement_threshold_pct`.
- **Acceptance criteria:** on synthetic data where you know the fix should
  help, `improved` comes back `True`; on a no-op remediation, it comes back
  `False`.

### Issue 3: Streamlit frontend
- Displays an `IncidentReport`: evidence timeline, evidence graph
  visualization, ranked hypotheses, remediation plan, before/after
  verification metrics.
- **Acceptance criteria:** can load a saved `IncidentReport` (JSON) and
  render all five sections without crashing on missing/optional fields.

---

## 🟪 Shared (everyone)

- Keep `docs/PROJECT_ROADMAP.md` updated weekly.
- `evaluation/benchmark_runner.py` — runs fault injection → full pipeline →
  scores `IncidentReport.root_cause_correct()` across N incidents. Whoever
  finishes their layer first should start this, since it's what the other
  two layers plug into for testing.
- The report/paper — don't let one person write it alone; each of you
  writes the section for your own layer, since you're the one who can
  actually defend it in viva.
