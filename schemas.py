"""
ReliAI — Shared Data Contracts
===============================
Single source of truth for every data structure that crosses a layer-boundary
in the pipeline:

    Detection            --EvidenceEvent-->       Reasoning (Graph + RCA)
    Reasoning (RCA)       --RCAResult-->           Remediation (Remediation/Verification)
    Remediation           --RemediationPlan-->     Remediation (Verification)
    Remediation (Verification)--VerificationResult--> Incident Report (assembled by Reasoning/evaluation)

RULE: nobody changes a field in this file without a 2-minute message to the
other two layers first. Everyone imports FROM this file — nobody redefines
these classes locally. If your component needs a field that isn't here,
add it here first, then use it.

Scope note: this contract assumes the reduced project scope — ONE failure
domain (data quality), NetworkX (not Neo4j), no Postgres/FAISS/OTel/Docker.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Shared enums
# ---------------------------------------------------------------------------

class FailureType(str, Enum):
    """Ground-truth label injected by the fault-injection benchmark.
    Owned by Detection. Keep this list matched to whatever you actually inject —
    don't leave unused types in here, it makes evaluation metrics misleading.
    """
    MISSING_VALUES = "missing_values"
    CORRUPTED_VALUES = "corrupted_values"
    SCHEMA_MISMATCH = "schema_mismatch"
    DUPLICATES = "duplicates"
    FEATURE_DRIFT = "feature_drift"
    LABEL_SHIFT = "label_shift"


class DetectionMethod(str, Enum):
    """Which statistical test flagged the anomaly. Used for evidence
    grounding — the RCA agent should be able to point back to *why* a node
    exists, not just that it exists.
    """
    KS_TEST = "ks_test"
    PSI = "population_stability_index"
    JENSEN_SHANNON = "jensen_shannon_divergence"
    ISOLATION_FOREST = "isolation_forest"
    ROLLING_ACCURACY = "rolling_accuracy"
    MISSING_VALUE_RATE = "missing_value_rate"
    DUPLICATE_ROW_RATE = "duplicate_row_rate"
    SCHEMA_CHECK = "schema_check"


class ConfidenceLevel(str, Enum):
    """Coarse bucket used for human-readable reports. Every numeric
    confidence score should also be mappable to one of these."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Shared inference: DetectionMethod -> FailureType, and which FailureType
# pairs are causally plausible. Lives here rather than in
# reasoning/rca_agent.py or reasoning/evidence_graph.py because both need
# the exact same rules, and per CONTRACT.md rule 1, shared contract logic
# belongs in this file, imported everywhere, never redefined locally.
# ---------------------------------------------------------------------------

DETECTION_METHOD_TO_FAILURE_TYPE: dict[DetectionMethod, FailureType] = {
    DetectionMethod.KS_TEST: FailureType.FEATURE_DRIFT,
    DetectionMethod.PSI: FailureType.FEATURE_DRIFT,
    DetectionMethod.JENSEN_SHANNON: FailureType.FEATURE_DRIFT,
    DetectionMethod.ISOLATION_FOREST: FailureType.CORRUPTED_VALUES,
    DetectionMethod.ROLLING_ACCURACY: FailureType.LABEL_SHIFT,
    DetectionMethod.MISSING_VALUE_RATE: FailureType.MISSING_VALUES,
    DetectionMethod.DUPLICATE_ROW_RATE: FailureType.DUPLICATES,
    DetectionMethod.SCHEMA_CHECK: FailureType.SCHEMA_MISMATCH,
}


def infer_failure_type(detection_method: Optional[DetectionMethod]) -> Optional[FailureType]:
    """Map a detection method to a best-guess failure category via the
    fixed rule table above. Returns None if no method is known or none of
    the table's entries match -- callers must treat that as "unknown", not
    silently default to a guess. Never derived from
    EvidenceEvent.ground_truth_label -- per CONTRACT.md rule 4, that field
    must not reach the reasoning layer. detection_method is a legitimate
    structural signal instead.
    """
    if detection_method is None:
        return None
    return DETECTION_METHOD_TO_FAILURE_TYPE.get(DetectionMethod(detection_method))


# Which FailureType pairs are plausibly causally related, independent of
# WHEN they were detected. reasoning/evidence_graph.py's edge-building used
# to treat "detected close together in time" as sufficient evidence of
# causality on its own -- which happens to be correct for a genuine
# multi-stage cascade (a schema/corruption event upstream manifesting as
# feature drift, which then shows up as a prediction/label shift), but
# produces a false story the moment truly independent real-world failures
# are detected in the same DataAgent run: DataAgent timestamps every
# EvidenceEvent with wall-clock detection time, not real anomaly-onset
# time, so *everything* found in one run is always "close together in
# time" -- proximity alone cannot tell "these are connected" apart from
# "we just happened to check every column in the same pass" (see
# evaluation/real_data_incident.py's four-fault real-data run, which
# time-proximity-only edges used to chain into a single, misleading root
# cause). An edge now additionally requires the two nodes' inferred
# FailureTypes to appear here (in either order), or to be identical -- a
# small, auditable, hand-curated rule table, matching this project's "LLM
# reasons, code/rules act" philosophy, not a second guess from the LLM.
CAUSALLY_PLAUSIBLE_FAILURE_TYPE_PAIRS: set[frozenset] = {
    frozenset({FailureType.CORRUPTED_VALUES, FailureType.FEATURE_DRIFT}),
    frozenset({FailureType.FEATURE_DRIFT, FailureType.LABEL_SHIFT}),
    frozenset({FailureType.CORRUPTED_VALUES, FailureType.LABEL_SHIFT}),
    frozenset({FailureType.SCHEMA_MISMATCH, FailureType.MISSING_VALUES}),
    frozenset({FailureType.SCHEMA_MISMATCH, FailureType.CORRUPTED_VALUES}),
    frozenset({FailureType.SCHEMA_MISMATCH, FailureType.FEATURE_DRIFT}),
    frozenset({FailureType.SCHEMA_MISMATCH, FailureType.LABEL_SHIFT}),
}
# Deliberately NOT included, because there is no well-known causal story:
#   - DUPLICATES paired with anything else -- duplicate rows are a distinct
#     batch/ETL failure mode with no standard causal link to a schema,
#     drift, corruption, or label-shift incident.
#   - MISSING_VALUES paired with FEATURE_DRIFT or LABEL_SHIFT -- an
#     elevated null rate and a distributional shift in the non-null values
#     are different statistical signatures; assuming one implies the other
#     would itself be exactly the kind of unjustified link this table
#     exists to prevent.
# Extend this table only with a real, statable reason -- the same way the
# entries above have one -- never just to make some scenario link up.


def are_failure_types_causally_plausible(
    a: Optional[FailureType], b: Optional[FailureType]
) -> bool:
    """Is there a real, hand-curated reason to believe a FailureType-`a`
    node and a FailureType-`b` node could be causally related, independent
    of how close together they were detected?

    True when a == b (same failure category, different feature -- always
    plausible, e.g. a multi-feature drift incident) or when the unordered
    pair appears in CAUSALLY_PLAUSIBLE_FAILURE_TYPE_PAIRS. False whenever
    either is None: an unmapped/unknown detection_method is evidence of
    nothing, not evidence of implausibility, but the conservative default
    here is still to NOT assert a link the graph has no real basis for --
    same principle as the rest of this table.
    """
    if a is None or b is None:
        return False
    if a == b:
        return True
    return frozenset({a, b}) in CAUSALLY_PLAUSIBLE_FAILURE_TYPE_PAIRS


# ---------------------------------------------------------------------------
# DETECTION -> REASONING contract
# ---------------------------------------------------------------------------

class EvidenceEvent(BaseModel):
    """One anomaly detection finding. This is the atomic unit of evidence
    that Detection's Data Agent produces and Reasoning's Evidence Graph
    consumes. Every field here must be something a statistical test can
    actually populate — no LLM-generated fields belong in this class.
    """
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime
    detection_method: DetectionMethod
    feature_name: Optional[str] = None          # e.g. "customer_age"; None if pipeline-level
    metric_value: float                          # the raw statistic (e.g. KS D-statistic, PSI value)
    threshold: float                              # threshold that was crossed
    confidence: float = Field(ge=0.0, le=1.0)    # normalized 0-1, not the raw metric
    description: str                              # one-line human-readable summary
    ground_truth_label: Optional[FailureType] = None  # only set in benchmark runs, not production

    model_config = ConfigDict(use_enum_values=True)


# ---------------------------------------------------------------------------
# REASONING: Evidence Graph internals
# ---------------------------------------------------------------------------

class EvidenceNode(BaseModel):
    """A node in the NetworkX evidence graph. Built from one or more
    EvidenceEvents that Reasoning decides are related (same feature, close
    in time, etc.) — that clustering logic lives in Reasoning's code, not here.
    """
    node_id: str = Field(default_factory=lambda: str(uuid4()))
    node_type: str            # e.g. "feature_drift", "schema_change", "prediction_shift"
    source_events: list[str]  # list of EvidenceEvent.event_id that support this node
    timestamp: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    detection_method: Optional[DetectionMethod] = None
    # The (mode of the) source events' DetectionMethod, carried forward by
    # the graph builder. NOT ground_truth_label — this is how the anomaly
    # was statistically detected, a legitimate structural signal the RCA
    # layer is allowed to see. It's what lets RootCauseHypothesis.failure_type
    # below be inferred by a rule instead of guessed from prose.


class EvidenceEdge(BaseModel):
    """A directed edge representing propagation: source_node caused/preceded
    target_node. This is what the RCA agent walks to reconstruct the causal
    chain — required by the doc's "Evidence Graph" section.
    """
    source_node_id: str
    target_node_id: str
    relationship: str    # e.g. "caused", "preceded", "correlated_with"
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str         # short justification string


# ---------------------------------------------------------------------------
# REASONING -> REMEDIATION contract
# ---------------------------------------------------------------------------

class RootCauseHypothesis(BaseModel):
    """One ranked candidate explanation. The RCA agent should emit several
    of these, ranked — not just a single answer — so the report can show
    the reasoning process, which matters for the viva.
    """
    rank: int
    node_id: str                 # which EvidenceNode this hypothesis centers on
    explanation: str             # natural-language causal chain, written by the LLM
    confidence: float = Field(ge=0.0, le=1.0)
    supporting_node_ids: list[str]   # the causal path through the graph
    failure_type: Optional[FailureType] = None
    # Rule-derived (NOT ground_truth_label) best guess at the failure
    # category, e.g. from EvidenceNode.detection_method. None when no
    # detection method is available or none maps cleanly. This is what lets
    # Person C's Remediation Agent choose a RemediationCategory
    # deterministically instead of text-mining `explanation`.


class RCAResult(BaseModel):
    """Final output of the RCA Agent. This is what Remediation's Remediation
    Agent consumes — it should never need to touch the raw graph.
    """
    incident_id: str
    hypotheses: list[RootCauseHypothesis]   # sorted, hypotheses[0] = top pick
    causal_path_node_ids: list[str]          # the full reconstructed chain, in order
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# REMEDIATION internals: Remediation -> Verification
# ---------------------------------------------------------------------------

class RemediationCategory(str, Enum):
    DATA_FIX = "data_fix"                # e.g. impute, drop, re-validate schema
    RETRAIN = "retrain"
    ROLLBACK = "rollback"
    CONFIG_CHANGE = "config_change"


class RemediationPlan(BaseModel):
    """Output of the Remediation Agent. Deliberately does NOT execute
    anything itself — it just proposes. Execution is the Verification
    Agent's job, kept separate so a bad proposal can't silently run.
    """
    incident_id: str
    category: RemediationCategory
    action_description: str        # what to actually do, in concrete terms
    expected_outcome: str          # what metric should improve, and roughly how much
    target_root_cause_node_id: str  # ties back to RCAResult.hypotheses[i].node_id


class VerificationResult(BaseModel):
    """Output of the Verification Agent — the empirical check on whether
    the remediation actually worked. This is what makes ReliAI more than
    an LLM opinion generator: every claim here should be a before/after
    metric delta, not a paraphrase of the remediation's own claim.
    """
    incident_id: str
    metric_name: str
    value_before: float
    value_after: float
    improved: bool
    replay_sample_size: int     # how many held-out records were replayed
    notes: str = ""


# ---------------------------------------------------------------------------
# Benchmark scoring helpers — shared by IncidentReport.root_cause_correct()
# and evaluation/benchmark_runner.py, which scores RCAResults directly
# without needing a full IncidentReport (see CONTRACT.md rule 4: only
# reachable here via evidence_events/evidence_nodes, never ground truth
# leaking into the reasoning layer itself).
# ---------------------------------------------------------------------------

def resolve_hypothesis_root_event_ids(
    hypothesis: RootCauseHypothesis,
    evidence_events: list[EvidenceEvent],
    evidence_nodes: Optional[list[EvidenceNode]] = None,
) -> set[str]:
    """Resolve a hypothesis's root (``node_id``) back to the EvidenceEvent
    id(s) it actually traces to, regardless of which ID space it was
    written in.

    RCAAgent (structured) writes an EvidenceNode.node_id there; NaiveRCAAgent
    (baseline) writes an EvidenceEvent.event_id directly — see
    reasoning/rca_agent.py's NaiveRCAAgent docstring for why. A naive
    ``hypothesis.node_id == some_event_id`` comparison would silently treat
    every structured hypothesis as wrong, since node ids and event ids are
    drawn from disjoint ``uuid4()`` spaces. This resolves both down to the
    same event_id space instead:

      1. If ``node_id`` matches an ``EvidenceNode.node_id`` in
         ``evidence_nodes``, return that node's ``source_events``.
      2. Else, if ``node_id`` is itself a known ``EvidenceEvent.event_id``,
         return it directly (the baseline case).
      3. Else, return an empty set — the hypothesis doesn't resolve to any
         evidence we were given, so it can't be scored as correct.
    """
    nodes_by_id = {node.node_id: node for node in (evidence_nodes or [])}
    node = nodes_by_id.get(hypothesis.node_id)
    if node is not None:
        return set(node.source_events)

    valid_event_ids = {event.event_id for event in evidence_events}
    if hypothesis.node_id in valid_event_ids:
        return {hypothesis.node_id}

    return set()


def root_cause_is_correct(
    rca_result: RCAResult,
    ground_truth_label: Optional[FailureType],
    evidence_events: list[EvidenceEvent],
    evidence_nodes: Optional[list[EvidenceNode]] = None,
) -> bool:
    """Did the top-ranked hypothesis's root actually originate from
    evidence carrying the injected ``ground_truth_label``?

    This is the single shared scoring predicate for benchmark runs — used by
    ``IncidentReport.root_cause_correct()`` and directly by
    ``evaluation/benchmark_runner.py`` (which has no need to construct a
    full IncidentReport just to score an RCAResult). Only ``hypotheses[0]``
    is checked: ranking correctness (whether the *right* candidate made it
    to rank 1) is what "root cause accuracy" means here, not whether the
    correct answer appears anywhere in the list.

    If the resolved root traces to more than one EvidenceEvent (a structured
    node can cluster several), the hypothesis counts as correct if *any* of
    them carries the injected label — clustering is expected to group
    same-cause evidence, so a partial match still means the hypothesis
    correctly localized the incident.
    """
    if ground_truth_label is None or not rca_result.hypotheses:
        return False

    top = rca_result.hypotheses[0]
    event_ids = resolve_hypothesis_root_event_ids(top, evidence_events, evidence_nodes)
    if not event_ids:
        return False

    labels_by_event_id = {event.event_id: event.ground_truth_label for event in evidence_events}
    return any(labels_by_event_id.get(event_id) == ground_truth_label for event_id in event_ids)


def root_cause_rank(
    rca_result: RCAResult,
    ground_truth_label: Optional[FailureType],
    evidence_events: list[EvidenceEvent],
    evidence_nodes: Optional[list[EvidenceNode]] = None,
) -> Optional[int]:
    """Where in the agent's OWN ranked hypothesis list does the correct
    root cause first appear?

    ``root_cause_is_correct`` only ever looks at ``hypotheses[0]`` (Hit@1).
    That's the right definition for "root cause accuracy" as a pass/fail
    metric, but it throws away information: an agent that puts the true
    cause at rank 2 is doing something very different from one that never
    surfaces it at all, and Hit@1 scores both as "WRONG" identically. This
    walks the full ranked list (not just the top entry) and returns the
    1-based ``rank`` of the first hypothesis that resolves back to evidence
    carrying ``ground_truth_label``, or ``None`` if no hypothesis in the
    list does -- the correct cause never made it into the agent's
    consideration set at all, not even at a low rank.

    Callers derive Hit@k as ``rank is not None and rank <= k``, and
    reciprocal rank as ``1/rank if rank else 0`` for averaging into MRR
    (Mean Reciprocal Rank) across incidents. Intentionally returns the
    hypothesis's own ``rank`` field rather than its list position, so a
    gap or reorder in how an agent numbers its hypotheses is reflected
    faithfully rather than silently renumbered -- with one correction: a
    hypothesis that repeats an earlier hypothesis's cause (same resolved
    root events) does not take a rank slot, so the value returned is the
    rank among DISTINCT causes.
    """
    if ground_truth_label is None:
        return None

    labels_by_event_id = {event.event_id: event.ground_truth_label for event in evidence_events}
    seen_causes: set[frozenset[str]] = set()
    repeats_skipped = 0
    for hypothesis in rca_result.hypotheses:
        event_ids = resolve_hypothesis_root_event_ids(hypothesis, evidence_events, evidence_nodes)
        if any(labels_by_event_id.get(event_id) == ground_truth_label for event_id in event_ids):
            return hypothesis.rank - repeats_skipped
        cause = frozenset(event_ids)
        if cause and cause in seen_causes:
            # The same cause listed again under a new explanation. A reader
            # sees one cause, not two, so it must not consume a rank slot --
            # otherwise an agent that repeats itself gets pushed out of the
            # top 3 by its own duplicates. (RCAAgent can't repeat a node;
            # NaiveRCAAgent can, so counting repeats penalized only naive.)
            repeats_skipped += 1
        seen_causes.add(cause)
    return None


# ---------------------------------------------------------------------------
# Final assembled output — owned by Reasoning, as part of the evaluation/
# benchmark layer (see docs/TASKS.md)
# ---------------------------------------------------------------------------

class IncidentReport(BaseModel):
    """The final artifact shown in the Streamlit frontend and saved as the
    project's evaluation unit. One of these is produced per injected
    incident during benchmark runs, which is what your results table in
    the report/paper will be built from.
    """
    incident_id: str = Field(default_factory=lambda: str(uuid4()))
    detected_at: datetime
    evidence_events: list[EvidenceEvent]
    evidence_nodes: list[EvidenceNode] = Field(default_factory=list)
    rca_result: RCAResult
    remediation_plan: RemediationPlan
    verification_result: VerificationResult
    ground_truth_label: Optional[FailureType] = None  # for benchmark scoring only

    def root_cause_correct(self) -> bool:
        """Convenience check for benchmark scoring: did the top hypothesis's
        root resolve back to evidence carrying the injected ground truth
        label? Only meaningful when ground_truth_label is set (i.e. during
        evaluation, not production). See root_cause_is_correct() for the
        resolution logic — this just supplies it with this report's data.

        evidence_nodes may be left empty for reports built from a naive
        (non-graph) RCAResult, since hypothesis.node_id is then already an
        EvidenceEvent.event_id and resolves without it.
        """
        return root_cause_is_correct(
            self.rca_result, self.ground_truth_label, self.evidence_events, self.evidence_nodes
        )


# ---------------------------------------------------------------------------
# Self-test — run `python schemas.py` to sanity-check the contract compiles
# and a full incident can round-trip through every stage.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    event = EvidenceEvent(
        timestamp=datetime.now(timezone.utc),
        detection_method=DetectionMethod.PSI,
        feature_name="customer_age",
        metric_value=0.31,
        threshold=0.2,
        confidence=0.87,
        description="PSI for customer_age exceeded threshold (0.31 > 0.2)",
        ground_truth_label=FailureType.FEATURE_DRIFT,
    )

    rca = RCAResult(
        incident_id="demo-001",
        hypotheses=[
            RootCauseHypothesis(
                rank=1,
                node_id=event.event_id,
                explanation="Feature drift in customer_age likely caused prediction shift",
                confidence=0.82,
                supporting_node_ids=[event.event_id],
            )
        ],
        causal_path_node_ids=[event.event_id],
    )

    remediation = RemediationPlan(
        incident_id="demo-001",
        category=RemediationCategory.RETRAIN,
        action_description="Retrain model on last 30 days of data to absorb drift",
        expected_outcome="Rolling accuracy should recover by ~4-6%",
        target_root_cause_node_id=event.event_id,
    )

    verification = VerificationResult(
        incident_id="demo-001",
        metric_name="rolling_accuracy",
        value_before=0.71,
        value_after=0.79,
        improved=True,
        replay_sample_size=500,
    )

    report = IncidentReport(
        detected_at=event.timestamp,
        evidence_events=[event],
        rca_result=rca,
        remediation_plan=remediation,
        verification_result=verification,
        ground_truth_label=FailureType.FEATURE_DRIFT,
    )

    print("Contract compiles. Sample IncidentReport:")
    print(report.model_dump_json(indent=2))
    print("\nroot_cause_correct():", report.root_cause_correct())
