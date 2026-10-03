"""
Smoke test for the shared contract. If this fails, nobody's layer will work
together correctly — fix this before building on top of schemas.py.
"""

from datetime import datetime, timezone

from schemas import (
    DetectionMethod,
    EvidenceEvent,
    EvidenceNode,
    FailureType,
    IncidentReport,
    RCAResult,
    RemediationCategory,
    RemediationPlan,
    RootCauseHypothesis,
    VerificationResult,
    resolve_hypothesis_root_event_ids,
    root_cause_is_correct,
    root_cause_rank,
)


def test_evidence_event_creation():
    event = EvidenceEvent(
        timestamp=datetime.now(timezone.utc),
        detection_method=DetectionMethod.PSI,
        feature_name="customer_age",
        metric_value=0.31,
        threshold=0.2,
        confidence=0.87,
        description="PSI exceeded threshold",
        ground_truth_label=FailureType.FEATURE_DRIFT,
    )
    assert 0.0 <= event.confidence <= 1.0


def test_full_incident_round_trip():
    event = EvidenceEvent(
        timestamp=datetime.now(timezone.utc),
        detection_method=DetectionMethod.PSI,
        feature_name="customer_age",
        metric_value=0.31,
        threshold=0.2,
        confidence=0.87,
        description="PSI exceeded threshold",
        ground_truth_label=FailureType.FEATURE_DRIFT,
    )

    rca = RCAResult(
        incident_id="test-001",
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
        incident_id="test-001",
        category=RemediationCategory.RETRAIN,
        action_description="Retrain on last 30 days",
        expected_outcome="Accuracy recovers ~4-6%",
        target_root_cause_node_id=event.event_id,
    )

    verification = VerificationResult(
        incident_id="test-001",
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

    assert report.root_cause_correct() is True


# ---------------------------------------------------------------------------
# ID-space resolution: RCAAgent hypotheses carry an EvidenceNode.node_id;
# NaiveRCAAgent hypotheses carry an EvidenceEvent.event_id directly (see
# reasoning/rca_agent.py's NaiveRCAAgent docstring). A naive `node_id ==
# event_id` comparison would silently score every structured hypothesis as
# wrong, since the two ids are drawn from disjoint uuid4() spaces. These
# tests pin down that resolve_hypothesis_root_event_ids() / root_cause_is_
# correct() bridge both spaces correctly instead of producing that silent
# false negative.
# ---------------------------------------------------------------------------

def _drift_event(confidence: float = 0.9) -> EvidenceEvent:
    return EvidenceEvent(
        timestamp=datetime.now(timezone.utc),
        detection_method=DetectionMethod.PSI,
        feature_name="customer_age",
        metric_value=0.31,
        threshold=0.2,
        confidence=confidence,
        description="PSI exceeded threshold",
        ground_truth_label=FailureType.FEATURE_DRIFT,
    )


def test_resolve_hypothesis_root_event_ids_naive_event_id_space():
    """A NaiveRCAAgent-style hypothesis whose node_id is itself an
    EvidenceEvent.event_id resolves to exactly that event, with no
    evidence_nodes needed at all."""
    event = _drift_event()
    hypothesis = RootCauseHypothesis(
        rank=1, node_id=event.event_id, explanation="naive guess",
        confidence=0.7, supporting_node_ids=[event.event_id],
    )

    resolved = resolve_hypothesis_root_event_ids(hypothesis, evidence_events=[event])

    assert resolved == {event.event_id}


def test_resolve_hypothesis_root_event_ids_structured_node_id_space():
    """An RCAAgent-style hypothesis whose node_id is an EvidenceNode.node_id
    (never equal to any EvidenceEvent.event_id) must resolve through
    evidence_nodes' source_events, not by comparing node_id to event_id."""
    event_a, event_b = _drift_event(0.9), _drift_event(0.85)
    node = EvidenceNode(
        node_type="feature_anomaly:customer_age",
        source_events=[event_a.event_id, event_b.event_id],
        timestamp=event_a.timestamp,
        confidence=0.88,
    )
    hypothesis = RootCauseHypothesis(
        rank=1, node_id=node.node_id, explanation="structured guess",
        confidence=0.88, supporting_node_ids=[node.node_id],
    )
    # Sanity: the node id genuinely lives in a different id space.
    assert node.node_id not in {event_a.event_id, event_b.event_id}

    resolved = resolve_hypothesis_root_event_ids(
        hypothesis, evidence_events=[event_a, event_b], evidence_nodes=[node]
    )

    assert resolved == {event_a.event_id, event_b.event_id}


def test_resolve_hypothesis_root_event_ids_unknown_id_returns_empty():
    """A hypothesis id that matches neither a known node nor a known event
    resolves to an empty set rather than guessing."""
    event = _drift_event()
    hypothesis = RootCauseHypothesis(
        rank=1, node_id="totally-unrelated-id", explanation="?",
        confidence=0.5, supporting_node_ids=["totally-unrelated-id"],
    )

    resolved = resolve_hypothesis_root_event_ids(hypothesis, evidence_events=[event])

    assert resolved == set()


def test_root_cause_is_correct_structured_node_id_matches_ground_truth():
    """The scoring predicate must credit a structured (node_id-space)
    hypothesis whose resolved source events carry the injected label --
    this is the exact case a naive id string-compare would silently fail."""
    event_a, event_b = _drift_event(0.9), _drift_event(0.85)
    node = EvidenceNode(
        node_type="feature_anomaly:customer_age",
        source_events=[event_a.event_id, event_b.event_id],
        timestamp=event_a.timestamp,
        confidence=0.88,
    )
    rca_result = RCAResult(
        incident_id="incident-x",
        hypotheses=[
            RootCauseHypothesis(
                rank=1, node_id=node.node_id, explanation="structured guess",
                confidence=0.88, supporting_node_ids=[node.node_id],
            )
        ],
        causal_path_node_ids=[node.node_id],
    )

    assert root_cause_is_correct(
        rca_result, FailureType.FEATURE_DRIFT, [event_a, event_b], [node]
    ) is True


def test_root_cause_is_correct_structured_id_without_evidence_nodes_is_false_not_true():
    """If evidence_nodes isn't supplied, a structured node_id cannot be
    resolved at all -- this must come back False (unscoreable), never a
    false positive from accidentally matching the raw node_id string."""
    event = _drift_event()
    node = EvidenceNode(
        node_type="feature_anomaly:customer_age",
        source_events=[event.event_id],
        timestamp=event.timestamp,
        confidence=0.9,
    )
    rca_result = RCAResult(
        incident_id="incident-y",
        hypotheses=[
            RootCauseHypothesis(
                rank=1, node_id=node.node_id, explanation="structured guess",
                confidence=0.9, supporting_node_ids=[node.node_id],
            )
        ],
        causal_path_node_ids=[node.node_id],
    )

    assert root_cause_is_correct(rca_result, FailureType.FEATURE_DRIFT, [event]) is False


# ---------------------------------------------------------------------------
# root_cause_rank: the Hit@3/MRR-enabling sibling of root_cause_is_correct.
# root_cause_is_correct only ever looks at hypotheses[0] (Hit@1); these
# pin down that root_cause_rank instead walks the FULL ranked list and
# returns the correct cause's own 1-based rank (or None if it's absent
# from the list entirely), matching this module's docstring exactly.
# ---------------------------------------------------------------------------

def _hypothesis(rank: int, node_id: str) -> RootCauseHypothesis:
    return RootCauseHypothesis(
        rank=rank, node_id=node_id, explanation=f"candidate at rank {rank}",
        confidence=0.9, supporting_node_ids=[node_id],
    )


def test_root_cause_rank_is_one_when_top_hypothesis_correct():
    """Mirrors root_cause_is_correct's True case: when hypotheses[0] is
    correct, root_cause_rank must agree exactly (rank == 1), not just
    agree on correctness."""
    event = _drift_event()
    rca_result = RCAResult(
        incident_id="incident-rank-1",
        hypotheses=[_hypothesis(1, event.event_id)],
        causal_path_node_ids=[event.event_id],
    )

    assert root_cause_rank(rca_result, FailureType.FEATURE_DRIFT, [event]) == 1
    # And root_cause_is_correct, defined as rank == 1, must agree.
    assert root_cause_is_correct(rca_result, FailureType.FEATURE_DRIFT, [event]) is True


def test_root_cause_rank_finds_correct_cause_below_rank_one():
    """The case root_cause_is_correct can't see: the correct cause sits at
    rank 2, behind a wrong but higher-ranked distractor. Hit@1 would call
    this a flat miss; root_cause_rank must report exactly where it is."""
    distractor = _drift_event()
    distractor_event = EvidenceEvent(
        timestamp=distractor.timestamp,
        detection_method=DetectionMethod.ISOLATION_FOREST,
        feature_name=None,
        metric_value=0.5,
        threshold=0.1,
        confidence=0.95,
        description="unrelated high-confidence anomaly",
        ground_truth_label=None,
    )
    true_cause = _drift_event()
    rca_result = RCAResult(
        incident_id="incident-rank-2",
        hypotheses=[
            _hypothesis(1, distractor_event.event_id),
            _hypothesis(2, true_cause.event_id),
        ],
        causal_path_node_ids=[distractor_event.event_id, true_cause.event_id],
    )

    rank = root_cause_rank(
        rca_result, FailureType.FEATURE_DRIFT, [distractor_event, true_cause]
    )

    assert rank == 2
    # Hit@1 correctly calls this wrong -- that's the gap Hit@3/MRR fill.
    assert root_cause_is_correct(
        rca_result, FailureType.FEATURE_DRIFT, [distractor_event, true_cause]
    ) is False


def test_root_cause_rank_is_none_when_correct_cause_never_ranked():
    """The correct cause doesn't appear anywhere in the agent's own ranked
    list at all -- a total miss, distinct from "ranked but low"."""
    true_cause = _drift_event()
    other_node_id = "some-other-candidate-id"
    rca_result = RCAResult(
        incident_id="incident-rank-none",
        hypotheses=[_hypothesis(1, other_node_id)],
        causal_path_node_ids=[other_node_id],
    )

    assert root_cause_rank(rca_result, FailureType.FEATURE_DRIFT, [true_cause]) is None


def test_root_cause_rank_is_none_when_ground_truth_label_is_none():
    """No ground truth (production, not a benchmark run) -> no rank to
    report, same convention as root_cause_is_correct returning False."""
    event = _drift_event()
    rca_result = RCAResult(
        incident_id="incident-no-gt",
        hypotheses=[_hypothesis(1, event.event_id)],
        causal_path_node_ids=[event.event_id],
    )

    assert root_cause_rank(rca_result, None, [event]) is None


def test_root_cause_rank_resolves_structured_node_id_space():
    """Same id-space bridging root_cause_is_correct needs (see the
    resolve_hypothesis_root_event_ids tests above) applies to
    root_cause_rank too -- it must work through evidence_nodes, not just
    the naive event_id-as-node_id shortcut."""
    event_a, event_b = _drift_event(0.9), _drift_event(0.85)
    node = EvidenceNode(
        node_type="feature_anomaly:customer_age",
        source_events=[event_a.event_id, event_b.event_id],
        timestamp=event_a.timestamp,
        confidence=0.88,
    )
    rca_result = RCAResult(
        incident_id="incident-structured-rank",
        hypotheses=[_hypothesis(1, node.node_id)],
        causal_path_node_ids=[node.node_id],
    )

    rank = root_cause_rank(
        rca_result, FailureType.FEATURE_DRIFT, [event_a, event_b], [node]
    )

    assert rank == 1
