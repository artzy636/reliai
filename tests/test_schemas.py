"""
Smoke test for the shared contract. If this fails, nobody's layer will work
together correctly — fix this before building on top of schemas.py.
"""

from datetime import datetime, timezone

from schemas import (
    DetectionMethod,
    EvidenceEvent,
    FailureType,
    IncidentReport,
    RCAResult,
    RemediationCategory,
    RemediationPlan,
    RootCauseHypothesis,
    VerificationResult,
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
                node_id="node-1",
                explanation="Feature drift in customer_age likely caused prediction shift",
                confidence=0.82,
                supporting_node_ids=["node-1"],
            )
        ],
        causal_path_node_ids=["node-1"],
    )

    remediation = RemediationPlan(
        incident_id="test-001",
        category=RemediationCategory.RETRAIN,
        action_description="Retrain on last 30 days",
        expected_outcome="Accuracy recovers ~4-6%",
        target_root_cause_node_id="node-1",
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
