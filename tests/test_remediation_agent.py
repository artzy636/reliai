"""
Tests for remediation/remediation_agent.py — RemediationAgent.

RemediationAgent's mapping from a root-cause hypothesis to a
RemediationCategory is rule-based and driven entirely by
RootCauseHypothesis.failure_type, so these tests build synthetic
RCAResults directly (no need to run a real RCA Agent / LLM) with a known
failure_type on the top hypothesis and check the plan that comes out.
"""

import logging

import pytest

from configs.settings import VerificationSettings
from remediation.remediation_agent import RemediationAgent
from schemas import FailureType, RCAResult, RemediationCategory, RootCauseHypothesis


def _hypothesis(
    node_id: str = "node-1",
    failure_type: FailureType | None = None,
    rank: int = 1,
) -> RootCauseHypothesis:
    """Build a minimal, valid RootCauseHypothesis for test scenarios."""
    return RootCauseHypothesis(
        rank=rank,
        node_id=node_id,
        explanation="synthetic hypothesis for testing",
        confidence=0.8,
        supporting_node_ids=[node_id],
        failure_type=failure_type,
    )


def _rca_result(
    incident_id: str = "incident-001",
    hypotheses: list[RootCauseHypothesis] | None = None,
) -> RCAResult:
    """Build a minimal, valid RCAResult wrapping the given hypotheses
    (defaulting to a single feature-drift hypothesis)."""
    if hypotheses is None:
        hypotheses = [_hypothesis(failure_type=FailureType.FEATURE_DRIFT)]
    return RCAResult(
        incident_id=incident_id,
        hypotheses=hypotheses,
        causal_path_node_ids=[h.node_id for h in hypotheses],
    )


# ---------------------------------------------------------------------------
# Category mapping, per failure_type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "failure_type,expected_category",
    [
        (FailureType.FEATURE_DRIFT, RemediationCategory.RETRAIN),
        (FailureType.LABEL_SHIFT, RemediationCategory.RETRAIN),
        (FailureType.SCHEMA_MISMATCH, RemediationCategory.DATA_FIX),
        (FailureType.MISSING_VALUES, RemediationCategory.DATA_FIX),
        (FailureType.CORRUPTED_VALUES, RemediationCategory.DATA_FIX),
        (FailureType.DUPLICATES, RemediationCategory.DATA_FIX),
    ],
)
def test_propose_maps_failure_type_to_expected_category(failure_type, expected_category):
    """Each FailureType must deterministically map to its documented
    RemediationCategory -- the core rule-based contract of this agent."""
    hypothesis = _hypothesis(node_id="root-node-42", failure_type=failure_type)
    rca_result = _rca_result(incident_id="incident-xyz", hypotheses=[hypothesis])

    plan = RemediationAgent().propose(rca_result)

    assert plan.category == expected_category
    assert plan.incident_id == "incident-xyz"
    assert plan.target_root_cause_node_id == "root-node-42"
    assert plan.action_description
    assert plan.expected_outcome


def test_propose_uses_top_ranked_hypothesis_not_others():
    """Only hypotheses[0] drives the plan, even when lower-ranked
    hypotheses would map to a different category."""
    top = _hypothesis(node_id="top-node", rank=1, failure_type=FailureType.FEATURE_DRIFT)
    second = _hypothesis(node_id="second-node", rank=2, failure_type=FailureType.SCHEMA_MISMATCH)
    rca_result = _rca_result(hypotheses=[top, second])

    plan = RemediationAgent().propose(rca_result)

    assert plan.target_root_cause_node_id == "top-node"
    assert plan.category == RemediationCategory.RETRAIN


def test_propose_defaults_to_data_fix_when_failure_type_unknown(caplog):
    """A hypothesis with no failure_type (e.g. detection_method that has no
    mapping, or none available at all) must not crash -- it degrades to the
    safe DATA_FIX default and logs a warning rather than guessing."""
    hypothesis = _hypothesis(node_id="unknown-node", failure_type=None)
    rca_result = _rca_result(hypotheses=[hypothesis])

    with caplog.at_level(logging.WARNING):
        plan = RemediationAgent().propose(rca_result)

    assert plan.category == RemediationCategory.DATA_FIX
    assert plan.target_root_cause_node_id == "unknown-node"
    assert any("failure_type" in record.message for record in caplog.records)


def test_propose_raises_on_empty_hypotheses():
    """No hypotheses means no root cause to remediate -- this must fail
    loudly rather than silently proposing a meaningless default plan."""
    rca_result = RCAResult(incident_id="incident-empty", hypotheses=[], causal_path_node_ids=[])

    with pytest.raises(ValueError):
        RemediationAgent().propose(rca_result)


def test_propose_expected_outcome_reflects_configured_verification_threshold():
    """expected_outcome must be phrased against configs.settings.VERIFICATION
    (injected here), not a hardcoded percentage."""
    settings = VerificationSettings(replay_sample_size=250, improvement_threshold_pct=0.10)
    hypothesis = _hypothesis(failure_type=FailureType.FEATURE_DRIFT)
    rca_result = _rca_result(hypotheses=[hypothesis])

    plan = RemediationAgent(verification_settings=settings).propose(rca_result)

    assert "10%" in plan.expected_outcome
    assert "250" in plan.expected_outcome
