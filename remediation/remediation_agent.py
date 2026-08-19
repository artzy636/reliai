"""
ReliAI — Remediation Agent (remediation/ layer)
==================================================
Consumes: schemas.RCAResult, produced by Person B's RCA Agent
          (reasoning/rca_agent.py).
Produces: schemas.RemediationPlan, consumed downstream by the Verification
          Agent (remediation/verification_agent.py, not yet built).

Per the project principle "LLM reasons, code/rules act": everything in this
module is a deterministic, rule-based mapping from a root-cause hypothesis to
a remediation category and action. No LLM call happens here -- the one
judgment call this agent makes (which category of fix applies) is exactly
the kind of decision that must stay auditable and reproducible, not phrased
as a model's opinion.

RootCauseHypothesis.failure_type (schemas.py) is what makes this possible:
it's a structural field, rule-derived upstream by the RCA Agent from
EvidenceNode.detection_method -- never from EvidenceEvent.ground_truth_label,
which per CONTRACT.md rule 4 must never reach the reasoning layer. Before
that field existed, there was no way to recover a failure category from an
RCAResult at all: RootCauseHypothesis only exposed node_id (an opaque
uuid4), a free-text LLM-written explanation, confidence, and
supporting_node_ids -- none of which reliably identify *what kind* of
failure occurred. See CONTRACT.md rule 2 / schemas.py's RootCauseHypothesis
and EvidenceNode docstrings for how failure_type / detection_method were
added to close that gap.
"""

from __future__ import annotations

import logging

from configs.settings import VERIFICATION, VerificationSettings
from schemas import FailureType, RCAResult, RemediationCategory, RemediationPlan, RootCauseHypothesis

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule-based mapping: failure_type -> remediation category
# ---------------------------------------------------------------------------

_FAILURE_TYPE_TO_CATEGORY: dict[FailureType, RemediationCategory] = {
    FailureType.FEATURE_DRIFT: RemediationCategory.RETRAIN,
    FailureType.LABEL_SHIFT: RemediationCategory.RETRAIN,
    FailureType.SCHEMA_MISMATCH: RemediationCategory.DATA_FIX,
    FailureType.MISSING_VALUES: RemediationCategory.DATA_FIX,
    FailureType.CORRUPTED_VALUES: RemediationCategory.DATA_FIX,
    FailureType.DUPLICATES: RemediationCategory.DATA_FIX,
}

# Fallback when failure_type couldn't be inferred upstream (no detection
# method available, or a detection method with no failure_type mapping --
# see reasoning/rca_agent.py's _DETECTION_METHOD_TO_FAILURE_TYPE). DATA_FIX
# is the least destructive option: it re-validates data rather than
# retraining a model or rolling back a deploy on an unconfirmed guess.
_DEFAULT_CATEGORY = RemediationCategory.DATA_FIX

_ACTION_DESCRIPTIONS: dict[FailureType, str] = {
    FailureType.FEATURE_DRIFT: (
        "Retrain the model on a recent data window that includes the "
        "drifted distribution, so the model absorbs the new feature "
        "distribution instead of continuing to score against a stale one."
    ),
    FailureType.LABEL_SHIFT: (
        "Retrain the model with a refreshed training set that reflects the "
        "current label distribution, rebalancing classes/targets to match "
        "observed production label rates."
    ),
    FailureType.SCHEMA_MISMATCH: (
        "Re-validate the ingestion schema against the expected feature "
        "contract, correct the mismatched field(s), and reprocess the "
        "affected records through the pipeline."
    ),
    FailureType.MISSING_VALUES: (
        "Impute or drop the affected feature's missing values per the "
        "pipeline's data-quality policy, then re-validate completeness "
        "before the next run."
    ),
    FailureType.CORRUPTED_VALUES: (
        "Quarantine the affected records, re-clean the corrupted "
        "feature(s), and re-validate value ranges/types before "
        "reprocessing."
    ),
    FailureType.DUPLICATES: (
        "Deduplicate the affected records using the pipeline's identity "
        "key and re-run downstream aggregation."
    ),
}

_DEFAULT_ACTION_DESCRIPTION = (
    "Root-cause category could not be inferred from available detection "
    "signals for this hypothesis; re-validate the data quality of the "
    "implicated evidence node as a general data-fix pass pending manual "
    "triage."
)


def _category_for(hypothesis: RootCauseHypothesis) -> RemediationCategory:
    """Deterministically map a hypothesis's failure_type to a
    RemediationCategory. Falls back to _DEFAULT_CATEGORY (with a logged
    warning) when failure_type is None or unmapped -- never guessed from
    hypothesis.explanation, which is free-text LLM prose with no contract
    guarantee of consistent terminology.
    """
    if hypothesis.failure_type is None:
        logger.warning(
            "Hypothesis for node %s has no failure_type; defaulting remediation "
            "category to %s",
            hypothesis.node_id,
            _DEFAULT_CATEGORY.value,
        )
        return _DEFAULT_CATEGORY
    category = _FAILURE_TYPE_TO_CATEGORY.get(hypothesis.failure_type)
    if category is None:
        logger.warning(
            "No remediation category mapped for failure_type=%s (node %s); defaulting to %s",
            hypothesis.failure_type,
            hypothesis.node_id,
            _DEFAULT_CATEGORY.value,
        )
        return _DEFAULT_CATEGORY
    return category


def _action_description_for(hypothesis: RootCauseHypothesis) -> str:
    """Deterministic action text for a hypothesis's failure_type, falling
    back to a generic re-validation description when failure_type is
    unknown."""
    if hypothesis.failure_type is None:
        return _DEFAULT_ACTION_DESCRIPTION
    return _ACTION_DESCRIPTIONS.get(hypothesis.failure_type, _DEFAULT_ACTION_DESCRIPTION)


def _expected_outcome_for(
    hypothesis: RootCauseHypothesis, verification_settings: VerificationSettings
) -> str:
    """Expected-outcome text, phrased against the configured verification
    threshold/sample size rather than a hardcoded number -- these are what
    the (not-yet-built) Verification Agent will actually check against."""
    threshold_pct = verification_settings.improvement_threshold_pct * 100
    category = _category_for(hypothesis)
    if category is RemediationCategory.RETRAIN:
        metric_phrase = "rolling prediction accuracy"
    else:
        metric_phrase = "the incident's tracked data-quality/accuracy metric"
    return (
        f"{metric_phrase.capitalize()} should improve by at least "
        f"{threshold_pct:.0f}%, to be confirmed by replaying "
        f"{verification_settings.replay_sample_size} held-out records "
        "against the remediated pipeline."
    )


class RemediationAgent:
    """Proposes a RemediationPlan for the top-ranked root cause of an
    incident. Deliberately does not execute anything -- see
    RemediationPlan's docstring: execution is the Verification Agent's job,
    kept separate so a bad proposal can't silently run.

    Usage:
        agent = RemediationAgent()
        plan = agent.propose(rca_result)
    """

    def __init__(self, verification_settings: VerificationSettings = VERIFICATION) -> None:
        """Initialize the agent.

        Args:
            verification_settings: thresholds used to phrase the plan's
                expected_outcome. Defaults to the shared
                configs.settings.VERIFICATION singleton -- pass an override
                only for testing, never to hardcode a threshold inline.
        """
        self._settings = verification_settings

    def propose(self, rca_result: RCAResult) -> RemediationPlan:
        """Build a RemediationPlan from an RCAResult's top-ranked hypothesis.

        Args:
            rca_result: output of RCAAgent.analyze() / NaiveRCAAgent.analyze().
                Only hypotheses[0] (the top-ranked candidate) is used -- this
                agent proposes one plan per incident, not one per hypothesis.

        Returns:
            A RemediationPlan whose category is deterministically derived
            from the top hypothesis's failure_type, and whose
            target_root_cause_node_id ties back to that hypothesis's node_id.

        Raises:
            ValueError: if rca_result has no hypotheses -- there is no root
                cause to remediate, so this fails loudly rather than
                proposing a meaningless default plan.
        """
        if not rca_result.hypotheses:
            raise ValueError(
                f"RCAResult for incident {rca_result.incident_id!r} has no hypotheses; "
                "cannot propose a remediation plan with no root cause."
            )

        top = rca_result.hypotheses[0]
        category = _category_for(top)
        logger.info(
            "Proposing %s remediation for incident %s (root node %s, failure_type=%s)",
            category.value,
            rca_result.incident_id,
            top.node_id,
            top.failure_type,
        )

        return RemediationPlan(
            incident_id=rca_result.incident_id,
            category=category,
            action_description=_action_description_for(top),
            expected_outcome=_expected_outcome_for(top, self._settings),
            target_root_cause_node_id=top.node_id,
        )
