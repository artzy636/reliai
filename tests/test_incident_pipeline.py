"""
Tests for evaluation/incident_pipeline.py.

Runs the full real pipeline end-to-end -- DataAgent -> EvidenceGraphBuilder
-> RCAAgent -> RemediationAgent -> VerificationAgent -- against a real
fault-injected dataset, with only the RCA LLM call stubbed (no network
access, no API key), following the same stub-boundary pattern as
tests/test_detection_to_reasoning.py and evaluation/benchmark_runner.py.

Dataset design: current_df is built by calling
detection.fault_injection.inject_feature_drift() directly on reference_df
(same rows, one column shifted), the same trick
tests/test_detection_to_reasoning.py uses. Every OTHER shared column
(including target_column, and feature_0/2/3/4) is then byte-identical
between reference_df and current_df, so DataAgent's KS test can only ever
flag the drifted column -- deterministically, not just "usually" -- which
is what lets this test assert exactly one EvidenceEvent/EvidenceNode with
no risk of flakiness from incidental distribution differences elsewhere.
n_informative=3 (from test_verification_agent.py's pattern) ensures
feature_1 is a real decision-boundary feature, so VerificationAgent's
before/after accuracy numbers are genuine measurements, not no-ops.
"""

import json
import re

import pandas as pd
from sklearn.datasets import make_classification

from detection.fault_injection import inject_feature_drift
from evaluation.incident_pipeline import run_incident_pipeline
from schemas import FailureType, IncidentReport, RemediationCategory

_FEATURE_COLUMNS = [f"feature_{i}" for i in range(5)]
_TARGET_COLUMN = "label"
_DRIFTED_FEATURE = "feature_1"

_ROOT_ID_RE = re.compile(r"^- root_node_id: (\S+)$", re.MULTILINE)


class _StructuralAgreementLLM:
    """Deterministic, offline stub for RCAAgent: agrees with the
    deterministic candidate ranking RCAAgent already computed by echoing
    back the ``root_node_id`` values it finds in the prompt, in order.

    Reuses evaluation/benchmark_runner.py's ``_StructuralAgreementLLM``
    pattern rather than hand-hardcoding a node_id in the test: the graph's
    node ids are fresh uuid4()s generated at pipeline run time, so there is
    no way to know them ahead of the call.
    """

    def __call__(self, prompt: str) -> str:
        node_ids = _ROOT_ID_RE.findall(prompt)
        return json.dumps(
            [
                {
                    "node_id": node_id,
                    "explanation": f"Structural rank {rank}: agrees with the deterministic candidate ordering.",
                }
                for rank, node_id in enumerate(node_ids, start=1)
            ]
        )


def _reference_and_drifted_current() -> tuple[pd.DataFrame, pd.DataFrame, FailureType]:
    """A classification dataset where feature_1 is a real decision-boundary
    feature; current_df is reference_df's exact rows with feature_1 shifted
    by inject_feature_drift(), so every other column is untouched."""
    X, y = make_classification(
        n_samples=600,
        n_features=len(_FEATURE_COLUMNS),
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=42,
    )
    reference_df = pd.DataFrame(X, columns=_FEATURE_COLUMNS)
    reference_df[_TARGET_COLUMN] = y

    current_df, injected_label = inject_feature_drift(
        reference_df, _DRIFTED_FEATURE, shift_amount=3.0, random_seed=42
    )
    return reference_df, current_df, injected_label


def test_run_incident_pipeline_produces_complete_report() -> None:
    """End-to-end: a real drifted dataset feeds the full assembly, and every
    IncidentReport field comes out populated, traceable to the drifted
    feature, and scoreable as correct via root_cause_correct()."""
    reference_df, current_df, injected_label = _reference_and_drifted_current()
    assert injected_label == FailureType.FEATURE_DRIFT

    report = run_incident_pipeline(
        reference_df,
        current_df,
        target_column=_TARGET_COLUMN,
        llm=_StructuralAgreementLLM(),
        ground_truth_label=injected_label,
        injected_feature_name=_DRIFTED_FEATURE,
    )

    # --- Every field populated, not None. ---------------------------------
    assert isinstance(report, IncidentReport)
    assert report.incident_id
    assert report.detected_at is not None
    assert report.evidence_events
    assert report.evidence_nodes
    assert report.rca_result is not None
    assert report.remediation_plan is not None
    assert report.verification_result is not None
    assert report.ground_truth_label == FailureType.FEATURE_DRIFT

    # --- Detection: exactly one event, tied to the drifted feature. -------
    assert len(report.evidence_events) == 1
    drifted_event = report.evidence_events[0]
    assert drifted_event.feature_name == _DRIFTED_FEATURE
    assert drifted_event.ground_truth_label == FailureType.FEATURE_DRIFT

    # --- Evidence graph: exactly one node, wrapping that event. -----------
    assert len(report.evidence_nodes) == 1
    drifted_node = report.evidence_nodes[0]
    assert drifted_node.node_type == f"feature_anomaly:{_DRIFTED_FEATURE}"
    assert drifted_node.source_events == [drifted_event.event_id]

    # --- RCA: top hypothesis resolves to the drifted node. -----------------
    assert report.rca_result.hypotheses
    top_hypothesis = report.rca_result.hypotheses[0]
    assert top_hypothesis.node_id == drifted_node.node_id
    assert top_hypothesis.failure_type == FailureType.FEATURE_DRIFT

    # --- Remediation: RETRAIN, targeting the drifted node. -----------------
    assert report.remediation_plan.category == RemediationCategory.RETRAIN
    assert report.remediation_plan.target_root_cause_node_id == drifted_node.node_id
    assert report.remediation_plan.action_description
    assert report.remediation_plan.expected_outcome

    # --- Verification: real, valid before/after accuracy numbers. ---------
    assert 0.0 <= report.verification_result.value_before <= 1.0
    assert 0.0 <= report.verification_result.value_after <= 1.0
    assert report.verification_result.replay_sample_size > 0

    # --- The whole point: scoreable as correct against the known fault. ---
    assert report.root_cause_correct() is True


def test_run_incident_pipeline_is_deterministic_given_an_explicit_incident_id() -> None:
    """Same inputs + explicit incident_id -> the same incident_id threaded
    through every layer of the report (RCAResult, RemediationPlan,
    VerificationResult all stamp it independently; nothing should drift)."""
    reference_df, current_df, injected_label = _reference_and_drifted_current()

    report = run_incident_pipeline(
        reference_df,
        current_df,
        target_column=_TARGET_COLUMN,
        llm=_StructuralAgreementLLM(),
        incident_id="incident-pipeline-test-001",
        ground_truth_label=injected_label,
        injected_feature_name=_DRIFTED_FEATURE,
    )

    assert report.incident_id == "incident-pipeline-test-001"
    assert report.rca_result.incident_id == "incident-pipeline-test-001"
    assert report.remediation_plan.incident_id == "incident-pipeline-test-001"
    assert report.verification_result.incident_id == "incident-pipeline-test-001"
