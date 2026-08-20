"""
ReliAI — Incident Pipeline (evaluation/ layer)
=================================================
Owned by Person C's evaluation/reasoning integration -- NOT the remediation
layer's frontend. Runs one incident through all four layers end-to-end and
assembles the final schemas.IncidentReport:

    reference_df, current_df
        -> detection.data_agent.DataAgent.investigate()       EvidenceEvent(s)
        -> reasoning.evidence_graph.EvidenceGraphBuilder       graph, EvidenceNode(s)
        -> reasoning.rca_agent.RCAAgent                        RCAResult
        -> remediation.remediation_agent.RemediationAgent      RemediationPlan
        -> remediation.verification_agent.VerificationAgent    VerificationResult
        -> schemas.IncidentReport

No LLM is constructed here -- the caller injects one (a stub in tests, a
real LangChain-style client in production), the same convention
reasoning.rca_agent.RCAAgent/NaiveRCAAgent and
evaluation.benchmark_runner.BenchmarkRunner already follow.

Ground truth: detection.data_agent.DataAgent is a real detector with no
notion of an "injected fault" -- schemas.EvidenceEvent.ground_truth_label is
documented as "only set in benchmark runs", and nothing in the detection
layer can populate it. During a benchmark run, the caller (who ran the
fault injection and therefore already knows the answer) supplies it via
``ground_truth_label`` + ``injected_feature_name``; this pipeline stamps it
only onto the EvidenceEvent(s) whose ``feature_name`` matches, rather than
blanket-labeling every event, so an unrelated/distractor event in a
multi-event incident can never be miscounted as the injected root cause.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import uuid4

import pandas as pd

from detection.data_agent import DataAgent
from reasoning.evidence_graph import EvidenceGraphBuilder
from reasoning.rca_agent import LLMLike, RCAAgent
from remediation.remediation_agent import RemediationAgent
from remediation.verification_agent import VerificationAgent
from schemas import EvidenceEvent, FailureType, IncidentReport

logger = logging.getLogger(__name__)


def run_incident_pipeline(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    target_column: str,
    llm: LLMLike,
    *,
    incident_id: Optional[str] = None,
    ground_truth_label: Optional[FailureType] = None,
    injected_feature_name: Optional[str | list[str]] = None,
) -> IncidentReport:
    """Run one incident through detection, evidence-graph construction, RCA,
    remediation, and verification, and assemble a complete IncidentReport.

    Args:
        reference_df: clean, pre-incident data (e.g. training distribution).
        current_df: the (possibly incident-affected) data being diagnosed.
            Must share reference_df's columns, including target_column.
        target_column: name of the label column. Excluded from
            VerificationAgent's feature set the same way it already is
            there; DataAgent has no target-column concept and scans it like
            any other shared numeric column.
        llm: a LangChain-style chat model (``.invoke(prompt) -> AIMessage``)
            or a plain ``Callable[[str], str]``, passed to RCAAgent. A stub
            in tests, a real provider client in production -- this function
            never constructs one itself.
        incident_id: identifier stamped across RCAResult, RemediationPlan,
            VerificationResult, and the final IncidentReport. Defaults to a
            fresh uuid4 so every layer's incident_id still lines up when the
            caller doesn't need to choose one.
        ground_truth_label: the injected FailureType, for benchmark runs
            (e.g. via detection.fault_injection). None in production. Only
            affects the report's own ground_truth_label field and the
            EvidenceEvent(s) selected by ``injected_feature_name`` -- never
            shown to the LLM.
        injected_feature_name: which EvidenceEvent.feature_name(s) to stamp
            with ``ground_truth_label`` (the column(s) fault-injection
            actually touched, or None for a pipeline-level fault). Accepts
            either a single feature name or a list, for incidents where
            drift was injected into more than one genuinely-weighted
            feature at once. Ignored if ground_truth_label is None. See the
            module docstring for why this can't be inferred automatically.

    Returns:
        A complete IncidentReport, including evidence_nodes (from
        EvidenceGraphBuilder.get_nodes()) -- required for
        IncidentReport.root_cause_correct() to resolve a structured
        RCAResult's hypothesis back to its originating evidence.

    Raises:
        ValueError: if DataAgent finds no evidence of drift between
            reference_df and current_df, propagated from
            RemediationAgent.propose() (an RCAResult with no hypotheses has
            no root cause to remediate).
    """
    incident_id = incident_id or str(uuid4())
    logger.info("Running incident pipeline for incident %s", incident_id)

    # 1. Detection: reference_df vs current_df -> raw EvidenceEvents.
    evidence_events = DataAgent().investigate(reference_df, current_df)
    logger.info(
        "Incident %s: DataAgent found %d evidence event(s)", incident_id, len(evidence_events)
    )
    if not evidence_events:
        logger.warning(
            "Incident %s: no evidence of drift between reference_df and current_df; "
            "downstream layers will have nothing to work with",
            incident_id,
        )
    if ground_truth_label is not None:
        injected_feature_names = (
            set(injected_feature_name)
            if isinstance(injected_feature_name, list)
            else {injected_feature_name}
        )
        evidence_events = [
            _stamp_ground_truth(event, ground_truth_label, injected_feature_names)
            for event in evidence_events
        ]

    # 2. Evidence graph: cluster events into EvidenceNodes/EvidenceEdges.
    graph_builder = EvidenceGraphBuilder()
    graph = graph_builder.build(evidence_events)
    evidence_nodes = graph_builder.get_nodes()

    # 3. RCA: rank candidate root causes over the graph.
    rca_result = RCAAgent(llm=llm).analyze(graph, incident_id=incident_id)

    # 4. Remediation: propose a fix for the top-ranked root cause.
    remediation_plan = RemediationAgent().propose(rca_result)

    # 5. Verification: measure the fix's real before/after effect.
    verification_result = VerificationAgent().verify(
        remediation_plan, reference_df, current_df, target_column
    )

    detected_at = min(event.timestamp for event in evidence_events)

    report = IncidentReport(
        incident_id=incident_id,
        detected_at=detected_at,
        evidence_events=evidence_events,
        evidence_nodes=evidence_nodes,
        rca_result=rca_result,
        remediation_plan=remediation_plan,
        verification_result=verification_result,
        ground_truth_label=ground_truth_label,
    )
    logger.info(
        "Incident %s assembled: %d event(s), %d node(s), %d hypothesis/es, remediation=%s",
        incident_id,
        len(evidence_events),
        len(evidence_nodes),
        len(rca_result.hypotheses),
        remediation_plan.category.value,
    )
    return report


def _stamp_ground_truth(
    event: EvidenceEvent,
    ground_truth_label: FailureType,
    injected_feature_names: set[Optional[str]],
) -> EvidenceEvent:
    """Return a copy of `event` with ground_truth_label set, if its
    feature_name is one of the column(s) the fault was actually injected
    into (or is None and None is in the set, for a pipeline-level fault)
    -- never a blanket stamp, so an unrelated evidence event in a
    multi-event incident can't be mistaken for the injected root cause."""
    if event.feature_name not in injected_feature_names:
        return event
    return event.model_copy(update={"ground_truth_label": ground_truth_label})
