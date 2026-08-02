"""
ReliAI — Benchmark Runner (evaluation/ layer)
================================================
Consumes: synthetic fault-injected BenchmarkIncidents (a list of
EvidenceEvent, each optionally carrying a schemas.FailureType
ground_truth_label), reasoning.evidence_graph.EvidenceGraphBuilder,
reasoning.rca_agent.RCAAgent / NaiveRCAAgent.
Produces: a BenchmarkSummary comparing root-cause accuracy of the
structured (graph-based) agent against the naive (raw-event) baseline —
this comparison is the project's actual research question (see README.md).

Scoring reuses schemas.root_cause_is_correct() rather than reimplementing
it: that function resolves a hypothesis's root id back to the
EvidenceEvent(s) it traces to — bridging the two disjoint id spaces the two
agents write into (RCAAgent: EvidenceNode.node_id: NaiveRCAAgent:
EvidenceEvent.event_id directly; see reasoning/rca_agent.py's
NaiveRCAAgent docstring) — and checks whether any of them carry the
incident's injected ground truth label. This module never re-derives that
logic; it only supplies each incident's evidence_events / evidence_nodes /
ground_truth_label to it.

No real LLM calls happen here by default: both agents are wired with
deterministic, prompt-parsing stub LLMs (_StructuralAgreementLLM /
_HighestConfidenceLLM below) so the runner works fully offline out of the
box, following the same stub pattern as tests/test_rca_agent.py. Pass real
LangChain-style clients via BenchmarkRunner(structured_llm=..., naive_llm=...)
to benchmark against actual models later.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from reasoning.evidence_graph import EvidenceGraphBuilder
from reasoning.rca_agent import LLMLike, NaiveRCAAgent, RCAAgent
from schemas import DetectionMethod, EvidenceEvent, FailureType, RCAResult, root_cause_is_correct

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Offline default LLM stubs — no network access, no API key.
# ---------------------------------------------------------------------------

class _StructuralAgreementLLM:
    """Default stub for RCAAgent: agrees with the deterministic candidate
    ranking RCAAgent already computed (see rca_agent._build_candidates) by
    echoing back the ``root_node_id`` values in the order they appear in
    the prompt. Stands in for "a reasonable LLM trusts a strong structural
    signal it's shown" without needing per-incident hardcoded node ids
    (which are random uuid4()s, unknowable ahead of time). Swap in a real
    LangChain chat model via ``BenchmarkRunner(structured_llm=...)`` later.
    """

    _ROOT_ID_RE = re.compile(r"^- root_node_id: (\S+)$", re.MULTILINE)

    def __call__(self, prompt: str) -> str:
        node_ids = self._ROOT_ID_RE.findall(prompt)
        if not node_ids:
            logger.warning("_StructuralAgreementLLM found no root_node_id lines in prompt")
            return json.dumps([])
        return json.dumps(
            [
                {
                    "node_id": node_id,
                    "explanation": f"Structural rank {rank}: agrees with the deterministic candidate ordering.",
                }
                for rank, node_id in enumerate(node_ids, start=1)
            ]
        )


class _HighestConfidenceLLM:
    """Default stub for NaiveRCAAgent: picks the single highest-confidence
    raw event as its one guess — a stand-in for the most a context-free
    LLM can reasonably infer from an unstructured telemetry dump with no
    causal structure to lean on. Swap in a real client via
    ``BenchmarkRunner(naive_llm=...)`` later.
    """

    _EVENT_LINE_RE = re.compile(r"^- event_id=(\S+).*?confidence=([\d.]+)", re.MULTILINE)

    def __call__(self, prompt: str) -> str:
        matches = self._EVENT_LINE_RE.findall(prompt)
        if not matches:
            logger.warning("_HighestConfidenceLLM found no event_id lines in prompt")
            return json.dumps([])
        top_event_id, top_confidence = max(matches, key=lambda pair: float(pair[1]))
        return json.dumps(
            [
                {
                    "event_ids": [top_event_id],
                    "explanation": "Highest-confidence anomaly in the raw telemetry.",
                    "confidence": float(top_confidence),
                }
            ]
        )


# ---------------------------------------------------------------------------
# Incident + scoring data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BenchmarkIncident:
    """One synthetic fault-injected incident: a raw event stream plus the
    ground truth used to score both agents' RCAResults.

    incident_id: stamped onto both agents' RCAResult for this incident.
    events: raw EvidenceEvents fed to both agents — the structured path
        clusters/links them via EvidenceGraphBuilder first; the naive path
        sees them exactly as given.
    ground_truth_label: the FailureType this incident's true root cause is.
        Passed to schemas.root_cause_is_correct() for scoring; per
        CONTRACT.md rule 4, never shown to either agent's LLM.
    description: human-readable summary for the printed results table.
    """

    incident_id: str
    events: list[EvidenceEvent]
    ground_truth_label: FailureType
    description: str = ""

    @property
    def true_root_cause_event_ids(self) -> list[str]:
        """Which raw events represent the injected root cause: by
        convention (see the built-in scenarios below), every event whose
        own ground_truth_label matches the incident's — downstream and
        distractor events are left unlabeled."""
        return [
            event.event_id for event in self.events if event.ground_truth_label == self.ground_truth_label
        ]


@dataclass(frozen=True)
class IncidentScore:
    """Per-incident scoring detail for both agents on one BenchmarkIncident."""

    incident_id: str
    ground_truth_label: FailureType
    structured_result: RCAResult
    structured_correct: bool
    naive_result: RCAResult
    naive_correct: bool


@dataclass(frozen=True)
class BenchmarkSummary:
    """Aggregated benchmark results: root-cause accuracy for the
    structured (graph-based) agent vs. the naive baseline across every
    incident that was run. This is the project's headline evidence for (or
    against) the "structured evidence beats naive LLM RCA" research
    question.
    """

    scores: list[IncidentScore]

    @property
    def n_incidents(self) -> int:
        return len(self.scores)

    @property
    def structured_accuracy(self) -> float:
        return self._accuracy(lambda score: score.structured_correct)

    @property
    def naive_accuracy(self) -> float:
        return self._accuracy(lambda score: score.naive_correct)

    def _accuracy(self, is_correct: Callable[[IncidentScore], bool]) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for score in self.scores if is_correct(score)) / len(self.scores)

    def as_dict(self) -> dict:
        """Machine-readable summary — the shape a results table/plot in
        the report would be built from."""
        return {
            "n_incidents": self.n_incidents,
            "structured_accuracy": self.structured_accuracy,
            "naive_accuracy": self.naive_accuracy,
            "per_incident": [
                {
                    "incident_id": score.incident_id,
                    "ground_truth_label": FailureType(score.ground_truth_label).value,
                    "structured_correct": score.structured_correct,
                    "naive_correct": score.naive_correct,
                }
                for score in self.scores
            ],
        }

    def print_table(self) -> None:
        """Print a simple, readable results table to stdout."""
        columns = f"{'incident_id':<32} {'ground_truth':<18} {'structured':<11} {'naive':<8}"
        rule = "-" * len(columns)
        print(columns)
        print(rule)
        for score in self.scores:
            print(
                f"{score.incident_id:<32} {FailureType(score.ground_truth_label).value:<18} "
                f"{'correct' if score.structured_correct else 'WRONG':<11} "
                f"{'correct' if score.naive_correct else 'WRONG':<8}"
            )
        print(rule)
        print(
            f"{'ACCURACY':<32} {'':<18} "
            f"{self.structured_accuracy:<11.0%} {self.naive_accuracy:<8.0%}"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Runs RCAAgent (structured) and NaiveRCAAgent (baseline) against a
    set of synthetic fault-injected incidents and scores both against
    ground truth via schemas.root_cause_is_correct() — this class never
    reimplements that scoring logic, only supplies it with each incident's
    data.

    Usage:
        runner = BenchmarkRunner()  # offline stub LLMs, no API calls
        summary = runner.run(load_builtin_incidents())
        summary.print_table()
    """

    def __init__(
        self,
        structured_llm: Optional[LLMLike] = None,
        naive_llm: Optional[LLMLike] = None,
        graph_builder: Optional[EvidenceGraphBuilder] = None,
    ) -> None:
        """Initialize the runner.

        Args:
            structured_llm: LLM passed to RCAAgent. Defaults to a
                deterministic, offline stub that agrees with RCAAgent's own
                structural candidate ranking (_StructuralAgreementLLM) —
                pass a real LangChain-style client to benchmark an actual
                model instead.
            naive_llm: LLM passed to NaiveRCAAgent. Defaults to a
                deterministic, offline stub that guesses the single
                highest-confidence raw event (_HighestConfidenceLLM).
            graph_builder: EvidenceGraphBuilder feeding the structured
                path. Defaults to a fresh builder using the shared
                configs.settings.GRAPH thresholds.
        """
        self._rca_agent = RCAAgent(llm=structured_llm or _StructuralAgreementLLM())
        self._naive_agent = NaiveRCAAgent(llm=naive_llm or _HighestConfidenceLLM())
        self._graph_builder = graph_builder or EvidenceGraphBuilder()

    def run(self, incidents: list[BenchmarkIncident]) -> BenchmarkSummary:
        """Run both agents against every incident and score them.

        Args:
            incidents: synthetic fault-injected incidents to benchmark.

        Returns:
            A BenchmarkSummary with per-incident detail and aggregate
            root-cause accuracy for both agents.
        """
        logger.info("Running benchmark over %d incident(s)", len(incidents))
        scores = [self._score_incident(incident) for incident in incidents]
        summary = BenchmarkSummary(scores=scores)
        logger.info(
            "Benchmark complete: structured accuracy=%.1f%%, naive accuracy=%.1f%%",
            summary.structured_accuracy * 100,
            summary.naive_accuracy * 100,
        )
        return summary

    def _score_incident(self, incident: BenchmarkIncident) -> IncidentScore:
        logger.info(
            "Scoring incident %s (%s)", incident.incident_id, incident.description or incident.ground_truth_label
        )

        graph = self._graph_builder.build(incident.events)
        evidence_nodes = self._graph_builder.get_nodes()
        structured_result = self._rca_agent.analyze(graph, incident_id=incident.incident_id)
        structured_correct = root_cause_is_correct(
            structured_result, incident.ground_truth_label, incident.events, evidence_nodes
        )

        naive_result = self._naive_agent.analyze(incident.events, incident_id=incident.incident_id)
        naive_correct = root_cause_is_correct(naive_result, incident.ground_truth_label, incident.events)

        logger.info(
            "Incident %s: structured_correct=%s, naive_correct=%s",
            incident.incident_id,
            structured_correct,
            naive_correct,
        )
        return IncidentScore(
            incident_id=incident.incident_id,
            ground_truth_label=incident.ground_truth_label,
            structured_result=structured_result,
            structured_correct=structured_correct,
            naive_result=naive_result,
            naive_correct=naive_correct,
        )


# ---------------------------------------------------------------------------
# Built-in synthetic benchmark incidents — no real detection-layer data
# needed to run this module. Incident 1 reuses the 7-event cascade scenario
# from tests/test_reasoning_integration.py; incidents 2-4 extend it to
# three more schemas.FailureType values. In every scenario, only the events
# representing the injected root cause carry ground_truth_label — see
# BenchmarkIncident.true_root_cause_event_ids.
# ---------------------------------------------------------------------------

_BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _event(
    minutes_after_base: float,
    feature_name: Optional[str],
    detection_method: DetectionMethod,
    confidence: float,
    description: str,
    ground_truth_label: Optional[FailureType] = None,
) -> EvidenceEvent:
    """Build a minimal, valid EvidenceEvent for a synthetic scenario."""
    return EvidenceEvent(
        timestamp=_BASE + timedelta(minutes=minutes_after_base),
        detection_method=detection_method,
        feature_name=feature_name,
        metric_value=0.5,
        threshold=0.2,
        confidence=confidence,
        description=description,
        ground_truth_label=ground_truth_label,
    )


def _cascading_schema_incident() -> BenchmarkIncident:
    """Reuses tests/test_reasoning_integration.py's scenario: an upstream
    schema change (t=0/8min) precedes feature drift in transaction_amount
    (t=20/28min), which precedes a prediction shift in model_prediction
    (t=40/45min). A region_code spike at t=200min is a distractor — far
    outside the clustering/edge window, but with a *higher* raw confidence
    (0.95) than the true root cause (0.90/0.85), which is exactly what
    trips up a confidence-only baseline with no structural context.
    """
    events = [
        _event(
            0, None, DetectionMethod.ISOLATION_FOREST, 0.90,
            "Isolation forest flagged anomalous row structure in the incoming transactions batch",
            ground_truth_label=FailureType.SCHEMA_MISMATCH,
        ),
        _event(
            8, None, DetectionMethod.ISOLATION_FOREST, 0.85,
            "Isolation forest flagged anomalous row structure in a second transactions batch",
            ground_truth_label=FailureType.SCHEMA_MISMATCH,
        ),
        _event(
            20, "transaction_amount", DetectionMethod.PSI, 0.88,
            "PSI for transaction_amount exceeded threshold following the schema change",
        ),
        _event(
            28, "transaction_amount", DetectionMethod.PSI, 0.82,
            "PSI for transaction_amount remained elevated",
        ),
        _event(
            40, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.75,
            "Rolling accuracy dropped for the fraud model",
        ),
        _event(
            45, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.70,
            "Rolling accuracy remained depressed",
        ),
        _event(
            200, "region_code", DetectionMethod.JENSEN_SHANNON, 0.95,
            "Jensen-Shannon divergence spike for region_code, unrelated to the incident",
        ),
    ]
    return BenchmarkIncident(
        incident_id="benchmark-schema-cascade-001",
        events=events,
        ground_truth_label=FailureType.SCHEMA_MISMATCH,
        description="schema change -> feature drift -> prediction shift, high-confidence distractor",
    )


def _corrupted_values_incident() -> BenchmarkIncident:
    """Corrupted values in account_balance (t=0/6min, pipeline-level)
    precede a rolling_accuracy drop (t=15min). A device_type distractor at
    t=300min has *lower* confidence (0.55) than the true root cause
    (0.92/0.90) — a "clean" case where a confidence-only baseline should
    also succeed.
    """
    events = [
        _event(
            0, None, DetectionMethod.KS_TEST, 0.92,
            "KS test flagged corrupted values in account_balance batch",
            ground_truth_label=FailureType.CORRUPTED_VALUES,
        ),
        _event(
            6, None, DetectionMethod.KS_TEST, 0.90,
            "KS test flagged corrupted values in a second account_balance batch",
            ground_truth_label=FailureType.CORRUPTED_VALUES,
        ),
        _event(
            15, "rolling_accuracy", DetectionMethod.ROLLING_ACCURACY, 0.80,
            "Rolling accuracy dropped shortly after the corrupted batch",
        ),
        _event(
            300, "device_type", DetectionMethod.JENSEN_SHANNON, 0.55,
            "Low-confidence device_type divergence, unrelated to the incident",
        ),
    ]
    return BenchmarkIncident(
        incident_id="benchmark-corrupted-values-002",
        events=events,
        ground_truth_label=FailureType.CORRUPTED_VALUES,
        description="corrupted values -> accuracy drop, low-confidence distractor",
    )


def _duplicate_rows_incident() -> BenchmarkIncident:
    """Duplicate rows detected pipeline-level (t=0/5min) precede a
    rolling_accuracy drop (t=18min). A session_length distractor at
    t=400min has a *higher* confidence (0.93) than the true root cause
    (0.85/0.83).
    """
    events = [
        _event(
            0, None, DetectionMethod.ISOLATION_FOREST, 0.85,
            "Isolation forest flagged duplicate rows in the ingestion batch",
            ground_truth_label=FailureType.DUPLICATES,
        ),
        _event(
            5, None, DetectionMethod.ISOLATION_FOREST, 0.83,
            "Isolation forest flagged duplicate rows in a second batch",
            ground_truth_label=FailureType.DUPLICATES,
        ),
        _event(
            18, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.70,
            "Rolling accuracy dropped shortly after the duplicate rows appeared",
        ),
        _event(
            400, "session_length", DetectionMethod.JENSEN_SHANNON, 0.93,
            "High-confidence session_length divergence, unrelated to the incident",
        ),
    ]
    return BenchmarkIncident(
        incident_id="benchmark-duplicates-003",
        events=events,
        ground_truth_label=FailureType.DUPLICATES,
        description="duplicate rows -> accuracy drop, high-confidence distractor",
    )


def _label_shift_incident() -> BenchmarkIncident:
    """Label shift detected in fraud_label (t=0/7min) precedes a
    rolling_accuracy drop (t=20min). A region_code distractor at t=500min
    has *lower* confidence (0.60) than the true root cause (0.88/0.86) — a
    second "clean" case.
    """
    events = [
        _event(
            0, "fraud_label", DetectionMethod.JENSEN_SHANNON, 0.88,
            "Jensen-Shannon divergence flagged a shift in the fraud_label distribution",
            ground_truth_label=FailureType.LABEL_SHIFT,
        ),
        _event(
            7, "fraud_label", DetectionMethod.JENSEN_SHANNON, 0.86,
            "Jensen-Shannon divergence remained elevated for fraud_label",
            ground_truth_label=FailureType.LABEL_SHIFT,
        ),
        _event(
            20, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.72,
            "Rolling accuracy dropped shortly after the label shift",
        ),
        _event(
            500, "region_code", DetectionMethod.PSI, 0.60,
            "Low-confidence region_code divergence, unrelated to the incident",
        ),
    ]
    return BenchmarkIncident(
        incident_id="benchmark-label-shift-004",
        events=events,
        ground_truth_label=FailureType.LABEL_SHIFT,
        description="label shift -> accuracy drop, low-confidence distractor",
    )


def load_builtin_incidents() -> list[BenchmarkIncident]:
    """Fresh instances of the built-in synthetic benchmark incidents, so
    the runner is immediately runnable without any real detection-layer
    data."""
    return [
        _cascading_schema_incident(),
        _corrupted_values_incident(),
        _duplicate_rows_incident(),
        _label_shift_incident(),
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result_summary = BenchmarkRunner().run(load_builtin_incidents())
    result_summary.print_table()
    print()
    print(json.dumps(result_summary.as_dict(), indent=2))
