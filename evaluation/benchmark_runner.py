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

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from reasoning.evidence_graph import EvidenceGraphBuilder
from reasoning.rca_agent import LLMLike, NaiveRCAAgent, RCAAgent
from schemas import (
    DetectionMethod,
    EvidenceEvent,
    FailureType,
    RCAResult,
    root_cause_is_correct,
    root_cause_rank,
)

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
    # 1-based rank of the correct cause in each agent's own ranked
    # hypothesis list (see schemas.root_cause_rank), or None if it never
    # appears in the list at all. structured_correct/naive_correct above
    # are exactly `rank == 1`; these carry the richer Hit@3/MRR signal.
    # Optional + defaulted so any older caller that built an IncidentScore
    # without computing a rank keeps working unchanged.
    structured_rank: Optional[int] = None
    naive_rank: Optional[int] = None


def _within_top_k(rank: Optional[int], k: int) -> bool:
    """True if `rank` (a 1-based schemas.root_cause_rank result, or None)
    is within the top k -- i.e. Hit@k for that one incident."""
    return rank is not None and rank <= k


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

    @property
    def structured_hit3(self) -> float:
        """Hit@3: did the correct cause appear ANYWHERE in the structured
        agent's top-3 ranked hypotheses, not just at rank 1? Always >=
        structured_accuracy (Hit@1), since rank 1 is itself <= 3."""
        return self._accuracy(lambda score: _within_top_k(score.structured_rank, 3))

    @property
    def naive_hit3(self) -> float:
        return self._accuracy(lambda score: _within_top_k(score.naive_rank, 3))

    @property
    def structured_mrr(self) -> float:
        """Mean Reciprocal Rank: average of 1/rank across incidents (0 for
        an incident where the correct cause never appears in the ranked
        list at all). Rewards a near-miss (rank 2) more than a total miss,
        which a Hit@1/Hit@3 pass-fail metric can't distinguish."""
        return self._mean_reciprocal_rank(lambda score: score.structured_rank)

    @property
    def naive_mrr(self) -> float:
        return self._mean_reciprocal_rank(lambda score: score.naive_rank)

    def _accuracy(self, is_correct: Callable[[IncidentScore], bool]) -> float:
        if not self.scores:
            return 0.0
        return sum(1 for score in self.scores if is_correct(score)) / len(self.scores)

    def _mean_reciprocal_rank(self, get_rank: Callable[[IncidentScore], Optional[int]]) -> float:
        if not self.scores:
            return 0.0
        reciprocal_ranks = [1.0 / rank if (rank := get_rank(score)) else 0.0 for score in self.scores]
        return sum(reciprocal_ranks) / len(reciprocal_ranks)

    def as_dict(self) -> dict:
        """Machine-readable summary — the shape a results table/plot in
        the report would be built from."""
        return {
            "n_incidents": self.n_incidents,
            "structured_accuracy": self.structured_accuracy,
            "naive_accuracy": self.naive_accuracy,
            "structured_hit3": self.structured_hit3,
            "naive_hit3": self.naive_hit3,
            "structured_mrr": self.structured_mrr,
            "naive_mrr": self.naive_mrr,
            "per_incident": [
                {
                    "incident_id": score.incident_id,
                    "ground_truth_label": FailureType(score.ground_truth_label).value,
                    "structured_correct": score.structured_correct,
                    "naive_correct": score.naive_correct,
                    "structured_rank": score.structured_rank,
                    "naive_rank": score.naive_rank,
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
            f"{'ACCURACY (Hit@1)':<32} {'':<18} "
            f"{self.structured_accuracy:<11.0%} {self.naive_accuracy:<8.0%}"
        )
        print(
            f"{'HIT@3':<32} {'':<18} "
            f"{self.structured_hit3:<11.0%} {self.naive_hit3:<8.0%}"
        )
        print(
            f"{'MRR':<32} {'':<18} "
            f"{self.structured_mrr:<11.3f} {self.naive_mrr:<8.3f}"
        )


@dataclass(frozen=True)
class AggregatedBenchmarkSummary:
    """Aggregates several independent BenchmarkSummary runs over the SAME
    incident set into one report: per-incident hit-rate (e.g. "4/5") for
    both agents, plus mean accuracy across repeats.

    Exists because a real LLM call is not guaranteed to be deterministic
    run-to-run (see remediation/pipeline_utils.py's model-choice comment --
    some Gemini tiers ignore the temperature=0 the code requests), so a
    single --real-llm run's accuracy numbers shouldn't be reported as if
    they were a stable measurement. Running --repeats N times and looking
    at the spread is how you'd actually know whether a result like
    "structured 100%, naive 75%" is a real, repeatable finding or one
    lucky/unlucky roll of the sampler.
    """

    summaries: list[BenchmarkSummary]

    @property
    def n_repeats(self) -> int:
        return len(self.summaries)

    @property
    def structured_accuracies(self) -> list[float]:
        return [summary.structured_accuracy for summary in self.summaries]

    @property
    def naive_accuracies(self) -> list[float]:
        return [summary.naive_accuracy for summary in self.summaries]

    @property
    def structured_hit3s(self) -> list[float]:
        return [summary.structured_hit3 for summary in self.summaries]

    @property
    def naive_hit3s(self) -> list[float]:
        return [summary.naive_hit3 for summary in self.summaries]

    @property
    def structured_mrrs(self) -> list[float]:
        return [summary.structured_mrr for summary in self.summaries]

    @property
    def naive_mrrs(self) -> list[float]:
        return [summary.naive_mrr for summary in self.summaries]

    @staticmethod
    def _mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _stdev(values: list[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance ** 0.5

    def per_incident_hit_rates(self) -> dict[str, dict[str, int]]:
        """{incident_id: {"structured_hits": int, "naive_hits": int,
        "structured_hit3_hits": int, "naive_hit3_hits": int, "total": int}}.
        The ``*_hit3_hits`` keys are additive (existing readers that only
        look at structured_hits/naive_hits/total, e.g. the frontend's
        PerIncidentTable, keep working unchanged)."""
        hit_rates: dict[str, dict[str, int]] = {}
        for summary in self.summaries:
            for score in summary.scores:
                entry = hit_rates.setdefault(
                    score.incident_id,
                    {
                        "structured_hits": 0,
                        "naive_hits": 0,
                        "structured_hit3_hits": 0,
                        "naive_hit3_hits": 0,
                        "total": 0,
                    },
                )
                entry["total"] += 1
                if score.structured_correct:
                    entry["structured_hits"] += 1
                if score.naive_correct:
                    entry["naive_hits"] += 1
                if _within_top_k(score.structured_rank, 3):
                    entry["structured_hit3_hits"] += 1
                if _within_top_k(score.naive_rank, 3):
                    entry["naive_hit3_hits"] += 1
        return hit_rates

    def as_dict(self) -> dict:
        return {
            "n_repeats": self.n_repeats,
            "structured_accuracy_mean": self._mean(self.structured_accuracies),
            "structured_accuracy_stdev": self._stdev(self.structured_accuracies),
            "naive_accuracy_mean": self._mean(self.naive_accuracies),
            "naive_accuracy_stdev": self._stdev(self.naive_accuracies),
            "structured_hit3_mean": self._mean(self.structured_hit3s),
            "structured_hit3_stdev": self._stdev(self.structured_hit3s),
            "naive_hit3_mean": self._mean(self.naive_hit3s),
            "naive_hit3_stdev": self._stdev(self.naive_hit3s),
            "structured_mrr_mean": self._mean(self.structured_mrrs),
            "naive_mrr_mean": self._mean(self.naive_mrrs),
            "per_repeat": [
                {"structured_accuracy": s, "naive_accuracy": n}
                for s, n in zip(self.structured_accuracies, self.naive_accuracies)
            ],
            "per_incident_hit_rates": self.per_incident_hit_rates(),
        }

    def print_table(self) -> None:
        """Per-incident hit-rate table (e.g. "4/5"), plus mean +/- stdev
        accuracy across all repeats."""
        hit_rates = self.per_incident_hit_rates()
        columns = f"{'incident_id':<38} {'structured':<12} {'naive':<12}"
        rule = "-" * len(columns)
        print(f"Repeats: {self.n_repeats}")
        print(columns)
        print(rule)
        for incident_id, counts in hit_rates.items():
            total = counts["total"]
            structured_rate = f"{counts['structured_hits']}/{total}"
            naive_rate = f"{counts['naive_hits']}/{total}"
            print(f"{incident_id:<38} {structured_rate:<12} {naive_rate:<12}")
        print(rule)
        s_mean, s_std = self._mean(self.structured_accuracies), self._stdev(self.structured_accuracies)
        n_mean, n_std = self._mean(self.naive_accuracies), self._stdev(self.naive_accuracies)
        print(
            f"{'MEAN ACCURACY / Hit@1 (+/- stdev)':<38} "
            f"{f'{s_mean:.0%} +/- {s_std:.0%}':<12} "
            f"{f'{n_mean:.0%} +/- {n_std:.0%}':<12}"
        )
        s3_mean, s3_std = self._mean(self.structured_hit3s), self._stdev(self.structured_hit3s)
        n3_mean, n3_std = self._mean(self.naive_hit3s), self._stdev(self.naive_hit3s)
        print(
            f"{'MEAN HIT@3 (+/- stdev)':<38} "
            f"{f'{s3_mean:.0%} +/- {s3_std:.0%}':<12} "
            f"{f'{n3_mean:.0%} +/- {n3_std:.0%}':<12}"
        )
        print(
            f"{'MEAN MRR':<38} "
            f"{self._mean(self.structured_mrrs):<12.3f} "
            f"{self._mean(self.naive_mrrs):<12.3f}"
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
        structured_rank = root_cause_rank(
            structured_result, incident.ground_truth_label, incident.events, evidence_nodes
        )

        naive_result = self._naive_agent.analyze(incident.events, incident_id=incident.incident_id)
        naive_correct = root_cause_is_correct(naive_result, incident.ground_truth_label, incident.events)
        naive_rank = root_cause_rank(naive_result, incident.ground_truth_label, incident.events)

        logger.info(
            "Incident %s: structured_correct=%s (rank=%s), naive_correct=%s (rank=%s)",
            incident.incident_id,
            structured_correct,
            structured_rank,
            naive_correct,
            naive_rank,
        )
        return IncidentScore(
            incident_id=incident.incident_id,
            ground_truth_label=incident.ground_truth_label,
            structured_result=structured_result,
            structured_correct=structured_correct,
            naive_result=naive_result,
            naive_correct=naive_correct,
            structured_rank=structured_rank,
            naive_rank=naive_rank,
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
            "Isolation-forest detector flagged elevated anomaly rate in current data relative to reference (anomaly_rate=0.1800, contamination_baseline=0.0500)",
            ground_truth_label=FailureType.SCHEMA_MISMATCH,
        ),
        _event(
            8, None, DetectionMethod.ISOLATION_FOREST, 0.85,
            "Isolation-forest detector flagged elevated anomaly rate in current data relative to reference (anomaly_rate=0.1500, contamination_baseline=0.0500)",
            ground_truth_label=FailureType.SCHEMA_MISMATCH,
        ),
        _event(
            20, "transaction_amount", DetectionMethod.PSI, 0.88,
            "PSI detector flagged distribution shift in transaction_amount (psi=0.3100, threshold=0.1000)",
        ),
        _event(
            28, "transaction_amount", DetectionMethod.PSI, 0.82,
            "PSI detector flagged distribution shift in transaction_amount (psi=0.2700, threshold=0.1000)",
        ),
        _event(
            40, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.75,
            "Rolling-accuracy detector flagged a drop in prediction accuracy (baseline=0.9100, current=0.8300, drop=0.0800 > threshold=0.0500)",
        ),
        _event(
            45, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.70,
            "Rolling-accuracy detector flagged a drop in prediction accuracy (baseline=0.9100, current=0.8500, drop=0.0600 > threshold=0.0500)",
        ),
        _event(
            200, "region_code", DetectionMethod.JENSEN_SHANNON, 0.95,
            "Jensen-Shannon divergence detector flagged distribution shift in region_code (js_divergence=0.4200, threshold=0.1000)",
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
            "KS Test detected distribution shift in account_balance (D=0.5800, p=0.0002)",
            ground_truth_label=FailureType.CORRUPTED_VALUES,
        ),
        _event(
            6, None, DetectionMethod.KS_TEST, 0.90,
            "KS Test detected distribution shift in account_balance (D=0.5400, p=0.0005)",
            ground_truth_label=FailureType.CORRUPTED_VALUES,
        ),
        _event(
            15, "rolling_accuracy", DetectionMethod.ROLLING_ACCURACY, 0.80,
            "Rolling-accuracy detector flagged a drop in prediction accuracy (baseline=0.9000, current=0.8200, drop=0.0800 > threshold=0.0500)",
        ),
        _event(
            300, "device_type", DetectionMethod.JENSEN_SHANNON, 0.55,
            "Jensen-Shannon divergence detector flagged distribution shift in device_type (js_divergence=0.1200, threshold=0.1000)",
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
            "Isolation-forest detector flagged elevated anomaly rate in current data relative to reference (anomaly_rate=0.1600, contamination_baseline=0.0500)",
            ground_truth_label=FailureType.DUPLICATES,
        ),
        _event(
            5, None, DetectionMethod.ISOLATION_FOREST, 0.83,
            "Isolation-forest detector flagged elevated anomaly rate in current data relative to reference (anomaly_rate=0.1400, contamination_baseline=0.0500)",
            ground_truth_label=FailureType.DUPLICATES,
        ),
        _event(
            18, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.70,
            "Rolling-accuracy detector flagged a drop in prediction accuracy (baseline=0.8900, current=0.8100, drop=0.0800 > threshold=0.0500)",
        ),
        _event(
            400, "session_length", DetectionMethod.JENSEN_SHANNON, 0.93,
            "Jensen-Shannon divergence detector flagged distribution shift in session_length (js_divergence=0.4500, threshold=0.1000)",
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
            "Jensen-Shannon divergence detector flagged distribution shift in fraud_label (js_divergence=0.3900, threshold=0.1000)",
            ground_truth_label=FailureType.LABEL_SHIFT,
        ),
        _event(
            7, "fraud_label", DetectionMethod.JENSEN_SHANNON, 0.86,
            "Jensen-Shannon divergence detector flagged distribution shift in fraud_label (js_divergence=0.3600, threshold=0.1000)",
            ground_truth_label=FailureType.LABEL_SHIFT,
        ),
        _event(
            20, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.72,
            "Rolling-accuracy detector flagged a drop in prediction accuracy (baseline=0.9000, current=0.8300, drop=0.0700 > threshold=0.0500)",
        ),
        _event(
            500, "region_code", DetectionMethod.PSI, 0.60,
            "PSI detector flagged distribution shift in region_code (psi=0.1500, threshold=0.1000)",
        ),
    ]
    return BenchmarkIncident(
        incident_id="benchmark-label-shift-004",
        events=events,
        ground_truth_label=FailureType.LABEL_SHIFT,
        description="label shift -> accuracy drop, low-confidence distractor",
    )


def _clean_schema_case() -> BenchmarkIncident:
    """Control case, NO distractor: schema mismatch (t=0/6min) precedes
    feature drift in customer_age (t=15/22min), which precedes a rolling-
    accuracy drop (t=35min). Nothing else in the event list. Exists to
    confirm both agents can nail an uncontested case -- if either fails
    here, that's a real problem, not a hard-case artifact.
    """
    events = [
        _event(
            0, None, DetectionMethod.ISOLATION_FOREST, 0.91,
            "Isolation-forest detector flagged elevated anomaly rate in current data "
            "relative to reference (anomaly_rate=0.2000, contamination_baseline=0.0500)",
            ground_truth_label=FailureType.SCHEMA_MISMATCH,
        ),
        _event(
            6, None, DetectionMethod.ISOLATION_FOREST, 0.87,
            "Isolation-forest detector flagged elevated anomaly rate in current data "
            "relative to reference (anomaly_rate=0.1700, contamination_baseline=0.0500)",
            ground_truth_label=FailureType.SCHEMA_MISMATCH,
        ),
        _event(
            15, "customer_age", DetectionMethod.KS_TEST, 0.84,
            "KS Test detected distribution shift in customer_age (D=0.4800, p=0.0009)",
        ),
        _event(
            22, "customer_age", DetectionMethod.KS_TEST, 0.79,
            "KS Test detected distribution shift in customer_age (D=0.4300, p=0.0021)",
        ),
        _event(
            35, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.73,
            "Rolling-accuracy detector flagged a drop in prediction accuracy "
            "(baseline=0.9000, current=0.8200, drop=0.0800 > threshold=0.0500)",
        ),
    ]
    return BenchmarkIncident(
        incident_id="benchmark-clean-schema-005",
        events=events,
        ground_truth_label=FailureType.SCHEMA_MISMATCH,
        description="schema change -> feature drift -> accuracy drop, no distractor (control)",
    )


def _clean_corrupted_case() -> BenchmarkIncident:
    """Control case, NO distractor: corrupted values (t=0/5min) precede a
    rolling-accuracy drop (t=12min). Second sanity check on an uncontested
    case, this time with the root's inferred FailureType (CORRUPTED_VALUES)
    matching its ground_truth_label directly.
    """
    events = [
        _event(
            0, None, DetectionMethod.ISOLATION_FOREST, 0.89,
            "Isolation-forest detector flagged elevated anomaly rate in current data "
            "relative to reference (anomaly_rate=0.1900, contamination_baseline=0.0500)",
            ground_truth_label=FailureType.CORRUPTED_VALUES,
        ),
        _event(
            5, None, DetectionMethod.ISOLATION_FOREST, 0.86,
            "Isolation-forest detector flagged elevated anomaly rate in current data "
            "relative to reference (anomaly_rate=0.1600, contamination_baseline=0.0500)",
            ground_truth_label=FailureType.CORRUPTED_VALUES,
        ),
        _event(
            12, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.77,
            "Rolling-accuracy detector flagged a drop in prediction accuracy "
            "(baseline=0.8800, current=0.7900, drop=0.0900 > threshold=0.0500)",
        ),
    ]
    return BenchmarkIncident(
        incident_id="benchmark-clean-corrupted-006",
        events=events,
        ground_truth_label=FailureType.CORRUPTED_VALUES,
        description="corrupted values -> accuracy drop, no distractor (control)",
    )


def _deep_chain_incident() -> BenchmarkIncident:
    """Four-stage chain, one hop longer than every other incident here:
    missing values in shipping_address (t=0/5min) precede a schema
    mismatch in warehouse_id (t=15/20min) -- MISSING_VALUES/SCHEMA_MISMATCH
    is a causally-plausible pair per
    schemas.CAUSALLY_PLAUSIBLE_FAILURE_TYPE_PAIRS -- which precedes
    feature drift in order_total (t=35/40min), which precedes a rolling-
    accuracy drop (t=55min). A weak, far-outside-the-window distractor at
    t=180min tests whether a longer chain dilutes the structured agent's
    advantage.
    """
    events = [
        _event(
            0, "shipping_address", DetectionMethod.MISSING_VALUE_RATE, 0.86,
            "Missing-values detector flagged elevated null rate (null_rate=0.3200, threshold=0.0500)",
            ground_truth_label=FailureType.MISSING_VALUES,
        ),
        _event(
            5, "shipping_address", DetectionMethod.MISSING_VALUE_RATE, 0.82,
            "Missing-values detector flagged elevated null rate (null_rate=0.2900, threshold=0.0500)",
            ground_truth_label=FailureType.MISSING_VALUES,
        ),
        _event(
            15, "warehouse_id", DetectionMethod.SCHEMA_CHECK, 0.80,
            "Schema-mismatch detector flagged column 'warehouse_id' dtype changed from int64 to object",
        ),
        _event(
            20, "warehouse_id", DetectionMethod.SCHEMA_CHECK, 0.78,
            "Schema-mismatch detector flagged column 'warehouse_id' dtype changed from int64 to object",
        ),
        _event(
            35, "order_total", DetectionMethod.KS_TEST, 0.76,
            "KS Test detected distribution shift in order_total (D=0.4100, p=0.0033)",
        ),
        _event(
            40, "order_total", DetectionMethod.KS_TEST, 0.72,
            "KS Test detected distribution shift in order_total (D=0.3700, p=0.0058)",
        ),
        _event(
            55, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.68,
            "Rolling-accuracy detector flagged a drop in prediction accuracy "
            "(baseline=0.9000, current=0.8300, drop=0.0700 > threshold=0.0500)",
        ),
        _event(
            180, "referral_source", DetectionMethod.JENSEN_SHANNON, 0.60,
            "Jensen-Shannon divergence detector flagged distribution shift in referral_source "
            "(js_divergence=0.2000, threshold=0.1000)",
        ),
    ]
    return BenchmarkIncident(
        incident_id="benchmark-deep-chain-007",
        events=events,
        ground_truth_label=FailureType.MISSING_VALUES,
        description=(
            "missing values -> schema mismatch -> feature drift -> accuracy drop, "
            "weak distractor (4-hop chain)"
        ),
    )


def _temporal_proximity_trap_incident() -> BenchmarkIncident:
    """Feature drift in queue_depth (t=0/6min) precedes a rolling-accuracy
    drop (t=20min) -- the true chain. A duplicate-row-rate spike at
    t=10min sits INSIDE the chain's own time span (unlike every other
    incident's distractor, placed well outside the 60-minute clustering
    window) and carries high confidence, but DUPLICATES has no causally-
    plausible pairing with any other failure type in
    schemas.CAUSALLY_PLAUSIBLE_FAILURE_TYPE_PAIRS. Tests whether an agent
    relies on temporal proximity alone (naive, reasoning over raw
    timestamps) versus type-compatibility (structured, whose graph-
    building enforces this rule directly, regardless of timing).
    """
    events = [
        _event(
            0, "queue_depth", DetectionMethod.KS_TEST, 0.84,
            "KS Test detected distribution shift in queue_depth (D=0.4600, p=0.0012)",
            ground_truth_label=FailureType.FEATURE_DRIFT,
        ),
        _event(
            6, "queue_depth", DetectionMethod.KS_TEST, 0.80,
            "KS Test detected distribution shift in queue_depth (D=0.4200, p=0.0025)",
            ground_truth_label=FailureType.FEATURE_DRIFT,
        ),
        _event(
            10, None, DetectionMethod.DUPLICATE_ROW_RATE, 0.91,
            "Duplicate-rows detector flagged elevated duplicate-row rate (duplicate_rate=0.3500, threshold=0.0500)",
        ),
        _event(
            20, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.71,
            "Rolling-accuracy detector flagged a drop in prediction accuracy "
            "(baseline=0.9000, current=0.8200, drop=0.0800 > threshold=0.0500)",
        ),
    ]
    return BenchmarkIncident(
        incident_id="benchmark-temporal-trap-008",
        events=events,
        ground_truth_label=FailureType.FEATURE_DRIFT,
        description="feature drift -> accuracy drop, temporally-adjacent but type-incompatible distractor",
    )


def _dual_independent_failures_incident() -> BenchmarkIncident:
    """Two genuinely independent problems in the same detection run: the
    primary incident is a schema mismatch (t=0/5min) precipitating feature
    drift in payment_amount (t=15/20min) and then a rolling-accuracy drop
    (t=35min) -- ground truth. A SEPARATE duplicate-row-rate spike
    (t=12min, inside the primary chain's own time span) is a second, real,
    independently-occurring problem, not a scripted decoy -- DUPLICATES
    has no causally-plausible pairing with anything (see
    schemas.CAUSALLY_PLAUSIBLE_FAILURE_TYPE_PAIRS), so it stays an
    isolated node regardless of timing. Tests whether an agent still picks
    the correct PRIMARY cause when a second, unrelated, real anomaly is
    competing for attention with comparable confidence.
    """
    events = [
        _event(
            0, None, DetectionMethod.ISOLATION_FOREST, 0.90,
            "Isolation-forest detector flagged elevated anomaly rate in current data "
            "relative to reference (anomaly_rate=0.2100, contamination_baseline=0.0500)",
            ground_truth_label=FailureType.SCHEMA_MISMATCH,
        ),
        _event(
            5, None, DetectionMethod.ISOLATION_FOREST, 0.86,
            "Isolation-forest detector flagged elevated anomaly rate in current data "
            "relative to reference (anomaly_rate=0.1800, contamination_baseline=0.0500)",
            ground_truth_label=FailureType.SCHEMA_MISMATCH,
        ),
        _event(
            12, None, DetectionMethod.DUPLICATE_ROW_RATE, 0.80,
            "Duplicate-rows detector flagged elevated duplicate-row rate (duplicate_rate=0.2600, threshold=0.0500)",
        ),
        _event(
            15, "payment_amount", DetectionMethod.PSI, 0.83,
            "PSI detector flagged distribution shift in payment_amount (psi=0.2900, threshold=0.1000)",
        ),
        _event(
            20, "payment_amount", DetectionMethod.PSI, 0.79,
            "PSI detector flagged distribution shift in payment_amount (psi=0.2500, threshold=0.1000)",
        ),
        _event(
            35, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.74,
            "Rolling-accuracy detector flagged a drop in prediction accuracy "
            "(baseline=0.9000, current=0.8200, drop=0.0800 > threshold=0.0500)",
        ),
    ]
    return BenchmarkIncident(
        incident_id="benchmark-dual-failure-009",
        events=events,
        ground_truth_label=FailureType.SCHEMA_MISMATCH,
        description=(
            "schema mismatch -> feature drift -> accuracy drop, PLUS a second "
            "independent real anomaly (not a decoy)"
        ),
    )


def _weak_root_incident() -> BenchmarkIncident:
    """The true root cause is only weakly detected: corrupted values at
    confidence 0.55/0.50 (well above the detection threshold, but modest),
    precipitating a rolling-accuracy drop at 0.62 (t=0/5/18min). A
    structurally-disconnected distractor at t=220min has moderate
    confidence (0.60) -- not dramatically higher than the true root's, the
    way the other high-confidence-distractor incidents are. Tests whether
    a genuinely weak-but-connected signal beats an unconnected moderate
    one -- the hardest version of the confidence-vs-structure trade-off in
    this benchmark.
    """
    events = [
        _event(
            0, None, DetectionMethod.ISOLATION_FOREST, 0.55,
            "Isolation-forest detector flagged elevated anomaly rate in current data "
            "relative to reference (anomaly_rate=0.0900, contamination_baseline=0.0500)",
            ground_truth_label=FailureType.CORRUPTED_VALUES,
        ),
        _event(
            5, None, DetectionMethod.ISOLATION_FOREST, 0.50,
            "Isolation-forest detector flagged elevated anomaly rate in current data "
            "relative to reference (anomaly_rate=0.0850, contamination_baseline=0.0500)",
            ground_truth_label=FailureType.CORRUPTED_VALUES,
        ),
        _event(
            18, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.62,
            "Rolling-accuracy detector flagged a drop in prediction accuracy "
            "(baseline=0.8700, current=0.8100, drop=0.0600 > threshold=0.0500)",
        ),
        _event(
            220, "browser_type", DetectionMethod.JENSEN_SHANNON, 0.60,
            "Jensen-Shannon divergence detector flagged distribution shift in browser_type "
            "(js_divergence=0.1800, threshold=0.1000)",
        ),
    ]
    return BenchmarkIncident(
        incident_id="benchmark-weak-root-010",
        events=events,
        ground_truth_label=FailureType.CORRUPTED_VALUES,
        description="corrupted values (weak signal) -> accuracy drop, moderate-confidence far-away distractor",
    )


def _label_shift_missing_values_trap_incident() -> BenchmarkIncident:
    """Label shift in churn_flag (t=0/6min) precedes a rolling-accuracy
    drop (t=22min) -- the true chain. A missing-values spike in
    referral_code at t=45min sits inside the 60-minute window with high
    confidence, but MISSING_VALUES has no plausible pairing with
    LABEL_SHIFT (only with SCHEMA_MISMATCH -- see
    schemas.CAUSALLY_PLAUSIBLE_FAILURE_TYPE_PAIRS), so it should stay
    unlinked despite the timing. A second instance of the type-
    plausibility trap in incident 008, with a different failure-type
    pair, so the finding isn't resting on a single example.
    """
    events = [
        _event(
            0, "churn_flag", DetectionMethod.JENSEN_SHANNON, 0.85,
            "Jensen-Shannon divergence detector flagged distribution shift in churn_flag "
            "(js_divergence=0.3700, threshold=0.1000)",
            ground_truth_label=FailureType.LABEL_SHIFT,
        ),
        _event(
            6, "churn_flag", DetectionMethod.JENSEN_SHANNON, 0.81,
            "Jensen-Shannon divergence detector flagged distribution shift in churn_flag "
            "(js_divergence=0.3400, threshold=0.1000)",
            ground_truth_label=FailureType.LABEL_SHIFT,
        ),
        _event(
            22, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.70,
            "Rolling-accuracy detector flagged a drop in prediction accuracy "
            "(baseline=0.9000, current=0.8200, drop=0.0800 > threshold=0.0500)",
        ),
        _event(
            45, "referral_code", DetectionMethod.MISSING_VALUE_RATE, 0.89,
            "Missing-values detector flagged elevated null rate (null_rate=0.4000, threshold=0.0500)",
        ),
    ]
    return BenchmarkIncident(
        incident_id="benchmark-label-shift-trap-011",
        events=events,
        ground_truth_label=FailureType.LABEL_SHIFT,
        description="label shift -> accuracy drop, temporally-close but type-incompatible distractor",
    )


def _schema_duplicates_trap_incident() -> BenchmarkIncident:
    """Schema mismatch in currency_code (t=0/6min) precedes a rolling-
    accuracy drop (t=18min). A duplicate-row-rate spike at t=12min sits
    inside the chain's own span with high confidence, but DUPLICATES has
    no plausible pairing with anything (see
    schemas.CAUSALLY_PLAUSIBLE_FAILURE_TYPE_PAIRS). A third instance of
    the type-plausibility trap (008, 011), for a third failure-type pair.
    """
    events = [
        _event(
            0, "currency_code", DetectionMethod.SCHEMA_CHECK, 0.87,
            "Schema-mismatch detector flagged column 'currency_code' dtype changed from object to float64",
            ground_truth_label=FailureType.SCHEMA_MISMATCH,
        ),
        _event(
            6, "currency_code", DetectionMethod.SCHEMA_CHECK, 0.83,
            "Schema-mismatch detector flagged column 'currency_code' dtype changed from object to float64",
            ground_truth_label=FailureType.SCHEMA_MISMATCH,
        ),
        _event(
            12, None, DetectionMethod.DUPLICATE_ROW_RATE, 0.88,
            "Duplicate-rows detector flagged elevated duplicate-row rate (duplicate_rate=0.3100, threshold=0.0500)",
        ),
        _event(
            18, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.66,
            "Rolling-accuracy detector flagged a drop in prediction accuracy "
            "(baseline=0.9000, current=0.8300, drop=0.0700 > threshold=0.0500)",
        ),
    ]
    return BenchmarkIncident(
        incident_id="benchmark-schema-duplicates-trap-012",
        events=events,
        ground_truth_label=FailureType.SCHEMA_MISMATCH,
        description="schema mismatch -> accuracy drop, temporally-close but type-incompatible duplicates distractor",
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
        _clean_schema_case(),
        _clean_corrupted_case(),
        _deep_chain_incident(),
        _temporal_proximity_trap_incident(),
        _dual_independent_failures_incident(),
        _weak_root_incident(),
        _label_shift_missing_values_trap_incident(),
        _schema_duplicates_trap_incident(),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help=(
            "Use a real Gemini call for both RCAAgent and NaiveRCAAgent "
            "instead of the deterministic offline stubs -- the actual "
            "research-question comparison this module exists for, rather "
            "than deterministic rule-following on both sides. Requires "
            "GOOGLE_API_KEY (see remediation/pipeline_utils.py's get_llm() "
            "docstring for a free key). The SAME model is used for both "
            "agents so the comparison isolates 'structured graph vs raw "
            "events' rather than 'model A vs model B'."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help=(
            "Run the full incident set this many times and report "
            "per-incident hit-rates plus mean +/- stdev accuracy across "
            "repeats, instead of one single-run pass/fail table. Only "
            "the LLM ranking/explanation step varies between repeats "
            "(candidate-finding is deterministic) -- with --real-llm, "
            "this is how you tell a stable finding apart from one "
            "lucky/unlucky sampler roll (some Gemini tiers ignore "
            "temperature=0; see remediation/pipeline_utils.py). Default 1 "
            "(single run, original behavior)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Also write the result dict (as_dict()) as JSON to this file "
            "path, independent of the table/JSON already printed to "
            "stdout. Lets a slow/rate-limited --real-llm run's numbers be "
            "saved once and then read by something else (e.g. the "
            "frontend's benchmark dashboard) without re-running the LLM."
        ),
    )
    parser.add_argument(
        "--no-graph-edges",
        action="store_true",
        help=(
            "Ablation control: build the evidence graph with NO edges "
            "(min_edge_confidence=1.0 -- an edge's confidence is always "
            "strictly below 1.0). The structured agent then sees the same "
            "nodes but no causal links. If its accuracy drops on the "
            "cascade incidents, the graph is doing real work; if not, the "
            "benefit comes from somewhere else (e.g. prompt/representation)."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    runner_kwargs = {}
    if args.no_graph_edges:
        from configs.settings import GRAPH, GraphSettings

        runner_kwargs["graph_builder"] = EvidenceGraphBuilder(
            GraphSettings(time_window_minutes=GRAPH.time_window_minutes, min_edge_confidence=1.0)
        )

    if args.real_llm:
        from remediation.pipeline_utils import get_llm

        llm = get_llm(use_stub=False)
        runner = BenchmarkRunner(structured_llm=llm, naive_llm=llm, **runner_kwargs)
    else:
        runner = BenchmarkRunner(**runner_kwargs)

    if args.repeats <= 1:
        result_summary = runner.run(load_builtin_incidents())
        result_summary.print_table()
        print()
        result_dict = result_summary.as_dict()
        print(json.dumps(result_dict, indent=2))
    else:
        summaries = []
        for repeat_num in range(1, args.repeats + 1):
            logger.info("=== Repeat %d/%d ===", repeat_num, args.repeats)
            summaries.append(runner.run(load_builtin_incidents()))
        aggregated = AggregatedBenchmarkSummary(summaries=summaries)
        aggregated.print_table()
        print()
        result_dict = aggregated.as_dict()
        print(json.dumps(result_dict, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)
        logger.info("Wrote result JSON to %s", args.output)
