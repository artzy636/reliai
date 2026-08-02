"""
Tests for evaluation/benchmark_runner.py.

Two levels of rigor, matching tests/test_reasoning_integration.py:

  1. The built-in synthetic incidents (load_builtin_incidents()) are run
     through the real pipeline end-to-end (real EvidenceGraphBuilder, real
     RCAAgent/NaiveRCAAgent, only the LLM boundary stubbed) and checked
     against a hand-verified per-incident correctness table + exact
     aggregate accuracy -- not just "it ran".

  2. A dedicated scenario proves scoring correctly bridges the two disjoint
     id spaces RCAAgent (EvidenceNode.node_id) and NaiveRCAAgent
     (EvidenceEvent.event_id) write into -- so a structured hypothesis that
     is actually correct isn't silently scored as wrong by a naive id
     string-compare, and a genuinely wrong hypothesis in either id space is
     still caught as wrong (i.e. this isn't a trivially-always-True check).
"""

import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from evaluation.benchmark_runner import (
    BenchmarkIncident,
    BenchmarkRunner,
    BenchmarkSummary,
    IncidentScore,
    load_builtin_incidents,
)
from schemas import DetectionMethod, EvidenceEvent, FailureType, RCAResult

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _event(
    minutes_after_base: float,
    feature_name: str | None,
    confidence: float,
    ground_truth_label: FailureType | None = None,
) -> EvidenceEvent:
    """Build a minimal, valid EvidenceEvent for test scenarios."""
    return EvidenceEvent(
        timestamp=BASE + timedelta(minutes=minutes_after_base),
        detection_method=DetectionMethod.PSI,
        feature_name=feature_name,
        metric_value=0.5,
        threshold=0.2,
        confidence=confidence,
        description=f"synthetic event for {feature_name} at t+{minutes_after_base}m",
        ground_truth_label=ground_truth_label,
    )


class _FakeAIMessage:
    """Mimics langchain_core's AIMessage just enough for _call_llm's
    duck-typed `.content` extraction."""

    def __init__(self, content: str) -> None:
        self.content = content


class _StubChatModel:
    """LangChain-style stub: exposes `.invoke(prompt) -> AIMessage`."""

    def __init__(self, response: str) -> None:
        self.response = response

    def invoke(self, prompt: str) -> _FakeAIMessage:
        return _FakeAIMessage(self.response)


# ---------------------------------------------------------------------------
# 1. Built-in incidents, hand-verified per-incident + aggregate results.
#
# Hand-verified by construction (see benchmark_runner.py's incident
# docstrings for the confidence/timing numbers this depends on):
#   - schema-cascade / duplicates: the true root cause has a *lower* raw
#     confidence than an out-of-window distractor. The structured agent's
#     graph-propagation ranking still finds it (a distractor with no
#     downstream edges can't out-rank a multi-hop causal chain); the
#     default naive stub -- which just picks the single highest-confidence
#     raw event -- picks the distractor instead. structured=correct,
#     naive=WRONG.
#   - corrupted-values / label-shift: the distractor has *lower* confidence
#     than the true root cause, so even the naive "highest confidence"
#     stub happens to land on the right event. Both correct.
# Aggregate: structured 4/4 = 100%, naive 2/4 = 50%.
# ---------------------------------------------------------------------------

def test_builtin_incidents_full_benchmark_matches_hand_verified_results():
    """Run the real pipeline (real graph builder, real agents, only the LLM
    boundary stubbed via the runner's default offline stubs) over every
    built-in incident and check both per-incident correctness and the
    aggregate accuracy against hand-verified expected values."""
    incidents = load_builtin_incidents()
    assert [incident.incident_id for incident in incidents] == [
        "benchmark-schema-cascade-001",
        "benchmark-corrupted-values-002",
        "benchmark-duplicates-003",
        "benchmark-label-shift-004",
    ]

    summary = BenchmarkRunner().run(incidents)

    expected = {
        "benchmark-schema-cascade-001": (True, False),
        "benchmark-corrupted-values-002": (True, True),
        "benchmark-duplicates-003": (True, False),
        "benchmark-label-shift-004": (True, True),
    }
    actual = {
        score.incident_id: (score.structured_correct, score.naive_correct) for score in summary.scores
    }
    assert actual == expected

    assert summary.n_incidents == 4
    assert summary.structured_accuracy == pytest.approx(1.0)
    assert summary.naive_accuracy == pytest.approx(0.5)

    as_dict = summary.as_dict()
    assert as_dict["n_incidents"] == 4
    assert as_dict["structured_accuracy"] == pytest.approx(1.0)
    assert as_dict["naive_accuracy"] == pytest.approx(0.5)
    assert as_dict["per_incident"][0] == {
        "incident_id": "benchmark-schema-cascade-001",
        "ground_truth_label": "schema_mismatch",
        "structured_correct": True,
        "naive_correct": False,
    }
    # Must be JSON-serializable as-is (the report/paper table depends on this).
    json.dumps(as_dict)


def test_builtin_incidents_true_root_cause_event_ids_match_labeled_events():
    """BenchmarkIncident.true_root_cause_event_ids (the optional 'which
    event(s) are the true root cause' annotation) must resolve to exactly
    the events carrying the incident's own ground_truth_label -- the two
    schema-change events, not the drift/shift/distractor events."""
    incident = load_builtin_incidents()[0]
    labeled = [e for e in incident.events if e.ground_truth_label == FailureType.SCHEMA_MISMATCH]

    assert incident.true_root_cause_event_ids == [e.event_id for e in labeled]
    assert len(incident.true_root_cause_event_ids) == 2


def test_print_table_runs_without_error(capsys):
    """Smoke check for the printed results table (req 5) -- must include
    both agents' accuracy and not blow up on formatting."""
    summary = BenchmarkRunner().run(load_builtin_incidents())

    summary.print_table()

    captured = capsys.readouterr().out
    assert "structured" in captured
    assert "naive" in captured
    assert "100%" in captured
    assert "50%" in captured


# ---------------------------------------------------------------------------
# 2. ID-space resolution: prove a structured (node_id-space) hypothesis and
# a naive (event_id-space) hypothesis are scored correctly -- and
# correctly as wrong when they ARE wrong -- despite referencing disjoint id
# spaces. This is req 4: no silent false-negative (or false-positive) from
# comparing the wrong id space.
# ---------------------------------------------------------------------------

def _single_cause_incident() -> BenchmarkIncident:
    """customer_age drifts (t=0, t=10min), causing a model_prediction shift
    at t=20min; an unrelated_metric spike at t=150min is a distractor well
    outside GRAPH.time_window_minutes (60min default) of everything else.
    Mirrors tests/test_rca_agent.py's scenario.
    """
    events = [
        _event(0, "customer_age", 0.9, ground_truth_label=FailureType.FEATURE_DRIFT),
        _event(10, "customer_age", 0.85, ground_truth_label=FailureType.FEATURE_DRIFT),
        _event(20, "model_prediction", 0.8),
        _event(150, "unrelated_metric", 0.95),
    ]
    return BenchmarkIncident(
        incident_id="id-space-test-incident",
        events=events,
        ground_truth_label=FailureType.FEATURE_DRIFT,
    )


class _PickCandidateByNodeType:
    """Test-only stub for RCAAgent: parses the prompt (rather than needing
    a node_id known ahead of time -- EvidenceNode.node_id is a fresh
    uuid4() on every EvidenceGraphBuilder.build() call, so one can't be
    hardcoded before BenchmarkRunner builds its own internal graph) and
    ranks whichever candidate's causal chain starts with
    ``preferred_node_type`` first."""

    _CANDIDATE_RE = re.compile(r"- root_node_id: (\S+)\n {2}causal_chain: (\S[^\n]*)")

    def __init__(self, preferred_node_type: str) -> None:
        self._preferred_node_type = preferred_node_type

    def __call__(self, prompt: str) -> str:
        candidates = self._CANDIDATE_RE.findall(prompt)
        preferred_ids = [nid for nid, chain in candidates if chain.startswith(self._preferred_node_type)]
        other_ids = [nid for nid, _ in candidates if nid not in preferred_ids]
        return json.dumps(
            [
                {"node_id": node_id, "explanation": f"picked for starting with {self._preferred_node_type}"}
                for node_id in preferred_ids + other_ids
            ]
        )


class _PickEventsByFeatureName:
    """Test-only stub for NaiveRCAAgent: parses the prompt for event_ids
    whose feature_name matches ``preferred_feature_name`` and returns them
    as the (only) hypothesis."""

    _EVENT_RE = re.compile(r"- event_id=(\S+) feature_name=(\S+)")

    def __init__(self, preferred_feature_name: str) -> None:
        self._preferred_feature_name = preferred_feature_name

    def __call__(self, prompt: str) -> str:
        matches = self._EVENT_RE.findall(prompt)
        picked = [event_id for event_id, feature_name in matches if feature_name == self._preferred_feature_name]
        return json.dumps(
            [{"event_ids": picked, "explanation": f"picked {self._preferred_feature_name}", "confidence": 0.8}]
        )


def test_correct_hypotheses_in_both_id_spaces_both_score_correct():
    """A structured hypothesis whose node_id resolves (via EvidenceNode.
    source_events) to the customer_age events, and a naive hypothesis whose
    node_id IS a customer_age event_id directly, must BOTH score correct
    against the same ground truth -- proving root_cause_is_correct() (as
    used by BenchmarkRunner) bridges both id spaces rather than only ever
    matching one of them by accident."""
    incident = _single_cause_incident()
    runner = BenchmarkRunner(
        structured_llm=_PickCandidateByNodeType("feature_anomaly:customer_age"),
        naive_llm=_PickEventsByFeatureName("customer_age"),
    )

    summary = runner.run([incident])

    score = summary.scores[0]
    # Sanity: the two agents really did write into disjoint id spaces.
    assert score.structured_result.hypotheses[0].node_id not in {e.event_id for e in incident.events}
    assert score.naive_result.hypotheses[0].node_id in {e.event_id for e in incident.events}
    # Both nonetheless resolve to the correct root cause.
    assert score.structured_correct is True
    assert score.naive_correct is True


def test_wrong_hypotheses_in_both_id_spaces_both_score_incorrect():
    """The negative control: when each agent's top hypothesis points at the
    distractor instead, both must score incorrect -- so the previous test's
    'both True' result isn't just root_cause_is_correct() vacuously
    returning True regardless of id space."""
    incident = _single_cause_incident()
    runner = BenchmarkRunner(
        structured_llm=_PickCandidateByNodeType("feature_anomaly:unrelated_metric"),
        naive_llm=_PickEventsByFeatureName("unrelated_metric"),
    )

    summary = runner.run([incident])

    score = summary.scores[0]
    assert score.structured_correct is False
    assert score.naive_correct is False


def test_benchmark_summary_accuracy_math_on_synthetic_scores():
    """Pure unit check of BenchmarkSummary's aggregation, decoupled from
    the agents: 3 correct / 4 total for structured, 1 / 4 for naive."""

    def _result(incident_id: str) -> RCAResult:
        return RCAResult(incident_id=incident_id, hypotheses=[], causal_path_node_ids=[])

    scores = [
        IncidentScore("a", FailureType.FEATURE_DRIFT, _result("a"), True, _result("a"), True),
        IncidentScore("b", FailureType.FEATURE_DRIFT, _result("b"), True, _result("b"), False),
        IncidentScore("c", FailureType.FEATURE_DRIFT, _result("c"), True, _result("c"), False),
        IncidentScore("d", FailureType.FEATURE_DRIFT, _result("d"), False, _result("d"), False),
    ]

    summary = BenchmarkSummary(scores=scores)

    assert summary.n_incidents == 4
    assert summary.structured_accuracy == pytest.approx(0.75)
    assert summary.naive_accuracy == pytest.approx(0.25)


def test_benchmark_summary_accuracy_on_empty_scores_is_zero_not_error():
    """No incidents run -> accuracy is 0.0, not a ZeroDivisionError."""
    summary = BenchmarkSummary(scores=[])

    assert summary.n_incidents == 0
    assert summary.structured_accuracy == 0.0
    assert summary.naive_accuracy == 0.0
