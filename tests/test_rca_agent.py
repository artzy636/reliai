"""
Tests for reasoning/rca_agent.py — RCAAgent (structured) and NaiveRCAAgent
(control-group baseline).

Both agents' LLM calls are stubbed: no network access, no API key, fully
deterministic. The stub mimics a LangChain chat model's `.invoke(prompt) ->
AIMessage`-with-`.content` shape so we also exercise the same duck-typed
path a real langchain_anthropic.ChatAnthropic client would take.

The synthetic incident mirrors tests/test_evidence_graph.py's causal-chain
scenario: a customer_age drift precedes a model_prediction shift shortly
after, plus a distractor event far outside the time window with nothing to
do with either. The known root cause is the customer_age drift.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from configs.settings import RCASettings
from reasoning.evidence_graph import EvidenceGraphBuilder
from reasoning.rca_agent import NaiveRCAAgent, RCAAgent
from schemas import DetectionMethod, EvidenceEvent, FailureType

BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _event(
    minutes_after_base: float,
    feature_name: str,
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


def _single_cause_incident() -> list[EvidenceEvent]:
    """customer_age drifts (t=0, t=10min), causing a model_prediction shift
    at t=20min; an unrelated_metric spike at t=150min is a distractor well
    outside GRAPH.time_window_minutes (60min default) of everything else.
    """
    return [
        _event(0, "customer_age", 0.9, ground_truth_label=FailureType.FEATURE_DRIFT),
        _event(10, "customer_age", 0.85, ground_truth_label=FailureType.FEATURE_DRIFT),
        _event(20, "model_prediction", 0.8),
        _event(150, "unrelated_metric", 0.95),
    ]


class _FakeAIMessage:
    """Mimics langchain_core's AIMessage just enough for _call_llm's
    duck-typed `.content` extraction."""

    def __init__(self, content: str) -> None:
        self.content = content


class _StubChatModel:
    """LangChain-style stub: exposes `.invoke(prompt) -> AIMessage`, records
    every prompt it was called with, and returns a pre-programmed response.
    """

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> _FakeAIMessage:
        self.prompts.append(prompt)
        return _FakeAIMessage(self.response)


# ---------------------------------------------------------------------------
# RCAAgent (structured, graph-based)
# ---------------------------------------------------------------------------

def test_structured_rca_top_hypothesis_identifies_root_cause():
    """The LLM stub returns a ranking that agrees with the deterministic
    candidate order; the top hypothesis must center on the customer_age
    node (the true root cause) and its explanation + supporting_node_ids
    must be traceable to that node's causal chain, not the distractor."""
    events = _single_cause_incident()
    graph = EvidenceGraphBuilder().build(events)

    node_a = next(n for n in graph.nodes if graph.nodes[n]["node_type"] == "feature_anomaly:customer_age")
    node_b = next(n for n in graph.nodes if graph.nodes[n]["node_type"] == "feature_anomaly:model_prediction")

    llm_response = json.dumps(
        [
            {
                "node_id": node_a,
                "explanation": (
                    "feature_anomaly:customer_age preceded feature_anomaly:model_prediction "
                    "via a preceded relationship, indicating the customer_age drift is the root cause."
                ),
            },
            {
                "node_id": next(
                    n for n in graph.nodes if graph.nodes[n]["node_type"] == "feature_anomaly:unrelated_metric"
                ),
                "explanation": "Isolated feature_anomaly:unrelated_metric with no downstream effects.",
            },
        ]
    )
    stub_llm = _StubChatModel(llm_response)
    agent = RCAAgent(llm=stub_llm)

    result = agent.analyze(graph, incident_id="incident-001")

    assert result.incident_id == "incident-001"
    top = result.hypotheses[0]
    assert top.node_id == node_a
    assert top.rank == 1
    assert "customer_age" in top.explanation.lower()
    assert top.supporting_node_ids == [node_a, node_b]
    assert result.causal_path_node_ids == [node_a, node_b]

    # The LLM must only ever see structural graph facts, never raw telemetry.
    prompt = stub_llm.prompts[0]
    assert "metric_value" not in prompt
    assert "threshold" not in prompt
    assert "ground_truth_label" not in prompt
    assert "feature_drift" not in prompt.lower()


def test_structured_rca_caps_hypotheses_at_max_hypotheses():
    """Two candidates exist (customer_age chain + unrelated_metric); capping
    max_hypotheses at 1 must return exactly one hypothesis."""
    events = _single_cause_incident()
    graph = EvidenceGraphBuilder().build(events)

    node_a = next(n for n in graph.nodes if graph.nodes[n]["node_type"] == "feature_anomaly:customer_age")
    node_c = next(
        n for n in graph.nodes if graph.nodes[n]["node_type"] == "feature_anomaly:unrelated_metric"
    )
    llm_response = json.dumps(
        [
            {"node_id": node_a, "explanation": "feature_anomaly:customer_age is the likely root cause."},
            {"node_id": node_c, "explanation": "feature_anomaly:unrelated_metric is unrelated noise."},
        ]
    )
    agent = RCAAgent(llm=_StubChatModel(llm_response), rca_settings=RCASettings(max_hypotheses=1))

    result = agent.analyze(graph, incident_id="incident-002")

    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].node_id == node_a


def test_structured_rca_falls_back_to_template_on_invalid_llm_response():
    """If the LLM returns garbage (unparseable / references an unknown
    node_id), the agent must not crash -- it degrades to a deterministic,
    graph-grounded template explanation instead."""
    events = _single_cause_incident()
    graph = EvidenceGraphBuilder().build(events)
    agent = RCAAgent(llm=_StubChatModel("not valid json at all"))

    result = agent.analyze(graph, incident_id="incident-003")

    assert len(result.hypotheses) >= 1
    # Template fallback still references real node types from the graph.
    assert "feature_anomaly:" in result.hypotheses[0].explanation


def test_structured_rca_requires_llm_when_candidates_exist():
    """Per the chosen 'pluggable interface, provider unwired' design, an
    agent with no llm configured must fail loudly rather than silently
    skipping the LLM ranking/explanation step."""
    events = _single_cause_incident()
    graph = EvidenceGraphBuilder().build(events)
    agent = RCAAgent(llm=None)

    with pytest.raises(ValueError):
        agent.analyze(graph, incident_id="incident-004")


def test_structured_rca_empty_graph_does_not_require_llm():
    """An empty graph has no candidates, so no LLM call is needed at all --
    this must not raise just because llm=None."""
    agent = RCAAgent(llm=None)
    result = agent.analyze(EvidenceGraphBuilder().build([]), incident_id="incident-005")

    assert result.hypotheses == []
    assert result.causal_path_node_ids == []


# ---------------------------------------------------------------------------
# NaiveRCAAgent (control group / baseline)
# ---------------------------------------------------------------------------

def test_naive_rca_produces_scoreable_rcaresult():
    """The naive baseline gets raw events with no graph at all, and must
    still emit an RCAResult with the same shape as the structured agent."""
    events = _single_cause_incident()
    root_event_id = events[0].event_id  # first customer_age event

    llm_response = json.dumps(
        [
            {
                "event_ids": [events[0].event_id, events[1].event_id],
                "explanation": "customer_age looks anomalous across two readings.",
                "confidence": 0.7,
            }
        ]
    )
    stub_llm = _StubChatModel(llm_response)
    agent = NaiveRCAAgent(llm=stub_llm)

    result = agent.analyze(events, incident_id="incident-baseline-001")

    assert result.incident_id == "incident-baseline-001"
    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].node_id == root_event_id
    assert result.hypotheses[0].supporting_node_ids == [events[0].event_id, events[1].event_id]
    assert result.causal_path_node_ids == result.hypotheses[0].supporting_node_ids


def test_naive_rca_never_leaks_ground_truth_label_into_prompt():
    """CONTRACT.md rule 4: ground_truth_label must never leak into anything
    resembling a production reasoning path, including this baseline."""
    events = _single_cause_incident()
    stub_llm = _StubChatModel(json.dumps([]))
    agent = NaiveRCAAgent(llm=stub_llm)

    agent.analyze(events, incident_id="incident-baseline-002")

    prompt = stub_llm.prompts[0]
    assert "ground_truth_label" not in prompt
    assert "feature_drift" not in prompt.lower()


def test_naive_rca_requires_llm():
    """Same pluggable-interface contract as RCAAgent: no silent skip."""
    agent = NaiveRCAAgent(llm=None)

    with pytest.raises(ValueError):
        agent.analyze(_single_cause_incident(), incident_id="incident-baseline-003")


def test_naive_rca_accepts_plain_callable_llm():
    """The pluggable interface must also accept a bare
    Callable[[str], str], not just a LangChain-style object with .invoke."""
    events = _single_cause_incident()
    response = json.dumps(
        [{"event_ids": [events[0].event_id], "explanation": "plain callable works", "confidence": 0.6}]
    )

    def fake_llm(prompt: str) -> str:
        return response

    agent = NaiveRCAAgent(llm=fake_llm)
    result = agent.analyze(events, incident_id="incident-baseline-004")

    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].explanation == "plain callable works"
