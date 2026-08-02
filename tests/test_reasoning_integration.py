"""
Integration test for reasoning/evidence_graph.py + reasoning/rca_agent.py.

Runs the full pipeline end-to-end -- a raw EvidenceEvent list -> a real
EvidenceGraphBuilder.build() graph -> RCAAgent.analyze() / NaiveRCAAgent.analyze()
-- with no mocking of the graph step, to catch interface mismatches that the
isolated unit tests (test_evidence_graph.py, test_rca_agent.py) wouldn't
surface. Only the LLM call is stubbed (no network access, no API key).

Synthetic scenario: a three-stage failure cascade plus one unrelated
distractor, using the default GRAPH/RCA settings (60min window, 0.3 min
edge confidence, 5 max hypotheses):

  1. Upstream schema change (pipeline-level, feature_name=None) at t=0/8min.
  2. That schema change causes feature drift in transaction_amount at
     t=20/28min.
  3. The drift causes a prediction shift in model_prediction at t=40/45min.
  4. An unrelated region_code spike at t=200min, far outside the 60-minute
     clustering/edge window, must not be linked to anything.

The true root cause is the schema-change node. Both agents are run against
comparable inputs derived from the same synthetic incident so their outputs
can be placed side by side -- that comparison (structured vs. naive RCA) is
the project's actual research question.
"""

import json
from datetime import datetime, timedelta, timezone

from reasoning.evidence_graph import EvidenceGraphBuilder
from reasoning.rca_agent import NaiveRCAAgent, RCAAgent
from schemas import DetectionMethod, EvidenceEvent, FailureType, RCAResult

BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _event(
    minutes_after_base: float,
    feature_name: str | None,
    detection_method: DetectionMethod,
    confidence: float,
    description: str,
    ground_truth_label: FailureType | None = None,
) -> EvidenceEvent:
    """Build a minimal, valid EvidenceEvent for the synthetic scenario."""
    return EvidenceEvent(
        timestamp=BASE + timedelta(minutes=minutes_after_base),
        detection_method=detection_method,
        feature_name=feature_name,
        metric_value=0.5,
        threshold=0.2,
        confidence=confidence,
        description=description,
        ground_truth_label=ground_truth_label,
    )


def _cascading_incident() -> list[EvidenceEvent]:
    """A schema change (t=0/8min) precedes feature drift in
    transaction_amount (t=20/28min), which precedes a prediction shift in
    model_prediction (t=40/45min). A region_code spike at t=200min is a
    distractor, far outside the default 60-minute window of everything else.
    """
    return [
        # Stage 1: upstream schema change (pipeline-level evidence).
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
        # Stage 2: feature drift caused by the schema change.
        _event(
            20, "transaction_amount", DetectionMethod.PSI, 0.88,
            "PSI for transaction_amount exceeded threshold following the schema change",
            ground_truth_label=FailureType.FEATURE_DRIFT,
        ),
        _event(
            28, "transaction_amount", DetectionMethod.PSI, 0.82,
            "PSI for transaction_amount remained elevated",
            ground_truth_label=FailureType.FEATURE_DRIFT,
        ),
        # Stage 3: prediction shift caused by the feature drift.
        _event(
            40, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.75,
            "Rolling accuracy dropped for the fraud model",
        ),
        _event(
            45, "model_prediction", DetectionMethod.ROLLING_ACCURACY, 0.70,
            "Rolling accuracy remained depressed",
        ),
        # Distractor: unrelated feature, far outside the clustering/edge window.
        _event(
            200, "region_code", DetectionMethod.JENSEN_SHANNON, 0.95,
            "Jensen-Shannon divergence spike for region_code, unrelated to the incident",
        ),
    ]


class _FakeAIMessage:
    """Mimics langchain_core's AIMessage just enough for _call_llm's
    duck-typed `.content` extraction."""

    def __init__(self, content: str) -> None:
        self.content = content


class _StubChatModel:
    """LangChain-style stub: exposes `.invoke(prompt) -> AIMessage`, records
    every prompt it was called with, and returns a pre-programmed response.
    No network access, no API key."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> _FakeAIMessage:
        self.prompts.append(prompt)
        return _FakeAIMessage(self.response)


def test_full_pipeline_structured_and_naive_agree_on_schema_root_cause() -> None:
    """End-to-end: a real EvidenceGraphBuilder graph feeds a real RCAAgent,
    and the same raw events feed a real NaiveRCAAgent -- both stubbed only
    at the LLM boundary. The structured agent's top hypothesis must
    identify the schema-change node as the root cause and reconstruct the
    full schema -> drift -> shift causal chain; the naive baseline must
    still return a same-shape RCAResult so the two can be compared."""
    events = _cascading_incident()

    # --- Step 1: real graph, no mocking. ---------------------------------
    graph = EvidenceGraphBuilder().build(events)

    assert graph.number_of_nodes() == 4
    assert graph.number_of_edges() == 2

    node_schema = next(
        n for n in graph.nodes if graph.nodes[n]["node_type"] == "pipeline_level_anomaly"
    )
    node_drift = next(
        n for n in graph.nodes if graph.nodes[n]["node_type"] == "feature_anomaly:transaction_amount"
    )
    node_shift = next(
        n for n in graph.nodes if graph.nodes[n]["node_type"] == "feature_anomaly:model_prediction"
    )
    node_distractor = next(
        n for n in graph.nodes if graph.nodes[n]["node_type"] == "feature_anomaly:region_code"
    )

    # Causal chain wired schema -> drift -> shift; distractor untouched.
    assert graph.has_edge(node_schema, node_drift)
    assert graph.has_edge(node_drift, node_shift)
    assert not graph.has_edge(node_schema, node_shift)  # decays below min_edge_confidence
    assert graph.in_degree(node_distractor) == 0
    assert graph.out_degree(node_distractor) == 0

    # --- Step 2: RCAAgent (structured) against the real graph. -----------
    structured_llm_response = json.dumps(
        [
            {
                "node_id": node_schema,
                "explanation": (
                    "pipeline_level_anomaly preceded feature_anomaly:transaction_amount, which in "
                    "turn preceded feature_anomaly:model_prediction, indicating the upstream schema "
                    "change is the root cause of the prediction shift."
                ),
            },
            {
                "node_id": node_distractor,
                "explanation": "Isolated feature_anomaly:region_code with no linked downstream effects.",
            },
        ]
    )
    structured_agent = RCAAgent(llm=_StubChatModel(structured_llm_response))

    structured_result = structured_agent.analyze(graph, incident_id="incident-integration-001")

    assert isinstance(structured_result, RCAResult)
    assert structured_result.incident_id == "incident-integration-001"
    top = structured_result.hypotheses[0]
    assert top.rank == 1
    assert top.node_id == node_schema
    assert top.supporting_node_ids == [node_schema, node_drift, node_shift]
    assert structured_result.causal_path_node_ids == [node_schema, node_drift, node_shift]
    assert "schema" in top.explanation.lower()

    # --- Step 3: NaiveRCAAgent (baseline) against the same raw events. ---
    schema_event_ids = [events[0].event_id, events[1].event_id]
    naive_llm_response = json.dumps(
        [
            {
                "event_ids": schema_event_ids,
                "explanation": "Two isolation-forest flags on row structure point to a schema change.",
                "confidence": 0.8,
            }
        ]
    )
    naive_agent = NaiveRCAAgent(llm=_StubChatModel(naive_llm_response))

    naive_result = naive_agent.analyze(events, incident_id="incident-integration-001")

    assert isinstance(naive_result, RCAResult)
    assert naive_result.incident_id == "incident-integration-001"
    assert len(naive_result.hypotheses) == 1
    assert naive_result.hypotheses[0].node_id == schema_event_ids[0]
    assert naive_result.hypotheses[0].supporting_node_ids == schema_event_ids
    assert naive_result.causal_path_node_ids == schema_event_ids
