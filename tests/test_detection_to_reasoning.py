"""
Integration test: detection/data_agent.py -> reasoning/evidence_graph.py ->
reasoning/rca_agent.py.

Runs the full real pipeline end-to-end with no hand-built EvidenceEvents and
no mocking of the graph step:

  1. A clean reference DataFrame (several numeric columns, one non-numeric).
  2. detection.fault_injection.inject_feature_drift() shifts one known column
     to build current_df.
  3. DataAgent.investigate() runs the real KS-test detector and returns real
     EvidenceEvents.
  4. Those events go straight into EvidenceGraphBuilder.build() -- no
     reshaping, no fixtures.
  5. The resulting graph goes straight into RCAAgent.analyze(), with only the
     LLM call stubbed (no network access, no API key) -- the same stub
     pattern used in tests/test_reasoning_integration.py.

test_data_agent.py and test_evidence_graph.py each cover one layer in
isolation with hand-built inputs; neither would catch a field mismatch
between DataAgent's real EvidenceEvent output and what EvidenceGraphBuilder
actually consumes. This test is what proves the two layers connect.
"""

import numpy as np
import pandas as pd

from detection.data_agent import DataAgent
from detection.fault_injection import inject_feature_drift
from reasoning.evidence_graph import EvidenceGraphBuilder
from reasoning.rca_agent import RCAAgent
from schemas import RCAResult


class _FakeAIMessage:
    """Mimics langchain_core's AIMessage just enough for _call_llm's
    duck-typed `.content` extraction."""

    def __init__(self, content: str) -> None:
        self.content = content


class _StubChatModel:
    """LangChain-style stub: exposes `.invoke(prompt) -> AIMessage` and
    returns a pre-programmed response. No network access, no API key."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> _FakeAIMessage:
        self.prompts.append(prompt)
        return _FakeAIMessage(self.response)


def _reference_df() -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "age": np.random.normal(30, 5, 500),
        "income": np.random.normal(50000, 5000, 500),
        "transaction_count": np.random.normal(12, 3, 500),
        "region": ["north", "south", "east", "west"] * 125,
    })


def test_data_agent_output_feeds_evidence_graph_and_rca_end_to_end() -> None:
    """Full real pipeline, no hand-built EvidenceEvents anywhere. The
    top-ranked hypothesis must trace back to the "income" node -- the only
    feature actually drifted; "age", "transaction_count", and the
    non-numeric "region" column must contribute no evidence at all.
    """
    reference_df = _reference_df()
    current_df, injected_fault = inject_feature_drift(
        reference_df, "income", shift_amount=25000, random_seed=42
    )

    # --- Step 1: real detection, no hand-built EvidenceEvents. -----------
    agent = DataAgent()
    events = agent.investigate(reference_df, current_df)

    assert len(events) == 1
    drifted_event = events[0]
    assert drifted_event.feature_name == "income"
    assert 0.0 < drifted_event.confidence <= 1.0

    # --- Step 2: real evidence graph, straight from DataAgent's output. --
    graph = EvidenceGraphBuilder().build(events)

    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 0
    income_node_id = next(iter(graph.nodes))
    assert graph.nodes[income_node_id]["node_type"] == "feature_anomaly:income"
    assert set(graph.nodes[income_node_id]["source_events"]) == {
        event.event_id for event in events
    }

    # --- Step 3: real RCA agent, only the LLM call stubbed. --------------
    stub_response = (
        '[{"node_id": "%s", "explanation": '
        '"feature_anomaly:income is the only evidence node in the graph, '
        'so it is the root cause."}]' % income_node_id
    )
    rca_agent = RCAAgent(llm=_StubChatModel(stub_response))
    result = rca_agent.analyze(graph, incident_id="incident-detection-to-reasoning-001")

    assert isinstance(result, RCAResult)
    assert result.hypotheses
    top = result.hypotheses[0]
    assert top.rank == 1
    assert top.node_id == income_node_id
    assert graph.nodes[top.node_id]["node_type"] == "feature_anomaly:income"
    assert top.supporting_node_ids == [income_node_id]
    assert result.causal_path_node_ids == [income_node_id]
