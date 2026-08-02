"""
Tests for reasoning/evidence_graph.py — the NetworkX evidence graph builder.

Scenarios use hand-picked timestamps/confidences so every clustering and
edge decision can be checked against a value computed by hand, not just
"some graph came out the other end."
"""

from datetime import datetime, timedelta, timezone

import pytest

from configs.settings import GraphSettings
from reasoning.evidence_graph import EvidenceGraphBuilder
from schemas import DetectionMethod, EvidenceEvent

BASE = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)


def _event(minutes_after_base: float, feature_name: str, confidence: float) -> EvidenceEvent:
    """Build a minimal, valid EvidenceEvent for test scenarios."""
    return EvidenceEvent(
        timestamp=BASE + timedelta(minutes=minutes_after_base),
        detection_method=DetectionMethod.PSI,
        feature_name=feature_name,
        metric_value=0.5,
        threshold=0.2,
        confidence=confidence,
        description=f"synthetic event for {feature_name} at t+{minutes_after_base}m",
    )


def test_same_feature_events_within_window_cluster_into_one_node():
    """Two events, same feature, well within the time window -> one node."""
    e1 = _event(0, "customer_age", 0.9)
    e2 = _event(10, "customer_age", 0.85)

    builder = EvidenceGraphBuilder()
    builder.build([e1, e2])
    nodes = builder.get_nodes()

    assert len(nodes) == 1
    assert set(nodes[0].source_events) == {e1.event_id, e2.event_id}


def test_causal_chain_links_earlier_node_to_later_node():
    """
    customer_age drifts (t=0, t=10min) then model_prediction shifts
    (t=20min) shortly after -- that should produce a directed edge from the
    customer_age node to the model_prediction node. A later, unrelated
    spike at t=150min is far outside GRAPH.time_window_minutes (60min
    default) from either and must NOT be linked to anything.

    Expected nodes (default GRAPH settings: window=60m, min_conf=0.3):
      A = customer_age cluster  (mean timestamp t=5min,  mean confidence 0.875)
      B = model_prediction      (t=20min, confidence 0.8)
      C = unrelated_metric      (t=150min, confidence 0.95)

    Expected edges:
      A -> B: delta=15m, decay=1-15/60=0.75, conf=0.75*(0.875+0.8)/2=0.628125 -> kept
      A -> C, B -> C: delta >> 60m -> dropped (outside time window)
    """
    e1 = _event(0, "customer_age", 0.9)
    e2 = _event(10, "customer_age", 0.85)
    e3 = _event(20, "model_prediction", 0.8)
    e4 = _event(150, "unrelated_metric", 0.95)

    builder = EvidenceGraphBuilder()
    graph = builder.build([e1, e2, e3, e4])
    nodes = builder.get_nodes()
    assert len(nodes) == 3

    node_a = next(n for n in nodes if n.node_type == "feature_anomaly:customer_age")
    node_b = next(n for n in nodes if n.node_type == "feature_anomaly:model_prediction")
    node_c = next(n for n in nodes if n.node_type == "feature_anomaly:unrelated_metric")

    # Correct causal/time direction: earlier node -> later node, not reversed.
    assert graph.has_edge(node_a.node_id, node_b.node_id)
    assert not graph.has_edge(node_b.node_id, node_a.node_id)
    assert graph.edges[node_a.node_id, node_b.node_id]["confidence"] == pytest.approx(
        0.628125, abs=1e-6
    )

    # Nodes outside the time window must NOT be linked, regardless of confidence.
    assert not graph.has_edge(node_a.node_id, node_c.node_id)
    assert not graph.has_edge(node_c.node_id, node_a.node_id)
    assert not graph.has_edge(node_b.node_id, node_c.node_id)
    assert not graph.has_edge(node_c.node_id, node_b.node_id)


def test_edge_dropped_when_below_min_edge_confidence():
    """
    Two low-confidence nodes, within the time window, produce a real but
    weak edge confidence that falls below min_edge_confidence -> the edge
    must be dropped even though both nodes themselves still exist.
    """
    settings = GraphSettings(time_window_minutes=60, min_edge_confidence=0.5)
    e1 = _event(0, "feature_x", 0.3)
    e2 = _event(50, "feature_y", 0.3)
    # delta=50m, decay=1-50/60=0.1667, edge_conf=0.1667*0.3=0.05 < 0.5

    builder = EvidenceGraphBuilder(graph_settings=settings)
    graph = builder.build([e1, e2])
    nodes = builder.get_nodes()

    assert len(nodes) == 2
    assert graph.number_of_edges() == 0
    assert builder.get_edges() == []
