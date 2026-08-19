"""
ReliAI — Evidence Graph Builder (reasoning/ layer)
====================================================
Consumes: schemas.EvidenceEvent (produced by Detection's Data Agent)
Produces: a NetworkX DiGraph of schemas.EvidenceNode / schemas.EvidenceEdge,
consumed downstream by the RCA Agent (reasoning/rca_agent.py).

schemas.EvidenceNode's docstring is explicit that clustering logic "lives
in Reasoning's code, not here" — so the two design decisions below are owned
by this module, not the shared contract:

  * Clustering: events are grouped into a node if they share a
    ``feature_name`` and are connected by a chain of consecutive timestamp
    gaps no larger than ``configs.settings.GRAPH.time_window_minutes``
    (events with ``feature_name=None`` are clustered among themselves as
    "pipeline-level" evidence). A node's timestamp is the mean timestamp of
    its member events; its confidence is the mean confidence.
  * Edges: for every pair of nodes where one strictly precedes the other in
    time, an edge is proposed with a confidence that linearly decays to 0
    over ``GRAPH.time_window_minutes`` and is scaled by the two nodes'
    confidence. Only edges strictly above ``GRAPH.min_edge_confidence``
    are kept. Nothing here reads ``EvidenceEvent.ground_truth_label`` — per
    CONTRACT.md rule 4, that field must never leak into reasoning logic.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from itertools import groupby
from typing import Optional

import networkx as nx

from configs.settings import GRAPH, GraphSettings
from schemas import EvidenceEdge, EvidenceEvent, EvidenceNode

logger = logging.getLogger(__name__)


class EvidenceGraphBuilder:
    """Builds a NetworkX evidence graph from a list of EvidenceEvents.

    Usage:
        builder = EvidenceGraphBuilder()
        graph = builder.build(events)
        nodes = builder.get_nodes()
        edges = builder.get_edges()
    """

    def __init__(self, graph_settings: GraphSettings = GRAPH) -> None:
        """Initialize the builder.

        Args:
            graph_settings: clustering/edge thresholds. Defaults to the
                shared configs.settings.GRAPH singleton — pass an override
                only for testing, never to hardcode a threshold in caller
                code.
        """
        self._settings = graph_settings
        self._nodes: list[EvidenceNode] = []
        self._edges: list[EvidenceEdge] = []
        self._graph: nx.DiGraph = nx.DiGraph()

    def build(self, events: list[EvidenceEvent]) -> nx.DiGraph:
        """Build the evidence graph from a list of EvidenceEvents.

        Args:
            events: raw detection findings from the Data Agent.

        Returns:
            A NetworkX DiGraph whose nodes/edges are keyed by
            EvidenceNode.node_id and carry the EvidenceNode/EvidenceEdge
            fields as attributes.
        """
        logger.info("Building evidence graph from %d event(s)", len(events))
        self._nodes = self._cluster_events(events)
        logger.info("Clustered events into %d node(s)", len(self._nodes))
        self._edges = self._build_edges(self._nodes)
        logger.info(
            "Kept %d edge(s) above min_edge_confidence=%.2f",
            len(self._edges),
            self._settings.min_edge_confidence,
        )
        self._graph = self._assemble_graph(self._nodes, self._edges)
        return self._graph

    def get_nodes(self) -> list[EvidenceNode]:
        """Return the EvidenceNodes produced by the most recent build()."""
        return list(self._nodes)

    def get_edges(self) -> list[EvidenceEdge]:
        """Return the EvidenceEdges produced by the most recent build()."""
        return list(self._edges)

    def _cluster_events(self, events: list[EvidenceEvent]) -> list[EvidenceNode]:
        """Group events into EvidenceNodes by feature_name + time proximity.

        Events sharing a feature_name are clustered using consecutive-gap
        chaining: sort by timestamp, start a new cluster whenever the gap
        to the previous event in the (sorted) group exceeds
        ``time_window_minutes``. This is equivalent to full pairwise
        connected-components clustering on a sorted 1-D timeline, but is
        O(n log n) instead of O(n^2).
        """
        window = timedelta(minutes=self._settings.time_window_minutes)
        sorted_events = sorted(events, key=lambda e: (e.feature_name or "", e.timestamp))

        nodes: list[EvidenceNode] = []
        for feature_name, group_iter in groupby(sorted_events, key=lambda e: e.feature_name):
            cluster: list[EvidenceEvent] = []
            for event in group_iter:
                if cluster and (event.timestamp - cluster[-1].timestamp) > window:
                    nodes.append(self._make_node(feature_name, cluster))
                    cluster = []
                cluster.append(event)
            if cluster:
                nodes.append(self._make_node(feature_name, cluster))
        return nodes

    @staticmethod
    def _make_node(feature_name: Optional[str], cluster: list[EvidenceEvent]) -> EvidenceNode:
        """Build a single EvidenceNode from a cluster of related events."""
        base_timestamp = cluster[0].timestamp
        mean_offset = sum(
            (event.timestamp - base_timestamp for event in cluster), timedelta()
        ) / len(cluster)
        mean_timestamp = base_timestamp + mean_offset
        mean_confidence = sum(event.confidence for event in cluster) / len(cluster)
        node_type = f"feature_anomaly:{feature_name}" if feature_name else "pipeline_level_anomaly"

        node = EvidenceNode(
            node_type=node_type,
            source_events=[event.event_id for event in cluster],
            timestamp=mean_timestamp,
            confidence=mean_confidence,
        )
        logger.debug(
            "Node %s (%s) built from %d event(s): %s",
            node.node_id,
            node_type,
            len(cluster),
            [event.event_id for event in cluster],
        )
        return node

    def _build_edges(self, nodes: list[EvidenceNode]) -> list[EvidenceEdge]:
        """Propose precedence edges between nodes, filtered by confidence.

        For every pair of nodes where one strictly precedes the other in
        time (within ``time_window_minutes``), a "preceded" edge confidence
        is computed as a linear time-decay factor scaled by the mean of the
        two nodes' own confidence, then kept only if it is strictly above
        ``min_edge_confidence``.
        """
        window_minutes = self._settings.time_window_minutes
        min_confidence = self._settings.min_edge_confidence
        ordered = sorted(nodes, key=lambda n: n.timestamp)

        edges: list[EvidenceEdge] = []
        for i, source in enumerate(ordered):
            for target in ordered[i + 1:]:
                delta_minutes = (target.timestamp - source.timestamp).total_seconds() / 60.0
                if delta_minutes <= 0:
                    # Same or earlier timestamp: no defined precedence direction.
                    continue
                if delta_minutes > window_minutes:
                    # `ordered` is time-sorted, so every later target is at
                    # least this far away — nothing closer remains.
                    break

                decay = 1.0 - (delta_minutes / window_minutes)
                confidence = decay * (source.confidence + target.confidence) / 2.0
                if confidence > min_confidence:
                    edge = EvidenceEdge(
                        source_node_id=source.node_id,
                        target_node_id=target.node_id,
                        relationship="preceded",
                        confidence=confidence,
                        evidence=(
                            f"{source.node_type} ({source.timestamp.isoformat()}) preceded "
                            f"{target.node_type} ({target.timestamp.isoformat()}), "
                            f"{delta_minutes:.1f} min apart"
                        ),
                    )
                    edges.append(edge)
                    logger.debug(
                        "Edge %s -> %s: confidence=%.3f (%.1f min apart)",
                        source.node_id,
                        target.node_id,
                        confidence,
                        delta_minutes,
                    )
        return edges

    @staticmethod
    def _assemble_graph(nodes: list[EvidenceNode], edges: list[EvidenceEdge]) -> nx.DiGraph:
        """Assemble a NetworkX DiGraph from EvidenceNodes/EvidenceEdges."""
        graph = nx.DiGraph()
        for node in nodes:
            graph.add_node(node.node_id, **node.model_dump())
        for edge in edges:
            graph.add_edge(edge.source_node_id, edge.target_node_id, **edge.model_dump())
        return graph
