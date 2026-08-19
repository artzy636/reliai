"""
ReliAI — RCA Agent (reasoning/ layer)
=======================================
Consumes: the NetworkX DiGraph produced by reasoning.evidence_graph.EvidenceGraphBuilder
Produces: schemas.RCAResult, consumed downstream by Remediation's Remediation Agent.

Two agents live here, both emitting the same schemas.RCAResult so they can
be scored identically (see README.md's research question):

  * RCAAgent: the actual research contribution. A LangGraph pipeline where
    graph traversal (finding candidate causal chains) is deterministic
    Python — no LLM in the loop — and the LLM is invoked exactly once, only
    to rank the deterministically-found candidates and write an explanation
    for each. The LLM is shown structural graph facts only (node types,
    timestamps, confidences, edge relationships) and is never shown raw
    EvidenceEvent telemetry (metric_value, threshold, description) or
    ground_truth_label — see CONTRACT.md rule 4.
  * NaiveRCAAgent: the control group. No graph, no clustering, no
    statistical pre-filtering — raw EvidenceEvents (minus
    ground_truth_label, per CONTRACT.md rule 4) go straight into one LLM
    call that guesses the root cause. This is what the structured approach
    has to beat to justify the evidence-graph design.

LLM wiring is intentionally left to the caller: both agents accept any
object exposing a LangChain-style ``.invoke(prompt) -> AIMessage``, or a
plain ``Callable[[str], str]``, via constructor injection. Neither agent
constructs a concrete provider client or reads an API key itself — no
credential handling lives in this file. Pass an ``llm=None`` (the default)
and either agent will raise a clear ``ValueError`` the moment it actually
needs to call one, rather than silently skipping the LLM step.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional, TypedDict, Union

import networkx as nx
from langgraph.graph import END, START, StateGraph

from configs.settings import RCA, RCASettings
from schemas import EvidenceEdge, EvidenceEvent, EvidenceNode, RCAResult, RootCauseHypothesis

logger = logging.getLogger(__name__)

LLMLike = Union[Callable[[str], str], Any]  # Any = duck-typed LangChain BaseChatModel


# ---------------------------------------------------------------------------
# LLM plumbing shared by both agents
# ---------------------------------------------------------------------------

def _call_llm(llm: LLMLike, prompt: str) -> str:
    """Invoke a pluggable LLM and normalize its response to plain text.

    Accepts either a plain ``Callable[[str], str]`` or any LangChain-style
    object exposing ``.invoke(prompt)`` (e.g. a ``BaseChatModel``, whose
    ``.invoke`` returns a message object with a ``.content`` attribute).
    """
    if hasattr(llm, "invoke"):
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    return llm(prompt)


def _extract_json_array(raw: str) -> str:
    """Strip an optional ```/```json markdown fence some chat models wrap
    JSON output in, so json.loads() sees a clean array."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


# ---------------------------------------------------------------------------
# RCAAgent: deterministic graph traversal + LLM ranking/explanation
# ---------------------------------------------------------------------------

@dataclass
class _CausalCandidate:
    """One deterministically-found candidate causal chain: a node with no
    incoming edges (nothing in the graph precedes it), together with its
    strongest downstream chain of effects. Internal to this module — not
    part of the shared contract.
    """

    root_node_id: str
    path_node_ids: list[str]
    path_nodes: list[EvidenceNode]
    path_edges: list[EvidenceEdge]
    propagation_score: float  # ranking-only heuristic; see _build_candidates

    @property
    def confidence(self) -> float:
        """The root node's own detection confidence.

        Used directly as the RootCauseHypothesis confidence: it is a single
        field already present on the graph, so it stays fully traceable
        with no blending or re-derivation.
        """
        return self.path_nodes[0].confidence


class _RCAState(TypedDict):
    graph: nx.DiGraph
    incident_id: str
    candidates: list[_CausalCandidate]
    hypotheses: list[RootCauseHypothesis]


def _node_from_graph(graph: nx.DiGraph, node_id: str) -> EvidenceNode:
    """Reconstruct an EvidenceNode from the attributes EvidenceGraphBuilder
    stored on the graph node (it wrote `**node.model_dump()`, so every
    field, including node_id, is already present)."""
    return EvidenceNode(**graph.nodes[node_id])


def _edge_from_graph(graph: nx.DiGraph, source_id: str, target_id: str) -> EvidenceEdge:
    """Reconstruct an EvidenceEdge from the attributes EvidenceGraphBuilder
    stored on the graph edge."""
    return EvidenceEdge(**graph.edges[source_id, target_id])


def _build_candidates(graph: nx.DiGraph) -> list[_CausalCandidate]:
    """Deterministically find candidate root causes and their causal chains.

    A candidate root is any node with in-degree 0 (the graph's edges only
    ever run earlier-node -> later-node, per evidence_graph.py, so this
    graph is guaranteed acyclic and such nodes always exist when the graph
    is non-empty). For each candidate root, the chain is the longest path
    (by summed edge confidence) through its descendants -- provably the
    globally-longest path within that subgraph, since the root is the
    subgraph's only source node and confidences are strictly positive, so
    any path not starting at the root could be extended backwards.

    Candidates are ranked by a `propagation_score` that rewards roots with
    stronger/longer downstream effects: `root_confidence * (1 + sum(edge
    confidences along the chain))`. This is a ranking heuristic ONLY --
    unbounded above 1.0 -- and never used as a hypothesis's `confidence`
    field (see _CausalCandidate.confidence).
    """
    candidates: list[_CausalCandidate] = []
    root_ids = [node_id for node_id in graph.nodes if graph.in_degree(node_id) == 0]

    for root_id in root_ids:
        descendant_ids = nx.descendants(graph, root_id)
        subgraph = graph.subgraph({root_id, *descendant_ids})
        path_node_ids = (
            nx.dag_longest_path(subgraph, weight="confidence")
            if subgraph.number_of_edges() > 0
            else [root_id]
        )
        path_nodes = [_node_from_graph(graph, node_id) for node_id in path_node_ids]
        path_edges = [
            _edge_from_graph(graph, path_node_ids[i], path_node_ids[i + 1])
            for i in range(len(path_node_ids) - 1)
        ]
        propagation_score = path_nodes[0].confidence * (
            1.0 + sum(edge.confidence for edge in path_edges)
        )
        candidates.append(
            _CausalCandidate(root_id, path_node_ids, path_nodes, path_edges, propagation_score)
        )

    candidates.sort(key=lambda c: c.propagation_score, reverse=True)
    logger.debug(
        "Built %d candidate(s): %s",
        len(candidates),
        [(c.root_node_id, round(c.propagation_score, 3)) for c in candidates],
    )
    return candidates


def _template_explanation(candidate: _CausalCandidate) -> str:
    """Deterministic, fully-traceable explanation built only from graph
    fields -- used when no LLM is needed to validate, or as a fallback if
    the LLM's response fails validation (see RCAAgent._rank_and_explain)."""
    chain = " -> ".join(
        f"{node.node_type} (confidence {node.confidence:.2f})" for node in candidate.path_nodes
    )
    if not candidate.path_edges:
        return f"Standalone evidence node with no linked downstream effects: {chain}."
    edge_bits = "; ".join(
        f'"{edge.relationship}" (confidence {edge.confidence:.2f})' for edge in candidate.path_edges
    )
    return f"Evidence chain: {chain}, linked by: {edge_bits}."


class RCAAgent:
    """Root Cause Analysis agent: deterministic graph traversal wrapped in
    a LangGraph pipeline, with the LLM invoked only to rank candidates and
    write their explanations.

    Usage:
        agent = RCAAgent(llm=my_chat_model)
        result = agent.analyze(graph, incident_id="incident-001")
    """

    def __init__(self, llm: Optional[LLMLike] = None, rca_settings: RCASettings = RCA) -> None:
        """Initialize the agent.

        Args:
            llm: a LangChain-style chat model (anything with
                ``.invoke(prompt) -> AIMessage``) or a plain
                ``Callable[[str], str]``. No default provider is wired here
                by design -- pass ``None`` (the default) and the agent
                still works on graphs with no candidates (e.g. an empty
                graph), but raises ``ValueError`` the moment it needs to
                actually call an LLM.
            rca_settings: hypothesis-count cap etc. Defaults to the shared
                configs.settings.RCA singleton.
        """
        self._llm = llm
        self._settings = rca_settings
        self._app = self._build_app()

    def analyze(self, graph: nx.DiGraph, incident_id: str) -> RCAResult:
        """Run RCA over an evidence graph (e.g. EvidenceGraphBuilder.build()'s
        return value) and return a ranked RCAResult.

        Args:
            graph: a NetworkX DiGraph whose nodes/edges carry EvidenceNode /
                EvidenceEdge fields as attributes.
            incident_id: identifier to stamp onto the resulting RCAResult.

        Returns:
            An RCAResult with hypotheses capped at RCA.max_hypotheses.
        """
        logger.info(
            "Running structured RCA for incident %s over a graph with %d node(s), %d edge(s)",
            incident_id,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        final_state = self._app.invoke(
            {"graph": graph, "incident_id": incident_id, "candidates": [], "hypotheses": []}
        )
        hypotheses = final_state["hypotheses"]
        causal_path = hypotheses[0].supporting_node_ids if hypotheses else []
        logger.info(
            "Structured RCA produced %d hypothesis/es for incident %s", len(hypotheses), incident_id
        )
        return RCAResult(
            incident_id=incident_id, hypotheses=hypotheses, causal_path_node_ids=causal_path
        )

    def _build_app(self):
        """Wire the LangGraph pipeline: find candidates (deterministic) ->
        rank + explain (LLM) -> finalize (cap at max_hypotheses)."""
        workflow = StateGraph(_RCAState)
        workflow.add_node("find_candidates", self._find_candidate_paths)
        workflow.add_node("rank_and_explain", self._rank_and_explain)
        workflow.add_node("finalize", self._finalize)
        workflow.add_edge(START, "find_candidates")
        workflow.add_edge("find_candidates", "rank_and_explain")
        workflow.add_edge("rank_and_explain", "finalize")
        workflow.add_edge("finalize", END)
        return workflow.compile()

    @staticmethod
    def _find_candidate_paths(state: _RCAState) -> dict:
        """LangGraph node: deterministic graph traversal, no LLM involved."""
        candidates = _build_candidates(state["graph"])
        logger.info("Found %d candidate causal chain(s)", len(candidates))
        return {"candidates": candidates}

    def _rank_and_explain(self, state: _RCAState) -> dict:
        """LangGraph node: the one point in this pipeline that calls an LLM.

        The LLM only ranks the already-found candidates and writes their
        explanations; it cannot introduce node_ids or evidence we didn't
        give it (validated in _parse_llm_response). If it isn't configured,
        or its response fails validation, that's a hard error / a logged
        fallback to a deterministic template -- never a silent skip that
        would look like a real LLM ranking happened.
        """
        candidates = state["candidates"]
        if not candidates:
            logger.warning("No causal-path candidates found; returning no hypotheses")
            return {"hypotheses": []}

        if self._llm is None:
            raise ValueError(
                "RCAAgent has no llm configured. Pass a LangChain-style chat model "
                "or a Callable[[str], str] via RCAAgent(llm=...)."
            )

        prompt = self._build_prompt(candidates)
        raw_response = _call_llm(self._llm, prompt)
        ordered = self._parse_llm_response(raw_response, candidates)
        if ordered is None:
            logger.warning(
                "LLM response failed validation; falling back to deterministic "
                "structural ranking with template explanations"
            )
            ordered = [(candidate, _template_explanation(candidate)) for candidate in candidates]

        hypotheses = [
            RootCauseHypothesis(
                rank=rank,
                node_id=candidate.root_node_id,
                explanation=explanation,
                confidence=candidate.confidence,
                supporting_node_ids=candidate.path_node_ids,
            )
            for rank, (candidate, explanation) in enumerate(ordered, start=1)
        ]
        return {"hypotheses": hypotheses}

    def _finalize(self, state: _RCAState) -> dict:
        """LangGraph node: enforce the RCA.max_hypotheses cap."""
        capped = state["hypotheses"][: self._settings.max_hypotheses]
        return {"hypotheses": capped}

    def _build_prompt(self, candidates: list[_CausalCandidate]) -> str:
        """Build a prompt containing ONLY structural graph facts -- node
        types, timestamps, node/edge confidences, edge relationships and
        their (already-structural) evidence strings. No raw EvidenceEvent
        telemetry (metric_value, threshold, description) or
        ground_truth_label ever reaches this text.
        """
        lines = [
            "You are the root-cause ranking step of ReliAI's evidence-graph RCA agent.",
            "Below are candidate causal chains, already extracted deterministically "
            "from an evidence graph's structure: node types, timestamps, confidences, "
            "and edge relationships. You have no access to raw telemetry, feature "
            "values, or dataframes -- do not invent any fact beyond what is listed.",
            "",
            f"Rank up to {self._settings.max_hypotheses} of the candidates below from "
            "most to least likely root cause. For each, write a one- or two-sentence "
            "explanation that explicitly references the node types and edge "
            "relationships given for that candidate.",
            "",
            "Candidates:",
        ]
        for candidate in candidates:
            chain_desc = " -> ".join(
                f"{node.node_type} (t={node.timestamp.isoformat()}, confidence={node.confidence:.2f})"
                for node in candidate.path_nodes
            )
            lines.append(f"- root_node_id: {candidate.root_node_id}")
            lines.append(f"  causal_chain: {chain_desc}")
            for edge in candidate.path_edges:
                lines.append(
                    f"  edge: {edge.source_node_id} -> {edge.target_node_id} "
                    f'relationship="{edge.relationship}" confidence={edge.confidence:.2f} '
                    f'evidence="{edge.evidence}"'
                )
        lines += [
            "",
            "Respond with ONLY a JSON array (no prose, no markdown fence), ordered "
            "most-to-least-likely, where each element is exactly: "
            '{"node_id": "<root_node_id from above>", "explanation": "<grounded explanation>"}.',
        ]
        return "\n".join(lines)

    @staticmethod
    def _parse_llm_response(
        raw: str, candidates: list[_CausalCandidate]
    ) -> Optional[list[tuple[_CausalCandidate, str]]]:
        """Parse + validate the LLM's ranking. Returns None (triggering the
        deterministic fallback) if the response isn't valid JSON, isn't a
        non-empty list, or references a node_id we didn't offer it --
        that last check is what stops the LLM from hallucinating evidence.
        """
        candidates_by_root = {c.root_node_id: c for c in candidates}
        try:
            items = json.loads(_extract_json_array(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning("Could not parse LLM response as JSON: %r", raw[:200])
            return None
        if not isinstance(items, list) or not items:
            logger.warning("LLM response was not a non-empty JSON array: %r", raw[:200])
            return None

        ordered: list[tuple[_CausalCandidate, str]] = []
        seen_ids: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                return None
            node_id = item.get("node_id")
            explanation = item.get("explanation")
            if not explanation or node_id not in candidates_by_root or node_id in seen_ids:
                logger.warning("LLM referenced an unknown/duplicate node_id %r", node_id)
                return None
            seen_ids.add(node_id)
            ordered.append((candidates_by_root[node_id], explanation))
        return ordered


# ---------------------------------------------------------------------------
# NaiveRCAAgent: the control group -- raw events straight into one LLM call
# ---------------------------------------------------------------------------

class NaiveRCAAgent:
    """Deliberately naive baseline: raw EvidenceEvents go straight into one
    LLM call with no graph, no clustering, no statistical pre-filtering.
    This is the control group for the project's core comparison (see
    README.md's research question) -- it emits the same schemas.RCAResult
    shape as RCAAgent so both can be scored by the same evaluation code.

    Per CONTRACT.md rule 4, EvidenceEvent.ground_truth_label is never
    serialized into the prompt, even though this path is deliberately
    unsophisticated -- a baseline that could see the answer key would
    invalidate the comparison, not just look bad.

    There is no evidence graph here, so RootCauseHypothesis.node_id /
    supporting_node_ids reference EvidenceEvent.event_id instead of an
    EvidenceNode.node_id. This is a deliberate, documented deviation from
    RootCauseHypothesis's docstring (written with RCAAgent's graph in
    mind), kept because both fields are plain `str`/`list[str]` with no
    contract-level cross-reference to enforce, and it's what lets this
    schema-free baseline still emit a scoreable RCAResult.
    """

    def __init__(self, llm: Optional[LLMLike] = None, rca_settings: RCASettings = RCA) -> None:
        """Initialize the baseline.

        Args:
            llm: same pluggable interface as RCAAgent -- no default
                provider wired here.
            rca_settings: hypothesis-count cap. Defaults to the shared
                configs.settings.RCA singleton.
        """
        self._llm = llm
        self._settings = rca_settings

    def analyze(self, events: list[EvidenceEvent], incident_id: str) -> RCAResult:
        """Guess the root cause directly from raw telemetry, no graph.

        Args:
            events: raw EvidenceEvents for the incident.
            incident_id: identifier to stamp onto the resulting RCAResult.

        Returns:
            An RCAResult with hypotheses capped at RCA.max_hypotheses.
        """
        if not events:
            logger.warning("NaiveRCAAgent given no events for incident %s", incident_id)
            return RCAResult(incident_id=incident_id, hypotheses=[], causal_path_node_ids=[])

        if self._llm is None:
            raise ValueError(
                "NaiveRCAAgent has no llm configured. Pass a LangChain-style chat model "
                "or a Callable[[str], str] via NaiveRCAAgent(llm=...)."
            )

        logger.info(
            "Running naive baseline RCA for incident %s over %d raw event(s)",
            incident_id,
            len(events),
        )
        prompt = self._build_prompt(events)
        raw_response = _call_llm(self._llm, prompt)
        hypotheses = self._parse_response(raw_response, events)[: self._settings.max_hypotheses]
        causal_path = hypotheses[0].supporting_node_ids if hypotheses else []
        logger.info(
            "Naive baseline RCA produced %d hypothesis/es for incident %s",
            len(hypotheses),
            incident_id,
        )
        return RCAResult(
            incident_id=incident_id, hypotheses=hypotheses, causal_path_node_ids=causal_path
        )

    def _build_prompt(self, events: list[EvidenceEvent]) -> str:
        """Serialize raw EvidenceEvents into a prompt. Deliberately dumps
        every statistical field (feature_name, detection_method,
        metric_value, threshold, confidence, description) with no
        structuring or pre-filtering -- that lack of structure is exactly
        what this baseline is testing. ground_truth_label is the one field
        never included (CONTRACT.md rule 4).
        """
        lines = [
            "You are a root-cause analysis assistant for an ML pipeline. Below is "
            "raw anomaly-detection telemetry with no pre-processing or structuring. "
            "Guess the most likely root cause(s).",
            "",
            "Telemetry:",
        ]
        for event in events:
            lines.append(
                f"- event_id={event.event_id} feature_name={event.feature_name} "
                f"detection_method={event.detection_method} metric_value={event.metric_value} "
                f"threshold={event.threshold} confidence={event.confidence} "
                f'timestamp={event.timestamp.isoformat()} description="{event.description}"'
            )
        lines += [
            "",
            f"Respond with ONLY a JSON array (no prose, no markdown fence) of up to "
            f"{self._settings.max_hypotheses} hypotheses, ordered most-to-least likely, "
            "where each element is exactly: "
            '{"event_ids": ["<event_id>", ...], "explanation": "<your guess>", '
            '"confidence": <float 0-1>}.',
        ]
        return "\n".join(lines)

    @staticmethod
    def _parse_response(raw: str, events: list[EvidenceEvent]) -> list[RootCauseHypothesis]:
        """Parse the LLM's guesses into RootCauseHypothesis objects,
        dropping any item that references an unknown event_id or is
        otherwise malformed rather than raising -- a malformed LLM
        response should degrade to fewer hypotheses, not crash the run.
        """
        valid_event_ids = {event.event_id for event in events}
        try:
            items = json.loads(_extract_json_array(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning("NaiveRCAAgent: could not parse LLM response as JSON: %r", raw[:200])
            return []
        if not isinstance(items, list):
            logger.warning("NaiveRCAAgent: LLM response was not a JSON array: %r", raw[:200])
            return []

        hypotheses: list[RootCauseHypothesis] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            event_ids = [eid for eid in item.get("event_ids", []) if eid in valid_event_ids]
            explanation = item.get("explanation")
            if not event_ids or not explanation:
                continue
            try:
                confidence = float(item.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = min(max(confidence, 0.0), 1.0)
            hypotheses.append(
                RootCauseHypothesis(
                    rank=len(hypotheses) + 1,
                    node_id=event_ids[0],
                    explanation=explanation,
                    confidence=confidence,
                    supporting_node_ids=event_ids,
                )
            )
        return hypotheses
