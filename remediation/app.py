"""
ReliAI — Streamlit frontend (remediation/ layer)
====================================================
Final display layer: renders a schemas.IncidentReport. Owns no reasoning or
assembly logic itself -- that lives in evaluation/incident_pipeline.py
(run_incident_pipeline), which this app only calls (Mode 2) or reads the
saved output of (Mode 1).

Two modes, selected in the sidebar:
  * Load saved report (default): reads an IncidentReport from a JSON file
    via IncidentReport.model_validate_json().
  * Run live: runs run_incident_pipeline() on the spot against freshly
    generated injected-drift data, using a stub LLM by default (see
    remediation.pipeline_utils.get_llm) -- swap in a real LangChain-style
    client there when one exists; nothing here needs to change.

Both modes render the same report layout via render_report().

Run with: streamlit run remediation/app.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st
from pydantic import ValidationError

from reasoning.evidence_graph import EvidenceGraphBuilder
from remediation.generate_sample_report import DEFAULT_OUTPUT_PATH
from remediation.pipeline_utils import get_llm, run_sample_incident
from schemas import (
    EvidenceEdge,
    EvidenceEvent,
    EvidenceNode,
    IncidentReport,
    RCAResult,
    RemediationPlan,
    VerificationResult,
)

logger = logging.getLogger(__name__)

st.set_page_config(page_title="ReliAI — Incident Report", layout="wide")

# ---------------------------------------------------------------------------
# Shared matplotlib styling (dark theme, matching Streamlit's own default
# dark background/text colors -- a plain white matplotlib figure clashes
# with it, and st.pyplot doesn't inherit the app's theme automatically).
# ---------------------------------------------------------------------------
_DARK_BG = "#0e1117"
_TEXT_COLOR = "#fafafa"
_MUTED_TEXT_COLOR = "#c9ccd1"
_EDGE_COLOR = "#9aa0a8"
_NODE_COLOR = "#3498db"
_HIGHLIGHT_COLOR = "#e74c3c"


def _new_dark_figure(figsize: tuple[float, float]):
    """A matplotlib Figure/Axes pre-styled to match Streamlit's dark theme."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(_DARK_BG)
    ax.set_facecolor(_DARK_BG)
    return fig, ax


def _style_axes_for_dark_theme(ax) -> None:
    """Recolor ticks/labels/spines so they stay readable on _DARK_BG."""
    ax.tick_params(colors=_TEXT_COLOR)
    ax.xaxis.label.set_color(_TEXT_COLOR)
    ax.yaxis.label.set_color(_TEXT_COLOR)
    ax.title.set_color(_TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(_MUTED_TEXT_COLOR)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_report(path: str) -> IncidentReport:
    """Load a saved IncidentReport from a JSON file.

    Args:
        path: path to a JSON file previously written via
            IncidentReport.model_dump_json() (e.g. by
            remediation.generate_sample_report.generate_sample_report()).

    Returns:
        The parsed IncidentReport.

    Raises:
        FileNotFoundError: if path does not exist.
        pydantic.ValidationError: if the file's contents don't match the
            IncidentReport schema.
    """
    text = Path(path).read_text(encoding="utf-8")
    return IncidentReport.model_validate_json(text)


# ---------------------------------------------------------------------------
# Section 1: Incident summary
# ---------------------------------------------------------------------------

def render_summary(report: IncidentReport) -> None:
    """Incident_id, detected_at, ground_truth_label (if present)."""
    st.header("1. Incident Summary")
    cols = st.columns(3)
    cols[0].metric("Incident ID", report.incident_id)
    cols[1].metric("Detected At", report.detected_at.strftime("%Y-%m-%d %H:%M:%S UTC"))
    ground_truth = report.ground_truth_label.value if report.ground_truth_label else "—"
    cols[2].metric("Ground Truth Label", ground_truth)

    if report.ground_truth_label is not None:
        if report.root_cause_correct():
            st.success("Root cause correctly identified against ground truth.")
        else:
            st.error("Root cause NOT correctly identified against ground truth.")


# ---------------------------------------------------------------------------
# Section 2: Evidence timeline
# ---------------------------------------------------------------------------

def render_evidence_timeline(events: list[EvidenceEvent]) -> None:
    """EvidenceEvents in time order: feature, detection method, confidence,
    description."""
    st.header("2. Evidence Timeline")
    if not events:
        st.info("No evidence events.")
        return

    for event in sorted(events, key=lambda e: e.timestamp):
        feature = event.feature_name or "(pipeline-level)"
        with st.container(border=True):
            st.markdown(f"**{event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}** — `{feature}`")
            c1, c2, c3 = st.columns(3)
            c1.write(f"Detection method: `{event.detection_method}`")
            c2.write(f"Confidence: {event.confidence:.2f}")
            c3.write(f"Metric / threshold: {event.metric_value:.4f} / {event.threshold:.4f}")
            st.caption(event.description)


# ---------------------------------------------------------------------------
# Section 3: Evidence graph
# ---------------------------------------------------------------------------

def _reconstruct_edges(nodes: list[EvidenceNode]) -> list[EvidenceEdge]:
    """Recompute the EvidenceEdges between `nodes` for visualization.

    schemas.IncidentReport does not persist EvidenceEdges -- only
    evidence_nodes. Edges only ever exist as an intermediate
    networkx.DiGraph built during run_incident_pipeline() and are not part
    of the saved contract, so a loaded (Mode 1) report has nowhere to read
    them from. Edge confidence is a pure, deterministic function of two
    nodes' timestamps/confidences plus the shared configs.settings.GRAPH
    thresholds (see EvidenceGraphBuilder._build_edges), so recomputing them
    here from the report's own evidence_nodes reproduces exactly the edges
    the original pipeline run computed -- this is not invented data, just
    not persisted data.
    """
    return EvidenceGraphBuilder()._build_edges(nodes)


def render_evidence_graph(
    nodes: list[EvidenceNode], highlight_node_id: Optional[str] = None
) -> None:
    """Visual of nodes + reconstructed edges (networkx + matplotlib),
    showing causal structure. The top RCA hypothesis's node is highlighted
    in red when highlight_node_id is given."""
    st.header("3. Evidence Graph")
    if not nodes:
        st.info("No evidence nodes.")
        return

    edges = _reconstruct_edges(nodes)
    graph = nx.DiGraph()
    for node in nodes:
        graph.add_node(node.node_id, **node.model_dump())
    for edge in edges:
        graph.add_edge(edge.source_node_id, edge.target_node_id, **edge.model_dump())

    fig, ax = _new_dark_figure((8, 5))
    if graph.number_of_nodes() > 1:
        layout = nx.spring_layout(graph, seed=42, k=1.1)
    else:
        layout = {nodes[0].node_id: (0.0, 0.0)}

    node_colors = [
        _HIGHLIGHT_COLOR if node_id == highlight_node_id else _NODE_COLOR for node_id in graph.nodes
    ]
    # Long node_type labels (e.g. "feature_anomaly:feature_3") extend past
    # the node circle; without extra margin, labels near the layout's edge
    # get clipped by the figure boundary.
    ax.margins(0.2)
    nx.draw_networkx_nodes(
        graph,
        layout,
        node_color=node_colors,
        node_size=1000,
        edgecolors=_DARK_BG,
        linewidths=1.5,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        layout,
        labels={n: graph.nodes[n]["node_type"] for n in graph.nodes},
        font_size=7,
        font_color=_TEXT_COLOR,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        layout,
        arrows=True,
        arrowsize=16,
        arrowstyle="-|>",
        edge_color=_EDGE_COLOR,
        width=1.4,
        ax=ax,
        connectionstyle="arc3,rad=0.15",
        node_size=1000,
    )
    edge_labels = {(u, v): f"{d['confidence']:.2f}" for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(
        graph, layout, edge_labels=edge_labels, font_size=7, font_color=_MUTED_TEXT_COLOR, ax=ax
    )
    ax.set_axis_off()
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Node details"):
        for node in nodes:
            method = node.detection_method.value if node.detection_method else "—"
            st.write(
                f"`{node.node_id[:8]}` — **{node.node_type}** "
                f"(confidence={node.confidence:.2f}, method={method}, "
                f"{len(node.source_events)} source event(s))"
            )


# ---------------------------------------------------------------------------
# Section 4: Root cause analysis
# ---------------------------------------------------------------------------

def render_root_cause(rca_result: RCAResult) -> None:
    """Ranked hypotheses, top one highlighted, with explanations and
    confidence."""
    st.header("4. Root Cause Analysis")
    if not rca_result.hypotheses:
        st.info("No hypotheses.")
        return

    top = rca_result.hypotheses[0]
    st.subheader(f"Top hypothesis (rank {top.rank})")
    with st.container(border=True):
        st.markdown(f"**Node:** `{top.node_id}`")
        st.markdown(f"**Confidence:** {top.confidence:.2f}")
        failure_type = top.failure_type.value if top.failure_type else "unknown"
        st.markdown(f"**Failure type:** {failure_type}")
        st.write(top.explanation)

    remaining = rca_result.hypotheses[1:]
    if remaining:
        with st.expander(f"All {len(rca_result.hypotheses)} hypotheses"):
            for hyp in rca_result.hypotheses:
                failure_type = hyp.failure_type.value if hyp.failure_type else "unknown"
                st.markdown(
                    f"**Rank {hyp.rank}** — node `{hyp.node_id}` — "
                    f"confidence {hyp.confidence:.2f} — failure_type: {failure_type}"
                )
                st.caption(hyp.explanation)


# ---------------------------------------------------------------------------
# Section 5: Remediation plan
# ---------------------------------------------------------------------------

def render_remediation(plan: RemediationPlan) -> None:
    """Category, action_description, expected_outcome."""
    st.header("5. Remediation Plan")
    with st.container(border=True):
        st.markdown(f"**Category:** {plan.category.value}")
        st.markdown(f"**Action:** {plan.action_description}")
        st.markdown(f"**Expected outcome:** {plan.expected_outcome}")
        st.caption(f"Target root-cause node: `{plan.target_root_cause_node_id}`")


# ---------------------------------------------------------------------------
# Section 6: Verification result
# ---------------------------------------------------------------------------

def render_verification(result: VerificationResult) -> None:
    """value_before vs value_after, whether it improved, as a metric delta
    and a bar chart zoomed to the two values' range with value labels on
    top of each bar -- a full 0-1 y-axis makes small-but-real accuracy
    deltas (e.g. 0.498 -> 0.524) look like no change at all, since both
    bars end up nearly the same height."""
    st.header("6. Verification Result")
    delta = result.value_after - result.value_before
    cols = st.columns(3)
    cols[0].metric(f"{result.metric_name} (before)", f"{result.value_before:.4f}")
    cols[1].metric(f"{result.metric_name} (after)", f"{result.value_after:.4f}", f"{delta:+.4f}")
    cols[2].metric("Replay sample size", result.replay_sample_size)

    st.write("✅ Improved" if result.improved else "⚠️ Did not improve")

    values = [result.value_before, result.value_after]
    bar_color = "#2ecc71" if result.improved else "#e67e22"
    fig, ax = _new_dark_figure((4, 3.5))
    bars = ax.bar(["before", "after"], values, color=["#7f8c8d", bar_color], width=0.5)

    lo = max(0.0, min(values) - 0.05)
    hi = min(1.0, max(values) + 0.05)
    if hi - lo < 1e-6:
        lo, hi = 0.0, 1.0
    ax.set_ylim(lo, hi)
    ax.set_ylabel(result.metric_name)
    _style_axes_for_dark_theme(ax)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=_TEXT_COLOR,
        )
    st.pyplot(fig)
    plt.close(fig)

    if result.notes:
        st.caption(result.notes)


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def render_report(report: IncidentReport) -> None:
    """Render all six sections for one IncidentReport, in order."""
    render_summary(report)
    st.divider()
    render_evidence_timeline(report.evidence_events)
    st.divider()
    top_node_id = (
        report.rca_result.hypotheses[0].node_id if report.rca_result.hypotheses else None
    )
    render_evidence_graph(report.evidence_nodes, highlight_node_id=top_node_id)
    st.divider()
    render_root_cause(report.rca_result)
    st.divider()
    render_remediation(report.remediation_plan)
    st.divider()
    render_verification(report.verification_result)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("ReliAI — Incident Report")

    mode = st.sidebar.radio("Mode", ["Load saved report", "Run live"], index=0)

    if mode == "Load saved report":
        report_path = st.sidebar.text_input("Report JSON path", value=DEFAULT_OUTPUT_PATH)
        if not report_path:
            st.info("Enter a report path in the sidebar.")
            return
        try:
            report = load_report(report_path)
        except FileNotFoundError:
            st.error(
                f"No report found at `{report_path}`. Generate one with "
                "`python -m remediation.generate_sample_report`, or switch to "
                "**Run live** mode in the sidebar."
            )
            return
        except ValidationError as exc:
            st.error(f"`{report_path}` is not a valid IncidentReport:\n\n{exc}")
            return
        render_report(report)

    else:  # Run live
        st.sidebar.caption(
            "Runs run_incident_pipeline() on the spot against freshly "
            "generated injected-drift data, using a stub LLM (no network "
            "access or API key required)."
        )
        if st.sidebar.button("Run live pipeline", type="primary"):
            with st.spinner(
                "Running detection -> evidence graph -> RCA -> remediation -> verification..."
            ):
                try:
                    report = run_sample_incident(llm=get_llm())
                except Exception:
                    logger.exception("Live pipeline run failed")
                    st.error("Pipeline run failed -- see application logs for details.")
                    return
            st.session_state["live_report"] = report

        report = st.session_state.get("live_report")
        if report is None:
            st.info("Click **Run live pipeline** in the sidebar to generate a report.")
            return
        render_report(report)


if __name__ == "__main__":
    main()
