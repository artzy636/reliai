"""
Look inside one real-data case: what the detectors emitted, what the graph
looks like, and what each agent actually answered (with explanations).

Use this to tell a *reasoning* failure (the LLM saw the true event and
ranked it badly) from a *benchmark-definition* failure (the LLM picked
something that is arguably also right but isn't counted).

    python -m evaluation.inspect_case label_shift_income --real-llm
    python -m evaluation.inspect_case corrupted_values_age_and_education_num --real-llm
    python -m evaluation.inspect_case --list
"""

from __future__ import annotations

import argparse
import logging
import textwrap

from configs.settings import GRAPH
from evaluation.real_data_benchmark import DEFAULT_DATA_PATH, _cases, _run_detection
from reasoning.evidence_graph import EvidenceGraphBuilder
from reasoning.rca_agent import NaiveRCAAgent, RCAAgent
from schemas import resolve_hypothesis_root_event_ids


def _describe_hypotheses(title, result, events, nodes, case):
    print(f"\n--- {title} ---")
    by_id = {e.event_id: e for e in events}
    for h in sorted(result.hypotheses, key=lambda h: h.rank):
        resolved = resolve_hypothesis_root_event_ids(h, events, nodes)
        members = [by_id[i] for i in resolved if i in by_id]
        correct = any(m.ground_truth_label == case.fault_type for m in members)
        what = ", ".join(f"{m.detection_method}:{m.feature_name}(conf {m.confidence:.2f})" for m in members) or "?"
        if title.startswith("NAIVE"):
            # Naive hypotheses may list several events; scoring credits only the
            # first (node_id). Show the rest so a scoring penalty is visible.
            extra = [by_id[i] for i in h.supporting_node_ids if i in by_id and i != h.node_id]
            if extra:
                what += "  | also lists: " + ", ".join(f"{m.detection_method}:{m.feature_name}" + ("[TRUE]" if m.ground_truth_label else "") for m in extra)
        print(f"#{h.rank} {'[CORRECT]' if correct else '         '} {what}")
        print(textwrap.indent(textwrap.fill(h.explanation, 100), "      "))


def _inspect_cascade(args) -> None:
    from configs.settings import GraphSettings
    from evaluation import cascade_benchmark as cb

    cases = {c.name: c for c in cb._cases()}
    if args.list or not args.case:
        print("\n".join(cases))
        return
    case = cases[args.case]
    events = cb.build_all_events(DEFAULT_DATA_PATH, args.seed, args.cache)[case.name]

    from remediation.pipeline_utils import get_llm
    llm = get_llm(use_stub=not args.real_llm)

    print(f"\n=== {case.name} (cascade, min_edge_confidence={args.min_edge_confidence}) ===\nEvents by onset:")
    for e in sorted(events, key=lambda e: (e.timestamp, -e.confidence)):
        mark = "TRUE FAULT" if e.ground_truth_label else ""
        print(f"  w{cb._onset_window(e):<2d} {e.confidence:.2f}  {e.detection_method:18s} {str(e.feature_name):15s} {mark}")
    builder = EvidenceGraphBuilder(
        GraphSettings(time_window_minutes=GRAPH.time_window_minutes, min_edge_confidence=args.min_edge_confidence)
    )
    graph = builder.build(events)
    nodes = builder.get_nodes()
    print(f"\nGraph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    structured = RCAAgent(llm=llm).analyze(graph, incident_id=case.name)
    naive = NaiveRCAAgent(llm=llm).analyze(events, incident_id=case.name)
    _describe_hypotheses("STRUCTURED agent", structured, events, nodes, case)
    _describe_hypotheses("NAIVE agent", naive, events, None, case)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", nargs="?")
    parser.add_argument("--real-llm", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="inspect a case from the streaming cascade benchmark (uses its saved windowed events)",
    )
    parser.add_argument("--cache", default="benchmark_results/cascade_events_cache.json")
    parser.add_argument("--min-edge-confidence", type=float, default=GRAPH.min_edge_confidence)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.cascade:
        _inspect_cascade(args)
        return

    cases = {c.name: c for c in _cases(args.seed)}
    if args.list or not args.case:
        print("\n".join(cases))
        return
    case = cases[args.case]

    from remediation.pipeline_utils import get_llm
    llm = get_llm(use_stub=not args.real_llm)

    detection = _run_detection(case, DEFAULT_DATA_PATH, args.seed, None)
    events = detection.events
    print(f"\n=== {case.name} (seed {args.seed}) ===\nEvents by confidence:")
    for e in sorted(events, key=lambda e: -e.confidence):
        mark = "TRUE FAULT" if e.ground_truth_label else ""
        print(f"  {e.confidence:.2f}  {e.detection_method:18s} {str(e.feature_name):15s} {e.description[:70]}  {mark}")

    builder = EvidenceGraphBuilder()
    graph = builder.build(events)
    nodes = builder.get_nodes()
    print(f"\nGraph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    structured = RCAAgent(llm=llm).analyze(graph, incident_id=case.name)
    naive = NaiveRCAAgent(llm=llm).analyze(events, incident_id=case.name)
    _describe_hypotheses("STRUCTURED agent", structured, events, nodes, case)
    _describe_hypotheses("NAIVE agent", naive, events, None, case)


if __name__ == "__main__":
    main()
