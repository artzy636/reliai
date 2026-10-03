"""
ReliAI — Real-data cascade benchmark (evaluation/ layer)
============================================================
The single-fault real-data benchmark (real_data_benchmark.py) compares two
static dataframes, so cause and symptoms appear at the same instant and the
evidence graph has no temporal structure to use -- structured and naive RCA
tie there by construction. This benchmark gives the graph real ordering to
work with, using only things a production monitor really has:

* The "current" data is a *stream* of time-ordered batches (windows).
* A fault begins at window ``k``: it is injected into every window >= k.
* Every detector runs on every window against the clean reference.
* An event's timestamp is its ONSET: the first window where that
  (detector, feature) fires and keeps firing for ``PERSISTENCE``
  consecutive windows. Persistence suppresses one-off noise.
* Labels arrive late (``LABEL_DELAY`` windows), as they do in production.
  Anything that needs the true label -- the rolling-accuracy detector and
  the KS test on the target column -- therefore fires ``LABEL_DELAY``
  windows after the data fault that caused it. That lag is what makes
  "data fault -> accuracy drop" a real, ordered cascade rather than a
  tie.

Nothing here reads ground truth to decide timing: onsets come only from
the detectors' own firing pattern. Ground truth is stamped afterwards, by
the same `matches` rules the single-fault benchmark uses, only for scoring.

Honest limits, kept visible rather than tuned away:
* Only faults that really propagate (corrupted values, feature drift when
  the model relies on the column) produce downstream events; the rest
  stay isolated, and structure can't help there.
* An edge is kept only above ``GRAPH.min_edge_confidence``; a low-confidence
  root (e.g. the isolation forest) may fail to link to its symptom. Use
  ``--structure-report`` to see exactly which cases got edges before
  spending any LLM calls, and ``--min-edge-confidence`` to test sensitivity.

Usage:
    python -m evaluation.cascade_benchmark --structure-report
    python -m evaluation.cascade_benchmark --real-llm --repeats 3 --output benchmark_results/cascade.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import numpy as np
import pandas as pd

from configs.settings import GRAPH, GraphSettings
from detection.data_agent import DataAgent
from detection.fault_injection import (
    inject_corrupted_values,
    inject_duplicate_rows,
    inject_feature_drift,
    inject_missing_values,
    inject_schema_mismatch,
)
from evaluation.benchmark_runner import AggregatedBenchmarkSummary, BenchmarkSummary, IncidentScore
from evaluation.real_data_incident import (
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
    _split_reference_current,
    load_adult_data,
)
from reasoning.evidence_graph import EvidenceGraphBuilder
from reasoning.rca_agent import NaiveRCAAgent, RCAAgent
from schemas import DetectionMethod, EvidenceEvent, FailureType, root_cause_is_correct, root_cause_rank

logger = logging.getLogger(__name__)

N_WINDOWS = 10
WINDOW_MINUTES = 10          # simulated minutes per window; adjacent onsets sit well inside GRAPH.time_window_minutes
LABEL_DELAY = 2              # windows before true labels are available
PERSISTENCE = 2              # consecutive windows a detector must fire to count
_BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)

_KS = DetectionMethod.KS_TEST.value
_ROLLING = DetectionMethod.ROLLING_ACCURACY.value


@dataclass(frozen=True)
class CascadeCase:
    name: str
    fault_type: FailureType
    onset_window: int
    inject_window: Callable[[pd.DataFrame, int], pd.DataFrame]  # (window_df, window_index) -> faulted window_df
    matches: Callable[[EvidenceEvent], bool]


# ---------------------------------------------------------------------------
# Fault injectors, applied per window
# ---------------------------------------------------------------------------

def _shift_label_prior(df: pd.DataFrame, target_rate: float, seed: int) -> pd.DataFrame:
    """Raise the positive-label rate by dropping negatives (no resampling,
    so no artificial duplicate rows)."""
    positives = df[df[TARGET_COLUMN] == 1]
    negatives = df[df[TARGET_COLUMN] == 0]
    keep = int(round(len(positives) * (1 - target_rate) / target_rate))
    keep = min(keep, len(negatives))
    return (
        pd.concat([positives, negatives.sample(n=keep, random_state=seed)])
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def _cases() -> list[CascadeCase]:
    def ks_on(feature):
        return lambda e: e.feature_name == feature and e.detection_method == _KS

    def method(m, feature=None):
        return lambda e: e.detection_method == m.value and e.feature_name == feature

    corrupt = lambda cols, frac: (
        lambda df, j: _corrupt(df, cols, frac, 100 + j)
    )
    return [
        CascadeCase("drift_hours_per_week", FailureType.FEATURE_DRIFT, 4,
                    lambda df, j: inject_feature_drift(df, "hours-per-week", shift_amount=15.0, random_seed=j)[0],
                    ks_on("hours-per-week")),
        CascadeCase("drift_age", FailureType.FEATURE_DRIFT, 3,
                    lambda df, j: inject_feature_drift(df, "age", shift_amount=15.0, random_seed=j)[0],
                    ks_on("age")),
        CascadeCase("drift_education_num", FailureType.FEATURE_DRIFT, 5,
                    lambda df, j: inject_feature_drift(df, "education-num", shift_amount=4.0, random_seed=j)[0],
                    ks_on("education-num")),
        CascadeCase("corrupted_age_education_num", FailureType.CORRUPTED_VALUES, 4,
                    corrupt(["age", "education-num"], 0.10),
                    method(DetectionMethod.ISOLATION_FOREST)),
        CascadeCase("corrupted_age_hours", FailureType.CORRUPTED_VALUES, 3,
                    corrupt(["age", "hours-per-week"], 0.15),
                    method(DetectionMethod.ISOLATION_FOREST)),
        CascadeCase("corrupted_education_hours_gain", FailureType.CORRUPTED_VALUES, 5,
                    corrupt(["education-num", "hours-per-week", "capital-gain"], 0.20),
                    method(DetectionMethod.ISOLATION_FOREST)),
        CascadeCase("missing_capital_loss", FailureType.MISSING_VALUES, 4,
                    lambda df, j: inject_missing_values(df, "capital-loss", fraction=0.3, random_seed=j)[0],
                    method(DetectionMethod.MISSING_VALUE_RATE, "capital-loss")),
        CascadeCase("missing_hours_per_week", FailureType.MISSING_VALUES, 3,
                    lambda df, j: inject_missing_values(df, "hours-per-week", fraction=0.3, random_seed=j)[0],
                    method(DetectionMethod.MISSING_VALUE_RATE, "hours-per-week")),
        CascadeCase("duplicates_15pct", FailureType.DUPLICATES, 4,
                    lambda df, j: inject_duplicate_rows(df, fraction=0.15, random_seed=j)[0],
                    method(DetectionMethod.DUPLICATE_ROW_RATE)),
        CascadeCase("duplicates_30pct", FailureType.DUPLICATES, 5,
                    lambda df, j: inject_duplicate_rows(df, fraction=0.30, random_seed=j)[0],
                    method(DetectionMethod.DUPLICATE_ROW_RATE)),
        CascadeCase("schema_drop_fnlwgt", FailureType.SCHEMA_MISMATCH, 4,
                    lambda df, j: inject_schema_mismatch(df, "fnlwgt", mode="drop")[0],
                    method(DetectionMethod.SCHEMA_CHECK, "fnlwgt")),
        CascadeCase("schema_drop_education_num", FailureType.SCHEMA_MISMATCH, 3,
                    lambda df, j: inject_schema_mismatch(df, "education-num", mode="drop")[0],
                    method(DetectionMethod.SCHEMA_CHECK, "education-num")),
        CascadeCase("label_shift_to_50pct", FailureType.LABEL_SHIFT, 4,
                    lambda df, j: _shift_label_prior(df, 0.5, 200 + j),
                    method(DetectionMethod.ROLLING_ACCURACY)),
        CascadeCase("label_shift_to_45pct", FailureType.LABEL_SHIFT, 5,
                    lambda df, j: _shift_label_prior(df, 0.45, 300 + j),
                    method(DetectionMethod.ROLLING_ACCURACY)),
    ]


def _corrupt(df: pd.DataFrame, columns: list[str], fraction: float, seed: int) -> pd.DataFrame:
    """Corrupt the SAME rows across all `columns` together (a lone corrupted
    column is a weak signal for the isolation forest on this dataset)."""
    out = df
    for column in columns:
        out = inject_corrupted_values(
            out, column, fraction=fraction, corruption_multiplier=20.0, random_seed=seed
        )[0]
    return out


# ---------------------------------------------------------------------------
# Windowed detection -> onset events
# ---------------------------------------------------------------------------

def _is_label_dependent(event: EvidenceEvent) -> bool:
    return event.detection_method == _ROLLING or (
        event.detection_method == _KS and event.feature_name == TARGET_COLUMN
    )


def collapse_to_onset_events(
    by_key: dict[tuple, dict[int, EvidenceEvent]], persistence: int = PERSISTENCE
) -> list[EvidenceEvent]:
    """Collapse each (detector, feature)'s firing pattern -- {time: event} --
    into at most one event stamped at its ONSET: the first time t where it
    fires at t, t+1, ... t+persistence-1 with no gap. A pattern that never
    persists (one-off noise) yields no event. Confidence and metric are the
    mean over every firing from onset onward."""
    collapsed: list[EvidenceEvent] = []
    for by_t in by_key.values():
        onset = next(
            (t for t in sorted(by_t) if all((t + i) in by_t for i in range(persistence))), None
        )
        if onset is None:
            continue
        fired = [by_t[t] for t in sorted(by_t) if t >= onset]
        representative = fired[0]
        collapsed.append(
            representative.model_copy(
                update={
                    "timestamp": _BASE_TIME + timedelta(minutes=WINDOW_MINUTES * onset),
                    "confidence": float(np.mean([e.confidence for e in fired])),
                    "metric_value": float(np.mean([e.metric_value for e in fired])),
                    "description": f"{representative.description} [onset window {onset}, persisted {len(fired)} window(s)]",
                }
            )
        )
    return collapsed


def build_stream_events(
    case: CascadeCase,
    data_path: str = DEFAULT_DATA_PATH,
    random_seed: int = 42,
    n_windows: int = N_WINDOWS,
) -> list[EvidenceEvent]:
    """Run every detector on every window and collapse each persistent
    (detector, feature) firing pattern into ONE event stamped at its onset.
    Ground truth is stamped afterwards, for scoring only."""
    df = load_adult_data(data_path)
    reference_df, current_df = _split_reference_current(df, random_seed)
    bounds = np.linspace(0, len(current_df), n_windows + 1).astype(int)
    windows = [current_df.iloc[bounds[i]:bounds[i + 1]] for i in range(n_windows)]
    windows = [
        case.inject_window(w.reset_index(drop=True), j) if j >= case.onset_window else w.reset_index(drop=True)
        for j, w in enumerate(windows)
    ]

    by_key: dict[tuple, dict[int, EvidenceEvent]] = defaultdict(dict)
    for j, window in enumerate(windows):
        for event in DataAgent().investigate(
            reference_df, window, target_column=TARGET_COLUMN, snapshot_timestamps=True
        ):
            t = j + (LABEL_DELAY if _is_label_dependent(event) else 0)
            by_key[(event.detection_method, event.feature_name)][t] = event

    collapsed = collapse_to_onset_events(by_key)

    stamped = []
    for event in collapsed:
        if case.matches(event):
            event = event.model_copy(update={"ground_truth_label": case.fault_type})
        stamped.append(event)
    return stamped


def _onset_window(event: EvidenceEvent) -> int:
    return int((event.timestamp - _BASE_TIME).total_seconds() // 60 // WINDOW_MINUTES)


# ---------------------------------------------------------------------------
# Structure report (no LLM)
# ---------------------------------------------------------------------------

def structure_report(
    events_by_case: dict[str, list[EvidenceEvent]], min_edge_confidence: float
) -> dict:
    """For each case: did the true root link to downstream symptoms, and did
    anything wrongly link INTO it? Tells you where graph structure can
    possibly matter, before any LLM call."""
    settings = GraphSettings(time_window_minutes=GRAPH.time_window_minutes, min_edge_confidence=min_edge_confidence)
    rows = {}
    for name, events in events_by_case.items():
        builder = EvidenceGraphBuilder(settings)
        graph = builder.build(events)
        nodes = builder.get_nodes()
        true_ids = {e.event_id for e in events if e.ground_truth_label is not None}
        true_nodes = {n.node_id for n in nodes if true_ids & set(n.source_events)}
        outgoing = sum(graph.out_degree(n) for n in true_nodes)
        incoming = sum(graph.in_degree(n) for n in true_nodes)
        label = {n.node_id: f"{n.node_type}[{getattr(n.detection_method, 'value', n.detection_method)}]" for n in nodes}
        rows[name] = {
            "edges": [f"{label[u]} -> {label[v]}" for u, v in graph.edges()],
            "n_events": len(events),
            "n_nodes": graph.number_of_nodes(),
            "n_edges": graph.number_of_edges(),
            "root_detected": bool(true_ids),
            "root_has_downstream": outgoing > 0,
            "root_has_incoming": incoming > 0,
        }
    return rows


def print_structure_report(report: dict, events_by_case: dict[str, list[EvidenceEvent]], verbose: bool) -> None:
    print(f"\n{'case':34s} {'events':>6s} {'edges':>5s} {'root found':>10s} {'→symptoms':>9s} {'←into root':>10s}")
    for name, r in report.items():
        print(
            f"{name:34s} {r['n_events']:6d} {r['n_edges']:5d} {str(r['root_detected']):>10s} "
            f"{str(r['root_has_downstream']):>9s} {str(r['root_has_incoming']):>10s}"
        )
        if verbose:
            for e in sorted(events_by_case[name], key=lambda e: (e.timestamp, -e.confidence)):
                tag = "  <== TRUE ROOT" if e.ground_truth_label else ""
                print(f"      w{_onset_window(e):<2d} conf={e.confidence:.2f} {e.detection_method:18s} {str(e.feature_name):15s}{tag}")
            for edge in r["edges"]:
                print(f"      edge: {edge}")
    informative = sum(1 for r in report.values() if r["root_has_downstream"])
    print(f"\nCases where the true root links to downstream symptoms: {informative}/{len(report)}\n")


# ---------------------------------------------------------------------------
# Scoring (same agents/metrics as the single-fault benchmark)
# ---------------------------------------------------------------------------

def run_case(case: CascadeCase, events: list[EvidenceEvent], llm, min_edge_confidence: float) -> Optional[IncidentScore]:
    if not any(e.ground_truth_label is not None for e in events):
        logger.warning("Case %s: injected fault never produced a persistent matching event -- skipped", case.name)
        return None
    builder = EvidenceGraphBuilder(
        GraphSettings(time_window_minutes=GRAPH.time_window_minutes, min_edge_confidence=min_edge_confidence)
    )
    graph = builder.build(events)
    nodes = builder.get_nodes()
    structured = RCAAgent(llm=llm).analyze(graph, incident_id=case.name)
    naive = NaiveRCAAgent(llm=llm).analyze(events, incident_id=case.name)
    return IncidentScore(
        incident_id=case.name,
        ground_truth_label=case.fault_type,
        structured_result=structured,
        structured_correct=root_cause_is_correct(structured, case.fault_type, events, nodes),
        naive_result=naive,
        naive_correct=root_cause_is_correct(naive, case.fault_type, events),
        structured_rank=root_cause_rank(structured, case.fault_type, events, nodes),
        naive_rank=root_cause_rank(naive, case.fault_type, events),
    )


def build_all_events(
    data_path: str, random_seed: int, cache_path: Optional[str] = None
) -> dict[str, list[EvidenceEvent]]:
    """Windowed detection for every case (about 5 minutes -- every detector
    runs on every window). With cache_path, results are saved/reused so
    re-running the scoring, or sweeping --min-edge-confidence, is instant.
    Delete the cache whenever detectors or injectors change."""
    if cache_path and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            raw = json.load(f)
        if raw.get("random_seed") == random_seed:
            logger.info("Using cached windowed events from %s", cache_path)
            return {
                name: [EvidenceEvent.model_validate(e) for e in events]
                for name, events in raw["events"].items()
            }
    out = {}
    for case in _cases():
        logger.info("Running windowed detection: %s", case.name)
        out[case.name] = build_stream_events(case, data_path, random_seed)
    if cache_path:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "random_seed": random_seed,
                    "events": {n: [e.model_dump(mode="json") for e in evs] for n, evs in out.items()},
                },
                f,
            )
    return out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--real-llm", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output", default=None)
    parser.add_argument("--structure-report", action="store_true", help="detection + graph only; no LLM")
    parser.add_argument("--verbose", action="store_true", help="with --structure-report, list every event")
    parser.add_argument("--cache", default=None, help="save/reuse windowed detection results at this path")
    parser.add_argument("--min-edge-confidence", type=float, default=GRAPH.min_edge_confidence)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    events_by_case = build_all_events(args.data_path, args.random_seed, args.cache)

    report = structure_report(events_by_case, args.min_edge_confidence)
    print_structure_report(report, events_by_case, args.verbose)
    if args.structure_report:
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump({"structure": report}, f, indent=2)
        return

    from remediation.pipeline_utils import get_llm

    llm = get_llm(use_stub=not args.real_llm)
    cases = {c.name: c for c in _cases()}
    summaries = []
    for repeat in range(1, args.repeats + 1):
        logger.info("=== Repeat %d/%d ===", repeat, args.repeats)
        scores = [
            s for s in (run_case(cases[n], ev, llm, args.min_edge_confidence) for n, ev in events_by_case.items())
            if s is not None
        ]
        summaries.append(BenchmarkSummary(scores=scores))

    if args.repeats <= 1:
        summaries[0].print_table()
        result = summaries[0].as_dict()
    else:
        aggregated = AggregatedBenchmarkSummary(summaries=summaries)
        aggregated.print_table()
        result = aggregated.as_dict()
    result["structure"] = report
    result["settings"] = {
        "n_windows": N_WINDOWS, "label_delay": LABEL_DELAY, "persistence": PERSISTENCE,
        "min_edge_confidence": args.min_edge_confidence,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
