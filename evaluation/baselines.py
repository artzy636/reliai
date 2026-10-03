"""
ReliAI — Trivial baselines (evaluation/ layer)
============================================================
Controls that need no LLM and no graph. Any RCA agent worth its cost has to
beat these; if it only ties them, the reasoning is adding nothing on that
benchmark.

``confidence_sort``: rank detections by their own detector confidence,
highest first. Ties are counted pessimistically (every other event with an
equal-or-higher confidence is ranked ahead of the true one), so a tie never
flatters the baseline.

``earliest_first``: rank detections by timestamp, earliest first. Same
pessimistic tie rule (every non-true event at or before the true event's
time counts ahead of it). This is the shortcut a graph with time-ordered
edges could be exploiting: if it scores well on a benchmark, that benchmark
cannot tell causal reasoning apart from "pick the first thing that fired".

    python -m evaluation.baselines
"""

from __future__ import annotations

import logging
from typing import Optional

from evaluation import cascade_benchmark as cb
from evaluation.benchmark_runner import load_builtin_incidents
from evaluation.real_data_benchmark import DEFAULT_DATA_PATH, _cases, _run_detection
from schemas import EvidenceEvent


def confidence_sort_rank(events: list[EvidenceEvent]) -> Optional[int]:
    """1-based rank of the injected fault's event when events are sorted by
    confidence (pessimistic on ties). None if no event is the true fault."""
    true = [e for e in events if e.ground_truth_label is not None]
    if not true:
        return None
    best = max(e.confidence for e in true)
    return 1 + sum(1 for e in events if e.ground_truth_label is None and e.confidence >= best)


def earliest_first_rank(events: list[EvidenceEvent]) -> Optional[int]:
    """1-based rank of the injected fault's event when events are sorted by
    timestamp, earliest first (pessimistic on ties). None if no event is the
    true fault."""
    true = [e for e in events if e.ground_truth_label is not None]
    if not true:
        return None
    first = min(e.timestamp for e in true)
    return 1 + sum(1 for e in events if e.ground_truth_label is None and e.timestamp <= first)


def summarize(ranks: list[Optional[int]]) -> dict:
    n = len(ranks)
    return {
        "n": n,
        "hit1": sum(1 for r in ranks if r == 1) / n,
        "hit3": sum(1 for r in ranks if r is not None and r <= 3) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
    }


def baselines_report(data_path: str = DEFAULT_DATA_PATH, seed: int = 42, cascade_cache: Optional[str] = None) -> dict:
    """{baseline name: {benchmark name: summarize(...)}} for every trivial
    baseline on every benchmark."""
    benchmarks = {
        "synthetic (12 incidents)": [i.events for i in load_builtin_incidents()],
        "real single-fault (6 cases)": [_run_detection(c, data_path, seed, None).events for c in _cases(seed)],
        "real streaming cascade (14 cases)": list(cb.build_all_events(data_path, seed, cascade_cache).values()),
    }
    rankers = {"confidence-sort": confidence_sort_rank, "earliest-first": earliest_first_rank}
    return {
        name: {bench: summarize([rank(events) for events in event_lists]) for bench, event_lists in benchmarks.items()}
        for name, rank in rankers.items()
    }


def confidence_sort_report(data_path: str = DEFAULT_DATA_PATH, seed: int = 42, cascade_cache: Optional[str] = None) -> dict:
    """Confidence-sort results only (kept for callers of the original API)."""
    return baselines_report(data_path, seed, cascade_cache)["confidence-sort"]


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    report = baselines_report(cascade_cache="benchmark_results/cascade_events_cache.json")
    for baseline, benches in report.items():
        print(f"\n{baseline + ' baseline (no LLM, no graph)':46s} {'Hit@1':>6s} {'Hit@3':>6s} {'MRR':>6s}")
        for name, r in benches.items():
            print(f"{name:46s} {r['hit1']:6.0%} {r['hit3']:6.0%} {r['mrr']:6.3f}")
    print()


if __name__ == "__main__":
    main()
