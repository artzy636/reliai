"""
ReliAI — Detector calibration check (evaluation/ layer)
============================================================
Measures the detection layer on its own, with no LLM involved:

1. **Does the true fault's event stand out by raw confidence?**  For each
   single-fault real-data case, across many random splits, where does the
   injected fault's event rank among every event DataAgent produced? (This
   is exactly what the naive "highest-confidence event wins" baseline
   uses, so it measures calibration quality directly.)
2. **False-positive behaviour on a clean control.**  No fault injected:
   how many events does each detector emit on an unmodified reference/
   current split, and how confident are they?

It can also reconstruct the *legacy* confidence formulas (pre-calibration)
from the same events, so the before/after comparison is reproducible
rather than a claim:

    python -m evaluation.detector_calibration
    python -m evaluation.detector_calibration --seeds 20
    python -m evaluation.detector_calibration --output benchmark_results/detector_calibration.json

Legacy formulas (see detection/confidence.py for why they were replaced):
KS ``1 - p/alpha``; every other detector ``(metric - thr) / (1 - thr)``;
isolation forest additionally used a fixed +0.05 flagging margin.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from statistics import mean
from typing import Optional

from configs.settings import DETECTION
from detection.data_agent import DataAgent
from evaluation.real_data_benchmark import (
    DEFAULT_DATA_PATH,
    TARGET_COLUMN,
    _cases,
    _isolate_ground_truth,
    _run_detection,
    _split_reference_current,
    load_adult_data,
)
from schemas import DetectionMethod, EvidenceEvent

logger = logging.getLogger(__name__)

_P_RE = re.compile(r"p=([0-9.eE+-]+)")
_DROP_RE = re.compile(r"drop=([0-9.]+)")


def legacy_confidence(event: EvidenceEvent) -> Optional[float]:
    """Confidence the pre-calibration detectors would have assigned to
    `event`, or None if the legacy detector would not have fired at all
    (only possible for the isolation forest, whose margin was fixed).
    """
    method = event.detection_method
    thr = event.threshold
    if method == DetectionMethod.KS_TEST.value:
        p = float(_P_RE.search(event.description).group(1))
        return max(0.0, min(1.0, 1 - p / DETECTION.ks_test_pvalue))
    if method == DetectionMethod.SCHEMA_CHECK.value:
        return event.confidence  # binary; unchanged
    if method == DetectionMethod.ROLLING_ACCURACY.value:
        drop = float(_DROP_RE.search(event.description).group(1))
        return max(0.0, min(1.0, (drop - thr) / (1 - thr)))
    if method == DetectionMethod.ISOLATION_FOREST.value:
        legacy_thr = DETECTION.isolation_forest_contamination + DETECTION.isolation_forest_anomaly_rate_margin
        if event.metric_value <= legacy_thr:
            return None
        return max(0.0, min(1.0, (event.metric_value - legacy_thr) / (1 - legacy_thr)))
    # missing values, duplicate rows
    return max(0.0, min(1.0, (event.metric_value - thr) / (1 - thr)))


def _rank_of_true_event(events: list[EvidenceEvent], conf_of) -> Optional[int]:
    """1-based rank of the injected fault's event by confidence, counting
    every tie as ahead of it (pessimistic: an arbitrary tiebreak shouldn't
    flatter the detector). None if the fault produced no (legacy) event."""
    scored = [(conf_of(e), e) for e in events]
    scored = [(c, e) for c, e in scored if c is not None]
    true = [c for c, e in scored if e.ground_truth_label is not None]
    if not true:
        return None
    best = max(true)
    others_ahead = sum(1 for c, e in scored if e.ground_truth_label is None and c >= best)
    return 1 + others_ahead


def _summarize(ranks: list[Optional[int]], confs: list[Optional[float]]) -> dict:
    n = len(ranks)
    found = [r for r in ranks if r is not None]
    return {
        "detected_rate": len(found) / n,
        "top1_by_confidence": sum(1 for r in found if r == 1) / n,
        "mrr": sum(1 / r for r in found) / n,
        "mean_true_confidence": mean([c for c in confs if c is not None]) if any(c is not None for c in confs) else None,
    }


def calibration_report(
    n_seeds: int = 10, data_path: str = DEFAULT_DATA_PATH, compare_legacy: bool = True
) -> dict:
    per_case: dict[str, dict] = {}
    for seed in range(n_seeds):
        for case in _cases(seed):
            result = _run_detection(case, data_path, seed, None)
            entry = per_case.setdefault(case.name, {"new": ([], []), "legacy": ([], [])})
            true_events = [e for e in result.events if e.ground_truth_label is not None]
            entry["new"][0].append(_rank_of_true_event(result.events, lambda e: e.confidence))
            entry["new"][1].append(max((e.confidence for e in true_events), default=None))
            if compare_legacy:
                legacy_scores = [legacy_confidence(e) for e in true_events]
                legacy_scores = [c for c in legacy_scores if c is not None]
                entry["legacy"][0].append(_rank_of_true_event(result.events, legacy_confidence))
                entry["legacy"][1].append(max(legacy_scores, default=None))

    cases_out = {}
    for name, entry in per_case.items():
        cases_out[name] = {"new": _summarize(*entry["new"])}
        if compare_legacy:
            cases_out[name]["legacy"] = _summarize(*entry["legacy"])

    def overall(key: str) -> dict:
        ranks = [r for e in per_case.values() for r in e[key][0]]
        confs = [c for e in per_case.values() for c in e[key][1]]
        return _summarize(ranks, confs)

    report = {"n_seeds": n_seeds, "cases": cases_out, "overall": {"new": overall("new")}}
    if compare_legacy:
        report["overall"]["legacy"] = overall("legacy")
    report["clean_control"] = clean_control_report(n_seeds, data_path, compare_legacy)
    return report


def clean_control_report(n_seeds: int, data_path: str = DEFAULT_DATA_PATH, compare_legacy: bool = True) -> dict:
    """No fault injected. Per detector: mean events per split and mean
    confidence of the (all-false-positive) events it did emit."""
    df = load_adult_data(data_path)
    counts: dict[str, list[int]] = defaultdict(lambda: [0] * n_seeds)
    confs: dict[str, list[float]] = defaultdict(list)
    legacy_counts: dict[str, list[int]] = defaultdict(lambda: [0] * n_seeds)
    legacy_confs: dict[str, list[float]] = defaultdict(list)
    for seed in range(n_seeds):
        reference_df, current_df = _split_reference_current(df, seed)
        for event in DataAgent().investigate(reference_df, current_df, target_column=TARGET_COLUMN):
            counts[event.detection_method][seed] += 1
            confs[event.detection_method].append(event.confidence)
            if compare_legacy:
                legacy = legacy_confidence(event)
                if legacy is not None:
                    legacy_counts[event.detection_method][seed] += 1
                    legacy_confs[event.detection_method].append(legacy)

    def table(c, f):
        return {
            m: {"events_per_split": mean(c[m]), "mean_confidence": mean(f[m]) if f[m] else None}
            for m in sorted(c)
        }

    out = {"new": table(counts, confs)}
    if compare_legacy:
        out["legacy"] = table(legacy_counts, legacy_confs)
    return out


def print_report(report: dict) -> None:
    has_legacy = "legacy" in report["overall"]
    print(f"\nDetector calibration over {report['n_seeds']} random splits (no LLM)\n")
    header = f"{'case':44s} {'top1':>11s} {'MRR':>11s} {'true conf':>11s}"
    print(header)
    print(f"{'':44s} {'old → new':>11s} {'old → new':>11s} {'old → new':>11s}")

    def fmt(entry, key, pct=False):
        new = entry["new"][key]
        new_s = "n/a" if new is None else (f"{new:.0%}" if pct else f"{new:.2f}")
        if not has_legacy:
            return new_s
        old = entry["legacy"][key]
        old_s = "n/a" if old is None else (f"{old:.0%}" if pct else f"{old:.2f}")
        return f"{old_s}→{new_s}"

    for name, entry in {**report["cases"], "OVERALL": report["overall"]}.items():
        print(
            f"{name:44s} {fmt(entry, 'top1_by_confidence', True):>11s} "
            f"{fmt(entry, 'mrr'):>11s} {fmt(entry, 'mean_true_confidence'):>11s}"
        )

    print("\nClean control (no fault): false-positive events per split / mean confidence")
    cc = report["clean_control"]
    for method in sorted(cc["new"]):
        new = cc["new"][method]
        line = f"  {method:20s} new: {new['events_per_split']:.2f} evts, conf {new['mean_confidence']:.2f}"
        if has_legacy and method in cc["legacy"]:
            old = cc["legacy"][method]
            line += f"   legacy: {old['events_per_split']:.2f} evts, conf {old['mean_confidence']:.2f}"
        print(line)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=10, help="number of random splits (default 10)")
    parser.add_argument("--no-legacy", action="store_true", help="skip the legacy-formula comparison")
    parser.add_argument("--output", default=None, help="write the report as JSON to this path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    report = calibration_report(args.seeds, compare_legacy=not args.no_legacy)
    print_report(report)
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
