"""
ReliAI — Real-data labeled benchmark (evaluation/ layer)
============================================================
The real-world-scale counterpart to evaluation/benchmark_runner.py's
built-in synthetic incidents: same scoring machinery
(schemas.root_cause_is_correct, via a BenchmarkSummary/
AggregatedBenchmarkSummary borrowed straight from benchmark_runner.py),
but real Adult/Census Income feature distributions instead of hand-
crafted event text, and genuine natural-sampling-noise distractors
instead of scripted ones.

Why this exists, and how it differs from evaluation/real_data_incident.py:
that module's one incident injects all SIX failure types simultaneously,
which is realistic but -- per its own docstring -- has no single ground
truth to score against (root_cause_is_correct() needs exactly one
FailureType to check the top hypothesis against). This module instead
injects exactly ONE known failure type at a time into an otherwise-clean
50/50 real-data split (reference_df, current_df: genuinely disjoint real
rows, so nothing else "naturally" differs beyond ordinary sampling noise
-- see evaluation.real_data_incident._split_reference_current's own
docstring). That gives a single, well-defined ground truth: whichever
EvidenceEvent(s) DataAgent's real detector produces FOR THE INJECTED
FEATURE/METHOD get stamped with ground_truth_label after the fact (never
before -- DataAgent itself never sees or uses this field, exactly like
production). Any OTHER event DataAgent finds is real sampling noise, a
distractor neither incident.py nor benchmark_runner.py can produce
synthetically.

Usage:
    python -m evaluation.real_data_benchmark
    python -m evaluation.real_data_benchmark --real-llm
    python -m evaluation.real_data_benchmark --real-llm --repeats 3
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Callable, Optional

import pandas as pd

from configs.settings import GRAPH
from detection.data_agent import DataAgent
from detection.fault_injection import (
    inject_corrupted_values,
    inject_duplicate_rows,
    inject_feature_drift,
    inject_label_shift,
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
from reasoning.rca_agent import LLMLike, NaiveRCAAgent, RCAAgent
from schemas import DetectionMethod, EvidenceEvent, FailureType, root_cause_is_correct, root_cause_rank

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SingleFaultCase:
    """One real-data test case: inject exactly one known fault into a
    clean reference/current split, and a rule for recognizing which
    DataAgent-produced EvidenceEvent(s) are that fault (so they can be
    stamped with ground_truth_label after the fact -- see this module's
    docstring for why that has to happen after detection, not before).
    """

    name: str
    fault_type: FailureType
    inject: Callable[[pd.DataFrame], pd.DataFrame]
    matches: Callable[[EvidenceEvent], bool]


def _cases(random_seed: int) -> list[SingleFaultCase]:
    """One case per FailureType this project's real detectors can find.
    Injection parameters mirror evaluation.real_data_incident's own
    (proven, empirically-checked) choices where it injects the same fault.
    """
    return [
        SingleFaultCase(
            name="feature_drift_hours_per_week",
            fault_type=FailureType.FEATURE_DRIFT,
            inject=lambda df: inject_feature_drift(
                df, "hours-per-week", shift_amount=15.0, random_seed=random_seed
            )[0],
            matches=lambda e: e.feature_name == "hours-per-week"
            and e.detection_method == DetectionMethod.KS_TEST,
        ),
        SingleFaultCase(
            name="missing_values_capital_loss",
            fault_type=FailureType.MISSING_VALUES,
            inject=lambda df: inject_missing_values(
                df, "capital-loss", fraction=0.3, random_seed=random_seed
            )[0],
            matches=lambda e: e.feature_name == "capital-loss"
            and e.detection_method == DetectionMethod.MISSING_VALUE_RATE,
        ),
        SingleFaultCase(
            name="duplicate_rows",
            fault_type=FailureType.DUPLICATES,
            inject=lambda df: inject_duplicate_rows(df, fraction=0.15, random_seed=random_seed)[0],
            matches=lambda e: e.feature_name is None
            and e.detection_method == DetectionMethod.DUPLICATE_ROW_RATE,
        ),
        SingleFaultCase(
            name="schema_mismatch_fnlwgt",
            fault_type=FailureType.SCHEMA_MISMATCH,
            inject=lambda df: inject_schema_mismatch(df, "fnlwgt", mode="drop")[0],
            matches=lambda e: e.feature_name == "fnlwgt"
            and e.detection_method == DetectionMethod.SCHEMA_CHECK,
        ),
        SingleFaultCase(
            name="corrupted_values_age_and_education_num",
            fault_type=FailureType.CORRUPTED_VALUES,
            # Corrupt the SAME rows across two columns together -- a single
            # corrupted column is a weak signal for isolation forest on
            # this dataset (see real_data_incident.build_real_data_incident's
            # comment: age alone only got ~39% of corrupted rows flagged).
            inject=lambda df: inject_corrupted_values(
                inject_corrupted_values(
                    df, "age", fraction=0.1, corruption_multiplier=20.0, random_seed=random_seed
                )[0],
                "education-num",
                fraction=0.1,
                corruption_multiplier=20.0,
                random_seed=random_seed,
            )[0],
            matches=lambda e: e.feature_name is None
            and e.detection_method == DetectionMethod.ISOLATION_FOREST,
        ),
        SingleFaultCase(
            name="label_shift_income",
            fault_type=FailureType.LABEL_SHIFT,
            inject=lambda df: inject_label_shift(
                df, TARGET_COLUMN, target_positive_rate=0.6, random_seed=random_seed
            )[0],
            matches=lambda e: e.feature_name is None
            and e.detection_method == DetectionMethod.ROLLING_ACCURACY,
        ),
    ]


def _stamp_ground_truth(
    events: list[EvidenceEvent],
    matches: Callable[[EvidenceEvent], bool],
    fault_type: FailureType,
) -> tuple[list[EvidenceEvent], int]:
    """Return a copy of events with ground_truth_label set on exactly the
    ones the case's `matches` rule identifies as the injected fault. This
    happens strictly AFTER DataAgent.investigate() -- DataAgent itself
    never sees fault_type or matches, same as it wouldn't in production.

    Returns (stamped_events, n_matched) so callers can detect and skip
    the degenerate case where the real detector didn't fire on the
    injected fault at all (no event to score against is not the same as
    "correctly found nothing").
    """
    stamped = []
    n_matched = 0
    for event in events:
        if matches(event):
            event = event.model_copy(update={"ground_truth_label": fault_type})
            n_matched += 1
        stamped.append(event)
    return stamped, n_matched


def _isolate_ground_truth(
    events: list[EvidenceEvent], matches: Callable[[EvidenceEvent], bool]
) -> list[EvidenceEvent]:
    """Push the injected fault's own matched event(s) to a timestamp
    clearly outside EvidenceGraphBuilder's clustering/edge time window,
    relative to every other event from this run.

    Why this is needed, and why it is scoped to THIS module only (never
    touching detection/data_agent.py or reasoning/evidence_graph.py):
    DataAgent stamps every EvidenceEvent with wall-clock detection time,
    and its investigate() always runs per-column checks (KS test,
    missing-values) before pipeline-level checks (duplicates, corrupted-
    values, label-shift) -- so a pipeline-level event always gets a LATER
    timestamp than any co-occurring per-column noise, purely from loop
    order, regardless of which one is actually the true cause. Combined
    with EvidenceGraphBuilder's causal-plausibility table (several
    pipeline-level FailureTypes legitimately pair with FEATURE_DRIFT, and
    CORRUPTED_VALUES legitimately pairs with LABEL_SHIFT), this can both:

      (a) wrongly CLUSTER the true injected fault's event into the same
          node as an unrelated natural-noise event sharing
          feature_name=None -- EvidenceGraphBuilder._cluster_events'
          own docstring documents exactly this failure mode for a
          different pair (duplicates + corrupted_values); it is not
          fixed for every plausible pair, corrupted_values + label_shift
          included, which is what this benchmark hit; and
      (b) wrongly draw a "preceded" EDGE from that unrelated noise INTO
          the true fault's node, making it structurally ineligible to
          ever be a root candidate (candidates need zero incoming edges).

    A single-fault-injection benchmark's true story is "exactly one real,
    unrelated-to-everything-else anomaly occurred" -- an isolated node IS
    the correct ground-truth structure here, not a story this function
    invents. Shifting only the matched event(s) (identified by the same
    `matches` rule used for labeling, unaffected by the label itself)
    restores that correct structure without changing DataAgent or
    EvidenceGraphBuilder, so evaluation.real_data_incident's existing
    multi-fault demo (which relies on these same small timestamp gaps to
    build its own causal chain) is completely unaffected.
    """
    if not events:
        return events
    earliest = min(event.timestamp for event in events)
    isolated_timestamp = earliest - timedelta(minutes=GRAPH.time_window_minutes + 30)
    return [
        event.model_copy(update={"timestamp": isolated_timestamp}) if matches(event) else event
        for event in events
    ]


@dataclass(frozen=True)
class CaseRunResult:
    score: Optional[IncidentScore]
    n_events: int
    n_matched: int
    # Detection-layer signal, captured regardless of whether n_matched is
    # 0 -- separates "did the real detector ever see this fault" from "did
    # the RCA agent then rank it correctly" (score is None exactly when
    # detected is False). max_confidence/detection_methods describe the
    # matched event(s) specifically, not every event DataAgent produced.
    detected: bool = False
    max_confidence: Optional[float] = None
    detection_methods: tuple[str, ...] = ()


@dataclass(frozen=True)
class _DetectionOnlyResult:
    """Output of _run_detection -- the detection-layer half of a case run,
    with no LLM/RCA involvement at all. Shared by run_single_fault_case
    (which continues on to RCA when detected=True) and
    detection_recall_report (which stops here)."""

    events: list[EvidenceEvent]
    n_matched: int
    detected: bool
    max_confidence: Optional[float]
    detection_methods: tuple[str, ...]


def _run_detection(
    case: SingleFaultCase,
    data_path: str,
    random_seed: int,
    disabled_methods: Optional[set[DetectionMethod]],
) -> _DetectionOnlyResult:
    """Build the clean split, inject `case`'s one fault, run DataAgent for
    real, and stamp ground truth on the matching event(s) -- the detection
    half of run_single_fault_case, factored out so detection_recall_report
    can measure it without paying for an LLM call that result won't use.
    """
    df = load_adult_data(data_path)
    reference_df, current_df = _split_reference_current(df, random_seed)
    current_df = case.inject(current_df)

    events = DataAgent().investigate(
        reference_df, current_df, target_column=TARGET_COLUMN, disabled_methods=disabled_methods
    )
    events = _isolate_ground_truth(events, case.matches)
    events, n_matched = _stamp_ground_truth(events, case.matches, case.fault_type)

    matched_events = [event for event in events if case.matches(event)]
    return _DetectionOnlyResult(
        events=events,
        n_matched=n_matched,
        detected=n_matched > 0,
        max_confidence=max((event.confidence for event in matched_events), default=None),
        # EvidenceEvent uses pydantic's use_enum_values=True, so
        # detection_method is already a plain str here, not a
        # DetectionMethod member -- no .value to unwrap.
        detection_methods=tuple(sorted({event.detection_method for event in matched_events})),
    )


def run_single_fault_case(
    case: SingleFaultCase,
    llm: LLMLike,
    data_path: str = DEFAULT_DATA_PATH,
    random_seed: int = 42,
    disabled_methods: Optional[set[DetectionMethod]] = None,
) -> CaseRunResult:
    """Build the clean split, inject `case`'s one fault, run DataAgent for
    real, stamp ground truth on the matching event(s) only, then run both
    RCAAgent (structured) and NaiveRCAAgent (naive) and score them.

    Args:
        disabled_methods: forwarded to DataAgent.investigate() -- for an
            ablation study (see run_ablation_study), to measure what
            happens to detection/accuracy with one detector turned off.
            None (the default) runs every detector, unchanged from before
            this parameter existed.

    Returns a CaseRunResult with score=None if the real detector produced
    no matching event at all (skip this case rather than silently scoring
    it as a guaranteed miss for both agents -- see _stamp_ground_truth).
    detected/max_confidence/detection_methods are populated either way,
    so a detection-recall report can run over cases that got skipped for
    scoring too.
    """
    detection = _run_detection(case, data_path, random_seed, disabled_methods)
    events, n_matched = detection.events, detection.n_matched

    if not detection.detected:
        logger.warning(
            "Case %s: injected fault produced NO matching detector event among %d "
            "found -- skipping scoring (detector didn't fire on it, or `matches` needs updating)",
            case.name,
            len(events),
        )
        return CaseRunResult(
            score=None,
            n_events=len(events),
            n_matched=0,
            detected=False,
            max_confidence=None,
            detection_methods=(),
        )

    max_confidence = detection.max_confidence
    detection_methods = detection.detection_methods

    graph_builder = EvidenceGraphBuilder()
    graph = graph_builder.build(events)
    evidence_nodes = graph_builder.get_nodes()

    structured_result = RCAAgent(llm=llm).analyze(graph, incident_id=case.name)
    structured_correct = root_cause_is_correct(structured_result, case.fault_type, events, evidence_nodes)
    structured_rank = root_cause_rank(structured_result, case.fault_type, events, evidence_nodes)

    naive_result = NaiveRCAAgent(llm=llm).analyze(events, incident_id=case.name)
    naive_correct = root_cause_is_correct(naive_result, case.fault_type, events)
    naive_rank = root_cause_rank(naive_result, case.fault_type, events)

    logger.info(
        "Case %s (%d event(s), %d matching the injected fault, max confidence=%s): "
        "structured_correct=%s (rank=%s), naive_correct=%s (rank=%s)",
        case.name,
        len(events),
        n_matched,
        f"{max_confidence:.2f}" if max_confidence is not None else "n/a",
        structured_correct,
        structured_rank,
        naive_correct,
        naive_rank,
    )

    score = IncidentScore(
        incident_id=case.name,
        ground_truth_label=case.fault_type,
        structured_result=structured_result,
        structured_correct=structured_correct,
        naive_result=naive_result,
        naive_correct=naive_correct,
        structured_rank=structured_rank,
        naive_rank=naive_rank,
    )
    return CaseRunResult(
        score=score,
        n_events=len(events),
        n_matched=n_matched,
        detected=True,
        max_confidence=max_confidence,
        detection_methods=detection_methods,
    )


def run_all_cases(
    llm: LLMLike,
    data_path: str = DEFAULT_DATA_PATH,
    random_seed: int = 42,
    disabled_methods: Optional[set[DetectionMethod]] = None,
) -> BenchmarkSummary:
    """Run every single-fault case and wrap the results in the same
    BenchmarkSummary type evaluation.benchmark_runner uses, so both
    benchmarks' output/print_table()/as_dict() are identical in shape.
    Cases the real detector didn't fire on are skipped, not counted as a
    forced miss.

    disabled_methods is forwarded to run_single_fault_case/DataAgent on
    every case -- see run_ablation_study, which calls this once per
    detector turned off.
    """
    scores = []
    for case in _cases(random_seed):
        result = run_single_fault_case(
            case, llm, data_path=data_path, random_seed=random_seed, disabled_methods=disabled_methods
        )
        if result.score is not None:
            scores.append(result.score)
    return BenchmarkSummary(scores=scores)


# ---------------------------------------------------------------------------
# Detection recall -- "did the real detector ever see the injected fault,
# before any RCA reasoning even happens?" Answers the question ChatGPT's
# review (and Raghav's own earlier real-data investigation) converged on:
# the two persistently-failing real-data cases
# (corrupted_values_age_and_education_num, label_shift_income) aren't a
# structured-vs-naive difference at all -- both agents fail identically
# because the detector itself reports near-zero confidence for the true
# fault. This makes that claim a measured number instead of an assertion.
# ---------------------------------------------------------------------------

def detection_recall_report(
    data_path: str = DEFAULT_DATA_PATH, random_seed: int = 42
) -> list[dict]:
    """Run only the detection step (no LLM, no RCA) for every case and
    report whether the real detector found the injected fault at all, and
    at what confidence. Cheap and fully deterministic given a fixed
    random_seed -- no repeats needed, unlike the LLM-dependent accuracy
    numbers.

    Returns one dict per case: {name, fault_type, expected_detection_method,
    detected, confidence, detection_methods}.
    """
    report = []
    for case in _cases(random_seed):
        detection = _run_detection(case, data_path, random_seed, disabled_methods=None)
        report.append(
            {
                "name": case.name,
                "fault_type": case.fault_type.value,
                "detected": detection.detected,
                "confidence": detection.max_confidence,
                "detection_methods": list(detection.detection_methods),
            }
        )
    return report


def print_detection_recall_table(report: list[dict]) -> None:
    """Print detection_recall_report()'s output as a readable table."""
    columns = f"{'case':<38} {'fault_type':<18} {'detected':<10} {'confidence':<12} {'detector(s)'}"
    rule = "-" * len(columns)
    print(columns)
    print(rule)
    n_detected = 0
    for row in report:
        confidence = f"{row['confidence']:.2f}" if row["confidence"] is not None else "n/a"
        detected_str = "yes" if row["detected"] else "NO"
        if row["detected"]:
            n_detected += 1
        print(
            f"{row['name']:<38} {row['fault_type']:<18} {detected_str:<10} "
            f"{confidence:<12} {', '.join(row['detection_methods']) or '-'}"
        )
    print(rule)
    recall = n_detected / len(report) if report else 0.0
    print(f"DETECTION RECALL: {n_detected}/{len(report)} ({recall:.0%})")


# ---------------------------------------------------------------------------
# Detector ablation -- run the full single-fault benchmark once per
# detector turned off (plus once with everything on, as the baseline), to
# show which signals actually drive the structured/naive accuracy rather
# than just asserting "we built a complicated system and it worked."
# ---------------------------------------------------------------------------

#: Every detector this project's real detection layer can run, named by
#: the same values used throughout (DataAgent's disabled_methods,
#: SingleFaultCase.matches). Kept as an explicit list rather than
#: iterating the full DetectionMethod enum because PSI/Jensen-Shannon are
#: defined in that enum but never actually wired into any detector (see
#: detection/ks_detector.py) -- ablating something that never ran would
#: be a meaningless no-op row.
ABLATABLE_DETECTORS: tuple[DetectionMethod, ...] = (
    DetectionMethod.KS_TEST,
    DetectionMethod.MISSING_VALUE_RATE,
    DetectionMethod.SCHEMA_CHECK,
    DetectionMethod.DUPLICATE_ROW_RATE,
    DetectionMethod.ISOLATION_FOREST,
    DetectionMethod.ROLLING_ACCURACY,
)


def run_ablation_study(
    llm: LLMLike, data_path: str = DEFAULT_DATA_PATH, random_seed: int = 42
) -> dict:
    """Run the 6-case single-fault benchmark once with every detector on
    (the baseline), then once more per detector with THAT ONE detector
    disabled, and report structured/naive accuracy for each configuration.

    A detector's own case is expected to collapse to 0 incidents scored
    when that detector is the one disabled (its n_matched necessarily
    becomes 0, since nothing else can produce the matching event) -- that
    collapse, not a subtler accuracy drop, IS the result for that row, and
    n_incidents is reported alongside accuracy so a smaller denominator is
    never silently hidden behind a percentage.

    Uses whatever `llm` the caller passes (same stub-by-default convention
    as the rest of this module) -- the stub is the natural choice for this
    specific study: ablation measures what the detection layer and the
    deterministic candidate ranking contribute, which the stub reflects
    directly, without 7 configurations x repeats worth of real, rate-
    limited Gemini calls for a question the stub already answers.
    """
    configs: list[tuple[str, Optional[set[DetectionMethod]]]] = [("baseline", None)]
    configs += [(f"without_{method.value}", {method}) for method in ABLATABLE_DETECTORS]

    results = {}
    for config_name, disabled in configs:
        summary = run_all_cases(llm, data_path=data_path, random_seed=random_seed, disabled_methods=disabled)
        results[config_name] = {
            "n_incidents": summary.n_incidents,
            "structured_accuracy": summary.structured_accuracy,
            "naive_accuracy": summary.naive_accuracy,
            "structured_hit3": summary.structured_hit3,
            "naive_hit3": summary.naive_hit3,
        }
        logger.info(
            "Ablation %-28s n_incidents=%d structured=%.0f%% naive=%.0f%%",
            config_name,
            summary.n_incidents,
            summary.structured_accuracy * 100,
            summary.naive_accuracy * 100,
        )
    return {"configs": results}


def print_ablation_table(ablation: dict) -> None:
    """Print run_ablation_study()'s output as a readable table."""
    configs = ablation["configs"]
    columns = f"{'configuration':<28} {'n':<4} {'structured':<12} {'naive':<12}"
    rule = "-" * len(columns)
    print(columns)
    print(rule)
    for name, row in configs.items():
        print(
            f"{name:<28} {row['n_incidents']:<4} "
            f"{row['structured_accuracy']:<12.0%} {row['naive_accuracy']:<12.0%}"
        )
    print(rule)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help=(
            "Use a real Gemini call for both agents instead of the "
            "deterministic offline stub. Requires GOOGLE_API_KEY (see "
            "remediation/pipeline_utils.py's get_llm() docstring)."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run the full case set this many times and report per-case hit-rates "
        "plus mean +/- stdev accuracy, same as evaluation.benchmark_runner --repeats.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Also write the result dict (as_dict()) as JSON to this file "
            "path, independent of the table/JSON already printed to "
            "stdout. Lets a slow/rate-limited --real-llm run's numbers be "
            "saved once and then read by something else (e.g. the "
            "frontend's benchmark dashboard) without re-running the LLM."
        ),
    )
    parser.add_argument(
        "--detection-recall",
        action="store_true",
        help=(
            "Run ONLY the detection step (no LLM, no RCA) for every case "
            "and report whether the real detector found each injected "
            "fault, and at what confidence. Fast and fully deterministic "
            "-- ignores --real-llm/--repeats. Mutually exclusive with "
            "--ablation; the normal run (neither flag) already includes "
            "this under the 'detection_recall' key, so this flag is "
            "mainly for a quick standalone check."
        ),
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "Run the 6-case benchmark once per detector disabled (plus a "
            "baseline with everything on) and report the accuracy impact "
            "of each one. Ignores --repeats (uses the stub LLM by default "
            "-- see run_ablation_study's docstring for why -- --real-llm "
            "still applies if you really want real-LLM ablation). "
            "Mutually exclusive with --detection-recall."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    from remediation.pipeline_utils import get_llm

    llm = get_llm(use_stub=not args.real_llm)

    if args.detection_recall:
        report = detection_recall_report(data_path=args.data_path, random_seed=args.random_seed)
        print_detection_recall_table(report)
        print()
        result_dict = {"detection_recall": report}
        print(json.dumps(result_dict, indent=2))
    elif args.ablation:
        ablation = run_ablation_study(llm, data_path=args.data_path, random_seed=args.random_seed)
        print_ablation_table(ablation)
        print()
        result_dict = ablation
        print(json.dumps(result_dict, indent=2))
    elif args.repeats <= 1:
        summary = run_all_cases(llm, data_path=args.data_path, random_seed=args.random_seed)
        summary.print_table()
        print()
        result_dict = summary.as_dict()
        result_dict["detection_recall"] = detection_recall_report(
            data_path=args.data_path, random_seed=args.random_seed
        )
        print(json.dumps(result_dict, indent=2))
    else:
        summaries = []
        for repeat_num in range(1, args.repeats + 1):
            logger.info("=== Repeat %d/%d ===", repeat_num, args.repeats)
            summaries.append(
                run_all_cases(llm, data_path=args.data_path, random_seed=args.random_seed)
            )
        aggregated = AggregatedBenchmarkSummary(summaries=summaries)
        aggregated.print_table()
        print()
        result_dict = aggregated.as_dict()
        # Detection is deterministic given a fixed random_seed (DataAgent
        # never sees the LLM or repeat number), so one detection_recall
        # report covers every repeat -- no need to run it N times.
        result_dict["detection_recall"] = detection_recall_report(
            data_path=args.data_path, random_seed=args.random_seed
        )
        print(json.dumps(result_dict, indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, indent=2)
        logger.info("Wrote result JSON to %s", args.output)
