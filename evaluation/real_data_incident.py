"""
ReliAI — Real-data incident (UCI Adult / Census Income dataset)
====================================================================
Runs the pipeline against real feature distributions instead of a
synthetic sklearn.make_classification dataset (contrast with
remediation/pipeline_utils.py's generate_injected_drift_data). Data file:
data/adult_dataset.csv (the standard Adult/Census Income dataset: age,
workclass, fnlwgt, education, education-num, marital-status, occupation,
relationship, race, sex, capital-gain, capital-loss, hours-per-week,
native-country, income).

Two separate incidents, for two different reasons:

  * detection_showcase() -- calls detection.data_agent.DataAgent directly
    (no RCA/remediation/verification) on a current_df carrying FOUR
    distinct, simultaneous injected failures: feature drift, missing
    values, duplicate rows, AND a schema mismatch (a dropped column).
    This is the "multiple failure types on real data" demonstration --
    DataAgent has no notion of a single ground-truth fault here, exactly
    like production.

  * full_incident() -- calls evaluation.incident_pipeline.run_incident_pipeline
    (the complete four-layer pipeline, including RemediationAgent and
    VerificationAgent) on a current_df carrying only TWO of those four
    failures: feature drift and duplicate rows. Missing values and the
    schema-mismatch (dropped-column) fault are both deliberately left out
    of this one -- neither is compatible with VerificationAgent as it
    exists today:
      - schema mismatch (dropped column): VerificationAgent derives its
        feature set from reference_df.columns and requires every one of
        those columns to still be present (and numeric) in current_df, so
        it can fit/score a LogisticRegression on both. A dropped column
        breaks that -- current_df[feature_columns] raises a KeyError.
      - missing values: VerificationAgent's value_before is scored by
        calling the baseline model directly on the raw replay slice,
        *before* branching on remediation_plan.category -- so even when
        the DATA_FIX branch would go on to impute them correctly,
        value_before's own scoring call still sees the untouched NaNs
        first and LogisticRegression.predict() raises ValueError: Input X
        contains NaN. This isn't a scope boundary the way the dropped
        column is (VerificationAgent's docstring explicitly names missing
        values as one of DATA_FIX's two corrections) -- it's a latent bug
        in the unconditional value_before step. Both are real, reproducible
        gaps in remediation/verification_agent.py, not something this
        script silently routes around by changing that file -- per
        CONTRACT.md rule 2, that's a team decision. detection_showcase()
        is where missing values and schema mismatch get their moment
        instead, run straight through DataAgent with no verification
        step to trip over.

Per verification_agent.py's own numeric-only assumption, both incidents
use only the Adult dataset's six numeric columns (age, fnlwgt,
education-num, capital-gain, capital-loss, hours-per-week) plus income
recoded to 0/1 -- the categorical columns (workclass, education,
occupation, ...) are dropped rather than encoded, matching the rest of the
pipeline's existing scope rather than expanding it.

ground_truth_label is left None in both incidents: schemas.EvidenceEvent's
own docstring says it "only exists during benchmark runs" (single
known-injected fault, scored against detection.fault_injection's output),
and both incidents here inject more than one failure type into the same
current_df -- there is no single ground truth to stamp. The RCA Agent
ranks candidates with no cheat sheet, same as it would in production.

Usage:
    python -m evaluation.real_data_incident
    python -m evaluation.real_data_incident --output adult_income_report.json
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

from detection.data_agent import DataAgent
from detection.fault_injection import (
    inject_duplicate_rows,
    inject_feature_drift,
    inject_missing_values,
    inject_schema_mismatch,
)
from evaluation.incident_pipeline import run_incident_pipeline
from remediation.pipeline_utils import get_llm
from schemas import EvidenceEvent, IncidentReport

logger = logging.getLogger(__name__)

DEFAULT_DATA_PATH = "data/adult_dataset.csv"
DEFAULT_OUTPUT_PATH = "adult_income_report.json"
TARGET_COLUMN = "income"
NUMERIC_FEATURE_COLUMNS = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]


def load_adult_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the Adult dataset and reduce it to the numeric feature columns
    plus a binary-encoded income target.

    Parameters
    ----------
    path : str
        Path to the Adult dataset CSV (header row + the standard 14
        feature columns + income).

    Returns
    -------
    pd.DataFrame
        NUMERIC_FEATURE_COLUMNS plus TARGET_COLUMN, TARGET_COLUMN recoded
        from "<=50K"/">50K" (any surrounding whitespace stripped, matching
        both the plain and the original UCI-raw formatting of this
        dataset) to 0/1.
    """
    df = pd.read_csv(path)
    df = df[NUMERIC_FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    df[TARGET_COLUMN] = (df[TARGET_COLUMN].str.strip().str.rstrip(".") == ">50K").astype(int)
    return df


def _split_reference_current(
    df: pd.DataFrame, random_seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle df once and split it in half -- reference_df and current_df
    start as disjoint real rows, not two copies of the same rows, so any
    drift/missing/duplicate/schema differences DataAgent finds beyond what
    was deliberately injected are genuine sampling noise, not an artifact
    of comparing a dataframe against itself."""
    shuffled = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    midpoint = len(shuffled) // 2
    reference_df = shuffled.iloc[:midpoint].reset_index(drop=True)
    current_df = shuffled.iloc[midpoint:].reset_index(drop=True)
    return reference_df, current_df


def _summarize_events(events: list[EvidenceEvent]) -> str:
    counts = Counter(event.detection_method for event in events)
    lines = [f"  - {method}: {count} event(s)" for method, count in counts.items()]
    return "\n".join(lines) if lines else "  (none)"


def detection_showcase(
    data_path: str = DEFAULT_DATA_PATH,
    random_seed: int = 42,
) -> list[EvidenceEvent]:
    """Run DataAgent directly against a current_df carrying four distinct,
    simultaneous real-data failures: feature drift (hours-per-week
    shifted), missing values (capital-loss partially nulled), duplicate
    rows (10% of rows re-appended), and a schema mismatch (fnlwgt dropped
    entirely).

    Returns
    -------
    list[EvidenceEvent]
        Every finding DataAgent.investigate() produced -- expected to span
        at least four distinct DetectionMethod values if all four
        injected failures are correctly localized.
    """
    df = load_adult_data(data_path)
    reference_df, current_df = _split_reference_current(df, random_seed)

    current_df, _ = inject_feature_drift(
        current_df, "hours-per-week", shift_amount=15.0, random_seed=random_seed
    )
    current_df, _ = inject_missing_values(
        current_df, "capital-loss", fraction=0.2, random_seed=random_seed
    )
    current_df, _ = inject_duplicate_rows(
        current_df, fraction=0.1, random_seed=random_seed
    )
    current_df, _ = inject_schema_mismatch(current_df, "fnlwgt", mode="drop")

    events = DataAgent().investigate(reference_df, current_df)
    logger.info(
        "detection_showcase: %d evidence event(s) across %d distinct detection method(s):\n%s",
        len(events),
        len({event.detection_method for event in events}),
        _summarize_events(events),
    )
    return events


def full_incident(
    data_path: str = DEFAULT_DATA_PATH,
    random_seed: int = 42,
    incident_id: Optional[str] = None,
) -> IncidentReport:
    """Run the complete four-layer pipeline (detection, evidence graph,
    RCA, remediation, verification) against a current_df carrying two
    real-data failures VerificationAgent can actually score end-to-end:
    feature drift (hours-per-week) and duplicate rows. See module
    docstring for why missing values and schema-mismatch are excluded
    here.

    Returns
    -------
    IncidentReport
        Complete report, including a real measured before/after accuracy
        from VerificationAgent.
    """
    df = load_adult_data(data_path)
    reference_df, current_df = _split_reference_current(df, random_seed)

    current_df, _ = inject_feature_drift(
        current_df, "hours-per-week", shift_amount=15.0, random_seed=random_seed
    )
    current_df, _ = inject_duplicate_rows(
        current_df, fraction=0.1, random_seed=random_seed
    )

    report = run_incident_pipeline(
        reference_df,
        current_df,
        target_column=TARGET_COLUMN,
        llm=get_llm(),
        incident_id=incident_id,
    )
    logger.info(
        "full_incident %s: %d evidence event(s), %d hypothesis/es, "
        "remediation=%s, verified=%s (%.4f -> %.4f)",
        report.incident_id,
        len(report.evidence_events),
        len(report.rca_result.hypotheses),
        report.remediation_plan.category.value,
        report.verification_result.improved,
        report.verification_result.value_before,
        report.verification_result.value_after,
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help=f"Path to the Adult dataset CSV (default: {DEFAULT_DATA_PATH})",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Where to write the full incident's IncidentReport JSON (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for the reference/current split and fault injection (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()

    detection_showcase(args.data_path, random_seed=args.random_seed)

    report = full_incident(args.data_path, random_seed=args.random_seed)
    Path(args.output).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("Wrote real-data IncidentReport (incident_id=%s) to %s", report.incident_id, args.output)
