"""
Tests for evaluation/real_data_incident.py.

Uses a small synthetic CSV shaped exactly like the real Adult dataset
(same column names/dtypes) instead of the full ~48k-row data/adult_dataset.csv,
so these tests stay fast and don't depend on that file's exact row count --
only its schema (the module's own column list, NUMERIC_FEATURE_COLUMNS +
TARGET_COLUMN) needs to match, which this fixture keeps in sync with by
importing that same list rather than hardcoding a second copy of it.
"""

import pandas as pd
import pytest

from evaluation.real_data_incident import (
    NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_real_data_incident,
    load_adult_data,
    run_real_data_incident,
)
from schemas import DetectionMethod, IncidentReport

_N_ROWS = 200


def _make_adult_like_csv(tmp_path, n_rows: int = _N_ROWS) -> str:
    """Write a small CSV with the same columns/shape as the real Adult
    dataset (numeric features + a "<=50K"/">50K" income column) to a temp
    file, and return its path."""
    data = {}
    for i, column in enumerate(NUMERIC_FEATURE_COLUMNS):
        # Distinct, non-degenerate integer ranges per column so KS-test /
        # LogisticRegression have real variance to work with, not a
        # constant column.
        data[column] = [(row * (i + 1) + row % (i + 3)) % 97 + 1 for row in range(n_rows)]
    # Alternate labels so both classes are present (LogisticRegression
    # needs >1 class to fit).
    data[TARGET_COLUMN] = ["<=50K" if row % 2 == 0 else ">50K" for row in range(n_rows)]

    df = pd.DataFrame(data)
    path = tmp_path / "adult_like.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_adult_data_encodes_income_and_keeps_only_numeric_columns(tmp_path):
    path = _make_adult_like_csv(tmp_path)

    df = load_adult_data(path)

    assert set(df.columns) == set(NUMERIC_FEATURE_COLUMNS) | {TARGET_COLUMN}
    assert set(df[TARGET_COLUMN].unique()) <= {0, 1}


def test_build_real_data_incident_injects_all_six_failure_types(tmp_path):
    path = _make_adult_like_csv(tmp_path)

    reference_df, current_df = build_real_data_incident(path, random_seed=42)

    # duplicate rows: current_df is longer than reference_df's own half.
    assert len(current_df) > len(reference_df)
    # missing values: capital-loss is not in this fixture's numeric set --
    # this script always injects into "capital-loss", which IS one of
    # NUMERIC_FEATURE_COLUMNS, so it must be present and partially null.
    assert current_df["capital-loss"].isna().any()
    # schema mismatch: fnlwgt dropped entirely.
    assert "fnlwgt" not in current_df.columns
    assert "fnlwgt" in reference_df.columns
    # corrupted values: some age values are now far outside the fixture's
    # deterministic 1-97 range, from being scaled by corruption_multiplier.
    assert (current_df["age"] > 97).any()
    # label shift: income's positive rate moved to ~60%, away from this
    # fixture's original alternating 50/50 split.
    assert current_df[TARGET_COLUMN].mean() == pytest.approx(0.6, abs=0.05)


def test_run_real_data_incident_detects_multiple_distinct_failure_types(tmp_path):
    path = _make_adult_like_csv(tmp_path)

    report = run_real_data_incident(path, random_seed=42, incident_id="test-incident")

    detection_methods = {event.detection_method for event in report.evidence_events}
    # Five of the six injected failure types must show up as distinct
    # DetectionMethods in the SAME report -- this is the whole point of
    # collapsing the old detection-showcase/full-incident split into one
    # incident once VerificationAgent could handle all of them at once.
    # ROLLING_ACCURACY (label shift) is deliberately not asserted here:
    # this fixture's features are a fully deterministic function of row
    # index, easy enough for LogisticRegression to fit near-perfectly
    # regardless of class balance, so accuracy doesn't meaningfully drop
    # from label shift alone on THIS data -- see
    # detection/rolling_accuracy_detector.py's own docstring for why that
    # only shows up on a classifier that isn't already near-perfect (real
    # coverage: tests/test_rolling_accuracy_detector.py, and the real
    # Adult-dataset run, both use less separable data for exactly this
    # reason).
    assert DetectionMethod.KS_TEST in detection_methods
    assert DetectionMethod.MISSING_VALUE_RATE in detection_methods
    assert DetectionMethod.DUPLICATE_ROW_RATE in detection_methods
    assert DetectionMethod.SCHEMA_CHECK in detection_methods
    assert DetectionMethod.ISOLATION_FOREST in detection_methods


def test_run_real_data_incident_produces_a_complete_verified_report(tmp_path):
    path = _make_adult_like_csv(tmp_path)

    report = run_real_data_incident(path, random_seed=42, incident_id="test-incident")

    assert isinstance(report, IncidentReport)
    assert report.incident_id == "test-incident"
    assert len(report.evidence_events) > 0
    assert len(report.rca_result.hypotheses) > 0
    assert report.remediation_plan is not None
    assert report.verification_result is not None
    # value_before/value_after are real measured accuracies, not the
    # ground_truth_label cheat-sheet -- this incident type isn't a
    # benchmark run (see module docstring), so ground_truth_label must
    # stay unset.
    assert report.ground_truth_label is None
    assert 0.0 <= report.verification_result.value_before <= 1.0
    assert 0.0 <= report.verification_result.value_after <= 1.0


def test_run_real_data_incident_is_reproducible(tmp_path):
    path = _make_adult_like_csv(tmp_path)

    report1 = run_real_data_incident(path, random_seed=7, incident_id="a")
    report2 = run_real_data_incident(path, random_seed=7, incident_id="b")

    assert len(report1.evidence_events) == len(report2.evidence_events)
    assert report1.verification_result.value_before == report2.verification_result.value_before
    assert report1.verification_result.value_after == report2.verification_result.value_after
