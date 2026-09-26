import numpy as np
import pandas as pd

from detection.schema_mismatch_detector import detect_schema_mismatch


def test_missing_column_detected():

    np.random.seed(42)

    reference = pd.DataFrame({
        "age": np.random.normal(30, 5, 200),
        "income": np.random.normal(50000, 5000, 200),
    })

    current = reference.drop(columns=["income"])

    event = detect_schema_mismatch(reference, current, feature_name="income")

    assert event is not None
    assert event.feature_name == "income"


def test_dtype_change_detected():

    np.random.seed(42)

    reference = pd.DataFrame({
        "age": np.random.normal(30, 5, 200),
    })

    current = reference.copy()
    current["age"] = current["age"].astype(str)

    event = detect_schema_mismatch(reference, current, feature_name="age")

    assert event is not None
    assert event.feature_name == "age"


def test_no_schema_mismatch():

    np.random.seed(42)

    reference = pd.DataFrame({
        "age": np.random.normal(30, 5, 200),
        "income": np.random.normal(50000, 5000, 200),
    })

    current = reference.copy()

    event = detect_schema_mismatch(reference, current, feature_name="age")

    assert event is None


def test_numeric_widening_is_not_a_schema_mismatch():
    # Regression test: int64 -> float64 is exactly what pandas does the
    # moment a NaN is introduced (or the column is combined with a float
    # in arithmetic) -- a routine numeric promotion, not a structural
    # schema break. Flagging it produced spurious SCHEMA_CHECK findings
    # on columns whose only real problem was missing values or feature
    # drift, confirmed on the real Adult-dataset incident.
    np.random.seed(42)

    reference = pd.DataFrame({
        "hours_per_week": np.random.randint(20, 60, 200).astype("int64"),
    })

    current = reference.copy()
    current["hours_per_week"] = current["hours_per_week"].astype("float64") + 15.0

    assert current["hours_per_week"].dtype != reference["hours_per_week"].dtype

    event = detect_schema_mismatch(reference, current, feature_name="hours_per_week")

    assert event is None


def test_numeric_to_non_numeric_is_still_a_schema_mismatch():
    np.random.seed(42)

    reference = pd.DataFrame({
        "age": np.random.randint(18, 80, 200).astype("int64"),
    })

    current = reference.copy()
    current["age"] = current["age"].astype(object)
    current.loc[0, "age"] = "unknown"

    event = detect_schema_mismatch(reference, current, feature_name="age")

    assert event is not None
    assert event.feature_name == "age"
