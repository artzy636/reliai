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
