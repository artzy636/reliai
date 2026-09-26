import numpy as np
import pandas as pd

from detection.duplicate_rows_detector import detect_duplicate_rows


def test_duplicate_rows_detected():

    np.random.seed(42)

    reference = pd.DataFrame({
        "age": np.random.normal(30, 5, 200),
        "income": np.random.normal(50000, 5000, 200),
    })

    current = pd.DataFrame({
        "age": np.random.normal(30, 5, 200),
        "income": np.random.normal(50000, 5000, 200),
    })
    current.iloc[100:180] = current.iloc[0:1].values

    event = detect_duplicate_rows(reference, current)

    assert event is not None
    assert event.feature_name is None
    assert event.metric_value > 0.25


def test_no_duplicate_rows():

    np.random.seed(42)

    reference = pd.DataFrame({
        "age": np.random.normal(30, 5, 200),
        "income": np.random.normal(50000, 5000, 200),
    })

    current = pd.DataFrame({
        "age": np.random.normal(30, 5, 200),
        "income": np.random.normal(50000, 5000, 200),
    })

    event = detect_duplicate_rows(reference, current)

    assert event is None
