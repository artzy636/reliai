import numpy as np
import pandas as pd

from detection.missing_values_detector import detect_missing_values


def test_missing_values_detected():

    np.random.seed(42)

    reference = pd.Series(np.random.normal(0, 1, 1000))

    current = pd.Series(np.random.normal(0, 1, 1000))
    current.iloc[:300] = np.nan

    event = detect_missing_values(
        reference,
        current,
        feature_name="age",
    )

    assert event is not None
    assert event.feature_name == "age"
    assert event.metric_value > 0.25


def test_no_missing_values():

    np.random.seed(42)

    reference = pd.Series(np.random.normal(0, 1, 1000))

    current = pd.Series(np.random.normal(0, 1, 1000))

    event = detect_missing_values(
        reference,
        current,
        feature_name="age",
    )

    assert event is None
