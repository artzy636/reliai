import numpy as np
import pandas as pd

from detection.isolation_forest_detector import detect_corrupted_values


def test_corrupted_values_detected():

    rng = np.random.RandomState(42)

    reference = pd.DataFrame({
        "age": rng.normal(30, 5, 500),
        "income": rng.normal(50000, 5000, 500),
    })

    current = reference.copy()
    # Corrupt 20% of rows with values far outside the reference range,
    # in a way a per-column KS test on either column alone would likely
    # miss for the corrupted fraction being this small relative to the
    # column's own spread -- but the (age, income) COMBINATION is
    # nothing like anything in reference.
    corrupt_idx = rng.choice(len(current), size=100, replace=False)
    current.loc[corrupt_idx, "age"] = current.loc[corrupt_idx, "age"] * 50
    current.loc[corrupt_idx, "income"] = current.loc[corrupt_idx, "income"] * 50

    event = detect_corrupted_values(reference, current)

    assert event is not None
    assert event.feature_name is None
    assert event.metric_value > 0.15


def test_no_corrupted_values():

    rng = np.random.RandomState(42)

    reference = pd.DataFrame({
        "age": rng.normal(30, 5, 500),
        "income": rng.normal(50000, 5000, 500),
    })

    current = pd.DataFrame({
        "age": rng.normal(30, 5, 500),
        "income": rng.normal(50000, 5000, 500),
    })

    event = detect_corrupted_values(reference, current)

    assert event is None


def test_no_shared_numeric_columns_returns_none():

    reference = pd.DataFrame({"region": ["north", "south", "east"]})
    current = pd.DataFrame({"region": ["north", "south", "east"]})

    event = detect_corrupted_values(reference, current)

    assert event is None


def test_too_few_reference_rows_returns_none():

    reference = pd.DataFrame({"age": [20, 22, 24]})
    current = pd.DataFrame({"age": [20, 22, 24, 5000]})

    event = detect_corrupted_values(reference, current)

    assert event is None


def test_empty_current_after_dropna_returns_none():

    rng = np.random.RandomState(42)

    reference = pd.DataFrame({"age": rng.normal(30, 5, 50)})
    current = pd.DataFrame({"age": [np.nan] * 10})

    event = detect_corrupted_values(reference, current)

    assert event is None
