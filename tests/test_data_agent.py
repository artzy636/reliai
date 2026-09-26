import numpy as np
import pandas as pd

from detection.data_agent import DataAgent
from detection.fault_injection import inject_corrupted_values, inject_feature_drift
from schemas import DetectionMethod


def test_investigate_localizes_drifted_feature():

    np.random.seed(42)

    reference_df = pd.DataFrame({
        "age": np.random.normal(30, 5, 500),
        "income": np.random.normal(50000, 5000, 500),
        "region": ["north", "south", "east", "west"] * 125,
    })

    current_df, _ = inject_feature_drift(
        reference_df,
        "income",
        shift_amount=20000,
        random_seed=42,
    )

    agent = DataAgent()
    events = agent.investigate(reference_df, current_df)

    assert len(events) >= 1

    drifted_event = next(
        (event for event in events if event.feature_name == "income"),
        None,
    )
    assert drifted_event is not None

    assert all(event.feature_name != "age" for event in events)


def test_investigate_detects_corrupted_values():

    rng = np.random.RandomState(42)

    reference_df = pd.DataFrame({
        "age": rng.normal(30, 5, 500),
        "income": rng.normal(50000, 5000, 500),
    })

    current_df, _ = inject_corrupted_values(
        reference_df,
        "income",
        fraction=0.2,
        corruption_multiplier=50.0,
        random_seed=42,
    )

    agent = DataAgent()
    events = agent.investigate(reference_df, current_df)

    corrupted_event = next(
        (event for event in events if event.detection_method == DetectionMethod.ISOLATION_FOREST),
        None,
    )
    assert corrupted_event is not None
    assert corrupted_event.feature_name is None
