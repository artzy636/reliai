import numpy as np
import pandas as pd

from detection.data_agent import DataAgent
from detection.fault_injection import inject_feature_drift


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
