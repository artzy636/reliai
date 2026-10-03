import numpy as np
import pandas as pd

from sklearn.datasets import make_classification

from detection.data_agent import DataAgent
from detection.fault_injection import inject_corrupted_values, inject_feature_drift, inject_label_shift
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


def test_investigate_disabled_methods_skips_only_those_detectors():
    """disabled_methods removes exactly the named detector's events and
    leaves every other detector untouched (ablation-study hook)."""

    np.random.seed(42)

    reference_df = pd.DataFrame({
        "age": np.random.normal(30, 5, 500),
        "income": np.random.normal(50000, 5000, 500),
    })
    current_df, _ = inject_feature_drift(
        reference_df, "income", shift_amount=20000, random_seed=42
    )
    current_df = current_df.drop(columns=["age"])  # also triggers schema_check

    agent = DataAgent()
    baseline = agent.investigate(reference_df, current_df)
    baseline_methods = {event.detection_method for event in baseline}
    assert DetectionMethod.KS_TEST in baseline_methods
    assert DetectionMethod.SCHEMA_CHECK in baseline_methods

    without_ks = agent.investigate(
        reference_df, current_df, disabled_methods={DetectionMethod.KS_TEST}
    )
    methods = {event.detection_method for event in without_ks}
    assert DetectionMethod.KS_TEST not in methods
    assert DetectionMethod.SCHEMA_CHECK in methods

    without_schema = agent.investigate(
        reference_df, current_df, disabled_methods={DetectionMethod.SCHEMA_CHECK}
    )
    methods = {event.detection_method for event in without_schema}
    assert DetectionMethod.SCHEMA_CHECK not in methods
    assert DetectionMethod.KS_TEST in methods


def test_investigate_disabled_methods_none_matches_default():
    np.random.seed(42)
    reference_df = pd.DataFrame({"income": np.random.normal(50000, 5000, 500)})
    current_df, _ = inject_feature_drift(
        reference_df, "income", shift_amount=20000, random_seed=42
    )

    agent = DataAgent()
    default_events = agent.investigate(reference_df, current_df)
    explicit_none = agent.investigate(reference_df, current_df, disabled_methods=None)

    assert [e.detection_method for e in default_events] == [e.detection_method for e in explicit_none]


def test_investigate_without_target_column_skips_label_shift_check():
    """No target_column passed -> DataAgent behaves exactly as it always
    has, same as every existing caller that doesn't have one."""

    np.random.seed(42)

    reference_df = pd.DataFrame({
        "age": np.random.normal(30, 5, 500),
        "income": np.random.normal(50000, 5000, 500),
    })
    current_df = reference_df.copy()

    agent = DataAgent()
    events = agent.investigate(reference_df, current_df)

    assert all(event.detection_method != DetectionMethod.ROLLING_ACCURACY for event in events)


def test_investigate_detects_label_shift_when_target_column_given():

    feature_columns = [f"feature_{i}" for i in range(5)]
    target_column = "label"

    X, y = make_classification(
        n_samples=2000,
        n_features=len(feature_columns),
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=0.7,
        weights=[0.9, 0.1],
        random_state=42,
    )
    df = pd.DataFrame(X, columns=feature_columns)
    df[target_column] = y
    reference_df = df.iloc[:1000].reset_index(drop=True)
    current_df = df.iloc[1000:].reset_index(drop=True)

    shifted_current_df, _ = inject_label_shift(
        current_df, target_column, target_positive_rate=0.6, random_seed=42
    )

    agent = DataAgent()
    events = agent.investigate(reference_df, shifted_current_df, target_column=target_column)

    label_shift_event = next(
        (event for event in events if event.detection_method == DetectionMethod.ROLLING_ACCURACY),
        None,
    )
    assert label_shift_event is not None
    assert label_shift_event.feature_name is None
