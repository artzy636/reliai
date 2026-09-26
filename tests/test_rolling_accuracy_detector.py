import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from detection.rolling_accuracy_detector import detect_label_shift

_FEATURE_COLUMNS = [f"feature_{i}" for i in range(5)]
_TARGET_COLUMN = "label"


def _classification_df(n_samples: int = 2000, weights=(0.9, 0.1), random_state: int = 42) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(_FEATURE_COLUMNS),
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=0.7,
        weights=list(weights),
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=_FEATURE_COLUMNS)
    df[_TARGET_COLUMN] = y
    return df


def _shift_label_balance(df: pd.DataFrame, target_positive_rate: float, random_seed: int = 42) -> pd.DataFrame:
    positive_df = df[df[_TARGET_COLUMN] == 1]
    negative_df = df[df[_TARGET_COLUMN] == 0]
    n_total = len(df)
    n_positive = int(round(n_total * target_positive_rate))
    n_negative = n_total - n_positive
    resampled = pd.concat(
        [
            positive_df.sample(n=n_positive, replace=True, random_state=random_seed),
            negative_df.sample(n=n_negative, replace=True, random_state=random_seed),
        ],
        ignore_index=True,
    )
    return resampled.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)


def test_label_shift_detected():

    df = _classification_df()
    reference_df = df.iloc[:1000].reset_index(drop=True)
    current_df = df.iloc[1000:].reset_index(drop=True)

    shifted_current_df = _shift_label_balance(current_df, target_positive_rate=0.6)

    event = detect_label_shift(reference_df, shifted_current_df, _TARGET_COLUMN)

    assert event is not None
    assert event.feature_name is None
    assert 0.0 < event.confidence <= 1.0


def test_no_label_shift():

    df = _classification_df()
    reference_df = df.iloc[:1000].reset_index(drop=True)
    current_df = df.iloc[1000:].reset_index(drop=True)

    event = detect_label_shift(reference_df, current_df, _TARGET_COLUMN)

    assert event is None


def test_missing_target_column_returns_none():

    df = _classification_df()
    reference_df = df.iloc[:1000].reset_index(drop=True)
    current_df = df.iloc[1000:].reset_index(drop=True).drop(columns=[_TARGET_COLUMN])

    event = detect_label_shift(reference_df, current_df, _TARGET_COLUMN)

    assert event is None


def test_too_few_rows_returns_none():

    reference_df = pd.DataFrame({
        "feature_0": [1.0, 2.0, 3.0],
        _TARGET_COLUMN: [0, 1, 0],
    })
    current_df = pd.DataFrame({
        "feature_0": [1.0, 2.0, 3.0],
        _TARGET_COLUMN: [0, 1, 0],
    })

    event = detect_label_shift(reference_df, current_df, _TARGET_COLUMN)

    assert event is None


def test_single_class_reference_returns_none():

    rng = np.random.RandomState(42)
    reference_df = pd.DataFrame({
        "feature_0": rng.normal(0, 1, 50),
        _TARGET_COLUMN: [1] * 50,
    })
    current_df = pd.DataFrame({
        "feature_0": rng.normal(0, 1, 50),
        _TARGET_COLUMN: [1] * 25 + [0] * 25,
    })

    event = detect_label_shift(reference_df, current_df, _TARGET_COLUMN)

    assert event is None
