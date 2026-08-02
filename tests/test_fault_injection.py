import pandas as pd

from detection.fault_injection import inject_feature_drift
from schemas import FailureType


def test_feature_drift_injection():

    df = pd.DataFrame({
        "age": [20, 22, 24, 26]
    })

    shifted_df, label = inject_feature_drift(
        df,
        "age",
        shift_amount=10,
        random_seed=42,
    )

    assert shifted_df["age"].tolist() == [30, 32, 34, 36]
    assert label == FailureType.FEATURE_DRIFT


def test_reproducibility():

    df = pd.DataFrame({
        "age": [20, 22, 24, 26]
    })

    df1, _ = inject_feature_drift(df, "age", 10, random_seed=42)
    df2, _ = inject_feature_drift(df, "age", 10, random_seed=42)

    assert df1.equals(df2)