import pandas as pd
import pytest

from detection.fault_injection import (
    inject_duplicate_rows,
    inject_feature_drift,
    inject_missing_values,
    inject_schema_mismatch,
)
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


def test_missing_values_injection():

    df = pd.DataFrame({
        "age": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
    })

    injected_df, label = inject_missing_values(
        df,
        "age",
        fraction=0.3,
        random_seed=42,
    )

    assert injected_df["age"].isna().sum() == 3
    assert label == FailureType.MISSING_VALUES
    # Original untouched.
    assert df["age"].isna().sum() == 0


def test_missing_values_injection_reproducibility():

    df = pd.DataFrame({
        "age": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
    })

    df1, _ = inject_missing_values(df, "age", fraction=0.3, random_seed=42)
    df2, _ = inject_missing_values(df, "age", fraction=0.3, random_seed=42)

    pd.testing.assert_frame_equal(df1, df2)


def test_missing_values_injection_missing_column_raises():

    df = pd.DataFrame({"age": [20, 22]})

    with pytest.raises(ValueError):
        inject_missing_values(df, "does_not_exist", fraction=0.5)


def test_missing_values_injection_invalid_fraction_raises():

    df = pd.DataFrame({"age": [20, 22]})

    with pytest.raises(ValueError):
        inject_missing_values(df, "age", fraction=1.5)


def test_duplicate_rows_injection():

    df = pd.DataFrame({
        "age": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
    })

    injected_df, label = inject_duplicate_rows(
        df,
        fraction=0.5,
        random_seed=42,
    )

    assert len(injected_df) == len(df) + 5
    # Every duplicated row's values must exist somewhere in the original.
    assert injected_df["age"].iloc[len(df):].isin(df["age"]).all()
    assert label == FailureType.DUPLICATES
    assert len(df) == 10  # original untouched


def test_duplicate_rows_injection_reproducibility():

    df = pd.DataFrame({
        "age": [20, 22, 24, 26, 28, 30, 32, 34, 36, 38],
    })

    df1, _ = inject_duplicate_rows(df, fraction=0.4, random_seed=42)
    df2, _ = inject_duplicate_rows(df, fraction=0.4, random_seed=42)

    pd.testing.assert_frame_equal(df1, df2)


def test_duplicate_rows_injection_invalid_fraction_raises():

    df = pd.DataFrame({"age": [20, 22]})

    with pytest.raises(ValueError):
        inject_duplicate_rows(df, fraction=-0.1)


def test_schema_mismatch_injection_drop():

    df = pd.DataFrame({
        "age": [20, 22, 24],
        "income": [0, 1, 0],
    })

    injected_df, label = inject_schema_mismatch(df, "age", mode="drop")

    assert "age" not in injected_df.columns
    assert "income" in injected_df.columns
    assert label == FailureType.SCHEMA_MISMATCH
    assert "age" in df.columns  # original untouched


def test_schema_mismatch_injection_retype():

    df = pd.DataFrame({
        "age": [20, 22, 24],
    })

    injected_df, label = inject_schema_mismatch(df, "age", mode="retype")

    assert injected_df["age"].dtype == object
    assert injected_df["age"].tolist() == ["20", "22", "24"]
    assert label == FailureType.SCHEMA_MISMATCH
    assert df["age"].dtype != object  # original untouched


def test_schema_mismatch_injection_missing_column_raises():

    df = pd.DataFrame({"age": [20, 22]})

    with pytest.raises(ValueError):
        inject_schema_mismatch(df, "does_not_exist")


def test_schema_mismatch_injection_invalid_mode_raises():

    df = pd.DataFrame({"age": [20, 22]})

    with pytest.raises(ValueError):
        inject_schema_mismatch(df, "age", mode="not_a_real_mode")
