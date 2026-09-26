import numpy as np
import pandas as pd

from schemas import FailureType


def inject_feature_drift(
    df: pd.DataFrame,
    column_name: str,
    shift_amount: float,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, FailureType]:
    """
    Inject feature drift into a numeric column by shifting its values.

    Parameters
    ----------
    df : pd.DataFrame
        Original clean dataset.

    column_name : str
        Column to inject drift into.

    shift_amount : float
        Amount by which to shift the feature values.

    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, FailureType]
        Modified dataset and its ground-truth fault label.
    """

    np.random.seed(random_seed)

    injected_df = df.copy()

    if column_name not in injected_df.columns:
        raise ValueError(f"Column '{column_name}' does not exist.")

    if not np.issubdtype(injected_df[column_name].dtype, np.number):
        raise ValueError(f"Column '{column_name}' must be numeric.")

    injected_df[column_name] = injected_df[column_name] + shift_amount

    return injected_df, FailureType.FEATURE_DRIFT


def inject_missing_values(
    df: pd.DataFrame,
    column_name: str,
    fraction: float,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, FailureType]:
    """
    Inject missing values into a column by nulling out a random fraction
    of its rows.

    Parameters
    ----------
    df : pd.DataFrame
        Original clean dataset.

    column_name : str
        Column to inject missing values into.

    fraction : float
        Fraction of rows (0-1) to set to NaN.

    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, FailureType]
        Modified dataset and its ground-truth fault label.
    """

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist.")

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}.")

    rng = np.random.RandomState(random_seed)

    injected_df = df.copy()

    n_to_null = int(round(fraction * len(injected_df)))
    if n_to_null > 0:
        null_positions = rng.choice(len(injected_df), size=n_to_null, replace=False)
        injected_df.iloc[null_positions, injected_df.columns.get_loc(column_name)] = np.nan

    return injected_df, FailureType.MISSING_VALUES


def inject_duplicate_rows(
    df: pd.DataFrame,
    fraction: float,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, FailureType]:
    """
    Inject duplicate rows by re-sampling a random fraction of existing rows
    and appending them, so the returned dataset contains every original
    row plus extra copies of some of them.

    Parameters
    ----------
    df : pd.DataFrame
        Original clean dataset.

    fraction : float
        Fraction of df's row count (0-1) to duplicate and append.

    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, FailureType]
        Modified dataset (original rows followed by duplicated rows) and
        its ground-truth fault label.
    """

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}.")

    n_to_duplicate = int(round(fraction * len(df)))

    if n_to_duplicate > 0:
        duplicated_rows = df.sample(n=n_to_duplicate, random_state=random_seed, replace=True)
        injected_df = pd.concat([df, duplicated_rows], ignore_index=True)
    else:
        injected_df = df.copy()

    return injected_df, FailureType.DUPLICATES


def inject_corrupted_values(
    df: pd.DataFrame,
    column_name: str,
    fraction: float,
    corruption_multiplier: float = 50.0,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, FailureType]:
    """
    Corrupt a random fraction of a numeric column's individual values by
    scaling them to extreme outliers (e.g. a sensor glitch or a unit-
    conversion bug), leaving every other row untouched.

    Unlike inject_feature_drift, which shifts EVERY value in a column and
    so moves its overall distribution, this only touches a small fraction
    of individual rows -- exactly the case a whole-distribution test like
    the KS test can miss (a handful of outliers among thousands of normal
    rows barely moves the KS statistic), but a per-row multivariate check
    (isolation forest) is built to catch.

    Parameters
    ----------
    df : pd.DataFrame
        Original clean dataset.

    column_name : str
        Numeric column to corrupt.

    fraction : float
        Fraction of rows (0-1) to corrupt.

    corruption_multiplier : float
        Factor the corrupted rows' values are scaled by, to push them well
        outside the column's normal range.

    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, FailureType]
        Modified dataset and its ground-truth fault label.
    """

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist.")

    if not np.issubdtype(df[column_name].dtype, np.number):
        raise ValueError(f"Column '{column_name}' must be numeric.")

    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction must be between 0 and 1, got {fraction}.")

    rng = np.random.RandomState(random_seed)

    injected_df = df.copy()

    n_to_corrupt = int(round(fraction * len(injected_df)))
    if n_to_corrupt > 0:
        corrupt_positions = rng.choice(len(injected_df), size=n_to_corrupt, replace=False)
        column_index = injected_df.columns.get_loc(column_name)
        injected_df.iloc[corrupt_positions, column_index] = (
            injected_df.iloc[corrupt_positions, column_index] * corruption_multiplier
        )

    return injected_df, FailureType.CORRUPTED_VALUES


def inject_schema_mismatch(
    df: pd.DataFrame,
    column_name: str,
    mode: str = "drop",
) -> tuple[pd.DataFrame, FailureType]:
    """
    Inject a schema mismatch, either by dropping a column entirely (the
    upstream pipeline stopped producing it) or by changing its dtype (the
    upstream pipeline started emitting it as a different type).

    Parameters
    ----------
    df : pd.DataFrame
        Original clean dataset.

    column_name : str
        Column to remove or retype.

    mode : str
        "drop" removes the column outright. "retype" casts it to string,
        simulating a pipeline change that starts emitting a numeric column
        as text. Defaults to "drop".

    Returns
    -------
    tuple[pd.DataFrame, FailureType]
        Modified dataset and its ground-truth fault label.
    """

    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist.")

    if mode not in ("drop", "retype"):
        raise ValueError(f"mode must be 'drop' or 'retype', got {mode!r}.")

    injected_df = df.copy()

    if mode == "drop":
        injected_df = injected_df.drop(columns=[column_name])
    else:
        injected_df[column_name] = injected_df[column_name].astype(str)

    return injected_df, FailureType.SCHEMA_MISMATCH
