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