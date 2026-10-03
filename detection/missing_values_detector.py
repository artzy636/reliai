from datetime import datetime, timezone

import pandas as pd

from configs.settings import DETECTION
from schemas import (
    DetectionMethod,
    EvidenceEvent,
)


def detect_missing_values(
    reference,
    current,
    feature_name: str | None = None,
):
    """
    Detect an elevated null rate in a column of current data.

    Parameters
    ----------
    reference : array-like
        Reference distribution (training data). Not used in the metric
        itself -- the check is purely against current data's null rate --
        but accepted to match the (reference, current, feature_name) calling
        convention shared by every detector in this package.

    current : array-like
        Current distribution (production data).

    feature_name : str | None
        Name of the feature being tested.

    Returns
    -------
    EvidenceEvent | None
        Returns an EvidenceEvent if current's null rate exceeds the
        configured threshold, otherwise returns None.
    """

    current = pd.Series(current)

    threshold = DETECTION.missing_value_rate

    if len(current) == 0:
        # An empty column has an undefined (NaN) null rate. NaN slips past
        # `null_rate <= threshold` below (a NaN comparison is always False
        # in Python/NumPy) and would otherwise reach the confidence
        # calculation, where min()/max() resolve a NaN-vs-number comparison
        # to the number -- producing a spurious confidence=1.0 "missing
        # values detected" event for a column with nothing in it. Same
        # failure mode already documented and fixed for the KS test in
        # detection/ks_detector.py.
        return None

    null_rate = current.isna().mean()

    if null_rate <= threshold:
        return None

    confidence = max(
        0.0,
        min(1.0, (null_rate - threshold) / (1 - threshold))
    )

    return EvidenceEvent(
        timestamp=datetime.now(timezone.utc),
        detection_method=DetectionMethod.MISSING_VALUE_RATE,
        feature_name=feature_name,
        metric_value=null_rate,
        threshold=threshold,
        confidence=confidence,
        description=(
            f"Missing-values detector flagged elevated null rate "
            f"(null_rate={null_rate:.4f}, threshold={threshold:.4f})"
        ),
    )
