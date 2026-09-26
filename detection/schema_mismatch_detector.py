from datetime import datetime, timezone

import pandas as pd

from configs.settings import DETECTION
from schemas import (
    DetectionMethod,
    EvidenceEvent,
)


def detect_schema_mismatch(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    feature_name: str,
):
    """
    Detect a column that is present in reference data but missing from
    current data, or whose dtype changed between the two.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference dataset (training data).

    current : pd.DataFrame
        Current dataset (production data).

    feature_name : str
        Column (expected to be present in reference) being checked.

    Returns
    -------
    EvidenceEvent | None
        Returns an EvidenceEvent if feature_name is missing from current,
        or its dtype changed relative to reference, otherwise returns None.
    """

    threshold = DETECTION.schema_mismatch_tolerance

    if feature_name not in current.columns:
        return EvidenceEvent(
            timestamp=datetime.now(timezone.utc),
            detection_method=DetectionMethod.SCHEMA_CHECK,
            feature_name=feature_name,
            metric_value=1.0,
            threshold=threshold,
            confidence=1.0,
            description=(
                f"Schema-mismatch detector flagged column '{feature_name}' "
                f"present in reference data but missing from current data"
            ),
        )

    reference_dtype = reference[feature_name].dtype
    current_dtype = current[feature_name].dtype

    if reference_dtype == current_dtype:
        return None

    return EvidenceEvent(
        timestamp=datetime.now(timezone.utc),
        detection_method=DetectionMethod.SCHEMA_CHECK,
        feature_name=feature_name,
        metric_value=1.0,
        threshold=threshold,
        confidence=1.0,
        description=(
            f"Schema-mismatch detector flagged column '{feature_name}' "
            f"dtype changed from {reference_dtype} to {current_dtype}"
        ),
    )
