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
        or its dtype *kind* changed relative to reference (e.g. numeric
        became text, or text became numeric), otherwise returns None.

    Notes
    -----
    A numeric-to-numeric dtype change (e.g. int64 -> float64) is NOT
    treated as a schema mismatch. Pandas silently upcasts an integer
    column to float the moment it gains a NaN or is combined with a
    float value in arithmetic -- both completely routine, and not this
    detector's concern (a NaN-driven upcast is missing_values_detector.py's
    job; the values themselves are still numeric and comparable either
    way). Flagging that upcast as a schema mismatch produced spurious
    SCHEMA_CHECK findings on columns with no actual structural problem --
    confirmed on the real Adult-dataset incident
    (evaluation/real_data_incident.py): injecting a plain float shift into
    the int64 hours-per-week column upcast it to float64, and injecting
    missing values into capital-loss did the same, so both picked up a
    schema-mismatch finding that had nothing to do with the schema and
    everything to do with numpy's type-promotion rules. A schema
    mismatch, in the sense this detector should catch, is a change in
    *kind* (numeric vs. text vs. boolean, etc.) -- an incompatibility a
    downstream numeric pipeline would actually choke on -- not a
    same-kind width/precision change.
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

    if pd.api.types.is_numeric_dtype(reference_dtype) and pd.api.types.is_numeric_dtype(current_dtype):
        # Same kind (numeric), different width/precision -- a benign
        # promotion, not a schema mismatch. See the docstring above.
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
