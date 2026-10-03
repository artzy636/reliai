from datetime import datetime, timezone

import pandas as pd

from configs.settings import DETECTION
from detection.confidence import effect_confidence
from schemas import (
    DetectionMethod,
    EvidenceEvent,
)


def detect_duplicate_rows(
    reference: pd.DataFrame,
    current: pd.DataFrame,
):
    """
    Detect an elevated duplicate-row rate in current data.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference dataset (training data). Not used in the metric itself --
        the check is purely against current data's duplicate-row rate --
        but accepted to match the (reference, current) calling convention
        shared by every detector in this package.

    current : pd.DataFrame
        Current dataset (production data).

    Returns
    -------
    EvidenceEvent | None
        Returns an EvidenceEvent if current's duplicate-row rate exceeds
        the configured threshold, otherwise returns None. Pipeline-level
        check: the returned event's feature_name is always None.
    """

    threshold = DETECTION.duplicate_row_rate

    duplicate_rate = current.duplicated().mean()

    if duplicate_rate <= threshold:
        return None

    confidence = effect_confidence(duplicate_rate, DETECTION.duplicate_row_rate_full_scale)

    return EvidenceEvent(
        timestamp=datetime.now(timezone.utc),
        detection_method=DetectionMethod.DUPLICATE_ROW_RATE,
        feature_name=None,
        metric_value=duplicate_rate,
        threshold=threshold,
        confidence=confidence,
        description=(
            f"Duplicate-rows detector flagged elevated duplicate-row rate "
            f"(duplicate_rate={duplicate_rate:.4f}, threshold={threshold:.4f})"
        ),
    )
