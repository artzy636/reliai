from datetime import datetime, timezone

import pandas as pd
from sklearn.ensemble import IsolationForest

from configs.settings import DETECTION
from schemas import (
    DetectionMethod,
    EvidenceEvent,
)

# IsolationForest needs at least this many reference rows to fit a
# meaningful model; below that, "normal" isn't well-defined enough to
# score anomalies against.
_MIN_REFERENCE_ROWS = 10


def detect_corrupted_values(
    reference: pd.DataFrame,
    current: pd.DataFrame,
):
    """
    Detect rows in current data that look structurally anomalous relative
    to reference data, using an Isolation Forest fit on reference's shared
    numeric columns and scored against current's.

    This is deliberately multivariate and row-level, unlike
    detect_distribution_shift (ks_detector.py), which tests one column's
    overall distribution at a time. A handful of corrupted rows scattered
    through an otherwise-normal column barely moves a whole-distribution
    KS statistic -- but each corrupted row's combination of feature
    values can still look nothing like anything in reference, which is
    exactly what an isolation forest is built to flag.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference dataset (training data).

    current : pd.DataFrame
        Current dataset (production data).

    Returns
    -------
    EvidenceEvent | None
        Pipeline-level (feature_name=None) event if current's anomaly
        rate, per a model fit on reference, exceeds the model's own
        contamination baseline by more than the configured tolerance.
        Returns None if there are no shared numeric columns, too few
        reference rows to fit a model, or no current rows to score.

    Notes
    -----
    contamination is IsolationForest's assumed anomaly rate in the data
    it's fit on -- by construction, a model fit with contamination=0.05
    will flag close to 5% of its OWN reference data as anomalies, even
    when reference is entirely clean. That's a built-in false-positive
    rate, not a finding. Only a current-data anomaly rate meaningfully
    *above* that baseline indicates a real, pipeline-detectable
    corruption -- so the flagging threshold is the baseline PLUS
    DETECTION.isolation_forest_anomaly_rate_margin, not the raw baseline
    itself. Without that margin, scoring current data that is genuinely
    clean but merely an independent sample from the same distribution as
    reference (never bit-for-bit identical to it) produces an anomaly
    rate that lands on either side of the raw baseline purely from
    sampling noise -- a real, reproduced flaky false positive, not a
    hypothetical one.
    """

    shared_numeric_columns = [
        column
        for column in reference.columns
        if column in current.columns
        and pd.api.types.is_numeric_dtype(reference[column])
        and pd.api.types.is_numeric_dtype(current[column])
    ]

    if not shared_numeric_columns:
        return None

    reference_features = reference[shared_numeric_columns].dropna()
    current_features = current[shared_numeric_columns].dropna()

    if len(reference_features) < _MIN_REFERENCE_ROWS or len(current_features) == 0:
        return None

    contamination = DETECTION.isolation_forest_contamination

    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(reference_features)

    predictions = model.predict(current_features)  # -1 = anomaly, 1 = normal
    anomaly_rate = (predictions == -1).mean()

    threshold = contamination + DETECTION.isolation_forest_anomaly_rate_margin

    if anomaly_rate <= threshold:
        return None

    confidence = max(
        0.0,
        min(1.0, (anomaly_rate - threshold) / (1 - threshold))
    )

    return EvidenceEvent(
        timestamp=datetime.now(timezone.utc),
        detection_method=DetectionMethod.ISOLATION_FOREST,
        feature_name=None,
        metric_value=anomaly_rate,
        threshold=threshold,
        confidence=confidence,
        description=(
            f"Isolation-forest detector flagged elevated anomaly rate in "
            f"current data relative to reference "
            f"(anomaly_rate={anomaly_rate:.4f}, "
            f"contamination_baseline={contamination:.4f})"
        ),
    )
