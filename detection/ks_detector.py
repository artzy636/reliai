from datetime import datetime, timezone

from scipy.stats import ks_2samp

from configs.settings import DETECTION
from schemas import (
    DetectionMethod,
    EvidenceEvent,
)


def detect_distribution_shift(
    reference,
    current,
    feature_name: str | None = None,
):
    """
    Detect distribution shift using the Kolmogorov-Smirnov (KS) test.

    Parameters
    ----------
    reference : array-like
        Reference distribution (training data).

    current : array-like
        Current distribution (production data).

    feature_name : str | None
        Name of the feature being tested.

    Returns
    -------
    EvidenceEvent | None
        Returns an EvidenceEvent if a statistically significant
        distribution shift is detected, otherwise returns None.
    """

    statistic, pvalue = ks_2samp(reference, current)

    threshold = DETECTION.ks_test_pvalue

    if pvalue >= threshold:
        return None

    confidence = max(
        0.0,
        min(1.0, 1 - (pvalue / threshold))
    )

    return EvidenceEvent(
        timestamp=datetime.now(timezone.utc),
        detection_method=DetectionMethod.KS_TEST,
        feature_name=feature_name,
        metric_value=statistic,
        threshold=threshold,
        confidence=confidence,
        description=(
            f"KS Test detected distribution shift "
            f"(D={statistic:.4f}, p={pvalue:.6f})"
        ),
    )
