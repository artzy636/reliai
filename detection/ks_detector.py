from datetime import datetime, timezone

import pandas as pd
from scipy.stats import ks_2samp

from configs.settings import DETECTION
from schemas import (
    DetectionMethod,
    EvidenceEvent,
)

# ks_2samp needs at least one observation per side to produce a defined
# statistic; below that there is nothing to test.
_MIN_OBSERVATIONS_FOR_KS_TEST = 1


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

    Notes
    -----
    NaNs are dropped from both samples before the test runs.
    ``scipy.stats.ks_2samp`` does not drop them itself: if either input
    contains any NaN, it silently returns ``statistic=nan, pvalue=nan``.
    The old code's guard, ``if pvalue >= threshold: return None``, does
    NOT catch that case -- ``nan >= threshold`` is always False in
    Python/NumPy, so a NaN p-value fell through to the confidence
    calculation instead of being treated as "no result." That calculation,
    ``max(0.0, min(1.0, 1 - (pvalue / threshold)))``, then collapses to
    exactly 1.0 for a NaN pvalue, because Python's min()/max() resolve a
    NaN-vs-number comparison to the number. The net effect: any column
    with missing values produced a maximum-confidence "distribution shift
    detected" event with no real statistical basis. This was not
    hypothetical -- it reproduced on the real Adult-dataset incident
    (evaluation/real_data_incident.py): injecting missing values into
    capital-loss (unrelated to feature drift) also produced a
    confidence=1.0 KS_TEST event whose description read "D=nan, p=nan".
    Whether a column *has* missing values is the missing-value-rate
    detector's job (detection/missing_values_detector.py); this detector
    should only judge drift among the values that are actually present.
    """

    reference_observed = pd.Series(reference).dropna().to_numpy()
    current_observed = pd.Series(current).dropna().to_numpy()

    threshold = DETECTION.ks_test_pvalue

    if (
        len(reference_observed) < _MIN_OBSERVATIONS_FOR_KS_TEST
        or len(current_observed) < _MIN_OBSERVATIONS_FOR_KS_TEST
    ):
        return None

    statistic, pvalue = ks_2samp(reference_observed, current_observed)

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
