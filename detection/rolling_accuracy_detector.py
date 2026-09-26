from datetime import datetime, timezone

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from configs.settings import DETECTION
from schemas import (
    DetectionMethod,
    EvidenceEvent,
)

# Below this many rows, a 5-fold cross-validated accuracy estimate is too
# noisy to trust as a baseline.
_MIN_ROWS_FOR_ROLLING_ACCURACY = 20
_CV_FOLDS = 5


def detect_label_shift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    target_column: str,
):
    """
    Detect label shift by comparing a baseline classifier's accuracy on
    reference data against its accuracy on current data's ACTUAL labels.

    Every other detector in this package only needs feature columns --
    this one needs target_column's real, current ground-truth values too
    (not the injected-fault ground_truth_label CONTRACT.md rule 4 keeps
    out of the reasoning layer; a genuine current label, the same way a
    production monitor would eventually observe the true outcome for a
    past prediction). Without it there is nothing to measure "rolling
    accuracy" against.

    Parameters
    ----------
    reference : pd.DataFrame
        Reference dataset (training data), including target_column.

    current : pd.DataFrame
        Current dataset (production data), including target_column with
        its real observed values.

    target_column : str
        Name of the binary label column, present in both reference and
        current.

    Returns
    -------
    EvidenceEvent | None
        Pipeline-level (feature_name=None) event if current's accuracy,
        using a model fit on reference, drops by more than
        DETECTION.rolling_accuracy_drop_pct relative to reference's own
        cross-validated baseline accuracy. Returns None if target_column
        isn't present on both sides, there are too few usable rows, or
        reference's target has fewer than two classes to fit against.

    Notes
    -----
    A drop in raw accuracy from label shift alone (P(y) changing, with
    each retained row's feature-label relationship untouched) only shows
    up when the baseline classifier isn't already near-perfect: a
    well-separated problem's accuracy is close to invariant to class
    balance, since a confident, mostly-correct model's error rate barely
    changes as the mix of classes shifts. A weaker or more realistic
    classifier -- verified directly against this project's real Adult-
    income baseline, ~0.81 accuracy -- shows a large, real drop (roughly
    0.82 -> 0.61 when the positive rate is shifted from ~24% to 60%) from
    exactly this cause. That's expected, not a limitation to work around:
    it mirrors why label shift matters in production at all -- it hurts
    the models that were already imperfect, which is the normal case.
    """

    if target_column not in reference.columns or target_column not in current.columns:
        return None

    feature_columns = [
        column
        for column in reference.columns
        if column != target_column
        and column in current.columns
        and pd.api.types.is_numeric_dtype(reference[column])
        and pd.api.types.is_numeric_dtype(current[column])
    ]

    if not feature_columns:
        return None

    reference_clean = reference[feature_columns + [target_column]].dropna()
    current_clean = current[feature_columns + [target_column]].dropna()

    if (
        len(reference_clean) < _MIN_ROWS_FOR_ROLLING_ACCURACY
        or len(current_clean) < _MIN_ROWS_FOR_ROLLING_ACCURACY
        or reference_clean[target_column].nunique() < 2
    ):
        return None

    reference_features = reference_clean[feature_columns]
    reference_target = reference_clean[target_column]

    model = LogisticRegression(max_iter=1000)

    # Baseline: what accuracy this model architecture actually achieves on
    # reference's OWN distribution, measured honestly via cross-validation
    # rather than by scoring the same rows it was fit on (which would
    # overstate it).
    baseline_accuracy = cross_val_score(
        model, reference_features, reference_target, cv=_CV_FOLDS
    ).mean()

    # Current: fit on the full reference set, score against current's
    # ACTUAL labels -- this is "rolling accuracy" in the monitoring sense.
    model.fit(reference_features, reference_target)
    current_accuracy = model.score(
        current_clean[feature_columns], current_clean[target_column]
    )

    drop = baseline_accuracy - current_accuracy
    threshold = DETECTION.rolling_accuracy_drop_pct

    if drop <= threshold:
        return None

    confidence = max(
        0.0,
        min(1.0, (drop - threshold) / (1 - threshold))
    )

    return EvidenceEvent(
        timestamp=datetime.now(timezone.utc),
        detection_method=DetectionMethod.ROLLING_ACCURACY,
        feature_name=None,
        metric_value=current_accuracy,
        threshold=threshold,
        confidence=confidence,
        description=(
            f"Rolling-accuracy detector flagged a drop in prediction accuracy "
            f"(baseline={baseline_accuracy:.4f}, current={current_accuracy:.4f}, "
            f"drop={drop:.4f} > threshold={threshold:.4f})"
        ),
    )
