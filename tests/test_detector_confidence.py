import numpy as np
import pandas as pd
import pytest

from detection.confidence import MAX_STATISTICAL_CONFIDENCE, effect_confidence
from detection.duplicate_rows_detector import detect_duplicate_rows
from detection.isolation_forest_detector import detect_corrupted_values
from detection.missing_values_detector import detect_missing_values


def test_effect_confidence_is_linear_ramp_clamped_to_cap():
    assert effect_confidence(0.0, 0.3) == 0.0
    assert effect_confidence(0.15, 0.3) == pytest.approx(MAX_STATISTICAL_CONFIDENCE / 2)
    assert effect_confidence(0.3, 0.3) == pytest.approx(MAX_STATISTICAL_CONFIDENCE)
    assert effect_confidence(10.0, 0.3) == pytest.approx(MAX_STATISTICAL_CONFIDENCE)
    assert effect_confidence(-1.0, 0.3) == 0.0


def test_effect_confidence_nan_is_never_evidence():
    assert effect_confidence(float("nan"), 0.3) == 0.0


def test_effect_confidence_rejects_nonpositive_full_scale():
    with pytest.raises(ValueError):
        effect_confidence(0.1, 0.0)


def test_missing_rate_confidence_is_monotonic_in_null_rate():
    rng = np.random.RandomState(0)
    reference = pd.Series(rng.normal(0, 1, 1000))

    def conf(null_fraction):
        current = pd.Series(rng.normal(0, 1, 1000))
        current.iloc[: int(1000 * null_fraction)] = np.nan
        event = detect_missing_values(reference, current, feature_name="x")
        return event.confidence

    assert conf(0.15) < conf(0.30) < conf(0.60)


def test_duplicate_confidence_is_monotonic_in_duplicate_rate():
    rng = np.random.RandomState(0)
    reference = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})

    def conf(n_dups):
        current = pd.DataFrame({"a": rng.normal(size=500), "b": rng.normal(size=500)})
        current.iloc[:n_dups] = current.iloc[0:1].values
        return detect_duplicate_rows(reference, current).confidence

    assert conf(40) < conf(100) < conf(200)


def test_isolation_forest_confidence_scales_with_anomaly_rate_over_baseline():
    rng = np.random.RandomState(42)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(0, 1, 5000)})

    def conf(fraction):
        current = pd.DataFrame({"a": rng.normal(0, 1, 5000), "b": rng.normal(0, 1, 5000)})
        idx = rng.choice(5000, size=int(5000 * fraction), replace=False)
        current.loc[idx, ["a", "b"]] *= 20
        event = detect_corrupted_values(reference, current)
        return event.confidence if event else 0.0

    assert conf(0.0) == 0.0
    assert 0.0 < conf(0.08) < conf(0.2)


def test_isolation_forest_margin_is_tighter_for_large_samples():
    """The fixed +0.05 margin hid a real ~2x anomaly-rate increase on large
    samples; the margin now shrinks with n (but never exceeds the old 0.05)."""
    rng = np.random.RandomState(1)
    n = 20_000
    reference = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    current = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
    idx = rng.choice(n, size=int(n * 0.05), replace=False)  # 5% corrupted -> ~10% flagged
    current.loc[idx, ["a", "b"]] *= 20

    event = detect_corrupted_values(reference, current)

    assert event is not None
    assert event.threshold < 0.10  # old fixed threshold would have been 0.10
    assert event.threshold >= 0.05 + 0.02  # floor
