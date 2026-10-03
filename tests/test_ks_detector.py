import numpy as np

from detection.ks_detector import detect_distribution_shift


def test_distribution_shift_detected():

    np.random.seed(42)

    reference = np.random.normal(0, 1, 1000)

    current = np.random.normal(5, 1, 1000)

    event = detect_distribution_shift(
        reference,
        current,
        feature_name="age",
    )

    assert event is not None
    assert event.confidence > 0.8


def test_no_distribution_shift():

    np.random.seed(42)

    reference = np.random.normal(0, 1, 1000)

    current = np.random.normal(0, 1, 1000)

    event = detect_distribution_shift(
        reference,
        current,
        feature_name="age",
    )

    assert event is None


def test_nan_in_current_does_not_produce_a_spurious_max_confidence_event():
    # Regression test: scipy.stats.ks_2samp returns statistic=nan,
    # pvalue=nan when either input contains NaN, and the old
    # `if pvalue >= threshold: return None` guard did not catch that
    # (nan >= threshold is always False), so it fell through and computed
    # confidence=1.0 for a result that was actually undefined -- a real,
    # reproduced false positive on any column with missing values.
    np.random.seed(42)

    reference = np.random.normal(0, 1, 1000)

    current = np.random.normal(0, 1, 1000)
    current[::5] = np.nan  # 20% missing, same distribution otherwise

    event = detect_distribution_shift(
        reference,
        current,
        feature_name="age",
    )

    # Same underlying distribution among the observed (non-NaN) values,
    # so no real shift should be reported -- not a NaN-driven false
    # maximum-confidence event.
    assert event is None


def test_nan_in_current_does_not_mask_a_real_shift():
    np.random.seed(42)

    reference = np.random.normal(0, 1, 1000)

    current = np.random.normal(5, 1, 1000)
    current[::5] = np.nan  # 20% missing, but the observed values do drift

    event = detect_distribution_shift(
        reference,
        current,
        feature_name="age",
    )

    assert event is not None
    assert event.confidence > 0.8


def test_all_nan_current_returns_none_instead_of_crashing():
    np.random.seed(42)

    reference = np.random.normal(0, 1, 100)

    current = np.full(100, np.nan)

    event = detect_distribution_shift(
        reference,
        current,
        feature_name="age",
    )

    assert event is None

def test_confidence_tracks_effect_size_not_sample_size():
    """A statistically significant but tiny shift on a large sample used to
    saturate at confidence 1.0 (1 - p/alpha). Confidence must now reflect
    the KS statistic D, so a tiny shift is low-confidence and a large shift
    is high-confidence."""
    rng = np.random.RandomState(0)

    reference = rng.normal(0, 1, 50_000)
    tiny_shift = rng.normal(0.05, 1, 50_000)   # D ~ 0.02, but p << 0.05 at this n
    big_shift = rng.normal(1.0, 1, 50_000)     # D ~ 0.38

    tiny = detect_distribution_shift(reference, tiny_shift, feature_name="x")
    big = detect_distribution_shift(reference, big_shift, feature_name="x")

    assert tiny is not None and tiny.metric_value < 0.05
    assert tiny.confidence < 0.2
    assert big is not None
    assert big.confidence > 0.9
    assert big.confidence < 1.0  # statistical estimates never reach certainty
