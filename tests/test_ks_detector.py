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