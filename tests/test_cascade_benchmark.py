from datetime import datetime, timezone

import pandas as pd
import pytest

from evaluation import cascade_benchmark as cb
from schemas import DetectionMethod, EvidenceEvent


def _event(method=DetectionMethod.KS_TEST, feature="age", confidence=0.5, metric=0.2):
    return EvidenceEvent(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        detection_method=method,
        feature_name=feature,
        metric_value=metric,
        threshold=0.05,
        confidence=confidence,
        description="synthetic",
    )


def test_onset_is_first_time_of_a_persistent_run():
    by_key = {("ks_test", "age"): {2: _event(confidence=0.4), 3: _event(confidence=0.6), 4: _event(confidence=0.8)}}

    (event,) = cb.collapse_to_onset_events(by_key, persistence=2)

    assert cb._onset_window(event) == 2
    assert event.confidence == pytest.approx(0.6)  # mean over onset..end


def test_one_off_firing_does_not_persist_and_yields_no_event():
    by_key = {("ks_test", "age"): {3: _event()}, ("ks_test", "hours"): {1: _event(), 3: _event()}}

    assert cb.collapse_to_onset_events(by_key, persistence=2) == []


def test_noise_blip_before_the_real_run_does_not_set_the_onset():
    by_key = {("ks_test", "age"): {0: _event(), 4: _event(), 5: _event()}}

    (event,) = cb.collapse_to_onset_events(by_key, persistence=2)

    assert cb._onset_window(event) == 4


def test_label_dependent_events_are_rolling_accuracy_and_target_ks_only():
    assert cb._is_label_dependent(_event(DetectionMethod.ROLLING_ACCURACY, None))
    assert cb._is_label_dependent(_event(DetectionMethod.KS_TEST, cb.TARGET_COLUMN))
    assert not cb._is_label_dependent(_event(DetectionMethod.KS_TEST, "age"))
    assert not cb._is_label_dependent(_event(DetectionMethod.DUPLICATE_ROW_RATE, None))


def test_label_prior_shift_hits_target_rate_without_creating_duplicates():
    df = pd.DataFrame({"x": range(1000), cb.TARGET_COLUMN: [1] * 240 + [0] * 760})

    out = cb._shift_label_prior(df, 0.5, seed=0)

    assert out[cb.TARGET_COLUMN].mean() == pytest.approx(0.5, abs=0.01)
    assert not out.duplicated().any()


def test_cases_have_unique_names_and_valid_onsets():
    cases = cb._cases()
    assert len({c.name for c in cases}) == len(cases)
    # onset must leave room for PERSISTENCE windows before the stream ends
    assert all(0 < c.onset_window <= cb.N_WINDOWS - cb.PERSISTENCE - 1 for c in cases)


def test_structure_report_detects_downstream_edge_from_true_root():
    from datetime import timedelta

    root = _event(DetectionMethod.KS_TEST, "age", confidence=0.9).model_copy(
        update={"ground_truth_label": "feature_drift"}
    )
    symptom = _event(DetectionMethod.ROLLING_ACCURACY, None, confidence=0.7).model_copy(
        update={"timestamp": root.timestamp + timedelta(minutes=20)}
    )

    report = cb.structure_report({"case": [root, symptom]}, min_edge_confidence=0.3)

    assert report["case"]["root_has_downstream"] is True
    assert report["case"]["root_has_incoming"] is False
    assert report["case"]["n_edges"] == 1
