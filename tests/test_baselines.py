from datetime import datetime, timedelta, timezone

from evaluation.baselines import confidence_sort_rank, earliest_first_rank, summarize
from schemas import DetectionMethod, EvidenceEvent, FailureType


def _event(conf, true=False, minutes=0):
    return EvidenceEvent(
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(minutes=minutes),
        detection_method=DetectionMethod.KS_TEST,
        feature_name="x",
        metric_value=0.1,
        threshold=0.05,
        confidence=conf,
        description="d",
        ground_truth_label=FailureType.FEATURE_DRIFT if true else None,
    )


def test_rank_is_one_when_true_event_has_highest_confidence():
    assert confidence_sort_rank([_event(0.9, True), _event(0.5), _event(0.4)]) == 1


def test_rank_counts_every_higher_confidence_event():
    assert confidence_sort_rank([_event(0.9), _event(0.8), _event(0.3, True)]) == 3


def test_ties_are_pessimistic():
    assert confidence_sort_rank([_event(0.5, True), _event(0.5)]) == 2


def test_no_true_event_gives_none():
    assert confidence_sort_rank([_event(0.5)]) is None


def test_summarize_hit_rates_and_mrr():
    s = summarize([1, 2, 4, None])
    assert s["hit1"] == 0.25 and s["hit3"] == 0.5
    assert abs(s["mrr"] - (1 + 0.5 + 0.25) / 4) < 1e-9


def test_earliest_first_rank_one_when_true_event_fired_first():
    assert earliest_first_rank([_event(0.1, True, 0), _event(0.9, minutes=5), _event(0.8, minutes=9)]) == 1


def test_earliest_first_counts_every_earlier_event_and_ignores_confidence():
    assert earliest_first_rank([_event(0.1, minutes=0), _event(0.2, minutes=3), _event(0.99, True, 7)]) == 3


def test_earliest_first_ties_are_pessimistic():
    assert earliest_first_rank([_event(0.5, True, 4), _event(0.5, minutes=4)]) == 2


def test_earliest_first_no_true_event_gives_none():
    assert earliest_first_rank([_event(0.5)]) is None
