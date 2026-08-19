"""
Tests for remediation/verification_agent.py.

Dataset choice: a small synthetic classification dataset from
sklearn.datasets.make_classification, NOT the Adult dataset from
fetch_openml. fetch_openml requires a network call (and a local cache) the
first time it runs, which would make this suite slow and non-deterministic
in CI/offline environments; make_classification is fully in-process,
seeded, and gives the same fine-grained control over class separability
that's needed to make the injected drift reliably degrade accuracy.

Scenario design:
  - A LogisticRegression trained on reference data has its accuracy
    measured on a replay slice of data where one informative feature has
    been shifted by inject_feature_drift() -- this reliably degrades
    accuracy because the shift pushes samples across the model's learned
    decision boundary.
  - RETRAIN scenario ("genuinely helps"): retraining on data that includes
    the shifted distribution lets the model relearn the boundary, so
    accuracy recovers -- improved=True.
  - DATA_FIX scenario ("no-op/harmful"): the injected fault is a pure
    numeric shift, not a missing-value or duplicate-row problem, so
    VerificationAgent._apply_data_fix's imputation/dedup has nothing to
    correct -- accuracy does not meaningfully change -- improved=False.
"""

import pandas as pd
import pytest
from sklearn.datasets import make_classification

from configs.settings import VerificationSettings
from detection.fault_injection import inject_feature_drift
from remediation.verification_agent import VerificationAgent
from schemas import RemediationCategory, RemediationPlan

_FEATURE_COLUMNS = [f"feature_{i}" for i in range(5)]
_TARGET_COLUMN = "label"


def _make_dataset(n_samples: int, random_state: int) -> pd.DataFrame:
    """A small, seeded, linearly-separable-ish binary classification set."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=len(_FEATURE_COLUMNS),
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=_FEATURE_COLUMNS)
    df[_TARGET_COLUMN] = y
    return df


def _reference_and_drifted_current() -> tuple[pd.DataFrame, pd.DataFrame]:
    """600 reference rows + 600 current rows drawn from the same
    distribution, then a large shift injected into the current data's
    dominant informative feature to simulate a feature-drift incident.

    feature_1 is used (not feature_0) because it's the feature the fitted
    LogisticRegression actually weighs heavily (coefficient ~2.4 vs ~-0.1
    for feature_0 on this seed) -- shifting a near-zero-coefficient feature
    would barely move accuracy and wouldn't be a meaningful "before" signal.
    """
    full_df = _make_dataset(n_samples=1200, random_state=42)
    reference_df = full_df.iloc[:600].reset_index(drop=True)
    current_clean_df = full_df.iloc[600:].reset_index(drop=True)

    drifted_df, fault_label = inject_feature_drift(
        current_clean_df, "feature_1", shift_amount=3.0, random_seed=42
    )
    from schemas import FailureType

    assert fault_label == FailureType.FEATURE_DRIFT
    return reference_df, drifted_df


def _plan(category: RemediationCategory) -> RemediationPlan:
    return RemediationPlan(
        incident_id="verify-test-incident",
        category=category,
        action_description="test plan",
        expected_outcome="test expected outcome",
        target_root_cause_node_id="node-1",
    )


@pytest.fixture(scope="module")
def data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return _reference_and_drifted_current()


def test_baseline_accuracy_is_degraded_by_injected_drift(data):
    """Sanity check that the injected drift is real: a model trained on
    reference data scores much worse on the drifted current data than on
    held-out reference data of the same distribution."""
    reference_df, drifted_df = data
    settings = VerificationSettings(replay_sample_size=150, improvement_threshold_pct=0.02)
    agent = VerificationAgent(settings=settings)

    result = agent.verify(_plan(RemediationCategory.RETRAIN), reference_df, drifted_df, _TARGET_COLUMN)

    assert 0.0 <= result.value_before <= 1.0
    assert result.value_before < 0.85


def test_retrain_plan_measures_real_improvement(data):
    """RETRAIN scenario: retraining on data that includes the drifted
    distribution should recover accuracy on the replay slice -- a genuine
    improvement."""
    reference_df, drifted_df = data
    settings = VerificationSettings(replay_sample_size=150, improvement_threshold_pct=0.02)
    agent = VerificationAgent(settings=settings)
    plan = _plan(RemediationCategory.RETRAIN)

    result = agent.verify(plan, reference_df, drifted_df, _TARGET_COLUMN)

    assert result.incident_id == plan.incident_id
    assert result.metric_name == "accuracy"
    assert result.replay_sample_size == 150
    assert 0.0 <= result.value_before <= 1.0
    assert 0.0 <= result.value_after <= 1.0
    assert result.value_after - result.value_before > settings.improvement_threshold_pct
    assert result.improved is True


def test_data_fix_plan_on_pure_drift_shows_no_improvement(data):
    """DATA_FIX scenario (no-op/harmful): imputation and dedup can't fix a
    pure numeric shift with no missing values or duplicates, so the
    remediation should NOT be credited with an improvement."""
    reference_df, drifted_df = data
    settings = VerificationSettings(replay_sample_size=150, improvement_threshold_pct=0.02)
    agent = VerificationAgent(settings=settings)
    plan = _plan(RemediationCategory.DATA_FIX)

    result = agent.verify(plan, reference_df, drifted_df, _TARGET_COLUMN)

    assert 0.0 <= result.value_before <= 1.0
    assert 0.0 <= result.value_after <= 1.0
    assert result.improved is False


def test_verify_is_deterministic(data):
    """Same inputs, same random_state -> identical results (required for a
    reproducible benchmark suite)."""
    reference_df, drifted_df = data
    settings = VerificationSettings(replay_sample_size=150, improvement_threshold_pct=0.02)
    plan = _plan(RemediationCategory.RETRAIN)

    result_a = VerificationAgent(settings=settings).verify(plan, reference_df, drifted_df, _TARGET_COLUMN)
    result_b = VerificationAgent(settings=settings).verify(plan, reference_df, drifted_df, _TARGET_COLUMN)

    assert result_a.value_before == result_b.value_before
    assert result_a.value_after == result_b.value_after
    assert result_a.improved == result_b.improved


def test_unsupported_category_raises(data):
    """ROLLBACK/CONFIG_CHANGE have no data-driven verification procedure in
    this scope -- verify() must fail loudly rather than fabricate a number."""
    reference_df, drifted_df = data
    agent = VerificationAgent()
    plan = _plan(RemediationCategory.ROLLBACK)

    with pytest.raises(ValueError):
        agent.verify(plan, reference_df, drifted_df, _TARGET_COLUMN)


def test_replay_sample_size_caps_to_available_rows():
    """If current_df has fewer rows than the configured replay_sample_size,
    verify() should use what's available instead of erroring."""
    reference_df, drifted_df = _reference_and_drifted_current()
    small_current_df = drifted_df.iloc[:50].reset_index(drop=True)
    settings = VerificationSettings(replay_sample_size=150, improvement_threshold_pct=0.02)
    agent = VerificationAgent(settings=settings)

    result = agent.verify(_plan(RemediationCategory.RETRAIN), reference_df, small_current_df, _TARGET_COLUMN)

    assert result.replay_sample_size == 50
