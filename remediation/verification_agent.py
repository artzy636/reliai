"""
ReliAI — Verification Agent (remediation/ layer)
===================================================
Consumes: schemas.RemediationPlan, produced by the Remediation Agent
          (remediation/remediation_agent.py, on the remediation-agent
          branch as of this writing).
Produces: schemas.VerificationResult, the empirical before/after check that
          closes the loop on a remediation proposal.

Per RemediationPlan's docstring in schemas.py, the Remediation Agent never
executes anything itself -- it only proposes. This module is where a
proposal actually gets exercised against data, so every VerificationResult
field is a real measured number, never a restatement of the plan's own
expected_outcome claim.

Verification method (deliberately a real measurement, not a simulated one):
  1. Train a small baseline classifier (LogisticRegression) on the
     reference (pre-incident) data.
  2. Score it on a held-out replay slice of the current (post-incident)
     data -- this is `value_before`, expected to be degraded relative to
     reference-distribution performance.
  3. Apply the remediation:
       - RETRAIN: fit a new model on reference data plus the current data
         that falls outside the replay slice, so the model actually
         absorbs the shifted distribution instead of continuing to score
         against a stale one.
       - DATA_FIX: leave the model alone; apply a real (if generic) data
         correction -- missing-value imputation from reference-column
         statistics, duplicate removal -- to the replay slice, then
         re-score the *original* baseline model on the corrected slice.
     Other RemediationCategory values (ROLLBACK, CONFIG_CHANGE) have no
     data-driven analogue in this benchmark's scope, so verify() raises
     rather than fabricating a number for them.
  4. Score the post-remediation state on the same held-out replay slice --
     `value_after`. Using the same rows for both measurements is what
     makes the delta a real before/after comparison rather than noise from
     comparing two different samples.
  5. `improved` is True only if value_after - value_before strictly
     exceeds configs.settings.VERIFICATION.improvement_threshold_pct.

Assumes reference_df / current_df contain target_column plus numeric
feature columns only (matching detection.fault_injection.inject_feature_drift's
own numeric-column constraint) -- no categorical encoding is performed.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from configs.settings import VERIFICATION, VerificationSettings
from schemas import RemediationCategory, RemediationPlan, VerificationResult

logger = logging.getLogger(__name__)

_METRIC_NAME = "accuracy"


class VerificationAgent:
    """Empirically checks whether a RemediationPlan's fix actually improves
    model performance, via a real before/after measurement on held-out data.

    Usage:
        agent = VerificationAgent()
        result = agent.verify(plan, reference_df, current_df, target_column="label")
    """

    def __init__(
        self,
        settings: VerificationSettings = VERIFICATION,
        random_state: int = 42,
    ) -> None:
        """Initialize the agent.

        Args:
            settings: replay sample size / improvement threshold. Defaults
                to the shared configs.settings.VERIFICATION singleton --
                pass an override only for testing, never to hardcode a
                threshold inline.
            random_state: seed used for the replay-slice sample and for the
                classifiers, so verification runs are reproducible.
        """
        self._settings = settings
        self._random_state = random_state

    def verify(
        self,
        remediation_plan: RemediationPlan,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_column: str,
    ) -> VerificationResult:
        """Run a real before/after measurement for a remediation plan.

        Args:
            remediation_plan: the proposal to verify. Only `.category` and
                `.incident_id` are used -- verification never trusts the
                plan's own `expected_outcome` claim.
            reference_df: clean, pre-incident data. Used to train the
                baseline model and, for DATA_FIX, as the source of
                imputation statistics.
            current_df: the post-incident (e.g. drifted) data being
                diagnosed. Must share reference_df's columns.
            target_column: name of the label column to predict.

        Returns:
            A VerificationResult with every numeric field populated from
            real measurements against held-out data.

        Raises:
            ValueError: if target_column is missing from either dataframe,
                current_df has no rows to replay against, or
                remediation_plan.category has no supported verification
                procedure.
        """
        if target_column not in reference_df.columns:
            raise ValueError(f"target_column {target_column!r} not found in reference_df.")
        if target_column not in current_df.columns:
            raise ValueError(f"target_column {target_column!r} not found in current_df.")

        feature_columns = [c for c in reference_df.columns if c != target_column]

        baseline_model = self._fit_model(reference_df, feature_columns, target_column)

        replay_size = min(self._settings.replay_sample_size, len(current_df))
        if replay_size < 1:
            raise ValueError("current_df has no rows to replay verification against.")
        if replay_size < self._settings.replay_sample_size:
            logger.warning(
                "current_df (%d rows) is smaller than configured replay_sample_size "
                "(%d); using %d rows instead.",
                len(current_df), self._settings.replay_sample_size, replay_size,
            )

        replay_df = current_df.sample(n=replay_size, random_state=self._random_state)
        remaining_current_df = current_df.drop(index=replay_df.index)

        value_before = self._score(baseline_model, replay_df, feature_columns, target_column)

        if remediation_plan.category == RemediationCategory.RETRAIN:
            retrain_df = pd.concat([reference_df, remaining_current_df], ignore_index=True)
            remediated_model = self._fit_model(retrain_df, feature_columns, target_column)
            value_after = self._score(remediated_model, replay_df, feature_columns, target_column)
            notes = (
                f"RETRAIN: refit on {len(retrain_df)} rows (reference + non-replay "
                f"current data), re-scored on the same {replay_size}-row replay slice."
            )
        elif remediation_plan.category == RemediationCategory.DATA_FIX:
            corrected_replay_df = self._apply_data_fix(replay_df, reference_df, feature_columns)
            value_after = self._score(baseline_model, corrected_replay_df, feature_columns, target_column)
            notes = (
                f"DATA_FIX: imputed missing values from reference statistics and "
                f"dropped duplicates on the {replay_size}-row replay slice, then "
                "re-scored the original baseline model."
            )
        else:
            raise ValueError(
                f"VerificationAgent has no verification procedure for "
                f"category={remediation_plan.category!r}; supported categories are "
                f"{RemediationCategory.RETRAIN!r} and {RemediationCategory.DATA_FIX!r}."
            )

        delta = value_after - value_before
        improved = delta > self._settings.improvement_threshold_pct

        logger.info(
            "Verified incident %s (%s): accuracy %.4f -> %.4f (delta=%.4f, improved=%s)",
            remediation_plan.incident_id, remediation_plan.category.value,
            value_before, value_after, delta, improved,
        )

        return VerificationResult(
            incident_id=remediation_plan.incident_id,
            metric_name=_METRIC_NAME,
            value_before=value_before,
            value_after=value_after,
            improved=improved,
            replay_sample_size=replay_size,
            notes=notes,
        )

    def _fit_model(
        self, df: pd.DataFrame, feature_columns: list[str], target_column: str
    ) -> LogisticRegression:
        """Fit a small, deterministic baseline classifier."""
        model = LogisticRegression(max_iter=1000, random_state=self._random_state)
        model.fit(df[feature_columns], df[target_column])
        return model

    @staticmethod
    def _score(
        model: LogisticRegression, df: pd.DataFrame, feature_columns: list[str], target_column: str
    ) -> float:
        """Accuracy of `model` on `df`."""
        predictions = model.predict(df[feature_columns])
        return float(accuracy_score(df[target_column], predictions))

    @staticmethod
    def _apply_data_fix(
        df: pd.DataFrame, reference_df: pd.DataFrame, feature_columns: list[str]
    ) -> pd.DataFrame:
        """Simple, real data-quality correction: impute missing numeric
        values from reference-column means, then drop duplicate rows.
        Does not address a pure distribution shift (there's nothing to
        impute or dedupe) -- that's intentional, since a DATA_FIX plan
        applied to a feature-drift incident should not be credited with an
        improvement it didn't actually produce.
        """
        fixed_df = df.copy()
        for column in feature_columns:
            if pd.api.types.is_numeric_dtype(fixed_df[column]):
                fixed_df[column] = fixed_df[column].fillna(reference_df[column].mean())
        return fixed_df.drop_duplicates()
