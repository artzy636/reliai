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
       - DATA_FIX: leave the model alone; drop duplicate rows from the
         replay slice, then re-score the *original* baseline model on the
         result. (Missing-value imputation used to live here too -- see
         _prepare_features's docstring for why it moved to a universal,
         always-applied step instead of a DATA_FIX-only one.)
     Other RemediationCategory values (ROLLBACK, CONFIG_CHANGE) have no
     data-driven analogue in this benchmark's scope, so verify() raises
     rather than fabricating a number for them.
  4. Score the post-remediation state on the same held-out replay slice --
     `value_after`. Using the same rows for both measurements is what
     makes the delta a real before/after comparison rather than noise from
     comparing two different samples.
  5. `improved` is True only if value_after - value_before strictly
     exceeds configs.settings.VERIFICATION.improvement_threshold_pct.

Missing / absent features (_prepare_features):
  A deployed model has to produce *some* prediction for a row even when a
  feature is missing or an entire column has stopped arriving -- refusing
  to score at all (which is what LogisticRegression.predict() does the
  moment it sees a NaN) isn't a real option for a serving system, so this
  agent doesn't treat it as one either. Every dataframe that reaches
  _fit_model or _score is first run through _prepare_features, which:
    - reindexes to reference_df's feature columns, so a column a
      schema_mismatch fault removed entirely from current_df becomes a
      100%-missing column instead of a KeyError -- from the model's point
      of view a fully-missing numeric column and a partially-missing one
      (missing_values) are the same problem, just at different rates, and
      get exactly the same treatment;
    - coerces every reference-numeric column and fills whatever is still
      missing with that column's mean over reference_df.
  This runs unconditionally, for every category, not only DATA_FIX --
  RemediationCategory.DATA_FIX's own, still-real distinguishing effect is
  dropping duplicate rows (see _apply_data_fix); RETRAIN's is training on
  the shifted distribution. Neither is "the reason verify() doesn't crash"
  anymore -- _prepare_features is.

Assumes reference_df / current_df contain target_column plus feature
columns that are numeric *in reference_df* (a column reference_df itself
doesn't have numeric data for is left untouched, unimputed, and will
still raise if it reaches a scikit-learn model with a NaN or a string in
it) -- no categorical encoding is performed.
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
                baseline model and, throughout _prepare_features, as the
                source of imputation statistics for every column that's
                numeric there.
            current_df: the post-incident (e.g. drifted, or missing a
                column, or carrying nulls/duplicates) data being
                diagnosed. Must share target_column with reference_df --
                nothing else is required to be shared; _prepare_features
                is what makes that safe.
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

        baseline_model = self._fit_model(reference_df, reference_df, feature_columns, target_column)

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

        value_before = self._score(
            baseline_model, replay_df, reference_df, feature_columns, target_column
        )

        if remediation_plan.category == RemediationCategory.RETRAIN:
            retrain_df = pd.concat([reference_df, remaining_current_df], ignore_index=True)
            remediated_model = self._fit_model(retrain_df, reference_df, feature_columns, target_column)
            value_after = self._score(
                remediated_model, replay_df, reference_df, feature_columns, target_column
            )
            notes = (
                f"RETRAIN: refit on {len(retrain_df)} rows (reference + non-replay "
                f"current data), re-scored on the same {replay_size}-row replay slice."
            )
        elif remediation_plan.category == RemediationCategory.DATA_FIX:
            corrected_replay_df = self._apply_data_fix(replay_df)
            value_after = self._score(
                baseline_model, corrected_replay_df, reference_df, feature_columns, target_column
            )
            notes = (
                f"DATA_FIX: dropped duplicate rows from the {replay_size}-row replay "
                "slice (missing/absent-column values are handled the same way for "
                "every category -- see _prepare_features), then re-scored the "
                "original baseline model."
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

    @staticmethod
    def _prepare_features(
        df: pd.DataFrame, reference_df: pd.DataFrame, feature_columns: list[str]
    ) -> pd.DataFrame:
        """Build the X matrix `_fit_model`/`_score` actually feed to
        scikit-learn: `df` reindexed to `feature_columns`, with every
        column that's numeric *in reference_df* coerced to numeric and
        any value still missing afterward filled with that column's mean
        over reference_df.

        Reindexing first is what makes a column schema_mismatch dropped
        from `df` entirely behave exactly like one missing_values only
        partially nulled out: reindex(columns=...) turns an absent column
        into an all-NaN one, and the fill step below can't tell the
        difference between "was never here" and "was here but null" --
        which is the point. A column that changed dtype (e.g. a numeric
        column re-emitted as strings) is recovered losslessly by the
        pd.to_numeric coercion when the strings are still numeric-looking
        -- if they aren't, they become NaN and get the same mean-fill as
        any other missing value.

        A column reference_df itself has no numeric data for is left
        exactly as `df` has it, untouched -- there's no reference
        statistic to impute from, and inventing one would misrepresent
        what "verified against real data" means.
        """
        features = df.reindex(columns=feature_columns)
        for column in feature_columns:
            if column not in reference_df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(reference_df[column]):
                continue
            features[column] = pd.to_numeric(features[column], errors="coerce")
            features[column] = features[column].fillna(reference_df[column].mean())
        return features

    def _fit_model(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> LogisticRegression:
        """Fit a small, deterministic baseline classifier on `df`, via
        _prepare_features so a fit against current-data-derived rows
        (e.g. RETRAIN's retrain_df) is exactly as robust to missing/absent
        features as a fit against reference_df itself."""
        model = LogisticRegression(max_iter=1000, random_state=self._random_state)
        X = self._prepare_features(df, reference_df, feature_columns)
        model.fit(X, df[target_column])
        return model

    @staticmethod
    def _score(
        model: LogisticRegression,
        df: pd.DataFrame,
        reference_df: pd.DataFrame,
        feature_columns: list[str],
        target_column: str,
    ) -> float:
        """Accuracy of `model` on `df`, via _prepare_features."""
        X = VerificationAgent._prepare_features(df, reference_df, feature_columns)
        predictions = model.predict(X)
        return float(accuracy_score(df[target_column], predictions))

    @staticmethod
    def _apply_data_fix(df: pd.DataFrame) -> pd.DataFrame:
        """DATA_FIX's own distinguishing correction: drop duplicate rows.

        Missing-value imputation used to live here too, but it's now
        handled unconditionally by _prepare_features for every category
        (see this module's docstring for why) -- doing it here as well
        would be redundant, not wrong, but it would also wrongly imply
        DATA_FIX is the only category that can survive missing data, which
        is no longer true. This function's only remaining job -- and the
        only thing that still makes DATA_FIX measurably different from
        just re-scoring the baseline model as-is -- is dropping duplicates.
        """
        return df.drop_duplicates()
