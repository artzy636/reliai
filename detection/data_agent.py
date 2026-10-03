"""Data Agent — detection layer orchestrator.

Scans a reference and a current DataFrame and delegates the actual
statistics to the detector functions in this package, each of which
already returns a fully-built schemas.EvidenceEvent (or None). DataAgent's
only job is column/pipeline iteration and collecting the results:

  * detect_distribution_shift — per shared numeric column.
  * detect_missing_values     — per shared column (any dtype).
  * detect_schema_mismatch    — per reference column (catches columns
    missing from current, which shared-column iteration would skip).
  * detect_duplicate_rows     — once, pipeline-level (feature_name=None).
  * detect_corrupted_values   — once, pipeline-level (feature_name=None);
    multivariate/row-level rather than per-column, so it runs once over
    all shared numeric columns together rather than being looped per
    column like detect_distribution_shift.
  * detect_label_shift        — once, pipeline-level (feature_name=None);
    only runs when the caller passes target_column, since it's the only
    detector in this package that needs supervised labels (on BOTH
    reference and current) rather than feature columns alone. DataAgent
    has no target-column concept otherwise -- see investigate()'s
    target_column parameter docstring.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from detection.duplicate_rows_detector import detect_duplicate_rows
from detection.isolation_forest_detector import detect_corrupted_values
from detection.ks_detector import detect_distribution_shift
from detection.missing_values_detector import detect_missing_values
from detection.rolling_accuracy_detector import detect_label_shift
from detection.schema_mismatch_detector import detect_schema_mismatch
from schemas import DetectionMethod, EvidenceEvent

logger = logging.getLogger(__name__)


class DataAgent:
    """Detection layer agent that localizes data-quality failures (feature
    drift, missing values, duplicate rows, schema mismatches, corrupted
    values, label shift) to a feature or the pipeline as a whole."""

    def investigate(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_column: Optional[str] = None,
        disabled_methods: Optional[set[DetectionMethod]] = None,
        snapshot_timestamps: bool = False,
    ) -> list[EvidenceEvent]:
        """Compare reference_df against current_df, running every available
        detector.

        Parameters
        ----------
        reference_df : pd.DataFrame
            Reference (e.g. training) data.
        current_df : pd.DataFrame
            Current (e.g. production) data.
        target_column : str | None
            Name of the binary label column, present in both reference_df
            and current_df with real ground-truth values, if the caller
            has one and wants label-shift detection. Every OTHER detector
            in this package treats target_column like any other shared
            numeric column (scanned for drift/missing-values/schema
            issues same as any feature) -- this parameter only turns on
            the additional rolling-accuracy check; it never excludes
            target_column from the per-column loops below. Defaults to
            None (no label-shift check), so existing callers that don't
            have a target column keep working unchanged.
        disabled_methods : set[DetectionMethod] | None
            Detector(s) to skip entirely, as if they didn't exist -- for
            an ablation study (see evaluation/real_data_benchmark.py's
            ``run_ablation_study``), to measure how much each individual
            signal actually contributes to downstream RCA accuracy, by
            comparing against a run with nothing disabled. None (the
            default) runs every detector, unchanged from before this
            parameter existed. A disabled detector's loop iteration still
            runs (e.g. still iterates every column) but simply never calls
            that detector function or appends its event, so the only
            difference from a normal run is the exact set of
            EvidenceEvents produced.

        snapshot_timestamps : bool
            By default every detector stamps its event with the wall-clock
            moment it ran, so events carry the order of this method's own
            loops (columns in dataframe order, then schema, duplicates,
            isolation forest, rolling accuracy) -- microseconds apart.
            EvidenceGraphBuilder reads those gaps as "A preceded B" and
            draws causal edges, but that order reflects how this method is
            written, not when anything went wrong in the data. A single
            comparison of two static dataframes contains no onset-time
            information at all. Pass True to stamp every event from this
            call with ONE shared observation time, so the graph builder
            draws no precedence edges from loop order. Defaults to False
            so existing callers (and the multi-fault demo built on those
            gaps) are unchanged. Use True whenever the incident's temporal
            order isn't independently known.

        Returns
        -------
        list[EvidenceEvent]
            One EvidenceEvent per detector finding across all columns and
            the pipeline as a whole.
        """
        disabled_methods = disabled_methods or set()
        shared_columns = [col for col in reference_df.columns if col in current_df.columns]

        events: list[EvidenceEvent] = []

        for column in shared_columns:
            reference_column = reference_df[column]
            current_column = current_df[column]

            if DetectionMethod.KS_TEST in disabled_methods:
                pass
            elif (
                pd.api.types.is_numeric_dtype(reference_column)
                and pd.api.types.is_numeric_dtype(current_column)
            ):
                drift_event = detect_distribution_shift(
                    reference_column,
                    current_column,
                    feature_name=column,
                )
                if drift_event is None:
                    logger.info("No distribution shift detected in column '%s'", column)
                else:
                    logger.info(
                        "Distribution shift detected in column '%s' (confidence=%.3f)",
                        column,
                        drift_event.confidence,
                    )
                    events.append(drift_event)
            else:
                logger.info("Skipping non-numeric column '%s' for distribution-shift check", column)

            if DetectionMethod.MISSING_VALUE_RATE not in disabled_methods:
                missing_event = detect_missing_values(
                    reference_column,
                    current_column,
                    feature_name=column,
                )
                if missing_event is None:
                    logger.info("No elevated missing-value rate detected in column '%s'", column)
                else:
                    logger.info(
                        "Elevated missing-value rate detected in column '%s' (confidence=%.3f)",
                        column,
                        missing_event.confidence,
                    )
                    events.append(missing_event)

        if DetectionMethod.SCHEMA_CHECK not in disabled_methods:
            for column in reference_df.columns:
                schema_event = detect_schema_mismatch(reference_df, current_df, feature_name=column)
                if schema_event is None:
                    logger.info("No schema mismatch detected in column '%s'", column)
                else:
                    logger.info("Schema mismatch detected in column '%s'", column)
                    events.append(schema_event)

        if DetectionMethod.DUPLICATE_ROW_RATE not in disabled_methods:
            duplicate_event = detect_duplicate_rows(reference_df, current_df)
            if duplicate_event is None:
                logger.info("No elevated duplicate-row rate detected")
            else:
                logger.info(
                    "Elevated duplicate-row rate detected (confidence=%.3f)",
                    duplicate_event.confidence,
                )
                events.append(duplicate_event)

        if DetectionMethod.ISOLATION_FOREST not in disabled_methods:
            corrupted_event = detect_corrupted_values(reference_df, current_df)
            if corrupted_event is None:
                logger.info("No elevated corrupted-row rate detected")
            else:
                logger.info(
                    "Elevated corrupted-row rate detected (confidence=%.3f)",
                    corrupted_event.confidence,
                )
                events.append(corrupted_event)

        if target_column is not None and DetectionMethod.ROLLING_ACCURACY not in disabled_methods:
            label_shift_event = detect_label_shift(reference_df, current_df, target_column)
            if label_shift_event is None:
                logger.info("No rolling-accuracy drop detected against target column '%s'", target_column)
            else:
                logger.info(
                    "Rolling-accuracy drop detected against target column '%s' (confidence=%.3f)",
                    target_column,
                    label_shift_event.confidence,
                )
                events.append(label_shift_event)

        if snapshot_timestamps and events:
            observed_at = datetime.now(timezone.utc)
            events = [event.model_copy(update={"timestamp": observed_at}) for event in events]

        return events
