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
"""

import logging

import pandas as pd

from detection.duplicate_rows_detector import detect_duplicate_rows
from detection.isolation_forest_detector import detect_corrupted_values
from detection.ks_detector import detect_distribution_shift
from detection.missing_values_detector import detect_missing_values
from detection.schema_mismatch_detector import detect_schema_mismatch
from schemas import EvidenceEvent

logger = logging.getLogger(__name__)


class DataAgent:
    """Detection layer agent that localizes data-quality failures (feature
    drift, missing values, duplicate rows, schema mismatches, corrupted
    values) to a feature or the pipeline as a whole."""

    def investigate(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> list[EvidenceEvent]:
        """Compare reference_df against current_df, running every available
        detector.

        Parameters
        ----------
        reference_df : pd.DataFrame
            Reference (e.g. training) data.
        current_df : pd.DataFrame
            Current (e.g. production) data.

        Returns
        -------
        list[EvidenceEvent]
            One EvidenceEvent per detector finding across all columns and
            the pipeline as a whole.
        """
        shared_columns = [col for col in reference_df.columns if col in current_df.columns]

        events: list[EvidenceEvent] = []

        for column in shared_columns:
            reference_column = reference_df[column]
            current_column = current_df[column]

            if (
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

        for column in reference_df.columns:
            schema_event = detect_schema_mismatch(reference_df, current_df, feature_name=column)
            if schema_event is None:
                logger.info("No schema mismatch detected in column '%s'", column)
            else:
                logger.info("Schema mismatch detected in column '%s'", column)
                events.append(schema_event)

        duplicate_event = detect_duplicate_rows(reference_df, current_df)
        if duplicate_event is None:
            logger.info("No elevated duplicate-row rate detected")
        else:
            logger.info(
                "Elevated duplicate-row rate detected (confidence=%.3f)",
                duplicate_event.confidence,
            )
            events.append(duplicate_event)

        corrupted_event = detect_corrupted_values(reference_df, current_df)
        if corrupted_event is None:
            logger.info("No elevated corrupted-row rate detected")
        else:
            logger.info(
                "Elevated corrupted-row rate detected (confidence=%.3f)",
                corrupted_event.confidence,
            )
            events.append(corrupted_event)

        return events
