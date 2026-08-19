"""Data Agent — detection layer orchestrator.

Scans every column shared between a reference and a current DataFrame and
delegates the actual statistics to detect_distribution_shift(), which already
returns a fully-built schemas.EvidenceEvent (or None). DataAgent's only job is
column iteration, numeric filtering, and collecting the results.
"""

import logging

import pandas as pd

from detection.ks_detector import detect_distribution_shift
from schemas import EvidenceEvent

logger = logging.getLogger(__name__)


class DataAgent:
    """Detection layer agent that localizes distribution shift to a feature."""

    def investigate(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> list[EvidenceEvent]:
        """Compare reference_df against current_df, column by column.

        Parameters
        ----------
        reference_df : pd.DataFrame
            Reference (e.g. training) data.
        current_df : pd.DataFrame
            Current (e.g. production) data.

        Returns
        -------
        list[EvidenceEvent]
            One EvidenceEvent per shared numeric column where
            detect_distribution_shift() found a statistically significant
            shift. Columns present in only one DataFrame, or that are
            non-numeric, are skipped.
        """
        shared_columns = [col for col in reference_df.columns if col in current_df.columns]

        events: list[EvidenceEvent] = []

        for column in shared_columns:
            reference_column = reference_df[column]
            current_column = current_df[column]

            if not (
                pd.api.types.is_numeric_dtype(reference_column)
                and pd.api.types.is_numeric_dtype(current_column)
            ):
                logger.info("Skipping non-numeric column '%s'", column)
                continue

            event = detect_distribution_shift(
                reference_column,
                current_column,
                feature_name=column,
            )

            if event is None:
                logger.info("No distribution shift detected in column '%s'", column)
                continue

            logger.info(
                "Distribution shift detected in column '%s' (confidence=%.3f)",
                column,
                event.confidence,
            )
            events.append(event)

        return events
