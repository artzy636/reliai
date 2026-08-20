"""
ReliAI — Sample report generator (remediation/ layer)
=========================================================
Standalone helper: runs one real incident through
evaluation.incident_pipeline.run_incident_pipeline (stub LLM, fresh
injected-drift data) and saves the resulting IncidentReport to disk, so
remediation/app.py's "Load saved report" mode has real data to load out of
the box without first running the "Run live" mode.

Usage:
    python -m remediation.generate_sample_report
    python -m remediation.generate_sample_report --output path/to/report.json
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from remediation.pipeline_utils import run_sample_incident
from schemas import IncidentReport

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = "sample_report.json"


def generate_sample_report(
    output_path: str = DEFAULT_OUTPUT_PATH,
    random_seed: Optional[int] = 42,
) -> IncidentReport:
    """Run one real incident and save the resulting IncidentReport as JSON.

    Args:
        output_path: where to write the report, via
            IncidentReport.model_dump_json().
        random_seed: fixed by default (42) so the checked-in/regenerated
            sample report is reproducible; pass None for a fresh,
            randomly-seeded incident instead.

    Returns:
        The IncidentReport that was written to output_path.
    """
    report = run_sample_incident(random_seed=random_seed)
    Path(output_path).write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info(
        "Wrote sample IncidentReport (incident_id=%s) to %s", report.incident_id, output_path
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for the injected-drift dataset (default: 42; pass -1 for a fresh random incident)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    seed = None if args.random_seed == -1 else args.random_seed
    generate_sample_report(args.output, random_seed=seed)
