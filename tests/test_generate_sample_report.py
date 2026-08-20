"""
Test for remediation/generate_sample_report.py.

Not a UI test (the Streamlit app itself isn't unit-tested) -- just confirms
the sample-report generator produces a real, valid IncidentReport and that
its JSON round-trips via model_validate_json(), since that's exactly what
remediation/app.py's "Load saved report" mode depends on.
"""

from pathlib import Path

from remediation.generate_sample_report import generate_sample_report
from schemas import IncidentReport


def test_generate_sample_report_round_trips_via_model_validate_json(tmp_path: Path) -> None:
    output_path = tmp_path / "sample_report.json"

    report = generate_sample_report(str(output_path))

    assert isinstance(report, IncidentReport)
    assert output_path.exists()

    loaded = IncidentReport.model_validate_json(output_path.read_text(encoding="utf-8"))

    assert isinstance(loaded, IncidentReport)
    assert loaded == report
