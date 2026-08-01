"""
ReliAI — Central configuration.

Rule: no agent, detector, or benchmark script should hardcode a threshold,
file path, or magic number. Import it from here instead. If a value you
need isn't here yet, add it here first.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DetectionThresholds:
    """Thresholds used by the anomaly detection engine (detection/ layer).
    Tune these against your validation split, don't guess-and-hardcode
    inline in the detector functions.
    """
    ks_test_pvalue: float = 0.05
    psi_warning: float = 0.1
    psi_critical: float = 0.2
    jensen_shannon_threshold: float = 0.1
    isolation_forest_contamination: float = 0.05
    rolling_accuracy_drop_pct: float = 0.05   # flag if rolling accuracy drops >5%


@dataclass(frozen=True)
class GraphSettings:
    """Evidence graph construction settings (reasoning/ layer)."""
    time_window_minutes: int = 60      # events within this window may be linked
    min_edge_confidence: float = 0.3    # drop edges below this confidence


@dataclass(frozen=True)
class RCASettings:
    """RCA Agent settings (reasoning/ layer)."""
    max_hypotheses: int = 5
    llm_model: str = "claude-sonnet-4-6"


@dataclass(frozen=True)
class VerificationSettings:
    """Verification Agent settings (remediation/ layer)."""
    replay_sample_size: int = 500
    improvement_threshold_pct: float = 0.02   # min improvement to count as "verified"


@dataclass(frozen=True)
class Paths:
    data_dir: str = "data/"
    benchmark_dir: str = "data/benchmark/"
    results_dir: str = "results/"


DETECTION = DetectionThresholds()
GRAPH = GraphSettings()
RCA = RCASettings()
VERIFICATION = VerificationSettings()
PATHS = Paths()
