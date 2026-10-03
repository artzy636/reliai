"""Shared confidence calibration for detectors.

Every detector separates two questions:

* *Did it fire?* -- decided by the detector's own threshold / significance
  test.
* *How strong is the evidence?* -- ``effect_confidence`` below.

Why this exists: the original per-detector formulas weren't comparable.
KS confidence was ``1 - p/alpha``, which saturates at 1.0 as soon as the
sample is large enough (a KS D of 0.03 on ~10k rows has p << 0.05), so
statistically-significant-but-tiny shifts looked as certain as real faults.
The other detectors used ``(metric - threshold) / (1 - threshold)``, which
compresses a perfectly real effect (a 21-point accuracy drop, a 30% null
rate) into 0.16-0.22. Downstream, the evidence graph only keeps edges
whose confidence exceeds ``GRAPH.min_edge_confidence``, so low-confidence
true-fault events were silently cut off from the causal chain while
saturated noise events connected freely.
"""


# A statistical estimate from a finite sample is never certain, so
# sampling-based detectors cap just below 1.0. 1.0 is reserved for
# deterministic structural checks (schema_check), which have no sampling
# error: a dropped column is dropped. Without the cap, a severe
# *downstream symptom* (e.g. a 59% duplicate rate caused by dropping
# fnlwgt) ties the structural root-cause event at exactly 1.0 and any
# tie-break between them is arbitrary.
MAX_STATISTICAL_CONFIDENCE = 0.95


def effect_confidence(
    effect: float, full_scale: float, cap: float = MAX_STATISTICAL_CONFIDENCE
) -> float:
    """Linear ramp from 0 (no effect) to ``cap`` (effect >= ``full_scale``)."""
    if full_scale <= 0:
        raise ValueError("full_scale must be positive")
    if effect != effect:  # NaN guard: an undefined effect is never evidence
        return 0.0
    return max(0.0, min(cap, cap * effect / full_scale))
