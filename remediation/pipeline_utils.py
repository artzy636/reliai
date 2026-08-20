"""
ReliAI — Streamlit frontend pipeline helpers (remediation/ layer)
=====================================================================
Shared plumbing for remediation/app.py's "Run live" mode and
remediation/generate_sample_report.py: builds a fresh injected-drift
dataset and a pluggable LLM for reasoning.rca_agent.RCAAgent, so both the
live Streamlit button and the sample-report generator run the exact same
real pipeline (evaluation.incident_pipeline.run_incident_pipeline) that
tests/test_incident_pipeline.py exercises -- no separate demo code path.
"""

from __future__ import annotations

import json
import logging
import random
import re
from typing import Optional

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from detection.fault_injection import inject_feature_drift
from evaluation.incident_pipeline import run_incident_pipeline
from reasoning.rca_agent import LLMLike
from schemas import FailureType, IncidentReport

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [f"feature_{i}" for i in range(5)]
TARGET_COLUMN = "label"
N_DRIFTED_FEATURES = 3
# How many of the model's most heavily-weighted features to inject drift
# into. >1 so the evidence graph gets more than one EvidenceNode (and a
# real edge between them) instead of a single lonely node -- see
# generate_injected_drift_data()'s docstring.

_ROOT_ID_RE = re.compile(r"^- root_node_id: (\S+)$", re.MULTILINE)


class StructuralAgreementStubLLM:
    """Deterministic, offline stand-in for a real chat model.

    Mirrors tests/test_incident_pipeline.py's ``_StructuralAgreementLLM``
    (and evaluation/benchmark_runner.py's equivalent): agrees with whatever
    deterministic candidate ranking RCAAgent already computed by echoing
    back the ``root_node_id`` values it finds in the prompt, in order. Used
    as the default LLM for both the "Run live" Streamlit button and
    generate_sample_report.py -- no network access or API key required.
    """

    def __call__(self, prompt: str) -> str:
        node_ids = _ROOT_ID_RE.findall(prompt)
        return json.dumps(
            [
                {
                    "node_id": node_id,
                    "explanation": (
                        f"Structural rank {rank}: agrees with the deterministic "
                        "candidate ordering."
                    ),
                }
                for rank, node_id in enumerate(node_ids, start=1)
            ]
        )


def get_llm(use_stub: bool = True) -> LLMLike:
    """Return the LLM to pass to RCAAgent / run_incident_pipeline.

    Args:
        use_stub: if True (the default -- no API key needed), returns the
            deterministic StructuralAgreementStubLLM. If False, this is
            where a real LangChain-style chat model would be constructed
            and returned instead (e.g. ``ChatAnthropic(model=...)``); no
            such client is wired up yet, so this branch raises rather than
            silently falling back to the stub.

    Returns:
        An LLMLike object accepted by RCAAgent (anything with
        ``.invoke(prompt) -> AIMessage``, or a plain ``Callable[[str], str]``).

    Raises:
        NotImplementedError: if use_stub is False.
    """
    if use_stub:
        return StructuralAgreementStubLLM()
    raise NotImplementedError(
        "No real LLM client is wired up yet. Construct a LangChain-style "
        "chat model here (anything exposing .invoke(prompt) -> AIMessage) "
        "and return it -- RCAAgent/run_incident_pipeline accept it as-is."
    )


def _select_weighted_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    n: int,
    random_state: int,
) -> list[str]:
    """Pick the `n` feature columns a LogisticRegression fit on `df` weighs
    most heavily (largest |coefficient|) -- the same "does the model
    actually depend on this feature" check remediation.verification_agent's
    baseline model implicitly relies on. Drifting a feature the model
    ignores would be undetectable-by-design in VerificationAgent's
    before/after accuracy delta; this guarantees the opposite, and does so
    dynamically so it stays correct across random "Run live" seeds, not
    just the fixed sample-report seed.
    """
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(df[feature_columns], df[target_column])
    ranked = sorted(
        zip(feature_columns, model.coef_[0]), key=lambda pair: abs(pair[1]), reverse=True
    )
    return [name for name, _coef in ranked[:n]]


def generate_injected_drift_data(
    random_seed: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, FailureType, list[str]]:
    """Build a fresh reference/current dataframe pair with real, multi-feature drift.

    Same base construction tests/test_incident_pipeline.py uses, extended
    to N_DRIFTED_FEATURES columns: a synthetic classification dataset, then
    the N_DRIFTED_FEATURES columns a LogisticRegression fit on reference_df
    actually weighs most heavily (see _select_weighted_features) are each
    shifted via detection.fault_injection.inject_feature_drift, applied one
    column at a time so current_df ends up as reference_df's exact rows
    with exactly those columns shifted -- every other column (including
    non-selected features and target_column) stays byte-identical, so
    DataAgent's KS test can only ever flag the drifted columns.

    Injecting drift into several genuinely-weighted features (instead of
    just one) is what gives the evidence graph more than one EvidenceNode
    to cluster, and a real edge between them: DataAgent.investigate()
    timestamps each EvidenceEvent with wall-clock time as it scans columns
    in sequence, so the drifted features' events land a few milliseconds
    apart -- comfortably inside GRAPH.time_window_minutes -- without
    needing to fabricate timestamps.

    Args:
        random_seed: seed for both the synthetic dataset and the drift
            injection. None (the default) picks a fresh random seed each
            call, so repeated "Run live" clicks in the Streamlit app
            produce a genuinely new incident each time rather than
            replaying the same one.

    Returns:
        (reference_df, current_df, injected_label, drifted_features) --
        injected_label is always FailureType.FEATURE_DRIFT, and
        drifted_features is the list of column names actually shifted (so
        callers can stamp ground_truth_label onto exactly those columns'
        EvidenceEvents, and no others).
    """
    seed = random_seed if random_seed is not None else random.randint(0, 2**31 - 1)
    logger.info("Generating injected-drift dataset with seed=%d", seed)

    X, y = make_classification(
        n_samples=600,
        n_features=len(FEATURE_COLUMNS),
        n_informative=3,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=1.5,
        random_state=seed,
    )
    reference_df = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    reference_df[TARGET_COLUMN] = y

    drifted_features = _select_weighted_features(
        reference_df, FEATURE_COLUMNS, TARGET_COLUMN, n=N_DRIFTED_FEATURES, random_state=seed
    )
    logger.info("Injecting drift into model-weighted features: %s", drifted_features)

    current_df = reference_df
    injected_label = FailureType.FEATURE_DRIFT
    for feature in drifted_features:
        current_df, injected_label = inject_feature_drift(
            current_df, feature, shift_amount=3.0, random_seed=seed
        )

    return reference_df, current_df, injected_label, drifted_features


def run_sample_incident(
    llm: Optional[LLMLike] = None,
    incident_id: Optional[str] = None,
    random_seed: Optional[int] = None,
) -> IncidentReport:
    """Generate a fresh injected-drift incident and run it through the full
    real pipeline (evaluation.incident_pipeline.run_incident_pipeline).

    Args:
        llm: LLM passed to RCAAgent. Defaults to ``get_llm()`` (the stub).
        incident_id: optional explicit incident id; a fresh uuid4 if None.
        random_seed: forwarded to generate_injected_drift_data().

    Returns:
        A complete IncidentReport.
    """
    reference_df, current_df, injected_label, drifted_features = generate_injected_drift_data(
        random_seed
    )
    return run_incident_pipeline(
        reference_df,
        current_df,
        target_column=TARGET_COLUMN,
        llm=llm or get_llm(),
        incident_id=incident_id,
        ground_truth_label=injected_label,
        injected_feature_name=drifted_features,
    )
