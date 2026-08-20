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

from detection.fault_injection import inject_feature_drift
from evaluation.incident_pipeline import run_incident_pipeline
from reasoning.rca_agent import LLMLike
from schemas import FailureType, IncidentReport

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [f"feature_{i}" for i in range(5)]
TARGET_COLUMN = "label"
DRIFTED_FEATURE = "feature_1"

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


def generate_injected_drift_data(
    random_seed: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, FailureType]:
    """Build a fresh reference/current dataframe pair with real feature drift.

    Same construction tests/test_incident_pipeline.py uses: a synthetic
    classification dataset where feature_1 is a real decision-boundary
    feature, then current_df is reference_df's exact rows with feature_1
    shifted via detection.fault_injection.inject_feature_drift -- so every
    other column stays byte-identical and DataAgent's KS test can only ever
    flag the drifted column.

    Args:
        random_seed: seed for both the synthetic dataset and the drift
            injection. None (the default) picks a fresh random seed each
            call, so repeated "Run live" clicks in the Streamlit app
            produce a genuinely new incident each time rather than
            replaying the same one.

    Returns:
        (reference_df, current_df, injected_label) -- injected_label is
        always FailureType.FEATURE_DRIFT (what inject_feature_drift injects).
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

    current_df, injected_label = inject_feature_drift(
        reference_df, DRIFTED_FEATURE, shift_amount=3.0, random_seed=seed
    )
    return reference_df, current_df, injected_label


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
    reference_df, current_df, injected_label = generate_injected_drift_data(random_seed)
    return run_incident_pipeline(
        reference_df,
        current_df,
        target_column=TARGET_COLUMN,
        llm=llm or get_llm(),
        incident_id=incident_id,
        ground_truth_label=injected_label,
        injected_feature_name=DRIFTED_FEATURE,
    )
