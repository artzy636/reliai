# ReliAI

An agentic framework for autonomous detection, root-cause diagnosis, and
verified remediation of ML pipeline failures.

**Research question:** Can a multi-agent AI system that reasons over
structured evidence (an evidence graph, built from statistical anomaly
detection) improve root-cause analysis accuracy over LLM-only approaches?

**Current scope (v1):** one failure domain (data quality), evaluated against
RCAEval. Pipeline/Model/Infrastructure agents are designed but not
implemented in v1 — see `docs/PROJECT_ROADMAP.md`.

## Team & layers

| Layer | Folder | Owner | Produces |
|---|---|---|---|
| Detection | `detection/` | TBD | `EvidenceEvent` |
| Reasoning (Evidence Graph + RCA) | `reasoning/` | TBD | `RCAResult` |
| Remediation & Verification | `remediation/` | TBD | `RemediationPlan`, `VerificationResult` |
| Evaluation & benchmark scoring | `evaluation/` | shared | `IncidentReport` scores |

Fill in owners in this table once you've split tasks — see `docs/TASKS.md`
for the actual task breakdown per layer.

**Before writing any component code, read `CONTRACT.md`.** It explains the
shared data schemas in `schemas.py` that every layer imports from — this is
what lets the three of you build independently without breaking each other.

## Setup

```bash
git clone <repo-url>
cd reliai
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# sanity-check the shared contract compiles
python schemas.py

# run tests
pytest
```

## Repo structure

```
reliai/
├── schemas.py              # shared data contract — read CONTRACT.md first
├── CONTRACT.md             # rules for using/changing schemas.py
├── configs/
│   └── settings.py          # thresholds & config — never hardcode these in agents
├── detection/                # anomaly detection + Data Agent + fault injection
├── reasoning/                 # evidence graph (NetworkX) + RCA Agent (LangGraph)
├── remediation/               # Remediation Agent + Verification Agent + Streamlit frontend
├── evaluation/                # benchmark runner, scoring against ground truth
├── tests/                     # pytest tests, one file per layer
└── docs/
    ├── PROJECT_ROADMAP.md    # week-by-week plan, kept up to date
    └── TASKS.md               # concrete first tasks per layer
```

## Development principles

- Type hints, docstrings, and logging on everything — no exceptions.
- No hardcoded thresholds or magic numbers — put them in `configs/settings.py`.
- LLMs reason; statistics detect. Don't use an LLM anywhere a statistical
  test or conventional algorithm would do the job.
- Every injected benchmark failure needs a known ground truth label.
- Run `python schemas.py` and `pytest` before pushing anything that touches
  a shared schema.
