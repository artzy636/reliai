"""One-off script: shows exactly what the naive vs structured RCA agents
said for benchmark-schema-cascade-001, the incident where naive got it
wrong. Run from the project root with the venv active and GOOGLE_API_KEY
already set:

    python inspect_schema_incident.py
"""
import logging

from evaluation.benchmark_runner import _cascading_schema_incident
from remediation.pipeline_utils import get_llm
from reasoning.rca_agent import NaiveRCAAgent, RCAAgent
from reasoning.evidence_graph import EvidenceGraphBuilder

logging.basicConfig(level=logging.WARNING)

incident = _cascading_schema_incident()
llm = get_llm(use_stub=False)

naive = NaiveRCAAgent(llm=llm)
naive_result = naive.analyze(incident.events, incident_id=incident.incident_id)
print("=== NAIVE hypotheses ===")
for h in naive_result.hypotheses:
    print(f"- event_ids={h.supporting_node_ids} conf={h.confidence:.2f}")
    print(f"  {h.explanation}")
    print()

graph = EvidenceGraphBuilder().build(incident.events)
structured = RCAAgent(llm=llm)
structured_result = structured.analyze(graph, incident_id=incident.incident_id)
print("=== STRUCTURED hypotheses ===")
for h in structured_result.hypotheses:
    print(f"- root={h.node_id!r} conf={h.confidence:.2f}")
    print(f"  {h.explanation}")
    print()

print("Ground truth:", incident.ground_truth_label)
