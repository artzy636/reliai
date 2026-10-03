// Mirrors schemas.py exactly (field names, enum values) -- this file has
// no logic, only shape, so keeping it in lockstep with the Python side is
// a matter of re-running the same field/enum dump if schemas.py changes:
//   python -c "from schemas import IncidentReport; print(IncidentReport.model_fields)"

export type FailureType =
  | "missing_values"
  | "corrupted_values"
  | "schema_mismatch"
  | "duplicates"
  | "feature_drift"
  | "label_shift";

export type DetectionMethod =
  | "ks_test"
  | "population_stability_index"
  | "jensen_shannon_divergence"
  | "isolation_forest"
  | "rolling_accuracy"
  | "missing_value_rate"
  | "duplicate_row_rate"
  | "schema_check";

export type RemediationCategory = "data_fix" | "retrain" | "rollback" | "config_change";

export interface EvidenceEvent {
  event_id: string;
  timestamp: string;
  detection_method: DetectionMethod;
  feature_name: string | null;
  metric_value: number;
  threshold: number;
  confidence: number;
  description: string;
  ground_truth_label: FailureType | null;
}

export interface EvidenceNode {
  node_id: string;
  node_type: string;
  source_events: string[];
  timestamp: string;
  confidence: number;
  detection_method: DetectionMethod;
}

export interface EvidenceEdge {
  source_node_id: string;
  target_node_id: string;
  relationship: string;
  confidence: number;
  evidence: string;
}

export interface RootCauseHypothesis {
  rank: number;
  node_id: string;
  explanation: string;
  confidence: number;
  supporting_node_ids: string[];
  failure_type: FailureType;
}

export interface RCAResult {
  incident_id: string;
  hypotheses: RootCauseHypothesis[];
  causal_path_node_ids: string[];
  generated_at: string;
}

export interface RemediationPlan {
  incident_id: string;
  category: RemediationCategory;
  action_description: string;
  expected_outcome: string;
  target_root_cause_node_id: string | null;
}

export interface VerificationResult {
  incident_id: string;
  metric_name: string;
  value_before: number;
  value_after: number;
  improved: boolean;
  replay_sample_size: number;
  notes: string | null;
}

export interface IncidentReport {
  incident_id: string;
  detected_at: string;
  evidence_events: EvidenceEvent[];
  evidence_nodes: EvidenceNode[];
  rca_result: RCAResult;
  remediation_plan: RemediationPlan;
  verification_result: VerificationResult;
  ground_truth_label: FailureType | null;
  // Computed server-side, not part of the persisted schema -- see
  // api/main.py's _report_to_dict().
  edges: EvidenceEdge[];
  top_node_id: string | null;
  root_cause_correct: boolean | null;
}

export interface ReportSummary {
  name: string;
  label: string;
  incident_id: string;
  detected_at: string;
  ground_truth_label: FailureType | null;
  root_cause_correct: boolean | null;
}

export interface PerRepeatAccuracy {
  structured_accuracy: number;
  naive_accuracy: number;
}

export interface IncidentHitRate {
  structured_hits: number;
  naive_hits: number;
  total: number;
}

export interface DetectionRecallRow {
  name: string;
  fault_type: FailureType;
  detected: boolean;
  confidence: number | null;
  detection_methods: string[];
}

export interface AblationRow {
  n_incidents: number;
  structured_accuracy: number;
  naive_accuracy: number;
  structured_hit3: number;
  naive_hit3: number;
}

export interface AblationResult {
  configs: Record<string, AblationRow>;
}

export interface BenchmarkResult {
  // Optional: absent in result files generated before Hit@3/MRR and
  // detection-recall existed; regenerate with the --output flag to get them.
  structured_hit3_mean?: number;
  naive_hit3_mean?: number;
  structured_mrr_mean?: number;
  naive_mrr_mean?: number;
  detection_recall?: DetectionRecallRow[];
  n_repeats: number;
  structured_accuracy_mean: number;
  structured_accuracy_stdev: number;
  naive_accuracy_mean: number;
  naive_accuracy_stdev: number;
  per_repeat: PerRepeatAccuracy[];
  per_incident_hit_rates: Record<string, IncidentHitRate>;
}
