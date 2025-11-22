export interface FeatureBounds {
  min?: number;
  max?: number;
  step?: number;
  scale?: string;
}

export interface DatasetRange {
  min: number;
  max: number;
  mean: number;
  step: number;
}

export interface FeatureMeta {
  name: string;
  label: string;
  unit: string;
  tendency: number;
  adjustable: boolean;
  bounds?: FeatureBounds | null;
  dataset_range?: DatasetRange | null;
}

export interface MetaResponse {
  classes: string[];
  classifier_features: string[];
  regressor_features: string[];
  adjustable_features: string[];
  features: Record<string, FeatureMeta>;
  defaults: Record<string, number>;
}

export interface BeamSummary {
  id: string;
}

export interface BeamListResponse {
  beams: BeamSummary[];
  count: number;
}

export interface GroundTruth {
  mode?: string | null;
  frt?: number | null;
}

export interface BeamDetailResponse {
  id: string;
  features: Record<string, number>;
  ground_truth?: GroundTruth | null;
}

export interface PredictionResponse {
  predicted_mode: string;
  probabilities: Record<string, number>;
  frt_minutes: number;
  threshold_minutes: number;
  gap_minutes: number;
  verdict: string;
}

export interface PlaygroundDeltas {
  frt_delta: number;
  prob_no_failure_delta: number;
  mode_changed: boolean;
  previous_mode?: string | null;
}

export interface PlaygroundResponse extends PredictionResponse {
  deltas?: PlaygroundDeltas | null;
}

export interface ScenarioResponse {
  exposure: number;
  margin: number;
  threshold: number;
  frt_pred: number;
  gap_minutes: number;
  pred_mode: string;
  prob_no_failure: number;
  verdict: string;
}

export interface RecommendationResponse {
  feature: string;
  title: string;
  action: string;
  current: number;
  target: number;
  delta_value: number;
  expected_frt: number;
  expected_delta_frt: number;
  expected_prob_no_fail: number;
  expected_delta_prob: number;
  new_mode: string;
  rationale: string;
  priority: number;
  unit: string;
}

export interface CombinationResponse {
  features: string[];
  actions: string[];
  expected_frt: number;
  expected_delta_frt: number;
  expected_prob_no_fail: number;
  expected_delta_prob: number;
  predicted_mode: string;
}

export interface EurocodeCheckResponse {
  verdict: string;
  criteria: Record<string, unknown>[];
  adjustments: Record<string, unknown>[];
}

export interface PredictionSummary {
  predicted_mode: string;
  probabilities: Record<string, number>;
  frt_minutes: number;
  threshold_minutes: number;
  gap_minutes: number;
  verdict: string;
}

export interface AnalyzeResponse {
  scenario: ScenarioResponse;
  recommendations: RecommendationResponse[];
  combination?: CombinationResponse | null;
  eurocode?: EurocodeCheckResponse | null;
  notes: string[];
  prediction: PredictionSummary;
}

export interface BeamInput {
  features: Record<string, number>;
  exposure_minutes: number;
  margin_minutes: number;
}

export interface PlaygroundInput {
  features: Record<string, number>;
  baseline_features?: Record<string, number> | null;
  exposure_minutes: number;
  margin_minutes: number;
}

export interface ParsedBeamRow {
  label: string;
  features: Record<string, number>;
  valid: boolean;
  errors: string[];
}

export interface CsvParseResult {
  fileName: string;
  rows: ParsedBeamRow[];
  totalRows: number;
  validRows: number;
  missingColumns: string[];
}
