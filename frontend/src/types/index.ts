export interface ModelMetadata {
  id: string;
  name: string;
  version: string;
  model_type: string;
  framework: string;
  size_mb: number;
  parameters: number;
  created_at: string;
  tags: string[];
}

export interface OptimizationSession {
  id: string;
  model_id: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled' | 'paused';
  criteria: OptimizationCriteria;
  plan?: OptimizationPlan;
  steps: OptimizationStep[];
  results?: OptimizationResults;
  created_at: string;
  updated_at: string;
  progress?: number;
}

export interface OptimizationCriteria {
  max_size_reduction: number;
  min_accuracy_retention: number;
  max_inference_time: number;
  techniques: string[];
  hardware_target: string;
}

export interface OptimizationPlan {
  techniques: string[];
  estimated_duration: number;
  expected_improvements: {
    size_reduction: number;
    speed_improvement: number;
    accuracy_impact: number;
  };
}

export interface OptimizationStep {
  id: string;
  technique: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused';
  progress: number;
  start_time?: string;
  end_time?: string;
  error_message?: string;
}

export interface OptimizationResults {
  original_size_mb: number;
  optimized_size_mb: number;
  size_reduction_percent: number;
  original_inference_time_ms: number;
  optimized_inference_time_ms: number;
  speed_improvement_percent: number;
  accuracy_retention_percent: number;
  techniques_applied: string[];
}

export interface AnalysisReport {
  model_id: string;
  architecture_summary: {
    total_parameters: number;
    model_size_mb: number;
    layer_count: number;
    framework: string;
  };
  performance_profile: {
    inference_time_ms: number;
    memory_usage_mb: number;
    throughput_samples_per_sec: number;
  };
  optimization_opportunities: OptimizationOpportunity[];
  compatibility_matrix: Record<string, boolean>;
  recommendations: string[];
}

export interface OptimizationOpportunity {
  technique: string;
  estimated_impact: {
    size_reduction: number;
    speed_improvement: number;
    accuracy_impact: number;
  };
  feasibility_score: number;
  requirements: string[];
}

export interface EvaluationReport {
  model_id: string;
  benchmarks: BenchmarkResult[];
  performance_metrics: {
    accuracy: number;
    inference_time_ms: number;
    memory_usage_mb: number;
    throughput: number;
  };
  comparison_baseline?: {
    accuracy_change: number;
    speed_change: number;
    size_change: number;
  };
  validation_status: 'passed' | 'failed' | 'warning';
  recommendations: string[];
}

export interface BenchmarkResult {
  name: string;
  score: number;
  unit: string;
  baseline_score?: number;
}

export interface ProgressUpdate {
  session_id: string;
  step_id?: string;
  progress: number;
  status: string;
  message?: string;
  timestamp: string;
}