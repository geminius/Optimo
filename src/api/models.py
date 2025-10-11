"""
Pydantic models for API request/response validation.
"""

import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ModelUploadResponse(BaseModel):
    """Response model for model upload."""
    model_id: str
    filename: str
    size_mb: float
    upload_time: datetime
    message: str


class OptimizationRequest(BaseModel):
    """Request model for starting optimization."""
    model_id: str = Field(..., description="ID of the model to optimize")
    criteria_name: Optional[str] = Field(None, description="Name of optimization criteria")
    target_accuracy_threshold: Optional[float] = Field(0.95, description="Minimum accuracy to maintain")
    max_size_reduction_percent: Optional[float] = Field(50.0, description="Maximum allowed size reduction")
    max_latency_increase_percent: Optional[float] = Field(10.0, description="Maximum allowed latency increase")
    optimization_techniques: Optional[List[str]] = Field(None, description="Specific techniques to apply")
    priority: Optional[int] = Field(1, description="Optimization priority (1-5)")
    notes: Optional[str] = Field(None, description="Additional notes")


class OptimizationResponse(BaseModel):
    """Response model for optimization start."""
    session_id: str
    model_id: str
    status: str
    message: str


class SessionStatusResponse(BaseModel):
    """Response model for session status."""
    session_id: str
    status: str
    progress_percentage: float
    current_step: Optional[str]
    start_time: datetime
    last_update: datetime
    error_message: Optional[str]
    model_id: str
    steps_completed: int


class OptimizationSessionSummary(BaseModel):
    """Summary information for an optimization session."""
    session_id: str = Field(..., description="Unique session identifier")
    model_id: str = Field(..., description="Model being optimized")
    model_name: str = Field(..., description="Human-readable model name")
    status: str = Field(..., description="Current session status")
    progress_percentage: float = Field(..., ge=0.0, le=100.0, description="Progress percentage")
    techniques: List[str] = Field(default_factory=list, description="Optimization techniques applied")
    size_reduction_percent: Optional[float] = Field(None, description="Size reduction achieved")
    speed_improvement_percent: Optional[float] = Field(None, description="Speed improvement achieved")
    created_at: datetime = Field(..., description="Session creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")


class SessionListResponse(BaseModel):
    """Response model for session list with pagination."""
    sessions: List[OptimizationSessionSummary] = Field(..., description="List of optimization sessions")
    total: int = Field(..., description="Total number of sessions matching filters")
    skip: int = Field(0, description="Number of sessions skipped")
    limit: int = Field(50, description="Maximum number of sessions returned")


class ModelInfo(BaseModel):
    """Model information."""
    id: str
    name: str
    size_mb: float
    created_at: datetime
    file_path: str


class ModelListResponse(BaseModel):
    """Response model for model list."""
    models: List[Dict[str, Any]]
    total: int
    skip: int
    limit: int


class EvaluationResponse(BaseModel):
    """Response model for evaluation results."""
    session_id: str
    model_id: str
    status: str
    optimization_summary: str
    performance_improvements: Dict[str, float]
    techniques_applied: List[str]
    evaluation_metrics: Dict[str, float]
    comparison_baseline: Dict[str, float]
    recommendations: List[str]


class ErrorResponse(BaseModel):
    """Standardized error response model with request tracking."""
    error: str = Field(..., description="Error type/code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    request_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()), description="Unique request ID for tracking")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    services: Dict[str, bool]


class DashboardStats(BaseModel):
    """Dashboard statistics response model."""
    total_models: int = Field(..., ge=0, description="Total number of models in the system")
    active_optimizations: int = Field(..., ge=0, description="Number of currently running optimizations")
    completed_optimizations: int = Field(..., ge=0, description="Number of completed optimizations")
    failed_optimizations: int = Field(..., ge=0, description="Number of failed optimizations")
    average_size_reduction: float = Field(..., description="Average size reduction percentage across completed optimizations")
    average_speed_improvement: float = Field(..., description="Average speed improvement percentage across completed optimizations")
    total_sessions: int = Field(..., ge=0, description="Total number of optimization sessions")
    last_updated: datetime = Field(default_factory=datetime.now, description="Timestamp of last statistics update")


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response model."""
    access_token: str
    token_type: str
    expires_in: int
    user: Dict[str, str]


class User(BaseModel):
    """User model."""
    id: str
    username: str
    role: str
    email: Optional[str] = None
    is_active: bool = True


class ProgressUpdate(BaseModel):
    """Progress update model for real-time monitoring."""
    session_id: str
    status: str
    progress_percentage: float
    current_step: Optional[str]
    timestamp: datetime
    message: Optional[str] = None


class OptimizationCriteriaRequest(BaseModel):
    """Request model for optimization criteria configuration."""
    name: str
    target_accuracy_threshold: float = Field(0.95, ge=0.0, le=1.0)
    max_size_reduction_percent: float = Field(50.0, ge=0.0, le=100.0)
    max_latency_increase_percent: float = Field(10.0, ge=0.0)
    optimization_techniques: List[str] = Field(default_factory=list)
    hardware_constraints: Optional[Dict[str, Any]] = None
    custom_parameters: Optional[Dict[str, Any]] = None


class OptimizationCriteriaResponse(BaseModel):
    """Response model for optimization criteria."""
    name: str
    target_accuracy_threshold: float
    max_size_reduction_percent: float
    max_latency_increase_percent: float
    optimization_techniques: List[str]
    hardware_constraints: Optional[Dict[str, Any]]
    custom_parameters: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime


class BenchmarkResult(BaseModel):
    """Benchmark result model."""
    benchmark_name: str
    score: float
    unit: str
    higher_is_better: bool
    baseline_score: Optional[float] = None
    improvement_percent: Optional[float] = None
    execution_time_seconds: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelAnalysisResponse(BaseModel):
    """Response model for model analysis."""
    model_id: str
    analysis_id: str
    architecture_summary: Dict[str, Any]
    performance_profile: Dict[str, Any]
    optimization_opportunities: List[Dict[str, Any]]
    compatibility_matrix: Dict[str, bool]
    recommendations: List[Dict[str, Any]]
    analysis_timestamp: datetime
    analysis_duration_seconds: float


class OptimizationPlanResponse(BaseModel):
    """Response model for optimization plan."""
    plan_id: str
    model_id: str
    steps: List[Dict[str, Any]]
    estimated_duration_minutes: float
    expected_improvements: Dict[str, float]
    risk_assessment: Dict[str, Any]
    created_at: datetime


class SystemMetrics(BaseModel):
    """System metrics model."""
    cpu_usage_percent: float
    memory_usage_percent: float
    disk_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    active_sessions: int
    total_models: int
    timestamp: datetime


class SystemStatusResponse(BaseModel):
    """System status response model."""
    status: str
    metrics: SystemMetrics
    services: Dict[str, Dict[str, Any]]
    version: str
    uptime_seconds: float