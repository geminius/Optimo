"""Core data model stubs used for testing the API."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class ModelMetadata:
    id: str = ""
    name: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    file_path: str = ""
    size_mb: float = 0.0
    author: str = ""
    parameters: int = 0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationSession:
    id: str = ""
    model_id: str = ""
    status: str = ""
    criteria_name: str = ""
    created_by: str = ""
    steps: List[Any] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 1


@dataclass
class SessionSnapshot:
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowState:
    session_id: str = ""
    status: Optional["WorkflowStatus"] = None
    progress_percentage: float = 0.0
    current_step: str = ""
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None


class WorkflowStatus(Enum):
    INITIALIZING = "initializing"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class AnalysisReport:
    pass


@dataclass
class OptimizationPlan:
    pass


@dataclass
class OptimizationResults:
    optimized_model: Any = None


@dataclass
class EvaluationReport:
    pass


class OptimizationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SessionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ProgressUpdate:
    status: OptimizationStatus = OptimizationStatus.PENDING
    progress_percentage: float = 0.0
    current_step: str = ""
    estimated_remaining_minutes: Optional[int] = None
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SessionContext:
    pass


@dataclass
class ArchitectureSummary:
    pass


@dataclass
class PerformanceProfile:
    pass


@dataclass
class OptimizationOpportunity:
    technique: str = ""
    confidence_score: float = 0.0


@dataclass
class Recommendation:
    technique: str = ""
    priority: int = 1


@dataclass
class OptimizationStep:
    """Placeholder for optimization plan step."""
    step_id: str = ""
    technique: str = ""
    status: str = "pending"
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    benchmark_name: str = ""
    score: float = 0.0
    execution_time_seconds: float = 0.0


@dataclass
class PerformanceMetrics:
    accuracy: float = 0.0
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0


@dataclass
class ComparisonResult:
    pass


class ValidationStatus(Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"

