"""Core data model stubs used for testing the API."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
from datetime import datetime


class ModelType(Enum):
    """Supported model types."""
    OPENVLA = "openvla"
    RT1 = "rt1"
    CUSTOM = "custom"


class ModelFramework(Enum):
    """Supported model frameworks."""
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    JAX = "jax"


class OptimizationStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class SessionStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowStatus(Enum):
    INITIALIZING = "initializing"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ValidationStatus(Enum):
    PENDING = "pending"
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


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
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    model_type: ModelType = ModelType.CUSTOM
    framework: ModelFramework = ModelFramework.PYTORCH
    checksum: str = ""
    
    def __post_init__(self):
        """Validate the model metadata after initialization."""
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())
        self._validate()
    
    def _validate(self):
        """Validate model metadata fields."""
        if not self.name or not self.name.strip():
            raise ValueError("Model name cannot be empty")
        
        if not self.version or not self.version.strip():
            raise ValueError("Model version cannot be empty")
        
        if self.size_mb < 0:
            raise ValueError("Model size cannot be negative")
        
        if self.parameters < 0:
            raise ValueError("Parameter count cannot be negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'tags': self.tags,
            'file_path': self.file_path,
            'size_mb': self.size_mb,
            'author': self.author,
            'parameters': self.parameters,
            'created_at': self.created_at.isoformat() if isinstance(self.created_at, datetime) else self.created_at,
            'updated_at': self.updated_at.isoformat() if isinstance(self.updated_at, datetime) else self.updated_at,
            'version': self.version,
            'model_type': self.model_type.value if isinstance(self.model_type, ModelType) else self.model_type,
            'framework': self.framework.value if isinstance(self.framework, ModelFramework) else self.framework,
            'checksum': self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary."""
        # Handle datetime fields
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()
        
        updated_at = data.get('updated_at')
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now()
        
        # Handle enum fields
        model_type = data.get('model_type', ModelType.CUSTOM)
        if isinstance(model_type, str):
            model_type = ModelType(model_type)
        
        framework = data.get('framework', ModelFramework.PYTORCH)
        if isinstance(framework, str):
            framework = ModelFramework(framework)
        
        return cls(
            id=data.get('id', ''),
            name=data.get('name', ''),
            description=data.get('description', ''),
            tags=data.get('tags', []),
            file_path=data.get('file_path', ''),
            size_mb=data.get('size_mb', 0.0),
            author=data.get('author', ''),
            parameters=data.get('parameters', 0),
            created_at=created_at,
            updated_at=updated_at,
            version=data.get('version', '1.0.0'),
            model_type=model_type,
            framework=framework,
            checksum=data.get('checksum', '')
        )


@dataclass
class OptimizationSession:
    id: str = ""
    model_id: str = ""
    status: OptimizationStatus = OptimizationStatus.PENDING
    criteria_name: str = ""
    created_by: str = ""
    steps: List[Any] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    priority: int = 1
    tags: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Optional['OptimizationResults'] = None
    
    def __post_init__(self):
        """Validate the optimization session after initialization."""
        if not self.id:
            import uuid
            self.id = str(uuid.uuid4())
        self._validate()
    
    def _validate(self):
        """Validate optimization session fields."""
        if not self.model_id or not self.model_id.strip():
            raise ValueError("Model ID cannot be empty")
        
        if self.priority < 1 or self.priority > 5:
            raise ValueError("Priority must be between 1 and 5")
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate session duration in seconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if the session is currently active."""
        return self.status in [OptimizationStatus.PENDING, OptimizationStatus.RUNNING]


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


@dataclass
class AnalysisReport:
    model_id: str = ""
    analysis_id: str = ""
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    analysis_duration_seconds: float = 0.0
    architecture_summary: Optional['ArchitectureSummary'] = None
    performance_profile: Optional['PerformanceProfile'] = None
    optimization_opportunities: List['OptimizationOpportunity'] = field(default_factory=list)
    recommendations: List['Recommendation'] = field(default_factory=list)
    compatibility_matrix: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the analysis report after initialization."""
        if not self.analysis_id:
            import uuid
            self.analysis_id = str(uuid.uuid4())
        self._validate()
    
    def _validate(self):
        """Validate analysis report fields."""
        if not self.model_id or not self.model_id.strip():
            raise ValueError("Model ID cannot be empty")
        
        if self.analysis_duration_seconds < 0:
            raise ValueError("Analysis duration cannot be negative")


@dataclass
class OptimizationPlan:
    pass


@dataclass
class OptimizationResults:
    optimized_model: Any = None
    original_model_size_mb: float = 0.0
    optimized_model_size_mb: float = 0.0
    size_reduction_percent: float = 0.0
    performance_improvements: Dict[str, float] = field(default_factory=dict)
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    optimization_summary: str = ""
    techniques_applied: List[str] = field(default_factory=list)
    rollback_available: bool = False
    validation_passed: bool = False


@dataclass
class EvaluationReport:
    model_id: str = ""
    evaluation_id: str = ""
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    evaluation_duration_seconds: float = 0.0
    benchmarks: List['BenchmarkResult'] = field(default_factory=list)
    performance_metrics: Optional['PerformanceMetrics'] = None
    comparison_baseline: Optional['ComparisonResult'] = None
    validation_status: ValidationStatus = ValidationStatus.PENDING
    validation_errors: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate the evaluation report after initialization."""
        if not self.evaluation_id:
            import uuid
            self.evaluation_id = str(uuid.uuid4())
        self._validate()
    
    def _validate(self):
        """Validate evaluation report fields."""
        if not self.model_id or not self.model_id.strip():
            raise ValueError("Model ID cannot be empty")
        
        if self.evaluation_duration_seconds < 0:
            raise ValueError("Evaluation duration cannot be negative")
    
    @property
    def overall_success(self) -> bool:
        """Check if the evaluation was overall successful."""
        return (self.validation_status == ValidationStatus.PASSED and 
                len(self.validation_errors) == 0)


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
    total_layers: int = 0
    layer_types: Dict[str, int] = field(default_factory=dict)
    total_parameters: int = 0
    trainable_parameters: int = 0
    model_depth: int = 0
    memory_footprint_mb: float = 0.0


@dataclass
class PerformanceProfile:
    memory_usage_mb: float = 0.0
    inference_time_ms: float = 0.0
    throughput_samples_per_sec: float = 0.0
    gpu_utilization_percent: float = 0.0
    cpu_utilization_percent: float = 0.0
    profiling_timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationOpportunity:
    technique: str = ""
    estimated_size_reduction_percent: float = 0.0
    estimated_speed_improvement_percent: float = 0.0
    estimated_accuracy_impact_percent: float = 0.0
    confidence_score: float = 0.0
    complexity: str = "medium"
    description: str = ""


@dataclass
class Recommendation:
    technique: str = ""
    priority: int = 1
    rationale: str = ""
    expected_benefits: List[str] = field(default_factory=list)
    potential_risks: List[str] = field(default_factory=list)
    estimated_effort: str = "medium"


@dataclass
class OptimizationStep:
    """Placeholder for optimization plan step."""
    step_id: str = ""
    technique: str = ""
    status: str = "pending"
    parameters: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    
    def __post_init__(self):
        """Validate the optimization step after initialization."""
        if not self.step_id:
            import uuid
            self.step_id = str(uuid.uuid4())


@dataclass
class BenchmarkResult:
    benchmark_name: str = ""
    score: float = 0.0
    execution_time_seconds: float = 0.0
    unit: str = ""
    higher_is_better: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    accuracy: float = 0.0
    inference_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    model_size_mb: float = 0.0
    throughput_samples_per_sec: float = 0.0
    flops: int = 0
    energy_consumption_watts: float = 0.0
    custom_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComparisonResult:
    original_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    optimized_metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    improvements: Dict[str, float] = field(default_factory=dict)
    regressions: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    recommendation: str = ""

