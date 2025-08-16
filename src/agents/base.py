"""
Base interfaces and abstract classes for all agent types.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import torch
import logging
import traceback
import copy

from ..utils.exceptions import (
    PlatformError, OptimizationError, ValidationError, ErrorSeverity
)
from ..utils.retry import RetryConfig, RetryableOperation, STANDARD_RETRY
from ..utils.recovery import (
    recovery_manager, model_recovery_manager, degradation_manager,
    RecoveryStrategy
)


@dataclass
class ImpactEstimate:
    """Estimated impact of an optimization technique."""
    performance_improvement: float  # Expected performance gain (0.0 to 1.0)
    size_reduction: float  # Expected model size reduction (0.0 to 1.0)
    speed_improvement: float  # Expected inference speed improvement (0.0 to 1.0)
    confidence: float  # Confidence in the estimate (0.0 to 1.0)
    estimated_time_minutes: int  # Estimated optimization time


@dataclass
class ValidationResult:
    """Result of model validation."""
    is_valid: bool
    accuracy_preserved: bool
    performance_metrics: Dict[str, float]
    issues: List[str]
    recommendations: List[str]


class OptimizationStatus(Enum):
    """Status of optimization operation."""
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    ANALYZING = "analyzing"
    OPTIMIZING = "optimizing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


@dataclass
class ProgressUpdate:
    """Progress update for optimization operation."""
    status: OptimizationStatus
    progress_percentage: float  # 0.0 to 100.0
    current_step: str
    estimated_remaining_minutes: Optional[int] = None
    message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class OptimizationSnapshot:
    """Snapshot of model state for rollback purposes."""
    model_state_dict: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    checkpoint_name: str


@dataclass
class OptimizationResult:
    """Result of optimization operation."""
    success: bool
    optimized_model: Optional[torch.nn.Module]
    original_model: Optional[torch.nn.Module]
    optimization_metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    optimization_time: float
    technique_used: str
    validation_result: Optional[ValidationResult]
    error_message: Optional[str] = None
    snapshots: List[OptimizationSnapshot] = field(default_factory=list)


@dataclass
class OptimizedModel:
    """Container for optimized model and metadata."""
    model: torch.nn.Module
    optimization_metadata: Dict[str, Any]
    performance_metrics: Dict[str, float]
    optimization_time: float
    technique_used: str


class BaseAgent(ABC):
    """Base abstract class for all agents in the platform."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.created_at = datetime.now()
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the agent with required resources."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources used by the agent."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the agent."""
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "config": self.config
        }


class BaseOptimizationAgent(BaseAgent):
    """Base abstract class for optimization agents with progress tracking and rollback support."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(f"{self.__class__.__name__}")
        self._current_status = OptimizationStatus.NOT_STARTED
        self._progress_callbacks: List[Callable[[ProgressUpdate], None]] = []
        self._snapshots: List[OptimizationSnapshot] = []
        self._current_operation_id: Optional[str] = None
        self._cancelled = False
    
    def add_progress_callback(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """Add a callback function to receive progress updates."""
        self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """Remove a progress callback function."""
        if callback in self._progress_callbacks:
            self._progress_callbacks.remove(callback)
    
    def _update_progress(self, status: OptimizationStatus, progress: float, 
                        step: str, estimated_remaining: Optional[int] = None,
                        message: Optional[str] = None) -> None:
        """Update optimization progress and notify callbacks."""
        self._current_status = status
        update = ProgressUpdate(
            status=status,
            progress_percentage=progress,
            current_step=step,
            estimated_remaining_minutes=estimated_remaining,
            message=message
        )
        
        self.logger.info(f"Progress update: {status.value} - {progress:.1f}% - {step}")
        if message:
            self.logger.info(f"Message: {message}")
        
        for callback in self._progress_callbacks:
            try:
                callback(update)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")
    
    def _create_snapshot(self, model: torch.nn.Module, checkpoint_name: str, 
                        metadata: Optional[Dict[str, Any]] = None) -> OptimizationSnapshot:
        """Create a snapshot of the model state for rollback purposes."""
        snapshot = OptimizationSnapshot(
            model_state_dict=copy.deepcopy(model.state_dict()),
            metadata=metadata or {},
            timestamp=datetime.now(),
            checkpoint_name=checkpoint_name
        )
        self._snapshots.append(snapshot)
        self.logger.info(f"Created snapshot: {checkpoint_name}")
        return snapshot
    
    def _rollback_to_snapshot(self, model: torch.nn.Module, 
                             snapshot: OptimizationSnapshot) -> bool:
        """Rollback model to a previous snapshot."""
        try:
            model.load_state_dict(snapshot.model_state_dict)
            self.logger.info(f"Successfully rolled back to snapshot: {snapshot.checkpoint_name}")
            self._update_progress(
                OptimizationStatus.ROLLED_BACK, 
                0.0, 
                "Rollback completed",
                message=f"Rolled back to {snapshot.checkpoint_name}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to rollback to snapshot {snapshot.checkpoint_name}: {e}")
            return False
    
    def _rollback_to_latest_snapshot(self, model: torch.nn.Module) -> bool:
        """Rollback to the most recent snapshot."""
        if not self._snapshots:
            self.logger.warning("No snapshots available for rollback")
            return False
        
        latest_snapshot = self._snapshots[-1]
        return self._rollback_to_snapshot(model, latest_snapshot)
    
    def cancel_optimization(self) -> None:
        """Cancel the current optimization operation."""
        self._cancelled = True
        self.logger.info("Optimization cancellation requested")
        self._update_progress(
            OptimizationStatus.CANCELLED,
            0.0,
            "Cancellation requested",
            message="Optimization will be cancelled at next checkpoint"
        )
    
    def is_cancelled(self) -> bool:
        """Check if optimization has been cancelled."""
        return self._cancelled
    
    def get_current_status(self) -> OptimizationStatus:
        """Get current optimization status."""
        return self._current_status
    
    def get_snapshots(self) -> List[OptimizationSnapshot]:
        """Get list of available snapshots."""
        return self._snapshots.copy()
    
    def clear_snapshots(self) -> None:
        """Clear all snapshots to free memory."""
        self._snapshots.clear()
        self.logger.info("Cleared all snapshots")
    
    def optimize_with_tracking(self, model: torch.nn.Module, 
                              config: Dict[str, Any],
                              operation_id: Optional[str] = None) -> OptimizationResult:
        """
        Execute optimization with full progress tracking, error handling, and rollback support.
        This is the main entry point for optimization operations.
        """
        self._current_operation_id = operation_id or f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        # Don't reset cancelled flag - it might have been set before calling this method
        self._snapshots.clear()
        
        start_time = datetime.now()
        original_model = None
        model_id = config.get('model_id', 'unknown')
        
        # Create model snapshot for recovery
        snapshot_id = model_recovery_manager.create_model_snapshot(
            model_id, model, {"operation_id": self._current_operation_id}
        )
        
        try:
            # Check for cancellation before starting
            if self._cancelled:
                return self._create_cancelled_result(None, start_time)
            
            # Initialize with retry logic
            retry_config = RetryConfig(max_attempts=3, base_delay=1.0)
            retry_op = RetryableOperation(
                "optimization_initialization",
                config=retry_config,
                on_retry=lambda e, attempt: self.logger.warning(f"Retrying initialization, attempt {attempt}: {e}")
            )
            
            def initialize_optimization():
                self._update_progress(OptimizationStatus.INITIALIZING, 0.0, "Initializing optimization")
                
                # Create initial snapshot
                nonlocal original_model
                original_model = copy.deepcopy(model)
                self._create_snapshot(model, "original", {"operation_id": self._current_operation_id})
                
                # Check if we can optimize this model
                if not self.can_optimize(model):
                    raise OptimizationError(
                        f"Model cannot be optimized by {self.__class__.__name__}",
                        technique=self.__class__.__name__,
                        session_id=self._current_operation_id,
                        step="initialization"
                    )
                return True
            
            retry_op.execute(initialize_optimization)
            
            if self._cancelled:
                return self._create_cancelled_result(original_model, start_time)
            
            # Analyze model with error handling
            def analyze_model():
                self._update_progress(OptimizationStatus.ANALYZING, 10.0, "Analyzing model")
                return self.estimate_impact(model)
            
            try:
                impact_estimate = analyze_model()
            except Exception as e:
                # Try to recover from analysis failure
                analysis_error = OptimizationError(
                    f"Model analysis failed: {str(e)}",
                    technique=self.__class__.__name__,
                    session_id=self._current_operation_id,
                    step="analysis"
                )
                
                if recovery_manager.handle_error(analysis_error, {"model_id": model_id}):
                    # Retry analysis after recovery
                    impact_estimate = analyze_model()
                else:
                    raise analysis_error
            
            if self._cancelled:
                return self._create_cancelled_result(original_model, start_time)
            
            # Execute optimization with comprehensive error handling
            def execute_optimization():
                self._update_progress(OptimizationStatus.OPTIMIZING, 20.0, "Executing optimization")
                
                # Create checkpoint before optimization
                self._create_snapshot(model, "pre_optimization", {
                    "operation_id": self._current_operation_id,
                    "technique": self.__class__.__name__
                })
                
                return self.optimize(model, config)
            
            try:
                optimized_model_data = execute_optimization()
            except Exception as e:
                # Handle optimization failure with recovery
                opt_error = OptimizationError(
                    f"Optimization execution failed: {str(e)}",
                    technique=self.__class__.__name__,
                    session_id=self._current_operation_id,
                    step="optimization"
                )
                
                # Attempt automatic rollback
                if self._rollback_to_latest_snapshot(model):
                    self.logger.info("Successfully rolled back after optimization failure")
                
                # Try recovery strategies
                if recovery_manager.handle_error(opt_error, {
                    "model_id": model_id,
                    "technique": self.__class__.__name__
                }):
                    # Retry optimization after recovery
                    optimized_model_data = execute_optimization()
                else:
                    # Recovery failed, return failure result
                    return OptimizationResult(
                        success=False,
                        optimized_model=None,
                        original_model=original_model,
                        optimization_metadata={},
                        performance_metrics={},
                        optimization_time=(datetime.now() - start_time).total_seconds(),
                        technique_used=self.__class__.__name__,
                        validation_result=None,
                        error_message=str(opt_error),
                        snapshots=self._snapshots.copy()
                    )
            
            if self._cancelled:
                return self._create_cancelled_result(original_model, start_time)
            
            # Validate result with error handling
            def validate_optimization():
                self._update_progress(OptimizationStatus.VALIDATING, 80.0, "Validating optimization result")
                return self.validate_result(original_model, optimized_model_data.model)
            
            try:
                validation_result = validate_optimization()
            except Exception as e:
                # Handle validation failure
                validation_error = ValidationError(
                    f"Validation failed: {str(e)}",
                    validation_type="optimization_result"
                )
                
                self.logger.error(f"Validation error: {validation_error}")
                
                # Create a failed validation result
                validation_result = ValidationResult(
                    is_valid=False,
                    accuracy_preserved=False,
                    performance_metrics={},
                    issues=[str(validation_error)],
                    recommendations=["Manual review required"]
                )
            
            # Handle validation failure
            if not validation_result.is_valid:
                self.logger.warning("Optimization validation failed, attempting rollback")
                
                rollback_success = self._rollback_to_latest_snapshot(model)
                if not rollback_success:
                    # Try model recovery manager as fallback
                    rollback_success = model_recovery_manager.restore_model_from_snapshot(
                        model, model_id, snapshot_id
                    )
                
                return OptimizationResult(
                    success=False,
                    optimized_model=None,
                    original_model=original_model,
                    optimization_metadata=optimized_model_data.optimization_metadata,
                    performance_metrics={},
                    optimization_time=(datetime.now() - start_time).total_seconds(),
                    technique_used=optimized_model_data.technique_used,
                    validation_result=validation_result,
                    error_message="Optimization validation failed",
                    snapshots=self._snapshots.copy()
                )
            
            # Complete successfully
            self._update_progress(OptimizationStatus.COMPLETED, 100.0, "Optimization completed successfully")
            
            return OptimizationResult(
                success=True,
                optimized_model=optimized_model_data.model,
                original_model=original_model,
                optimization_metadata=optimized_model_data.optimization_metadata,
                performance_metrics=optimized_model_data.performance_metrics,
                optimization_time=(datetime.now() - start_time).total_seconds(),
                technique_used=optimized_model_data.technique_used,
                validation_result=validation_result,
                snapshots=self._snapshots.copy()
            )
            
        except Exception as e:
            self.logger.error(f"Optimization failed with unhandled exception: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Attempt comprehensive recovery
            recovery_context = {
                "model_id": model_id,
                "operation_id": self._current_operation_id,
                "technique": self.__class__.__name__
            }
            
            # Try to rollback using internal snapshots first
            rollback_success = False
            if original_model is not None:
                rollback_success = self._rollback_to_latest_snapshot(model)
            
            # If internal rollback failed, try model recovery manager
            if not rollback_success:
                rollback_success = model_recovery_manager.restore_model_from_snapshot(
                    model, model_id, snapshot_id
                )
            
            # Try general error recovery
            recovery_manager.handle_error(e, recovery_context)
            
            self._update_progress(
                OptimizationStatus.FAILED, 
                0.0, 
                "Optimization failed",
                message=str(e)
            )
            
            return OptimizationResult(
                success=False,
                optimized_model=None,
                original_model=original_model,
                optimization_metadata={},
                performance_metrics={},
                optimization_time=(datetime.now() - start_time).total_seconds(),
                technique_used=self.__class__.__name__,
                validation_result=None,
                error_message=str(e),
                snapshots=self._snapshots.copy()
            )
    
    def _create_cancelled_result(self, original_model: Optional[torch.nn.Module], 
                                start_time: datetime) -> OptimizationResult:
        """Create result object for cancelled optimization."""
        return OptimizationResult(
            success=False,
            optimized_model=None,
            original_model=original_model,
            optimization_metadata={},
            performance_metrics={},
            optimization_time=(datetime.now() - start_time).total_seconds(),
            technique_used="cancelled",
            validation_result=None,
            error_message="Optimization was cancelled",
            snapshots=self._snapshots.copy()
        )
    
    @abstractmethod
    def can_optimize(self, model: torch.nn.Module) -> bool:
        """Check if this agent can optimize the given model."""
        pass
    
    @abstractmethod
    def estimate_impact(self, model: torch.nn.Module) -> ImpactEstimate:
        """Estimate the impact of optimization on the model."""
        pass
    
    @abstractmethod
    def optimize(self, model: torch.nn.Module, config: Dict[str, Any]) -> OptimizedModel:
        """Execute optimization on the model. This method should be implemented by subclasses."""
        pass
    
    @abstractmethod
    def validate_result(self, original: torch.nn.Module, optimized: torch.nn.Module) -> ValidationResult:
        """Validate the optimization result."""
        pass
    
    def get_supported_techniques(self) -> List[str]:
        """Get list of optimization techniques supported by this agent."""
        return []


class BaseAnalysisAgent(BaseAgent):
    """Base abstract class for analysis agents."""
    
    @abstractmethod
    def analyze_model(self, model_path: str) -> Dict[str, Any]:
        """Analyze a model and return analysis report."""
        pass
    
    @abstractmethod
    def identify_bottlenecks(self, model: torch.nn.Module) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks in the model."""
        pass


class BasePlanningAgent(BaseAgent):
    """Base abstract class for planning agents."""
    
    @abstractmethod
    def plan_optimization(self, analysis_report: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        """Create an optimization plan based on analysis and criteria."""
        pass
    
    @abstractmethod
    def prioritize_techniques(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize optimization techniques based on impact and feasibility."""
        pass


class BaseEvaluationAgent(BaseAgent):
    """Base abstract class for evaluation agents."""
    
    @abstractmethod
    def evaluate_model(self, model: torch.nn.Module, benchmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model performance against benchmarks."""
        pass
    
    @abstractmethod
    def compare_models(self, original: torch.nn.Module, optimized: torch.nn.Module) -> Dict[str, Any]:
        """Compare performance between original and optimized models."""
        pass