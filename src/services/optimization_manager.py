"""
Optimization Manager - Central orchestrator for robotics model optimization platform.

This module provides the OptimizationManager class that coordinates all agents
and manages the complete optimization workflow with proper error handling,
session management, and rollback mechanisms.
"""

import logging
import os
import time
import uuid
import asyncio
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
import threading
import copy
import torch

from ..utils.exceptions import (
    PlatformError, OptimizationError, SystemError, ErrorSeverity
)
from ..utils.retry import RetryConfig, RetryableOperation, STANDARD_RETRY, PERSISTENT_RETRY
from ..utils.recovery import (
    recovery_manager, model_recovery_manager, degradation_manager,
    RecoveryStrategy, RecoveryAction
)

from ..agents.base import (
    BaseAgent, BaseOptimizationAgent, BaseAnalysisAgent, 
    BasePlanningAgent, BaseEvaluationAgent, OptimizationStatus,
    ProgressUpdate, OptimizationResult
)
from ..agents.analysis.agent import AnalysisAgent
from ..agents.planning.agent import PlanningAgent, OptimizationPlan, OptimizationStep
from ..agents.evaluation.agent import EvaluationAgent
from ..agents.optimization.quantization import QuantizationAgent
from ..agents.optimization.pruning import PruningAgent
from ..models.core import (
    OptimizationSession, OptimizationStatus as SessionStatus,
    AnalysisReport, EvaluationReport, ModelMetadata
)
from ..config.optimization_criteria import OptimizationCriteria


logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Status of optimization workflow."""
    INITIALIZING = "initializing"
    ANALYZING = "analyzing"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"


@dataclass
class WorkflowState:
    """State of optimization workflow."""
    session_id: str
    status: WorkflowStatus
    current_step: Optional[str] = None
    progress_percentage: float = 0.0
    start_time: datetime = field(default_factory=datetime.now)
    last_update: datetime = field(default_factory=datetime.now)
    error_message: Optional[str] = None
    rollback_points: List[str] = field(default_factory=list)
    
    def update_progress(self, status: WorkflowStatus, progress: float, step: Optional[str] = None):
        """Update workflow state."""
        self.status = status
        self.progress_percentage = progress
        self.current_step = step
        self.last_update = datetime.now()


@dataclass
class SessionSnapshot:
    """Snapshot of optimization session for rollback."""
    session_id: str
    model_state: Dict[str, Any]
    step_index: int
    timestamp: datetime
    description: str


class OptimizationManager:
    """
    Central orchestrator that coordinates all agents and manages the optimization workflow.
    
    Responsibilities:
    - Coordinate analysis, planning, optimization, and evaluation agents
    - Manage optimization sessions and state tracking
    - Handle error recovery and rollback mechanisms
    - Provide progress tracking and notifications
    - Ensure proper resource cleanup
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Agent instances
        self.analysis_agent: Optional[AnalysisAgent] = None
        self.planning_agent: Optional[PlanningAgent] = None
        self.evaluation_agent: Optional[EvaluationAgent] = None
        self.optimization_agents: Dict[str, BaseOptimizationAgent] = {}
        
        # Session management
        self.active_sessions: Dict[str, OptimizationSession] = {}
        self.workflow_states: Dict[str, WorkflowState] = {}
        self.session_snapshots: Dict[str, List[SessionSnapshot]] = {}
        
        # Progress callbacks
        self.progress_callbacks: List[Callable[[str, ProgressUpdate], None]] = []
        
        # Configuration
        self.max_concurrent_sessions = config.get("max_concurrent_sessions", 3)
        self.auto_rollback_on_failure = config.get("auto_rollback_on_failure", True)
        self.snapshot_frequency = config.get("snapshot_frequency", 1)  # Every N steps
        self.session_timeout_minutes = config.get("session_timeout_minutes", 240)  # 4 hours
        
        # Thread safety
        self._lock = threading.RLock()
        
        self.logger.info("OptimizationManager initialized")
    
    def initialize(self) -> bool:
        """Initialize the optimization manager and all agents."""
        try:
            self.logger.info("Initializing OptimizationManager and agents")
            
            # Initialize analysis agent (if not already injected)
            if self.analysis_agent is None:
                analysis_config = self.config.get("analysis_agent", {})
                self.analysis_agent = AnalysisAgent(analysis_config)
                if not self.analysis_agent.initialize():
                    raise RuntimeError("Failed to initialize AnalysisAgent")
            else:
                self.logger.info("AnalysisAgent already injected, skipping initialization")
            
            # Initialize planning agent (if not already injected)
            if self.planning_agent is None:
                planning_config = self.config.get("planning_agent", {})
                self.planning_agent = PlanningAgent(planning_config)
                if not self.planning_agent.initialize():
                    raise RuntimeError("Failed to initialize PlanningAgent")
            else:
                self.logger.info("PlanningAgent already injected, skipping initialization")
            
            # Initialize evaluation agent (if not already injected)
            if self.evaluation_agent is None:
                evaluation_config = self.config.get("evaluation_agent", {})
                self.evaluation_agent = EvaluationAgent(evaluation_config)
                if not self.evaluation_agent.initialize():
                    raise RuntimeError("Failed to initialize EvaluationAgent")
            else:
                self.logger.info("EvaluationAgent already injected, skipping initialization")
            
            # Initialize optimization agents (if not already injected)
            if not self.optimization_agents:
                self._initialize_optimization_agents()
            else:
                self.logger.info(f"Optimization agents already injected ({len(self.optimization_agents)} agents), skipping initialization")
            
            self.logger.info("OptimizationManager initialization completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OptimizationManager: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up all resources and agents."""
        try:
            self.logger.info("Cleaning up OptimizationManager")
            
            # Cancel all active sessions
            with self._lock:
                for session_id in list(self.active_sessions.keys()):
                    self.cancel_session(session_id)
            
            # Cleanup agents
            if self.analysis_agent:
                self.analysis_agent.cleanup()
            if self.planning_agent:
                self.planning_agent.cleanup()
            if self.evaluation_agent:
                self.evaluation_agent.cleanup()
            
            for agent in self.optimization_agents.values():
                agent.cleanup()
            
            # Clear state
            self.active_sessions.clear()
            self.workflow_states.clear()
            self.session_snapshots.clear()
            self.progress_callbacks.clear()
            
            self.logger.info("OptimizationManager cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during OptimizationManager cleanup: {e}")
    
    def start_optimization_session(
        self, 
        model_path: str, 
        criteria: OptimizationCriteria,
        session_id: Optional[str] = None
    ) -> str:
        """
        Start a new optimization session.
        
        Args:
            model_path: Path to the model file
            criteria: Optimization criteria and constraints
            session_id: Optional session ID (generated if not provided)
            
        Returns:
            Session ID for tracking the optimization
            
        Raises:
            RuntimeError: If session cannot be started
        """
        if session_id is None:
            session_id = str(uuid.uuid4())
        
        with self._lock:
            # Check concurrent session limit
            if len(self.active_sessions) >= self.max_concurrent_sessions:
                raise RuntimeError(f"Maximum concurrent sessions ({self.max_concurrent_sessions}) reached")
            
            # Check if session already exists
            if session_id in self.active_sessions:
                raise RuntimeError(f"Session {session_id} already exists")
            
            try:
                # Create optimization session
                session = OptimizationSession(
                    id=session_id,
                    model_id=model_path,  # Using path as model ID for now
                    status=SessionStatus.PENDING,
                    criteria_name=criteria.name,
                    created_by="OptimizationManager"
                )
                
                # Create workflow state
                workflow_state = WorkflowState(
                    session_id=session_id,
                    status=WorkflowStatus.INITIALIZING
                )
                
                # Register session
                self.active_sessions[session_id] = session
                self.workflow_states[session_id] = workflow_state
                self.session_snapshots[session_id] = []
                
                self.logger.info(f"Started optimization session: {session_id}")
                
                # Start workflow in background
                threading.Thread(
                    target=self._execute_optimization_workflow,
                    args=(session_id, model_path, criteria),
                    daemon=True
                ).start()
                
                return session_id
                
            except Exception as e:
                # Cleanup on failure
                self.active_sessions.pop(session_id, None)
                self.workflow_states.pop(session_id, None)
                self.session_snapshots.pop(session_id, None)
                raise RuntimeError(f"Failed to start optimization session: {e}")
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get current status of optimization session."""
        with self._lock:
            if session_id not in self.active_sessions:
                raise ValueError(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            workflow_state = self.workflow_states[session_id]
            
            return {
                "session_id": session_id,
                "status": workflow_state.status.value,
                "progress_percentage": workflow_state.progress_percentage,
                "current_step": workflow_state.current_step,
                "start_time": workflow_state.start_time.isoformat(),
                "last_update": workflow_state.last_update.isoformat(),
                "error_message": workflow_state.error_message,
                "session_data": {
                    "model_id": session.model_id,
                    "criteria_name": session.criteria_name,
                    "created_at": session.created_at.isoformat(),
                    "steps_completed": len([s for s in session.steps if s.status == SessionStatus.COMPLETED])
                }
            }
    
    def cancel_session(self, session_id: str) -> bool:
        """Cancel an active optimization session."""
        with self._lock:
            if session_id not in self.active_sessions:
                self.logger.warning(f"Attempted to cancel non-existent session: {session_id}")
                return False
            
            try:
                session = self.active_sessions[session_id]
                workflow_state = self.workflow_states[session_id]
                
                # Update status
                session.status = SessionStatus.CANCELLED
                workflow_state.update_progress(WorkflowStatus.CANCELLED, 0.0, "Cancellation requested")
                
                # Cancel any running optimization agents
                for agent in self.optimization_agents.values():
                    if hasattr(agent, 'cancel_optimization'):
                        agent.cancel_optimization()
                
                self.logger.info(f"Cancelled optimization session: {session_id}")
                self._notify_progress(session_id, ProgressUpdate(
                    status=OptimizationStatus.CANCELLED,
                    progress_percentage=0.0,
                    current_step="Session cancelled"
                ))
                
                return True
                
            except Exception as e:
                self.logger.error(f"Error cancelling session {session_id}: {e}")
                return False
    
    def rollback_session(self, session_id: str, snapshot_index: Optional[int] = None) -> bool:
        """
        Rollback session to a previous snapshot.
        
        Args:
            session_id: Session to rollback
            snapshot_index: Index of snapshot to rollback to (latest if None)
            
        Returns:
            True if rollback successful
        """
        with self._lock:
            if session_id not in self.active_sessions:
                self.logger.error(f"Session {session_id} not found for rollback")
                return False
            
            snapshots = self.session_snapshots.get(session_id, [])
            if not snapshots:
                self.logger.error(f"No snapshots available for session {session_id}")
                return False
            
            try:
                # Select snapshot
                if snapshot_index is None:
                    snapshot = snapshots[-1]  # Latest snapshot
                else:
                    if snapshot_index < 0 or snapshot_index >= len(snapshots):
                        raise IndexError(f"Invalid snapshot index: {snapshot_index}")
                    snapshot = snapshots[snapshot_index]
                
                # Perform rollback
                session = self.active_sessions[session_id]
                workflow_state = self.workflow_states[session_id]
                
                # Restore session state
                session.status = SessionStatus.RUNNING
                session.steps = session.steps[:snapshot.step_index]
                
                # Update workflow state
                workflow_state.update_progress(
                    WorkflowStatus.ROLLED_BACK, 
                    0.0, 
                    f"Rolled back to: {snapshot.description}"
                )
                
                self.logger.info(f"Rolled back session {session_id} to snapshot: {snapshot.description}")
                self._notify_progress(session_id, ProgressUpdate(
                    status=OptimizationStatus.ROLLED_BACK,
                    progress_percentage=0.0,
                    current_step=f"Rolled back to: {snapshot.description}"
                ))
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to rollback session {session_id}: {e}")
                return False
    
    def add_progress_callback(self, callback: Callable[[str, ProgressUpdate], None]) -> None:
        """Add a callback function to receive progress updates."""
        self.progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[str, ProgressUpdate], None]) -> None:
        """Remove a progress callback function."""
        if callback in self.progress_callbacks:
            self.progress_callbacks.remove(callback)
    
    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        with self._lock:
            return list(self.active_sessions.keys())
    
    def _initialize_optimization_agents(self) -> None:
        """Initialize all optimization agents."""
        # Initialize quantization agent
        quantization_config = self.config.get("quantization_agent", {})
        quantization_agent = QuantizationAgent(quantization_config)
        if quantization_agent.initialize():
            self.optimization_agents["quantization"] = quantization_agent
            self.logger.info("QuantizationAgent initialized")
        else:
            self.logger.warning("Failed to initialize QuantizationAgent")
        
        # Initialize pruning agent
        pruning_config = self.config.get("pruning_agent", {})
        pruning_agent = PruningAgent(pruning_config)
        if pruning_agent.initialize():
            self.optimization_agents["pruning"] = pruning_agent
            self.logger.info("PruningAgent initialized")
        else:
            self.logger.warning("Failed to initialize PruningAgent")
        
        # Additional optimization agents can be added here
        # self.optimization_agents["distillation"] = DistillationAgent(config)
        # self.optimization_agents["compression"] = CompressionAgent(config)
    
    def _execute_optimization_workflow(
        self, 
        session_id: str, 
        model_path: str, 
        criteria: OptimizationCriteria
    ) -> None:
        """Execute the complete optimization workflow with comprehensive error handling."""
        workflow_start_time = datetime.now()
        
        # Create recovery context
        recovery_context = {
            "session_id": session_id,
            "model_path": model_path,
            "workflow_start_time": workflow_start_time
        }
        
        try:
            self.logger.info(f"Starting optimization workflow for session: {session_id}")
            
            # Create model snapshot for workflow-level recovery
            try:
                model = self._load_model(model_path)
                model_snapshot_id = model_recovery_manager.create_model_snapshot(
                    session_id, model, {"workflow_phase": "initial"}
                )
                recovery_context["model_snapshot_id"] = model_snapshot_id
            except Exception as e:
                self.logger.warning(f"Could not create initial model snapshot: {e}")
            
            # Phase 1: Analysis with error handling
            analysis_report = self._execute_analysis_phase_with_recovery(
                session_id, model_path, recovery_context
            )
            
            # Phase 2: Planning with error handling
            optimization_plan = self._execute_planning_phase_with_recovery(
                session_id, analysis_report, criteria, recovery_context
            )
            
            # Phase 3: Optimization with graceful degradation
            optimization_results = self._execute_optimization_phase_with_recovery(
                session_id, model_path, optimization_plan, recovery_context
            )
            
            # Phase 4: Evaluation with fallback options
            evaluation_report = self._execute_evaluation_phase_with_recovery(
                session_id, optimization_results, recovery_context
            )
            
            # Complete session
            self._complete_session(session_id, optimization_results, evaluation_report)
            
        except Exception as e:
            self.logger.error(f"Optimization workflow failed for session {session_id}: {e}")
            
            # Create workflow error
            workflow_error = OptimizationError(
                f"Workflow execution failed: {str(e)}",
                session_id=session_id,
                step="workflow_execution",
                severity=ErrorSeverity.HIGH
            )
            
            # Attempt workflow-level recovery
            if recovery_manager.handle_error(workflow_error, recovery_context):
                self.logger.info(f"Workflow recovery successful for session {session_id}")
            
            self._handle_workflow_failure(session_id, str(e))
    
    def _execute_analysis_phase(self, session_id: str, model_path: str) -> AnalysisReport:
        """Execute model analysis phase."""
        self.logger.info(f"Starting analysis phase for session: {session_id}")
        
        with self._lock:
            workflow_state = self.workflow_states[session_id]
            workflow_state.update_progress(WorkflowStatus.ANALYZING, 10.0, "Analyzing model")
        
        self._notify_progress(session_id, ProgressUpdate(
            status=OptimizationStatus.ANALYZING,
            progress_percentage=10.0,
            current_step="Analyzing model architecture and performance"
        ))
        
        try:
            analysis_report = self.analysis_agent.analyze_model(model_path)
            
            # Create snapshot after analysis
            self._create_session_snapshot(
                session_id, 
                {"analysis_report": analysis_report}, 
                0, 
                "Analysis completed"
            )
            
            self.logger.info(f"Analysis phase completed for session: {session_id}")
            return analysis_report
            
        except Exception as e:
            self.logger.error(f"Analysis phase failed for session {session_id}: {e}")
            raise
    
    def _execute_planning_phase(
        self, 
        session_id: str, 
        analysis_report: AnalysisReport, 
        criteria: OptimizationCriteria
    ) -> OptimizationPlan:
        """Execute optimization planning phase."""
        self.logger.info(f"Starting planning phase for session: {session_id}")
        
        with self._lock:
            workflow_state = self.workflow_states[session_id]
            workflow_state.update_progress(WorkflowStatus.PLANNING, 25.0, "Creating optimization plan")
        
        self._notify_progress(session_id, ProgressUpdate(
            status=OptimizationStatus.ANALYZING,  # Still in analysis/planning
            progress_percentage=25.0,
            current_step="Creating optimization plan"
        ))
        
        try:
            optimization_plan = self.planning_agent.plan_optimization(analysis_report, criteria)
            
            # Validate plan
            validation_result = self.planning_agent.validate_plan(optimization_plan)
            if not validation_result.is_valid:
                raise RuntimeError(f"Invalid optimization plan: {validation_result.issues}")
            
            # Update session with plan
            with self._lock:
                session = self.active_sessions[session_id]
                session.plan_id = optimization_plan.plan_id
                
                # Convert plan steps to session steps
                for plan_step in optimization_plan.steps:
                    session_step = self._convert_plan_step_to_session_step(plan_step)
                    session.steps.append(session_step)
            
            # Create snapshot after planning
            self._create_session_snapshot(
                session_id,
                {"optimization_plan": optimization_plan},
                0,
                "Planning completed"
            )
            
            self.logger.info(f"Planning phase completed for session: {session_id}")
            return optimization_plan
            
        except Exception as e:
            self.logger.error(f"Planning phase failed for session {session_id}: {e}")
            raise    

    def _execute_optimization_phase(
        self, 
        session_id: str, 
        model_path: str, 
        optimization_plan: OptimizationPlan
    ) -> Dict[str, Any]:
        """Execute optimization steps according to plan."""
        self.logger.info(f"Starting optimization phase for session: {session_id}")
        
        with self._lock:
            workflow_state = self.workflow_states[session_id]
            workflow_state.update_progress(WorkflowStatus.EXECUTING, 40.0, "Executing optimizations")
        
        optimization_results = {
            "original_model_path": model_path,
            "optimized_models": {},
            "step_results": [],
            "final_model": None
        }
        
        try:
            # Load original model
            original_model = self._load_model(model_path)
            current_model = copy.deepcopy(original_model)
            
            total_steps = len(optimization_plan.steps)
            
            for step_index, plan_step in enumerate(optimization_plan.steps):
                self.logger.info(f"Executing step {step_index + 1}/{total_steps}: {plan_step.technique}")
                
                # Check for cancellation
                with self._lock:
                    session = self.active_sessions[session_id]
                    if session.status == SessionStatus.CANCELLED:
                        self.logger.info(f"Session {session_id} cancelled during optimization")
                        return optimization_results
                
                # Update progress
                step_progress = 40.0 + (step_index / total_steps) * 40.0  # 40-80% range
                self._notify_progress(session_id, ProgressUpdate(
                    status=OptimizationStatus.OPTIMIZING,
                    progress_percentage=step_progress,
                    current_step=f"Executing {plan_step.technique} optimization"
                ))
                
                # Create snapshot before step
                if step_index % self.snapshot_frequency == 0:
                    self._create_session_snapshot(
                        session_id,
                        {"model_state": current_model.state_dict()},
                        step_index,
                        f"Before {plan_step.technique} optimization"
                    )
                
                # Execute optimization step
                step_result = self._execute_optimization_step(
                    session_id, current_model, plan_step
                )
                
                optimization_results["step_results"].append(step_result)
                
                if step_result.success:
                    current_model = step_result.optimized_model
                    optimization_results["optimized_models"][plan_step.technique] = current_model
                    
                    # Update session step status
                    with self._lock:
                        session = self.active_sessions[session_id]
                        if step_index < len(session.steps):
                            session.steps[step_index].status = SessionStatus.COMPLETED
                            session.steps[step_index].end_time = datetime.now()
                else:
                    self.logger.warning(f"Optimization step {plan_step.technique} failed: {step_result.error_message}")
                    
                    # Handle step failure
                    if self.auto_rollback_on_failure:
                        self.logger.info(f"Auto-rollback enabled, reverting step {plan_step.technique}")
                        # Model is already preserved in step_result.original_model
                        current_model = step_result.original_model
                    else:
                        # Continue with failed step but log the issue
                        with self._lock:
                            session = self.active_sessions[session_id]
                            if step_index < len(session.steps):
                                session.steps[step_index].status = SessionStatus.FAILED
                                session.steps[step_index].error_message = step_result.error_message
            
            optimization_results["final_model"] = current_model
            
            self.logger.info(f"Optimization phase completed for session: {session_id}")
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Optimization phase failed for session {session_id}: {e}")
            raise
    
    def _execute_evaluation_phase(
        self, 
        session_id: str, 
        optimization_results: Dict[str, Any]
    ) -> EvaluationReport:
        """Execute model evaluation phase."""
        self.logger.info(f"Starting evaluation phase for session: {session_id}")
        
        with self._lock:
            workflow_state = self.workflow_states[session_id]
            workflow_state.update_progress(WorkflowStatus.EVALUATING, 85.0, "Evaluating optimized model")
        
        self._notify_progress(session_id, ProgressUpdate(
            status=OptimizationStatus.VALIDATING,
            progress_percentage=85.0,
            current_step="Evaluating optimized model performance"
        ))
        
        try:
            final_model = optimization_results["final_model"]
            original_model_path = optimization_results["original_model_path"]
            
            if final_model is None:
                raise RuntimeError("No final optimized model available for evaluation")
            
            # Load original model for comparison
            original_model = self._load_model(original_model_path)
            
            # Perform model comparison
            comparison_result = self.evaluation_agent.compare_models(original_model, final_model)
            
            # Run standard benchmarks on final model
            standard_benchmarks = self._get_standard_benchmarks()
            evaluation_report = self.evaluation_agent.evaluate_model(final_model, standard_benchmarks)
            
            # Add comparison results to evaluation report
            evaluation_report.comparison_baseline = comparison_result
            evaluation_report.model_id = f"{session_id}_optimized"
            evaluation_report.session_id = session_id
            
            # Create final snapshot
            self._create_session_snapshot(
                session_id,
                {
                    "evaluation_report": evaluation_report,
                    "comparison_result": comparison_result
                },
                len(optimization_results["step_results"]),
                "Evaluation completed"
            )
            
            self.logger.info(f"Evaluation phase completed for session: {session_id}")
            return evaluation_report
            
        except Exception as e:
            self.logger.error(f"Evaluation phase failed for session {session_id}: {e}")
            raise
    
    def _execute_analysis_phase_with_recovery(
        self,
        session_id: str,
        model_path: str,
        recovery_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute analysis phase with error handling and recovery."""
        retry_config = RetryConfig(max_attempts=3, base_delay=2.0)
        
        def analysis_operation():
            return self._execute_analysis_phase(session_id, model_path)
        
        retry_op = RetryableOperation(
            "model_analysis",
            config=retry_config,
            on_retry=lambda e, attempt: self.logger.warning(
                f"Analysis retry {attempt} for session {session_id}: {e}"
            )
        )
        
        try:
            return retry_op.execute(analysis_operation)
        except Exception as e:
            # Create analysis error
            analysis_error = OptimizationError(
                f"Analysis phase failed: {str(e)}",
                session_id=session_id,
                step="analysis"
            )
            
            # Attempt recovery
            if recovery_manager.handle_error(analysis_error, recovery_context):
                # Retry after recovery
                return retry_op.execute(analysis_operation)
            else:
                raise analysis_error
    
    def _execute_planning_phase_with_recovery(
        self,
        session_id: str,
        analysis_report: Dict[str, Any],
        criteria: OptimizationCriteria,
        recovery_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute planning phase with error handling and recovery."""
        retry_config = RetryConfig(max_attempts=2, base_delay=1.0)
        
        def planning_operation():
            return self._execute_planning_phase(session_id, analysis_report, criteria)
        
        retry_op = RetryableOperation(
            "optimization_planning",
            config=retry_config,
            on_retry=lambda e, attempt: self.logger.warning(
                f"Planning retry {attempt} for session {session_id}: {e}"
            )
        )
        
        try:
            return retry_op.execute(planning_operation)
        except Exception as e:
            # Create planning error
            planning_error = OptimizationError(
                f"Planning phase failed: {str(e)}",
                session_id=session_id,
                step="planning"
            )
            
            # Attempt recovery
            if recovery_manager.handle_error(planning_error, recovery_context):
                # Retry after recovery
                return retry_op.execute(planning_operation)
            else:
                raise planning_error
    
    def _execute_optimization_phase_with_recovery(
        self,
        session_id: str,
        model_path: str,
        optimization_plan,  # OptimizationPlan object
        recovery_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute optimization phase with graceful degradation and recovery."""
        # Extract techniques from plan steps
        # Handle both dict and object formats for backward compatibility
        if isinstance(optimization_plan, dict):
            # Legacy format: {"techniques": ["quantization", "pruning"]}
            original_techniques = optimization_plan.get("techniques", [])
        else:
            # Object format with .steps attribute
            original_techniques = [step.technique for step in optimization_plan.steps]
        
        failed_techniques = []
        
        # Try the original plan first
        try:
            return self._execute_optimization_phase(session_id, model_path, optimization_plan)
        except Exception as e:
            self.logger.warning(f"Original optimization plan failed: {e}")
            
            # Determine which techniques failed
            if hasattr(e, 'context') and 'technique' in e.context:
                failed_techniques.append(e.context['technique'])
            
            # Create degraded plan using graceful degradation
            available_techniques = list(self.optimization_agents.keys())
            degraded_techniques = degradation_manager.create_degraded_plan(
                original_techniques,
                failed_techniques,
                available_techniques
            )
            
            if not degraded_techniques:
                # No fallback available, try recovery
                opt_error = OptimizationError(
                    f"All optimization techniques failed: {str(e)}",
                    session_id=session_id,
                    step="optimization"
                )
                
                if recovery_manager.handle_error(opt_error, recovery_context):
                    # Retry original plan after recovery
                    return self._execute_optimization_phase(session_id, model_path, optimization_plan)
                else:
                    raise opt_error
            
            # Try degraded plan - create new plan with degraded techniques
            self.logger.info(f"Attempting graceful degradation with techniques: {degraded_techniques}")
            
            # Create degraded plan by filtering steps
            if isinstance(optimization_plan, dict):
                # Legacy format: create new dict with filtered techniques
                degraded_plan = {
                    **optimization_plan,
                    "techniques": degraded_techniques
                }
            else:
                # Object format: filter steps and create new plan
                degraded_steps = [step for step in optimization_plan.steps if step.technique in degraded_techniques]
                degraded_plan = replace(optimization_plan, steps=degraded_steps)
            
            try:
                result = self._execute_optimization_phase(session_id, model_path, degraded_plan)
                result["degraded"] = True
                result["original_techniques"] = original_techniques
                result["failed_techniques"] = failed_techniques
                return result
            except Exception as degraded_error:
                # Even degraded plan failed
                final_error = OptimizationError(
                    f"Both original and degraded optimization plans failed: {str(degraded_error)}",
                    session_id=session_id,
                    step="optimization_degraded"
                )
                
                if recovery_manager.handle_error(final_error, recovery_context):
                    # Last attempt with minimal optimization - use first step only
                    if isinstance(optimization_plan, dict):
                        # Legacy format: use first technique only
                        techniques = optimization_plan.get("techniques", [])
                        if techniques:
                            minimal_plan = {
                                **optimization_plan,
                                "techniques": [techniques[0]]
                            }
                            return self._execute_optimization_phase(session_id, model_path, minimal_plan)
                        else:
                            raise final_error
                    else:
                        # Object format: use first step only
                        if optimization_plan.steps:
                            minimal_plan = replace(optimization_plan, steps=[optimization_plan.steps[0]])
                            return self._execute_optimization_phase(session_id, model_path, minimal_plan)
                        else:
                            raise final_error
                else:
                    raise final_error
    
    def _execute_evaluation_phase_with_recovery(
        self,
        session_id: str,
        optimization_results: Dict[str, Any],
        recovery_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute evaluation phase with error handling and fallback options."""
        retry_config = RetryConfig(max_attempts=2, base_delay=1.0)
        
        def evaluation_operation():
            return self._execute_evaluation_phase(session_id, optimization_results)
        
        retry_op = RetryableOperation(
            "model_evaluation",
            config=retry_config,
            on_retry=lambda e, attempt: self.logger.warning(
                f"Evaluation retry {attempt} for session {session_id}: {e}"
            )
        )
        
        try:
            return retry_op.execute(evaluation_operation)
        except Exception as e:
            self.logger.warning(f"Full evaluation failed: {e}")
            
            # Try minimal evaluation as fallback
            try:
                return self._execute_minimal_evaluation(session_id, optimization_results)
            except Exception as minimal_error:
                # Create evaluation error
                eval_error = OptimizationError(
                    f"Evaluation phase failed: {str(minimal_error)}",
                    session_id=session_id,
                    step="evaluation"
                )
                
                # Attempt recovery
                if recovery_manager.handle_error(eval_error, recovery_context):
                    # Retry minimal evaluation after recovery
                    return self._execute_minimal_evaluation(session_id, optimization_results)
                else:
                    # Return a basic evaluation report
                    return {
                        "evaluation_status": "failed",
                        "error_message": str(eval_error),
                        "basic_metrics": {
                            "optimization_completed": optimization_results.get("success", False),
                            "techniques_applied": optimization_results.get("techniques_applied", [])
                        }
                    }
    
    def _execute_minimal_evaluation(
        self,
        session_id: str,
        optimization_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a minimal evaluation when full evaluation fails."""
        self.logger.info(f"Executing minimal evaluation for session: {session_id}")
        
        # Basic evaluation without complex benchmarks
        final_model = optimization_results.get("final_model")
        if final_model is None:
            return {
                "evaluation_status": "minimal",
                "error_message": "No optimized model available for evaluation",
                "basic_metrics": {
                    "optimization_completed": False
                }
            }
        
        # Simple model validation
        try:
            # Check if model is loadable and has expected structure
            model_size = sum(p.numel() for p in final_model.parameters())
            
            return {
                "evaluation_status": "minimal",
                "basic_metrics": {
                    "optimization_completed": True,
                    "model_parameters": model_size,
                    "techniques_applied": optimization_results.get("techniques_applied", []),
                    "optimization_time": optimization_results.get("total_time", 0)
                },
                "warnings": ["Full evaluation failed, using minimal evaluation"]
            }
        except Exception as e:
            return {
                "evaluation_status": "failed",
                "error_message": f"Minimal evaluation failed: {str(e)}",
                "basic_metrics": {
                    "optimization_completed": False
                }
            }
    
    def _execute_optimization_step(
        self, 
        session_id: str, 
        model: torch.nn.Module, 
        plan_step: OptimizationStep
    ) -> OptimizationResult:
        """Execute a single optimization step."""
        technique = plan_step.technique
        
        if technique not in self.optimization_agents:
            error_msg = f"Optimization agent for technique '{technique}' not available"
            self.logger.error(error_msg)
            return OptimizationResult(
                success=False,
                optimized_model=None,
                original_model=model,
                optimization_metadata={},
                performance_metrics={},
                optimization_time=0.0,
                technique_used=technique,
                validation_result=None,
                error_message=error_msg
            )
        
        agent = self.optimization_agents[technique]
        
        # Add progress callback for this step
        def step_progress_callback(update: ProgressUpdate):
            # Forward progress updates with session context
            self._notify_progress(session_id, update)
        
        agent.add_progress_callback(step_progress_callback)
        
        try:
            # Execute optimization with tracking
            result = agent.optimize_with_tracking(
                model=model,
                config=plan_step.parameters,
                operation_id=f"{session_id}_{plan_step.step_id}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Optimization step {technique} failed: {e}")
            return OptimizationResult(
                success=False,
                optimized_model=None,
                original_model=model,
                optimization_metadata={},
                performance_metrics={},
                optimization_time=0.0,
                technique_used=technique,
                validation_result=None,
                error_message=str(e)
            )
        finally:
            # Remove progress callback
            agent.remove_progress_callback(step_progress_callback)
    
    def _complete_session(
        self, 
        session_id: str, 
        optimization_results: Dict[str, Any], 
        evaluation_report: EvaluationReport
    ) -> None:
        """Complete optimization session with results."""
        self.logger.info(f"Completing optimization session: {session_id}")
        
        try:
            with self._lock:
                session = self.active_sessions[session_id]
                workflow_state = self.workflow_states[session_id]
                
                # Update session with results
                session.status = SessionStatus.COMPLETED
                session.completed_at = datetime.now()
                
                # Create optimization results summary
                from ..models.core import OptimizationResults
                
                results_summary = OptimizationResults(
                    optimization_summary=f"Applied {len(optimization_results['step_results'])} optimization techniques",
                    techniques_applied=[step.technique for step in session.steps],
                    validation_passed=evaluation_report.validation_status.value == "passed"
                )
                
                # Aggregate metrics from optimization steps
                self._aggregate_optimization_metrics(
                    results_summary, 
                    optimization_results, 
                    session.model_id  # model_id contains the model path
                )
                
                # Add performance improvements from evaluation
                if evaluation_report.comparison_baseline:
                    results_summary.performance_improvements.update(
                        evaluation_report.comparison_baseline.improvements
                    )
                
                session.results = results_summary
                
                # Update workflow state
                workflow_state.update_progress(WorkflowStatus.COMPLETED, 100.0, "Optimization completed successfully")
                
                # Notify completion
                self._notify_progress(session_id, ProgressUpdate(
                    status=OptimizationStatus.COMPLETED,
                    progress_percentage=100.0,
                    current_step="Optimization workflow completed successfully"
                ))
                
                self.logger.info(f"Successfully completed optimization session: {session_id}")
                
        except Exception as e:
            self.logger.error(f"Error completing session {session_id}: {e}")
            self._handle_workflow_failure(session_id, f"Completion error: {str(e)}")
    
    def _calculate_size_reduction_percent(
        self,
        original_size_mb: float,
        optimized_size_mb: float
    ) -> float:
        """
        Calculate size reduction percentage.
        
        Args:
            original_size_mb: Original model size in MB
            optimized_size_mb: Optimized model size in MB
            
        Returns:
            float: Size reduction percentage
        """
        if original_size_mb > 0:
            return ((original_size_mb - optimized_size_mb) / original_size_mb) * 100
        return 0.0
    
    def _aggregate_optimization_metrics(
        self,
        results_summary,
        optimization_results: Dict[str, Any],
        model_path: str
    ) -> None:
        """Aggregate metrics from optimization steps into the results summary."""
        # Get original model size from file
        if os.path.exists(model_path):
            original_file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            results_summary.original_model_size_mb = original_file_size_mb
        
        # Aggregate metrics from each optimization step
        original_params = 0
        optimized_params = 0
        
        for step_result in optimization_results.get('step_results', []):
            if not step_result.success:
                continue
            
            # Extract metrics from step
            if hasattr(step_result, 'performance_metrics') and step_result.performance_metrics:
                metrics = step_result.performance_metrics
                
                # Parameter counts - track the original and final optimized counts
                if 'original_parameters' in metrics:
                    # Use the original count from the first step
                    if original_params == 0:
                        original_params = metrics['original_parameters']
                
                if 'optimized_parameters' in metrics:
                    # Always use the latest optimized count (cumulative effect)
                    optimized_params = metrics['optimized_parameters']
                
                # Add other metrics to performance improvements
                for metric_name, metric_value in metrics.items():
                    if metric_name not in ['original_parameters', 'optimized_parameters', 'parameter_reduction_ratio']:
                        if isinstance(metric_value, (int, float)):
                            results_summary.performance_improvements[metric_name] = metric_value
        
        # Calculate model sizes from parameters if we have them
        if original_params > 0 and optimized_params > 0:
            # Estimate size: parameters * 4 bytes (float32) / (1024 * 1024) for MB
            estimated_original_mb = (original_params * 4) / (1024 * 1024)
            estimated_optimized_mb = (optimized_params * 4) / (1024 * 1024)
            
            # Use file size if available, otherwise use parameter-based estimate
            if results_summary.original_model_size_mb == 0.0:
                results_summary.original_model_size_mb = estimated_original_mb
            
            results_summary.optimized_model_size_mb = estimated_optimized_mb
            
            # Calculate size reduction percentage
            results_summary.size_reduction_percent = self._calculate_size_reduction_percent(
                results_summary.original_model_size_mb,
                results_summary.optimized_model_size_mb
            )
            
            # Calculate total parameter reduction from aggregated counts
            # This is correct because it accounts for the cumulative effect of all optimizations
            if original_params > 0:
                param_reduction_percent = ((original_params - optimized_params) / original_params) * 100
            else:
                param_reduction_percent = 0.0
            
            # Add parameter metrics to performance improvements
            results_summary.performance_improvements['parameter_reduction_percent'] = param_reduction_percent
            results_summary.performance_improvements['original_parameters'] = original_params
            results_summary.performance_improvements['optimized_parameters'] = optimized_params
        
        # Check if optimized model was saved
        if 'final_model' in optimization_results and optimization_results['final_model']:
            final_model_path = optimization_results.get('final_model_path')
            if final_model_path and os.path.exists(final_model_path):
                optimized_file_size_mb = os.path.getsize(final_model_path) / (1024 * 1024)
                results_summary.optimized_model_size_mb = optimized_file_size_mb
                
                # Recalculate size reduction with actual file size
                results_summary.size_reduction_percent = self._calculate_size_reduction_percent(
                    results_summary.original_model_size_mb,
                    optimized_file_size_mb
                )
        
        # Log aggregated metrics
        self.logger.info(
            f"Aggregated metrics: {results_summary.original_model_size_mb:.2f} MB → "
            f"{results_summary.optimized_model_size_mb:.2f} MB "
            f"({results_summary.size_reduction_percent:.2f}% reduction)"
        )
        
        if original_params > 0:
            param_reduction_percent = ((original_params - optimized_params) / original_params) * 100
            self.logger.info(
                f"Parameters: {original_params:,} → {optimized_params:,} "
                f"({param_reduction_percent:.2f}% reduction)"
            )
    
    def _handle_workflow_failure(self, session_id: str, error_message: str) -> None:
        """Handle workflow failure with proper cleanup and rollback."""
        self.logger.error(f"Handling workflow failure for session {session_id}: {error_message}")
        
        try:
            with self._lock:
                if session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    workflow_state = self.workflow_states[session_id]
                    
                    # Update status
                    session.status = SessionStatus.FAILED
                    workflow_state.update_progress(WorkflowStatus.FAILED, 0.0, "Workflow failed")
                    workflow_state.error_message = error_message
                    
                    # Attempt rollback if enabled
                    if self.auto_rollback_on_failure:
                        self.logger.info(f"Attempting auto-rollback for failed session: {session_id}")
                        rollback_success = self.rollback_session(session_id)
                        if rollback_success:
                            self.logger.info(f"Auto-rollback successful for session: {session_id}")
                        else:
                            self.logger.error(f"Auto-rollback failed for session: {session_id}")
                    
                    # Notify failure
                    self._notify_progress(session_id, ProgressUpdate(
                        status=OptimizationStatus.FAILED,
                        progress_percentage=0.0,
                        current_step="Optimization workflow failed",
                        message=error_message
                    ))
                    
        except Exception as e:
            self.logger.error(f"Error handling workflow failure for session {session_id}: {e}")
    
    def _create_session_snapshot(
        self, 
        session_id: str, 
        state_data: Dict[str, Any], 
        step_index: int, 
        description: str
    ) -> None:
        """Create a snapshot of session state for rollback."""
        try:
            snapshot = SessionSnapshot(
                session_id=session_id,
                model_state=state_data,
                step_index=step_index,
                timestamp=datetime.now(),
                description=description
            )
            
            with self._lock:
                if session_id not in self.session_snapshots:
                    self.session_snapshots[session_id] = []
                
                self.session_snapshots[session_id].append(snapshot)
                
                # Limit number of snapshots to prevent memory issues
                max_snapshots = 10
                if len(self.session_snapshots[session_id]) > max_snapshots:
                    self.session_snapshots[session_id] = self.session_snapshots[session_id][-max_snapshots:]
            
            self.logger.debug(f"Created snapshot for session {session_id}: {description}")
            
        except Exception as e:
            self.logger.error(f"Failed to create snapshot for session {session_id}: {e}")
    
    def _notify_progress(self, session_id: str, update: ProgressUpdate) -> None:
        """Notify all progress callbacks about update."""
        for callback in self.progress_callbacks:
            try:
                callback(session_id, update)
            except Exception as e:
                self.logger.error(f"Error in progress callback: {e}")
    
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load model from file path."""
        try:
            # Use weights_only=False for testing - in production this should be more secure
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            
            if isinstance(model, dict):
                if 'model' in model:
                    model = model['model']
                elif 'state_dict' in model:
                    raise ValueError("State dict found but no model architecture available")
            
            if not isinstance(model, torch.nn.Module):
                raise ValueError(f"Loaded object is not a PyTorch model: {type(model)}")
            
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def _convert_plan_step_to_session_step(self, plan_step: OptimizationStep) -> 'OptimizationStep':
        """Convert planning step to session step format."""
        from ..models.core import OptimizationStep as SessionOptimizationStep
        
        return SessionOptimizationStep(
            step_id=plan_step.step_id,
            technique=plan_step.technique,
            status=SessionStatus.PENDING,
            parameters=plan_step.parameters
        )
    
    def _get_standard_benchmarks(self) -> List[Dict[str, Any]]:
        """Get standard benchmark configurations for evaluation."""
        return [
            {"name": "inference_speed", "type": "inference_speed"},
            {"name": "memory_usage", "type": "memory_usage"},
            {"name": "throughput", "type": "throughput"},
            {"name": "model_size", "type": "model_size"},
            {"name": "accuracy", "type": "accuracy"}
        ]
    
    def get_session_snapshots(self, session_id: str) -> List[Dict[str, Any]]:
        """Get available snapshots for a session."""
        with self._lock:
            if session_id not in self.session_snapshots:
                return []
            
            snapshots = self.session_snapshots[session_id]
            return [
                {
                    "index": i,
                    "description": snapshot.description,
                    "timestamp": snapshot.timestamp.isoformat(),
                    "step_index": snapshot.step_index
                }
                for i, snapshot in enumerate(snapshots)
            ]
    
    def pause_session(self, session_id: str) -> bool:
        """Pause an active optimization session."""
        with self._lock:
            if session_id not in self.active_sessions:
                return False
            
            try:
                session = self.active_sessions[session_id]
                workflow_state = self.workflow_states[session_id]
                
                session.status = SessionStatus.PAUSED
                workflow_state.update_progress(WorkflowStatus.CANCELLED, workflow_state.progress_percentage, "Session paused")
                
                self.logger.info(f"Paused optimization session: {session_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error pausing session {session_id}: {e}")
                return False
    
    def resume_session(self, session_id: str) -> bool:
        """Resume a paused optimization session."""
        with self._lock:
            if session_id not in self.active_sessions:
                return False
            
            try:
                session = self.active_sessions[session_id]
                workflow_state = self.workflow_states[session_id]
                
                if session.status != SessionStatus.PAUSED:
                    self.logger.warning(f"Cannot resume session {session_id} - not in paused state")
                    return False
                
                session.status = SessionStatus.RUNNING
                workflow_state.update_progress(WorkflowStatus.EXECUTING, workflow_state.progress_percentage, "Session resumed")
                
                self.logger.info(f"Resumed optimization session: {session_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"Error resuming session {session_id}: {e}")
                return False
    
    def get_optimization_agents_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all optimization agents."""
        agent_status = {}
        
        for technique, agent in self.optimization_agents.items():
            try:
                status = agent.get_status()
                status["current_status"] = agent.get_current_status().value if hasattr(agent, 'get_current_status') else "unknown"
                status["supported_techniques"] = agent.get_supported_techniques()
                agent_status[technique] = status
            except Exception as e:
                agent_status[technique] = {"error": str(e)}
        
        return agent_status