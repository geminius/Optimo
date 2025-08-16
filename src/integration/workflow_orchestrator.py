"""
Workflow Orchestrator - Complete end-to-end optimization workflow implementation.

This module provides comprehensive workflow orchestration that ensures
proper execution of the complete optimization pipeline from upload to evaluation.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid

from ..models.core import ModelMetadata, OptimizationSession
from ..config.optimization_criteria import OptimizationCriteria
from .platform_integration import PlatformIntegrator


logger = logging.getLogger(__name__)


class WorkflowPhase(Enum):
    """Phases of the optimization workflow."""
    UPLOAD = "upload"
    VALIDATION = "validation"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    OPTIMIZATION = "optimization"
    EVALUATION = "evaluation"
    COMPLETION = "completion"


@dataclass
class WorkflowContext:
    """Context for workflow execution."""
    workflow_id: str
    model_path: str
    criteria: OptimizationCriteria
    user_id: str
    start_time: datetime
    current_phase: WorkflowPhase
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    workflow_id: str
    success: bool
    session_id: Optional[str]
    final_phase: WorkflowPhase
    execution_time_seconds: float
    error_message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


class WorkflowOrchestrator:
    """
    Complete end-to-end optimization workflow orchestrator.
    
    Manages the entire optimization pipeline from model upload through
    evaluation, ensuring proper coordination and error handling.
    """
    
    def __init__(self, platform_integrator: PlatformIntegrator):
        self.platform_integrator = platform_integrator
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Workflow tracking
        self.active_workflows: Dict[str, WorkflowContext] = {}
        self.workflow_callbacks: List[Callable[[str, WorkflowPhase, Dict[str, Any]], None]] = []
        
        self.logger.info("WorkflowOrchestrator initialized")
    
    async def execute_complete_workflow(
        self,
        model_path: str,
        criteria: OptimizationCriteria,
        user_id: str,
        workflow_id: Optional[str] = None
    ) -> WorkflowResult:
        """
        Execute the complete optimization workflow from upload to evaluation.
        
        Args:
            model_path: Path to the model file
            criteria: Optimization criteria
            user_id: ID of the user initiating the workflow
            workflow_id: Optional workflow ID (generated if not provided)
            
        Returns:
            WorkflowResult with execution details
        """
        if workflow_id is None:
            workflow_id = str(uuid.uuid4())
        
        start_time = datetime.now()
        
        # Create workflow context
        context = WorkflowContext(
            workflow_id=workflow_id,
            model_path=model_path,
            criteria=criteria,
            user_id=user_id,
            start_time=start_time,
            current_phase=WorkflowPhase.UPLOAD
        )
        
        self.active_workflows[workflow_id] = context
        
        try:
            self.logger.info(f"Starting complete workflow: {workflow_id}")
            
            # Phase 1: Upload and Validation
            await self._execute_upload_phase(context)
            
            # Phase 2: Model Analysis
            await self._execute_analysis_phase(context)
            
            # Phase 3: Optimization Planning
            await self._execute_planning_phase(context)
            
            # Phase 4: Optimization Execution
            await self._execute_optimization_phase(context)
            
            # Phase 5: Model Evaluation
            await self._execute_evaluation_phase(context)
            
            # Phase 6: Completion
            await self._execute_completion_phase(context)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = WorkflowResult(
                workflow_id=workflow_id,
                success=True,
                session_id=context.session_id,
                final_phase=WorkflowPhase.COMPLETION,
                execution_time_seconds=execution_time,
                results=context.metadata.get("results", {})
            )
            
            self.logger.info(f"Workflow completed successfully: {workflow_id}")
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.error(f"Workflow failed: {workflow_id} - {e}")
            
            result = WorkflowResult(
                workflow_id=workflow_id,
                success=False,
                session_id=context.session_id,
                final_phase=context.current_phase,
                execution_time_seconds=execution_time,
                error_message=str(e)
            )
            
            return result
            
        finally:
            # Clean up workflow context
            self.active_workflows.pop(workflow_id, None)
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a workflow."""
        context = self.active_workflows.get(workflow_id)
        if not context:
            return None
        
        status = {
            "workflow_id": workflow_id,
            "current_phase": context.current_phase.value,
            "start_time": context.start_time.isoformat(),
            "elapsed_seconds": (datetime.now() - context.start_time).total_seconds(),
            "session_id": context.session_id,
            "user_id": context.user_id
        }
        
        # Add session status if available
        if context.session_id:
            try:
                optimization_manager = self.platform_integrator.get_optimization_manager()
                session_status = optimization_manager.get_session_status(context.session_id)
                status["session_status"] = session_status
            except Exception as e:
                self.logger.warning(f"Failed to get session status: {e}")
        
        return status
    
    def add_workflow_callback(self, callback: Callable[[str, WorkflowPhase, Dict[str, Any]], None]) -> None:
        """Add a callback for workflow phase transitions."""
        self.workflow_callbacks.append(callback)
    
    def remove_workflow_callback(self, callback: Callable[[str, WorkflowPhase, Dict[str, Any]], None]) -> None:
        """Remove a workflow callback."""
        if callback in self.workflow_callbacks:
            self.workflow_callbacks.remove(callback)
    
    async def _execute_upload_phase(self, context: WorkflowContext) -> None:
        """Execute upload and validation phase."""
        self.logger.info(f"Executing upload phase for workflow: {context.workflow_id}")
        
        context.current_phase = WorkflowPhase.UPLOAD
        await self._notify_phase_transition(context, {"phase": "upload_started"})
        
        try:
            # Validate model file exists and is readable
            from pathlib import Path
            model_file = Path(context.model_path)
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {context.model_path}")
            
            if not model_file.is_file():
                raise ValueError(f"Path is not a file: {context.model_path}")
            
            # Validate file format
            allowed_extensions = {".pt", ".pth", ".onnx", ".pb", ".h5", ".safetensors"}
            if model_file.suffix.lower() not in allowed_extensions:
                raise ValueError(f"Unsupported file format: {model_file.suffix}")
            
            # Store model metadata
            model_store = self.platform_integrator.get_model_store()
            
            # Create model metadata
            metadata = ModelMetadata(
                id=str(uuid.uuid4()),
                name=model_file.stem,
                description=f"Model for workflow {context.workflow_id}",
                tags=["workflow", context.criteria.name],
                file_path=context.model_path,
                size_mb=model_file.stat().st_size / (1024 * 1024),
                author=context.user_id
            )
            
            # Store in model store
            model_store.store_model_metadata(metadata)
            context.metadata["model_metadata"] = metadata
            
            context.current_phase = WorkflowPhase.VALIDATION
            await self._notify_phase_transition(context, {
                "phase": "validation_completed",
                "model_id": metadata.id,
                "model_size_mb": metadata.size_mb
            })
            
            self.logger.info(f"Upload phase completed for workflow: {context.workflow_id}")
            
        except Exception as e:
            self.logger.error(f"Upload phase failed for workflow {context.workflow_id}: {e}")
            raise
    
    async def _execute_analysis_phase(self, context: WorkflowContext) -> None:
        """Execute model analysis phase."""
        self.logger.info(f"Executing analysis phase for workflow: {context.workflow_id}")
        
        context.current_phase = WorkflowPhase.ANALYSIS
        await self._notify_phase_transition(context, {"phase": "analysis_started"})
        
        try:
            optimization_manager = self.platform_integrator.get_optimization_manager()
            
            # Get analysis agent
            analysis_agent = optimization_manager.analysis_agent
            if not analysis_agent:
                raise RuntimeError("Analysis agent not available")
            
            # Perform model analysis
            analysis_report = analysis_agent.analyze_model(context.model_path)
            context.metadata["analysis_report"] = analysis_report
            
            await self._notify_phase_transition(context, {
                "phase": "analysis_completed",
                "opportunities_found": len(analysis_report.optimization_opportunities),
                "model_type": analysis_report.architecture_summary.model_type
            })
            
            self.logger.info(f"Analysis phase completed for workflow: {context.workflow_id}")
            
        except Exception as e:
            self.logger.error(f"Analysis phase failed for workflow {context.workflow_id}: {e}")
            raise
    
    async def _execute_planning_phase(self, context: WorkflowContext) -> None:
        """Execute optimization planning phase."""
        self.logger.info(f"Executing planning phase for workflow: {context.workflow_id}")
        
        context.current_phase = WorkflowPhase.PLANNING
        await self._notify_phase_transition(context, {"phase": "planning_started"})
        
        try:
            optimization_manager = self.platform_integrator.get_optimization_manager()
            
            # Get planning agent
            planning_agent = optimization_manager.planning_agent
            if not planning_agent:
                raise RuntimeError("Planning agent not available")
            
            # Get analysis report
            analysis_report = context.metadata.get("analysis_report")
            if not analysis_report:
                raise RuntimeError("Analysis report not available")
            
            # Create optimization plan
            optimization_plan = planning_agent.plan_optimization(analysis_report, context.criteria)
            context.metadata["optimization_plan"] = optimization_plan
            
            await self._notify_phase_transition(context, {
                "phase": "planning_completed",
                "techniques_planned": len(optimization_plan.steps),
                "estimated_improvement": optimization_plan.estimated_improvement
            })
            
            self.logger.info(f"Planning phase completed for workflow: {context.workflow_id}")
            
        except Exception as e:
            self.logger.error(f"Planning phase failed for workflow {context.workflow_id}: {e}")
            raise
    
    async def _execute_optimization_phase(self, context: WorkflowContext) -> None:
        """Execute optimization phase."""
        self.logger.info(f"Executing optimization phase for workflow: {context.workflow_id}")
        
        context.current_phase = WorkflowPhase.OPTIMIZATION
        await self._notify_phase_transition(context, {"phase": "optimization_started"})
        
        try:
            optimization_manager = self.platform_integrator.get_optimization_manager()
            
            # Start optimization session
            session_id = optimization_manager.start_optimization_session(
                model_path=context.model_path,
                criteria=context.criteria
            )
            
            context.session_id = session_id
            context.metadata["session_id"] = session_id
            
            # Wait for optimization to complete
            await self._wait_for_optimization_completion(context)
            
            await self._notify_phase_transition(context, {
                "phase": "optimization_completed",
                "session_id": session_id
            })
            
            self.logger.info(f"Optimization phase completed for workflow: {context.workflow_id}")
            
        except Exception as e:
            self.logger.error(f"Optimization phase failed for workflow {context.workflow_id}: {e}")
            raise
    
    async def _execute_evaluation_phase(self, context: WorkflowContext) -> None:
        """Execute evaluation phase."""
        self.logger.info(f"Executing evaluation phase for workflow: {context.workflow_id}")
        
        context.current_phase = WorkflowPhase.EVALUATION
        await self._notify_phase_transition(context, {"phase": "evaluation_started"})
        
        try:
            optimization_manager = self.platform_integrator.get_optimization_manager()
            
            # Get session results
            if not context.session_id:
                raise RuntimeError("No optimization session ID available")
            
            session_status = optimization_manager.get_session_status(context.session_id)
            
            if session_status["status"] != "completed":
                raise RuntimeError(f"Optimization session not completed: {session_status['status']}")
            
            # Store evaluation results
            context.metadata["session_results"] = session_status
            
            await self._notify_phase_transition(context, {
                "phase": "evaluation_completed",
                "session_status": session_status["status"]
            })
            
            self.logger.info(f"Evaluation phase completed for workflow: {context.workflow_id}")
            
        except Exception as e:
            self.logger.error(f"Evaluation phase failed for workflow {context.workflow_id}: {e}")
            raise
    
    async def _execute_completion_phase(self, context: WorkflowContext) -> None:
        """Execute completion phase."""
        self.logger.info(f"Executing completion phase for workflow: {context.workflow_id}")
        
        context.current_phase = WorkflowPhase.COMPLETION
        
        try:
            # Compile final results
            results = {
                "workflow_id": context.workflow_id,
                "model_metadata": context.metadata.get("model_metadata"),
                "analysis_report": context.metadata.get("analysis_report"),
                "optimization_plan": context.metadata.get("optimization_plan"),
                "session_id": context.session_id,
                "session_results": context.metadata.get("session_results"),
                "execution_summary": {
                    "start_time": context.start_time.isoformat(),
                    "completion_time": datetime.now().isoformat(),
                    "total_duration_seconds": (datetime.now() - context.start_time).total_seconds()
                }
            }
            
            context.metadata["results"] = results
            
            await self._notify_phase_transition(context, {
                "phase": "workflow_completed",
                "results": results
            })
            
            self.logger.info(f"Completion phase finished for workflow: {context.workflow_id}")
            
        except Exception as e:
            self.logger.error(f"Completion phase failed for workflow {context.workflow_id}: {e}")
            raise
    
    async def _wait_for_optimization_completion(self, context: WorkflowContext) -> None:
        """Wait for optimization session to complete."""
        if not context.session_id:
            raise RuntimeError("No session ID to wait for")
        
        optimization_manager = self.platform_integrator.get_optimization_manager()
        max_wait_seconds = 3600  # 1 hour timeout
        check_interval = 5  # Check every 5 seconds
        
        elapsed = 0
        while elapsed < max_wait_seconds:
            try:
                session_status = optimization_manager.get_session_status(context.session_id)
                status = session_status["status"]
                
                if status in ["completed", "failed", "cancelled"]:
                    if status == "failed":
                        error_msg = session_status.get("error_message", "Unknown error")
                        raise RuntimeError(f"Optimization session failed: {error_msg}")
                    elif status == "cancelled":
                        raise RuntimeError("Optimization session was cancelled")
                    else:
                        # Completed successfully
                        return
                
                # Still running, wait and check again
                await asyncio.sleep(check_interval)
                elapsed += check_interval
                
            except Exception as e:
                if "not found" in str(e).lower():
                    raise RuntimeError(f"Optimization session {context.session_id} not found")
                else:
                    raise
        
        # Timeout reached
        raise RuntimeError(f"Optimization session timed out after {max_wait_seconds} seconds")
    
    async def _notify_phase_transition(self, context: WorkflowContext, data: Dict[str, Any]) -> None:
        """Notify callbacks about phase transitions."""
        for callback in self.workflow_callbacks:
            try:
                callback(context.workflow_id, context.current_phase, data)
            except Exception as e:
                self.logger.warning(f"Workflow callback failed: {e}")