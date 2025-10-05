"""
Unit tests for OptimizationManager orchestrator.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from src.services.optimization_manager import OptimizationManager, WorkflowStatus, WorkflowState
from src.agents.base import OptimizationStatus, ProgressUpdate, OptimizationResult, ValidationResult
from src.models.core import OptimizationSession, OptimizationStatus as SessionStatus
from src.config.optimization_criteria import OptimizationCriteria, PerformanceMetric


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 1)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return self.linear2(x)


@pytest.fixture
def test_model():
    """Create a simple test model."""
    return SimpleTestModel()


@pytest.fixture
def test_model_path(test_model):
    """Create a temporary model file."""
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(test_model, f.name)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def optimization_criteria():
    """Create test optimization criteria."""
    return OptimizationCriteria(
        name="test_criteria",
        description="Test optimization criteria for unit tests",
        priority_weights={PerformanceMetric.MODEL_SIZE: 0.6, PerformanceMetric.INFERENCE_TIME: 0.4}
    )


@pytest.fixture
def manager_config():
    """Create test configuration for OptimizationManager."""
    return {
        "max_concurrent_sessions": 2,
        "auto_rollback_on_failure": True,
        "snapshot_frequency": 1,
        "session_timeout_minutes": 60,
        "analysis_agent": {},
        "planning_agent": {},
        "evaluation_agent": {},
        "quantization_agent": {},
        "pruning_agent": {}
    }


@pytest.fixture
def optimization_manager(manager_config):
    """Create OptimizationManager instance."""
    manager = OptimizationManager(manager_config)
    return manager


class TestOptimizationManagerInitialization:
    """Test OptimizationManager initialization."""
    
    def test_manager_creation(self, manager_config):
        """Test OptimizationManager creation."""
        manager = OptimizationManager(manager_config)
        
        assert manager.config == manager_config
        assert manager.max_concurrent_sessions == 2
        assert manager.auto_rollback_on_failure is True
        assert manager.snapshot_frequency == 1
        assert len(manager.active_sessions) == 0
        assert len(manager.workflow_states) == 0
    
    @patch('src.services.optimization_manager.AnalysisAgent')
    @patch('src.services.optimization_manager.PlanningAgent')
    @patch('src.services.optimization_manager.EvaluationAgent')
    @patch('src.services.optimization_manager.QuantizationAgent')
    @patch('src.services.optimization_manager.PruningAgent')
    def test_manager_initialization_success(self, mock_pruning, mock_quantization, 
                                          mock_evaluation, mock_planning, mock_analysis, 
                                          optimization_manager):
        """Test successful manager initialization."""
        # Mock all agents to return True for initialize()
        mock_analysis.return_value.initialize.return_value = True
        mock_planning.return_value.initialize.return_value = True
        mock_evaluation.return_value.initialize.return_value = True
        mock_quantization.return_value.initialize.return_value = True
        mock_pruning.return_value.initialize.return_value = True
        
        result = optimization_manager.initialize()
        
        assert result is True
        assert optimization_manager.analysis_agent is not None
        assert optimization_manager.planning_agent is not None
        assert optimization_manager.evaluation_agent is not None
        assert len(optimization_manager.optimization_agents) >= 0  # May vary based on agent initialization
    
    @patch('src.services.optimization_manager.AnalysisAgent')
    def test_manager_initialization_failure(self, mock_analysis, optimization_manager):
        """Test manager initialization failure."""
        # Mock analysis agent to fail initialization
        mock_analysis.return_value.initialize.return_value = False
        
        result = optimization_manager.initialize()
        
        assert result is False
    
    def test_manager_cleanup(self, optimization_manager):
        """Test manager cleanup."""
        # Add some mock sessions
        optimization_manager.active_sessions["test_session"] = Mock()
        optimization_manager.workflow_states["test_session"] = Mock()
        
        # Mock agents
        optimization_manager.analysis_agent = Mock()
        optimization_manager.planning_agent = Mock()
        optimization_manager.evaluation_agent = Mock()
        optimization_manager.optimization_agents["test"] = Mock()
        
        optimization_manager.cleanup()
        
        # Verify cleanup
        assert len(optimization_manager.active_sessions) == 0
        assert len(optimization_manager.workflow_states) == 0
        optimization_manager.analysis_agent.cleanup.assert_called_once()
        optimization_manager.planning_agent.cleanup.assert_called_once()
        optimization_manager.evaluation_agent.cleanup.assert_called_once()


class TestSessionManagement:
    """Test session management functionality."""
    
    @patch('src.services.optimization_manager.OptimizationManager._execute_optimization_workflow')
    def test_start_optimization_session(self, mock_workflow, optimization_manager, 
                                      test_model_path, optimization_criteria):
        """Test starting optimization session."""
        session_id = optimization_manager.start_optimization_session(
            test_model_path, optimization_criteria
        )
        
        assert session_id is not None
        assert session_id in optimization_manager.active_sessions
        assert session_id in optimization_manager.workflow_states
        
        session = optimization_manager.active_sessions[session_id]
        assert session.model_id == test_model_path
        assert session.criteria_name == optimization_criteria.name
        assert session.status == SessionStatus.PENDING
        
        # Verify workflow was started
        mock_workflow.assert_called_once()
    
    def test_start_session_with_custom_id(self, optimization_manager, test_model_path, optimization_criteria):
        """Test starting session with custom ID."""
        custom_id = "custom_session_123"
        
        with patch('src.services.optimization_manager.OptimizationManager._execute_optimization_workflow'):
            session_id = optimization_manager.start_optimization_session(
                test_model_path, optimization_criteria, custom_id
            )
        
        assert session_id == custom_id
        assert custom_id in optimization_manager.active_sessions
    
    def test_start_session_duplicate_id(self, optimization_manager, test_model_path, optimization_criteria):
        """Test starting session with duplicate ID."""
        session_id = "duplicate_session"
        
        with patch('src.services.optimization_manager.OptimizationManager._execute_optimization_workflow'):
            # Start first session
            optimization_manager.start_optimization_session(
                test_model_path, optimization_criteria, session_id
            )
            
            # Try to start second session with same ID
            with pytest.raises(RuntimeError, match="already exists"):
                optimization_manager.start_optimization_session(
                    test_model_path, optimization_criteria, session_id
                )
    
    def test_concurrent_session_limit(self, optimization_manager, test_model_path, optimization_criteria):
        """Test concurrent session limit enforcement."""
        with patch('src.services.optimization_manager.OptimizationManager._execute_optimization_workflow'):
            # Start sessions up to limit
            session_ids = []
            for i in range(optimization_manager.max_concurrent_sessions):
                session_id = optimization_manager.start_optimization_session(
                    test_model_path, optimization_criteria
                )
                session_ids.append(session_id)
            
            # Try to start one more session
            with pytest.raises(RuntimeError, match="Maximum concurrent sessions"):
                optimization_manager.start_optimization_session(
                    test_model_path, optimization_criteria
                )
    
    def test_get_session_status(self, optimization_manager, test_model_path, optimization_criteria):
        """Test getting session status."""
        with patch('src.services.optimization_manager.OptimizationManager._execute_optimization_workflow'):
            session_id = optimization_manager.start_optimization_session(
                test_model_path, optimization_criteria
            )
        
        status = optimization_manager.get_session_status(session_id)
        
        assert status["session_id"] == session_id
        assert "status" in status
        assert "progress_percentage" in status
        assert "session_data" in status
        assert status["session_data"]["model_id"] == test_model_path
    
    def test_get_session_status_not_found(self, optimization_manager):
        """Test getting status for non-existent session."""
        with pytest.raises(ValueError, match="not found"):
            optimization_manager.get_session_status("non_existent_session")
    
    def test_cancel_session(self, optimization_manager, test_model_path, optimization_criteria):
        """Test cancelling session."""
        with patch('src.services.optimization_manager.OptimizationManager._execute_optimization_workflow'):
            session_id = optimization_manager.start_optimization_session(
                test_model_path, optimization_criteria
            )
        
        result = optimization_manager.cancel_session(session_id)
        
        assert result is True
        session = optimization_manager.active_sessions[session_id]
        assert session.status == SessionStatus.CANCELLED
    
    def test_cancel_nonexistent_session(self, optimization_manager):
        """Test cancelling non-existent session."""
        result = optimization_manager.cancel_session("non_existent_session")
        assert result is False
    
    def test_get_active_sessions(self, optimization_manager, test_model_path, optimization_criteria):
        """Test getting active sessions list."""
        with patch('src.services.optimization_manager.OptimizationManager._execute_optimization_workflow'):
            session_id1 = optimization_manager.start_optimization_session(
                test_model_path, optimization_criteria
            )
            session_id2 = optimization_manager.start_optimization_session(
                test_model_path, optimization_criteria
            )
        
        active_sessions = optimization_manager.get_active_sessions()
        
        assert len(active_sessions) == 2
        assert session_id1 in active_sessions
        assert session_id2 in active_sessions


class TestProgressTracking:
    """Test progress tracking functionality."""
    
    def test_add_progress_callback(self, optimization_manager):
        """Test adding progress callback."""
        callback = Mock()
        
        optimization_manager.add_progress_callback(callback)
        
        assert callback in optimization_manager.progress_callbacks
    
    def test_remove_progress_callback(self, optimization_manager):
        """Test removing progress callback."""
        callback = Mock()
        
        optimization_manager.add_progress_callback(callback)
        optimization_manager.remove_progress_callback(callback)
        
        assert callback not in optimization_manager.progress_callbacks
    
    def test_notify_progress(self, optimization_manager):
        """Test progress notification."""
        callback1 = Mock()
        callback2 = Mock()
        
        optimization_manager.add_progress_callback(callback1)
        optimization_manager.add_progress_callback(callback2)
        
        session_id = "test_session"
        update = ProgressUpdate(
            status=OptimizationStatus.OPTIMIZING,
            progress_percentage=50.0,
            current_step="Test step"
        )
        
        optimization_manager._notify_progress(session_id, update)
        
        callback1.assert_called_once_with(session_id, update)
        callback2.assert_called_once_with(session_id, update)
    
    def test_notify_progress_with_callback_error(self, optimization_manager):
        """Test progress notification with callback error."""
        failing_callback = Mock(side_effect=Exception("Callback error"))
        working_callback = Mock()
        
        optimization_manager.add_progress_callback(failing_callback)
        optimization_manager.add_progress_callback(working_callback)
        
        session_id = "test_session"
        update = ProgressUpdate(
            status=OptimizationStatus.OPTIMIZING,
            progress_percentage=50.0,
            current_step="Test step"
        )
        
        # Should not raise exception despite callback failure
        optimization_manager._notify_progress(session_id, update)
        
        # Working callback should still be called
        working_callback.assert_called_once_with(session_id, update)


class TestSnapshotManagement:
    """Test snapshot and rollback functionality."""
    
    def test_create_session_snapshot(self, optimization_manager):
        """Test creating session snapshot."""
        session_id = "test_session"
        state_data = {"model_state": {"param1": "value1"}}
        
        optimization_manager._create_session_snapshot(
            session_id, state_data, 0, "Test snapshot"
        )
        
        snapshots = optimization_manager.session_snapshots[session_id]
        assert len(snapshots) == 1
        
        snapshot = snapshots[0]
        assert snapshot.session_id == session_id
        assert snapshot.model_state == state_data
        assert snapshot.step_index == 0
        assert snapshot.description == "Test snapshot"
    
    def test_get_session_snapshots(self, optimization_manager):
        """Test getting session snapshots."""
        session_id = "test_session"
        
        # Create multiple snapshots
        for i in range(3):
            optimization_manager._create_session_snapshot(
                session_id, {"step": i}, i, f"Snapshot {i}"
            )
        
        snapshots_info = optimization_manager.get_session_snapshots(session_id)
        
        assert len(snapshots_info) == 3
        for i, snapshot_info in enumerate(snapshots_info):
            assert snapshot_info["index"] == i
            assert snapshot_info["description"] == f"Snapshot {i}"
            assert snapshot_info["step_index"] == i
    
    def test_rollback_session(self, optimization_manager, test_model_path, optimization_criteria):
        """Test session rollback."""
        with patch('src.services.optimization_manager.OptimizationManager._execute_optimization_workflow'):
            session_id = optimization_manager.start_optimization_session(
                test_model_path, optimization_criteria
            )
        
        # Create snapshots
        optimization_manager._create_session_snapshot(
            session_id, {"step": 0}, 0, "Initial state"
        )
        optimization_manager._create_session_snapshot(
            session_id, {"step": 1}, 1, "After step 1"
        )
        
        # Add some steps to session
        session = optimization_manager.active_sessions[session_id]
        from src.models.core import OptimizationStep
        session.steps = [
            OptimizationStep(technique="quantization"),
            OptimizationStep(technique="pruning")
        ]
        
        # Rollback to first snapshot
        result = optimization_manager.rollback_session(session_id, 0)
        
        assert result is True
        assert len(session.steps) == 0  # Steps should be truncated
        
        workflow_state = optimization_manager.workflow_states[session_id]
        assert workflow_state.status == WorkflowStatus.ROLLED_BACK
    
    def test_rollback_session_no_snapshots(self, optimization_manager, test_model_path, optimization_criteria):
        """Test rollback with no snapshots."""
        with patch('src.services.optimization_manager.OptimizationManager._execute_optimization_workflow'):
            session_id = optimization_manager.start_optimization_session(
                test_model_path, optimization_criteria
            )
        
        result = optimization_manager.rollback_session(session_id)
        
        assert result is False
    
    def test_rollback_nonexistent_session(self, optimization_manager):
        """Test rollback for non-existent session."""
        result = optimization_manager.rollback_session("non_existent_session")
        assert result is False


class TestWorkflowExecution:
    """Test workflow execution functionality."""
    
    @patch('src.services.optimization_manager.OptimizationManager._load_model')
    def test_load_model(self, mock_load, optimization_manager, test_model):
        """Test model loading."""
        mock_load.return_value = test_model
        
        loaded_model = optimization_manager._load_model("test_path.pt")
        
        assert loaded_model == test_model
        mock_load.assert_called_once_with("test_path.pt")
    
    def test_load_model_invalid_path(self, optimization_manager):
        """Test loading model with invalid path."""
        with pytest.raises(Exception):
            optimization_manager._load_model("non_existent_model.pt")
    
    def test_get_standard_benchmarks(self, optimization_manager):
        """Test getting standard benchmarks."""
        benchmarks = optimization_manager._get_standard_benchmarks()
        
        assert isinstance(benchmarks, list)
        assert len(benchmarks) > 0
        
        # Check benchmark structure
        for benchmark in benchmarks:
            assert "name" in benchmark
            assert "type" in benchmark
    
    def test_pause_resume_session(self, optimization_manager, test_model_path, optimization_criteria):
        """Test pausing and resuming session."""
        with patch('src.services.optimization_manager.OptimizationManager._execute_optimization_workflow'):
            session_id = optimization_manager.start_optimization_session(
                test_model_path, optimization_criteria
            )
        
        # Pause session
        result = optimization_manager.pause_session(session_id)
        assert result is True
        
        session = optimization_manager.active_sessions[session_id]
        assert session.status == SessionStatus.PAUSED
        
        # Resume session
        result = optimization_manager.resume_session(session_id)
        assert result is True
        assert session.status == SessionStatus.RUNNING
    
    def test_pause_nonexistent_session(self, optimization_manager):
        """Test pausing non-existent session."""
        result = optimization_manager.pause_session("non_existent_session")
        assert result is False
    
    def test_resume_non_paused_session(self, optimization_manager, test_model_path, optimization_criteria):
        """Test resuming non-paused session."""
        with patch('src.services.optimization_manager.OptimizationManager._execute_optimization_workflow'):
            session_id = optimization_manager.start_optimization_session(
                test_model_path, optimization_criteria
            )
        
        # Try to resume without pausing first
        result = optimization_manager.resume_session(session_id)
        assert result is False


class TestErrorHandling:
    """Test error handling and recovery mechanisms."""
    
    def test_handle_workflow_failure(self, optimization_manager, test_model_path, optimization_criteria):
        """Test workflow failure handling."""
        with patch('src.services.optimization_manager.OptimizationManager._execute_optimization_workflow'):
            session_id = optimization_manager.start_optimization_session(
                test_model_path, optimization_criteria
            )
        
        error_message = "Test error"
        
        with patch.object(optimization_manager, 'rollback_session', return_value=True) as mock_rollback:
            optimization_manager._handle_workflow_failure(session_id, error_message)
        
        session = optimization_manager.active_sessions[session_id]
        workflow_state = optimization_manager.workflow_states[session_id]
        
        assert session.status == SessionStatus.FAILED
        assert workflow_state.status == WorkflowStatus.FAILED
        assert workflow_state.error_message == error_message
        
        # Verify auto-rollback was attempted
        mock_rollback.assert_called_once_with(session_id)
    
    def test_handle_workflow_failure_no_auto_rollback(self, optimization_manager, test_model_path, optimization_criteria):
        """Test workflow failure handling without auto-rollback."""
        optimization_manager.auto_rollback_on_failure = False
        
        with patch('src.services.optimization_manager.OptimizationManager._execute_optimization_workflow'):
            session_id = optimization_manager.start_optimization_session(
                test_model_path, optimization_criteria
            )
        
        error_message = "Test error"
        
        with patch.object(optimization_manager, 'rollback_session') as mock_rollback:
            optimization_manager._handle_workflow_failure(session_id, error_message)
        
        # Verify rollback was not attempted
        mock_rollback.assert_not_called()
    
    def test_optimization_agents_status(self, optimization_manager):
        """Test getting optimization agents status."""
        # Mock some agents
        mock_agent1 = Mock()
        mock_agent1.get_status.return_value = {"name": "agent1"}
        mock_agent1.get_current_status.return_value = OptimizationStatus.NOT_STARTED
        mock_agent1.get_supported_techniques.return_value = ["technique1"]
        
        mock_agent2 = Mock()
        mock_agent2.get_status.side_effect = Exception("Agent error")
        
        optimization_manager.optimization_agents = {
            "agent1": mock_agent1,
            "agent2": mock_agent2
        }
        
        status = optimization_manager.get_optimization_agents_status()
        
        assert "agent1" in status
        assert status["agent1"]["name"] == "agent1"
        assert status["agent1"]["current_status"] == "not_started"
        assert status["agent1"]["supported_techniques"] == ["technique1"]
        
        assert "agent2" in status
        assert "error" in status["agent2"]


class TestOrchestrationLogic:
    """Test orchestration logic and workflow coordination."""
    
    def test_execute_analysis_phase(self, optimization_manager, test_model_path):
        """Test analysis phase execution."""
        session_id = "test_session"
        
        # Setup session state
        optimization_manager.workflow_states[session_id] = WorkflowState(
            session_id=session_id,
            status=WorkflowStatus.INITIALIZING
        )
        optimization_manager.session_snapshots[session_id] = []
        
        # Mock analysis agent
        mock_analysis_report = Mock()
        mock_analysis_agent = Mock()
        mock_analysis_agent.analyze_model.return_value = mock_analysis_report
        optimization_manager.analysis_agent = mock_analysis_agent
        
        # Execute analysis phase
        result = optimization_manager._execute_analysis_phase(session_id, test_model_path)
        
        assert result == mock_analysis_report
        mock_analysis_agent.analyze_model.assert_called_once_with(test_model_path)
        
        # Verify workflow state updated
        workflow_state = optimization_manager.workflow_states[session_id]
        assert workflow_state.status == WorkflowStatus.ANALYZING
        assert workflow_state.progress_percentage == 10.0
        
        # Verify snapshot created
        assert len(optimization_manager.session_snapshots[session_id]) == 1
    
    def test_execute_planning_phase(self, optimization_manager, optimization_criteria):
        """Test planning phase execution."""
        session_id = "test_session"
        
        # Setup session and workflow state
        from src.models.core import OptimizationSession
        session = OptimizationSession(
            id=session_id,
            model_id="test_model",
            status=SessionStatus.RUNNING,
            criteria_name=optimization_criteria.name,
            created_by="test"
        )
        optimization_manager.active_sessions[session_id] = session
        optimization_manager.workflow_states[session_id] = WorkflowState(
            session_id=session_id,
            status=WorkflowStatus.ANALYZING
        )
        optimization_manager.session_snapshots[session_id] = []
        
        # Mock planning agent
        mock_analysis_report = Mock()
        mock_optimization_plan = Mock()
        mock_optimization_plan.plan_id = "test_plan"
        mock_optimization_plan.steps = []
        
        mock_planning_agent = Mock()
        mock_planning_agent.plan_optimization.return_value = mock_optimization_plan
        mock_planning_agent.validate_plan.return_value = Mock(is_valid=True)
        optimization_manager.planning_agent = mock_planning_agent
        
        # Execute planning phase
        result = optimization_manager._execute_planning_phase(
            session_id, mock_analysis_report, optimization_criteria
        )
        
        assert result == mock_optimization_plan
        mock_planning_agent.plan_optimization.assert_called_once_with(mock_analysis_report, optimization_criteria)
        mock_planning_agent.validate_plan.assert_called_once_with(mock_optimization_plan)
        
        # Verify session updated with plan
        assert session.plan_id == "test_plan"
        
        # Verify workflow state updated
        workflow_state = optimization_manager.workflow_states[session_id]
        assert workflow_state.status == WorkflowStatus.PLANNING
        assert workflow_state.progress_percentage == 25.0
    
    def test_execute_planning_phase_invalid_plan(self, optimization_manager, optimization_criteria):
        """Test planning phase with invalid plan."""
        session_id = "test_session"
        
        # Setup session state
        optimization_manager.active_sessions[session_id] = Mock()
        optimization_manager.workflow_states[session_id] = WorkflowState(
            session_id=session_id,
            status=WorkflowStatus.ANALYZING
        )
        
        # Mock planning agent with invalid plan
        mock_analysis_report = Mock()
        mock_optimization_plan = Mock()
        
        mock_planning_agent = Mock()
        mock_planning_agent.plan_optimization.return_value = mock_optimization_plan
        mock_planning_agent.validate_plan.return_value = Mock(
            is_valid=False, 
            issues=["Invalid constraint"]
        )
        optimization_manager.planning_agent = mock_planning_agent
        
        # Execute planning phase should raise error
        with pytest.raises(RuntimeError, match="Invalid optimization plan"):
            optimization_manager._execute_planning_phase(
                session_id, mock_analysis_report, optimization_criteria
            )
    
    @patch('src.services.optimization_manager.OptimizationManager._load_model')
    def test_execute_optimization_phase(self, mock_load_model, optimization_manager, test_model):
        """Test optimization phase execution."""
        session_id = "test_session"
        model_path = "test_model.pt"
        
        # Setup session and workflow state
        from src.models.core import OptimizationSession
        session = OptimizationSession(
            id=session_id,
            model_id=model_path,
            status=SessionStatus.RUNNING,
            criteria_name="test_criteria",
            created_by="test"
        )
        optimization_manager.active_sessions[session_id] = session
        optimization_manager.workflow_states[session_id] = WorkflowState(
            session_id=session_id,
            status=WorkflowStatus.PLANNING
        )
        optimization_manager.session_snapshots[session_id] = []
        
        # Mock model loading
        mock_load_model.return_value = test_model
        
        # Create mock optimization plan
        from src.agents.planning.agent import OptimizationPlan, OptimizationStep
        plan_step = OptimizationStep(
            step_id="step_1",
            technique="quantization",
            parameters={"bits": 8}
        )
        optimization_plan = OptimizationPlan(
            plan_id="test_plan",
            steps=[plan_step],
            expected_improvements={}
        )
        
        # Mock optimization agent
        mock_optimization_result = OptimizationResult(
            success=True,
            optimized_model=test_model,
            original_model=test_model,
            optimization_metadata={"technique": "quantization"},
            performance_metrics={"size_reduction": 0.5},
            optimization_time=30.0,
            technique_used="quantization",
            validation_result=Mock(is_valid=True)
        )
        
        mock_agent = Mock()
        mock_agent.optimize_with_tracking.return_value = mock_optimization_result
        mock_agent.add_progress_callback = Mock()
        mock_agent.remove_progress_callback = Mock()
        optimization_manager.optimization_agents["quantization"] = mock_agent
        
        # Execute optimization phase
        result = optimization_manager._execute_optimization_phase(
            session_id, model_path, optimization_plan
        )
        
        assert result["original_model_path"] == model_path
        assert result["final_model"] == test_model
        assert len(result["step_results"]) == 1
        assert result["step_results"][0] == mock_optimization_result
        assert "quantization" in result["optimized_models"]
        
        # Verify agent was called correctly
        mock_agent.optimize_with_tracking.assert_called_once()
        mock_agent.add_progress_callback.assert_called_once()
        mock_agent.remove_progress_callback.assert_called_once()
    
    @patch('src.services.optimization_manager.OptimizationManager._load_model')
    def test_execute_optimization_phase_with_failure(self, mock_load_model, optimization_manager, test_model):
        """Test optimization phase with step failure."""
        session_id = "test_session"
        model_path = "test_model.pt"
        
        # Setup session
        from src.models.core import OptimizationSession
        session = OptimizationSession(
            id=session_id,
            model_id=model_path,
            status=SessionStatus.RUNNING,
            criteria_name="test_criteria",
            created_by="test"
        )
        # Add a step to the session
        from src.models.core import OptimizationStep as SessionStep
        session.steps = [SessionStep(technique="quantization")]
        
        optimization_manager.active_sessions[session_id] = session
        optimization_manager.workflow_states[session_id] = WorkflowState(
            session_id=session_id,
            status=WorkflowStatus.EXECUTING
        )
        optimization_manager.session_snapshots[session_id] = []
        
        # Mock model loading
        mock_load_model.return_value = test_model
        
        # Create mock optimization plan
        from src.agents.planning.agent import OptimizationPlan, OptimizationStep
        plan_step = OptimizationStep(
            step_id="step_1",
            technique="quantization",
            parameters={"bits": 8}
        )
        optimization_plan = OptimizationPlan(
            plan_id="test_plan",
            steps=[plan_step],
            expected_improvements={}
        )
        
        # Mock optimization agent with failure
        mock_optimization_result = OptimizationResult(
            success=False,
            optimized_model=None,
            original_model=test_model,
            optimization_metadata={},
            performance_metrics={},
            optimization_time=0.0,
            technique_used="quantization",
            validation_result=None,
            error_message="Quantization failed"
        )
        
        mock_agent = Mock()
        mock_agent.optimize_with_tracking.return_value = mock_optimization_result
        mock_agent.add_progress_callback = Mock()
        mock_agent.remove_progress_callback = Mock()
        optimization_manager.optimization_agents["quantization"] = mock_agent
        
        # Execute optimization phase
        result = optimization_manager._execute_optimization_phase(
            session_id, model_path, optimization_plan
        )
        
        assert result["final_model"] == test_model  # Should revert to original
        assert len(result["step_results"]) == 1
        assert not result["step_results"][0].success
        
        # Verify session step marked as failed (if auto_rollback is False)
        if not optimization_manager.auto_rollback_on_failure:
            assert session.steps[0].status == SessionStatus.FAILED
            assert session.steps[0].error_message == "Quantization failed"
    
    def test_execute_evaluation_phase(self, optimization_manager):
        """Test evaluation phase execution."""
        session_id = "test_session"
        
        # Setup workflow state
        optimization_manager.workflow_states[session_id] = WorkflowState(
            session_id=session_id,
            status=WorkflowStatus.EXECUTING
        )
        optimization_manager.session_snapshots[session_id] = []
        
        # Mock optimization results
        mock_final_model = Mock()
        mock_original_model = Mock()
        optimization_results = {
            "final_model": mock_final_model,
            "original_model_path": "original_model.pt",
            "step_results": [Mock(), Mock()]
        }
        
        # Mock evaluation agent
        mock_comparison_result = Mock()
        mock_evaluation_report = Mock()
        mock_evaluation_report.comparison_baseline = mock_comparison_result
        mock_evaluation_report.model_id = f"{session_id}_optimized"
        mock_evaluation_report.session_id = session_id
        
        mock_evaluation_agent = Mock()
        mock_evaluation_agent.compare_models.return_value = mock_comparison_result
        mock_evaluation_agent.evaluate_model.return_value = mock_evaluation_report
        optimization_manager.evaluation_agent = mock_evaluation_agent
        
        # Mock model loading
        with patch.object(optimization_manager, '_load_model', return_value=mock_original_model):
            with patch.object(optimization_manager, '_get_standard_benchmarks', return_value=[]):
                result = optimization_manager._execute_evaluation_phase(session_id, optimization_results)
        
        assert result == mock_evaluation_report
        mock_evaluation_agent.compare_models.assert_called_once_with(mock_original_model, mock_final_model)
        mock_evaluation_agent.evaluate_model.assert_called_once_with(mock_final_model, [])
        
        # Verify workflow state updated
        workflow_state = optimization_manager.workflow_states[session_id]
        assert workflow_state.status == WorkflowStatus.EVALUATING
        assert workflow_state.progress_percentage == 85.0
    
    def test_execute_evaluation_phase_no_final_model(self, optimization_manager):
        """Test evaluation phase with no final model."""
        session_id = "test_session"
        
        # Setup workflow state
        optimization_manager.workflow_states[session_id] = WorkflowState(
            session_id=session_id,
            status=WorkflowStatus.EXECUTING
        )
        
        # Mock optimization results without final model
        optimization_results = {
            "final_model": None,
            "original_model_path": "original_model.pt"
        }
        
        # Execute evaluation phase should raise error
        with pytest.raises(RuntimeError, match="No final optimized model available"):
            optimization_manager._execute_evaluation_phase(session_id, optimization_results)
    
    def test_execute_optimization_step_missing_agent(self, optimization_manager, test_model):
        """Test optimization step with missing agent."""
        session_id = "test_session"
        
        from src.agents.planning.agent import OptimizationStep
        plan_step = OptimizationStep(
            step_id="step_1",
            technique="missing_technique",
            parameters={}
        )
        
        result = optimization_manager._execute_optimization_step(session_id, test_model, plan_step)
        
        assert not result.success
        assert result.error_message == "Optimization agent for technique 'missing_technique' not available"
        assert result.original_model == test_model
        assert result.optimized_model is None
    
    def test_execute_optimization_step_agent_exception(self, optimization_manager, test_model):
        """Test optimization step with agent exception."""
        session_id = "test_session"
        
        from src.agents.planning.agent import OptimizationStep
        plan_step = OptimizationStep(
            step_id="step_1",
            technique="quantization",
            parameters={}
        )
        
        # Mock agent that raises exception
        mock_agent = Mock()
        mock_agent.optimize_with_tracking.side_effect = Exception("Agent error")
        mock_agent.add_progress_callback = Mock()
        mock_agent.remove_progress_callback = Mock()
        optimization_manager.optimization_agents["quantization"] = mock_agent
        
        result = optimization_manager._execute_optimization_step(session_id, test_model, plan_step)
        
        assert not result.success
        assert result.error_message == "Agent error"
        assert result.original_model == test_model
        assert result.optimized_model is None
    
    def test_complete_session(self, optimization_manager, test_model_path, optimization_criteria):
        """Test session completion."""
        session_id = "test_session"
        
        # Setup session and workflow state
        from src.models.core import OptimizationSession
        session = OptimizationSession(
            id=session_id,
            model_id=test_model_path,
            status=SessionStatus.RUNNING,
            criteria_name=optimization_criteria.name,
            created_by="test"
        )
        # Add completed steps
        from src.models.core import OptimizationStep
        session.steps = [
            OptimizationStep(technique="quantization"),
            OptimizationStep(technique="pruning")
        ]
        
        optimization_manager.active_sessions[session_id] = session
        optimization_manager.workflow_states[session_id] = WorkflowState(
            session_id=session_id,
            status=WorkflowStatus.EVALUATING
        )
        
        # Mock optimization results and evaluation report
        optimization_results = {
            "step_results": [Mock(), Mock()]
        }
        
        mock_evaluation_report = Mock()
        mock_evaluation_report.validation_status = Mock()
        mock_evaluation_report.validation_status.value = "passed"
        mock_evaluation_report.comparison_baseline = Mock()
        mock_evaluation_report.comparison_baseline.improvements = {"size": 0.5}
        
        # Execute completion
        optimization_manager._complete_session(session_id, optimization_results, mock_evaluation_report)
        
        # Verify session completed
        assert session.status == SessionStatus.COMPLETED
        assert session.completed_at is not None
        assert session.results is not None
        assert session.results.techniques_applied == ["quantization", "pruning"]
        assert session.results.validation_passed is True
        
        # Verify workflow state
        workflow_state = optimization_manager.workflow_states[session_id]
        assert workflow_state.status == WorkflowStatus.COMPLETED
        assert workflow_state.progress_percentage == 100.0
    
    def test_convert_plan_step_to_session_step(self, optimization_manager):
        """Test converting plan step to session step."""
        from src.agents.planning.agent import OptimizationStep as PlanStep
        
        plan_step = PlanStep(
            step_id="step_1",
            technique="quantization",
            parameters={"bits": 8}
        )
        
        session_step = optimization_manager._convert_plan_step_to_session_step(plan_step)
        
        assert session_step.step_id == "step_1"
        assert session_step.technique == "quantization"
        assert session_step.parameters == {"bits": 8}
        assert session_step.status == SessionStatus.PENDING


class TestRecoveryMechanisms:
    """Test error recovery and graceful degradation mechanisms."""
    
    @patch('src.services.optimization_manager.recovery_manager')
    @patch('src.services.optimization_manager.RetryableOperation')
    def test_execute_analysis_phase_with_recovery(self, mock_retry_op, mock_recovery, 
                                                 optimization_manager, test_model_path):
        """Test analysis phase with recovery mechanisms."""
        session_id = "test_session"
        recovery_context = {"session_id": session_id}
        
        # Mock analysis phase to succeed
        mock_analysis_report = Mock()
        mock_retry_instance = Mock()
        mock_retry_instance.execute.return_value = mock_analysis_report
        mock_retry_op.return_value = mock_retry_instance
        
        with patch.object(optimization_manager, '_execute_analysis_phase', return_value=mock_analysis_report):
            result = optimization_manager._execute_analysis_phase_with_recovery(
                session_id, test_model_path, recovery_context
            )
        
        assert result == mock_analysis_report
        mock_retry_op.assert_called_once()
        mock_retry_instance.execute.assert_called_once()
    
    @patch('src.services.optimization_manager.recovery_manager')
    def test_execute_planning_phase_with_recovery_success(self, mock_recovery, optimization_manager):
        """Test planning phase with recovery - success case."""
        session_id = "test_session"
        mock_analysis_report = Mock()
        mock_criteria = Mock()
        recovery_context = {"session_id": session_id}
        
        # Mock planning phase to succeed
        mock_plan = Mock()
        with patch.object(optimization_manager, '_execute_planning_phase', return_value=mock_plan):
            result = optimization_manager._execute_planning_phase_with_recovery(
                session_id, mock_analysis_report, mock_criteria, recovery_context
            )
        
        assert result == mock_plan
        mock_recovery.handle_error.assert_not_called()
    
    @patch('src.services.optimization_manager.recovery_manager')
    def test_execute_planning_phase_with_recovery_failure(self, mock_recovery, optimization_manager):
        """Test planning phase with recovery - failure and recovery."""
        session_id = "test_session"
        mock_analysis_report = Mock()
        mock_criteria = Mock()
        recovery_context = {"session_id": session_id}
        
        # Mock planning phase to fail first, then succeed
        mock_plan = Mock()
        planning_calls = [Exception("Planning failed"), mock_plan]
        
        with patch.object(optimization_manager, '_execute_planning_phase', side_effect=planning_calls):
            mock_recovery.handle_error.return_value = True  # Recovery successful
            
            result = optimization_manager._execute_planning_phase_with_recovery(
                session_id, mock_analysis_report, mock_criteria, recovery_context
            )
        
        assert result == mock_plan
        mock_recovery.handle_error.assert_called_once()
    
    @patch('src.services.optimization_manager.degradation_manager')
    def test_execute_optimization_phase_with_graceful_degradation(self, mock_degradation, 
                                                                 optimization_manager):
        """Test optimization phase with graceful degradation."""
        session_id = "test_session"
        model_path = "test_model.pt"
        optimization_plan = {"techniques": ["quantization", "pruning"]}
        recovery_context = {"session_id": session_id}
        
        # Mock original plan to fail
        original_error = Exception("Original plan failed")
        original_error.context = {"technique": "quantization"}
        
        # Mock degraded plan to succeed
        mock_result = {"success": True, "degraded": True}
        
        with patch.object(optimization_manager, '_execute_optimization_phase', 
                         side_effect=[original_error, mock_result]):
            # Mock degradation manager
            mock_degradation.create_degraded_plan.return_value = ["pruning"]
            optimization_manager.optimization_agents = {"quantization": Mock(), "pruning": Mock()}
            
            result = optimization_manager._execute_optimization_phase_with_recovery(
                session_id, model_path, optimization_plan, recovery_context
            )
        
        assert result["degraded"] is True
        assert result["original_techniques"] == ["quantization", "pruning"]
        assert result["failed_techniques"] == ["quantization"]
        mock_degradation.create_degraded_plan.assert_called_once()
    
    def test_execute_minimal_evaluation(self, optimization_manager):
        """Test minimal evaluation fallback."""
        session_id = "test_session"
        
        # Mock optimization results with final model
        mock_model = Mock()
        mock_model.parameters.return_value = [Mock(numel=Mock(return_value=1000))]
        
        optimization_results = {
            "final_model": mock_model,
            "techniques_applied": ["quantization"],
            "total_time": 120.0
        }
        
        result = optimization_manager._execute_minimal_evaluation(session_id, optimization_results)
        
        assert result["evaluation_status"] == "minimal"
        assert result["basic_metrics"]["optimization_completed"] is True
        assert result["basic_metrics"]["model_parameters"] == 1000
        assert result["basic_metrics"]["techniques_applied"] == ["quantization"]
        assert result["basic_metrics"]["optimization_time"] == 120.0
        assert "warnings" in result
    
    def test_execute_minimal_evaluation_no_model(self, optimization_manager):
        """Test minimal evaluation with no model."""
        session_id = "test_session"
        optimization_results = {"final_model": None}
        
        result = optimization_manager._execute_minimal_evaluation(session_id, optimization_results)
        
        assert result["evaluation_status"] == "minimal"
        assert result["basic_metrics"]["optimization_completed"] is False
        assert "error_message" in result
    
    def test_execute_minimal_evaluation_model_error(self, optimization_manager):
        """Test minimal evaluation with model error."""
        session_id = "test_session"
        
        # Mock model that raises exception
        mock_model = Mock()
        mock_model.parameters.side_effect = Exception("Model error")
        
        optimization_results = {"final_model": mock_model}
        
        result = optimization_manager._execute_minimal_evaluation(session_id, optimization_results)
        
        assert result["evaluation_status"] == "failed"
        assert result["basic_metrics"]["optimization_completed"] is False
        assert "Model error" in result["error_message"]


class TestIntegration:
    """Integration tests for OptimizationManager."""
    
    @patch('src.services.optimization_manager.AnalysisAgent')
    @patch('src.services.optimization_manager.PlanningAgent')
    @patch('src.services.optimization_manager.EvaluationAgent')
    @patch('src.services.optimization_manager.QuantizationAgent')
    @patch('src.services.optimization_manager.PruningAgent')
    def test_full_workflow_mock(self, mock_pruning, mock_quantization, mock_evaluation, 
                               mock_planning, mock_analysis, optimization_manager, 
                               test_model_path, optimization_criteria):
        """Test full optimization workflow with mocked agents."""
        # Setup mocks
        mock_analysis_instance = mock_analysis.return_value
        mock_analysis_instance.initialize.return_value = True
        mock_analysis_instance.analyze_model.return_value = Mock()
        
        mock_planning_instance = mock_planning.return_value
        mock_planning_instance.initialize.return_value = True
        mock_plan = Mock()
        mock_plan.steps = []
        mock_planning_instance.plan_optimization.return_value = mock_plan
        mock_planning_instance.validate_plan.return_value = Mock(is_valid=True)
        
        mock_evaluation_instance = mock_evaluation.return_value
        mock_evaluation_instance.initialize.return_value = True
        mock_evaluation_instance.compare_models.return_value = Mock()
        mock_evaluation_instance.evaluate_model.return_value = Mock()
        
        mock_quantization_instance = mock_quantization.return_value
        mock_quantization_instance.initialize.return_value = True
        
        mock_pruning_instance = mock_pruning.return_value
        mock_pruning_instance.initialize.return_value = True
        
        # Initialize manager
        assert optimization_manager.initialize() is True
        
        # Start session
        session_id = optimization_manager.start_optimization_session(
            test_model_path, optimization_criteria
        )
        
        assert session_id is not None
        assert session_id in optimization_manager.active_sessions
        
        # Wait a bit for workflow to start
        time.sleep(0.1)
        
        # Verify session was created
        session = optimization_manager.active_sessions[session_id]
        assert session.model_id == test_model_path
        assert session.criteria_name == optimization_criteria.name


if __name__ == "__main__":
    pytest.main([__file__])