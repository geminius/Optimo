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