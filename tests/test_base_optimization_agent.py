"""
Unit tests for BaseOptimizationAgent framework.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
from datetime import datetime
from typing import Dict, Any

from src.agents.base import (
    BaseOptimizationAgent, 
    OptimizationStatus, 
    ProgressUpdate, 
    OptimizationResult,
    OptimizedModel,
    ImpactEstimate,
    ValidationResult,
    OptimizationSnapshot
)


class MockOptimizationAgent(BaseOptimizationAgent):
    """Mock implementation of BaseOptimizationAgent for testing."""
    
    def __init__(self, config: Dict[str, Any], should_fail: bool = False, 
                 should_validate: bool = True, should_cancel: bool = False):
        super().__init__(config)
        self.should_fail = should_fail
        self.should_validate = should_validate
        self.should_cancel = should_cancel
        self.optimize_called = False
        self.validate_called = False
    
    def initialize(self) -> bool:
        return True
    
    def cleanup(self) -> None:
        pass
    
    def can_optimize(self, model: torch.nn.Module) -> bool:
        return True
    
    def estimate_impact(self, model: torch.nn.Module) -> ImpactEstimate:
        return ImpactEstimate(
            performance_improvement=0.2,
            size_reduction=0.1,
            speed_improvement=0.15,
            confidence=0.8,
            estimated_time_minutes=5
        )
    
    def optimize(self, model: torch.nn.Module, config: Dict[str, Any]) -> OptimizedModel:
        self.optimize_called = True
        
        if self.should_fail:
            raise RuntimeError("Simulated optimization failure")
        
        # Create a simple optimized model (just a copy for testing)
        optimized_model = nn.Linear(model.in_features, model.out_features)
        
        return OptimizedModel(
            model=optimized_model,
            optimization_metadata={"technique": "mock", "config": config},
            performance_metrics={"accuracy": 0.95, "speed": 1.2},
            optimization_time=1.5,
            technique_used="mock_optimization"
        )
    
    def validate_result(self, original: torch.nn.Module, optimized: torch.nn.Module) -> ValidationResult:
        self.validate_called = True
        return ValidationResult(
            is_valid=self.should_validate,
            accuracy_preserved=self.should_validate,
            performance_metrics={"accuracy": 0.95 if self.should_validate else 0.5},
            issues=[] if self.should_validate else ["Accuracy degraded"],
            recommendations=["Good optimization"] if self.should_validate else ["Consider different technique"]
        )


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    return nn.Linear(10, 5)


@pytest.fixture
def mock_agent():
    """Create a mock optimization agent for testing."""
    config = {"test_param": "test_value"}
    return MockOptimizationAgent(config)


@pytest.fixture
def failing_agent():
    """Create a mock optimization agent that fails during optimization."""
    config = {"test_param": "test_value"}
    return MockOptimizationAgent(config, should_fail=True)


@pytest.fixture
def invalid_agent():
    """Create a mock optimization agent that produces invalid results."""
    config = {"test_param": "test_value"}
    return MockOptimizationAgent(config, should_validate=False)


class TestBaseOptimizationAgent:
    """Test cases for BaseOptimizationAgent."""
    
    def test_initialization(self, mock_agent):
        """Test agent initialization."""
        assert mock_agent.name == "MockOptimizationAgent"
        assert mock_agent.config == {"test_param": "test_value"}
        assert mock_agent.get_current_status() == OptimizationStatus.NOT_STARTED
        assert len(mock_agent.get_snapshots()) == 0
    
    def test_progress_callbacks(self, mock_agent):
        """Test progress callback functionality."""
        progress_updates = []
        
        def callback(update: ProgressUpdate):
            progress_updates.append(update)
        
        mock_agent.add_progress_callback(callback)
        mock_agent._update_progress(OptimizationStatus.ANALYZING, 50.0, "Test step")
        
        assert len(progress_updates) == 1
        assert progress_updates[0].status == OptimizationStatus.ANALYZING
        assert progress_updates[0].progress_percentage == 50.0
        assert progress_updates[0].current_step == "Test step"
        
        # Test removing callback
        mock_agent.remove_progress_callback(callback)
        mock_agent._update_progress(OptimizationStatus.OPTIMIZING, 75.0, "Another step")
        
        assert len(progress_updates) == 1  # No new updates
    
    def test_snapshot_creation_and_rollback(self, mock_agent, simple_model):
        """Test snapshot creation and rollback functionality."""
        # Create initial snapshot
        original_state = simple_model.state_dict().copy()
        snapshot = mock_agent._create_snapshot(simple_model, "test_snapshot")
        
        assert snapshot.checkpoint_name == "test_snapshot"
        assert len(mock_agent.get_snapshots()) == 1
        
        # Modify model
        with torch.no_grad():
            simple_model.weight.fill_(999.0)
        
        # Verify model was modified
        assert not torch.equal(simple_model.weight, torch.zeros_like(simple_model.weight))
        
        # Rollback to snapshot
        success = mock_agent._rollback_to_snapshot(simple_model, snapshot)
        assert success
        
        # Verify rollback worked
        for key in original_state:
            assert torch.equal(simple_model.state_dict()[key], original_state[key])
    
    def test_successful_optimization(self, mock_agent, simple_model):
        """Test successful optimization workflow."""
        config = {"learning_rate": 0.01}
        
        result = mock_agent.optimize_with_tracking(simple_model, config)
        
        assert result.success
        assert result.optimized_model is not None
        assert result.original_model is not None
        assert result.technique_used == "mock_optimization"
        assert result.validation_result is not None
        assert result.validation_result.is_valid
        assert result.error_message is None
        assert len(result.snapshots) > 0
        
        # Verify agent methods were called
        assert mock_agent.optimize_called
        assert mock_agent.validate_called
        assert mock_agent.get_current_status() == OptimizationStatus.COMPLETED
    
    def test_optimization_failure(self, failing_agent, simple_model):
        """Test optimization failure handling."""
        config = {"learning_rate": 0.01}
        
        result = failing_agent.optimize_with_tracking(simple_model, config)
        
        assert not result.success
        assert result.optimized_model is None
        assert result.original_model is not None
        assert "Simulated optimization failure" in result.error_message
        # Status could be FAILED or ROLLED_BACK depending on rollback success
        assert failing_agent.get_current_status() in [OptimizationStatus.FAILED, OptimizationStatus.ROLLED_BACK]
    
    def test_validation_failure_with_rollback(self, invalid_agent, simple_model):
        """Test validation failure and automatic rollback."""
        config = {"learning_rate": 0.01}
        original_state = simple_model.state_dict().copy()
        
        result = invalid_agent.optimize_with_tracking(simple_model, config)
        
        assert not result.success
        assert result.optimized_model is None
        assert result.validation_result is not None
        assert not result.validation_result.is_valid
        assert result.error_message == "Optimization validation failed"
        
        # Verify model was rolled back to original state
        for key in original_state:
            assert torch.equal(simple_model.state_dict()[key], original_state[key])
    
    def test_optimization_cancellation(self, simple_model):
        """Test optimization cancellation."""
        config = {"learning_rate": 0.01}
        agent = MockOptimizationAgent(config, should_cancel=True)
        
        # Cancel optimization before starting
        agent.cancel_optimization()
        assert agent.is_cancelled()
        
        result = agent.optimize_with_tracking(simple_model, config)
        
        assert not result.success
        assert result.error_message == "Optimization was cancelled"
        assert agent.get_current_status() == OptimizationStatus.CANCELLED
    
    def test_snapshot_management(self, mock_agent, simple_model):
        """Test snapshot management functionality."""
        # Create multiple snapshots
        mock_agent._create_snapshot(simple_model, "snapshot1")
        mock_agent._create_snapshot(simple_model, "snapshot2")
        mock_agent._create_snapshot(simple_model, "snapshot3")
        
        snapshots = mock_agent.get_snapshots()
        assert len(snapshots) == 3
        assert snapshots[0].checkpoint_name == "snapshot1"
        assert snapshots[2].checkpoint_name == "snapshot3"
        
        # Test rollback to latest
        success = mock_agent._rollback_to_latest_snapshot(simple_model)
        assert success
        
        # Clear snapshots
        mock_agent.clear_snapshots()
        assert len(mock_agent.get_snapshots()) == 0
        
        # Test rollback when no snapshots exist
        success = mock_agent._rollback_to_latest_snapshot(simple_model)
        assert not success
    
    def test_progress_callback_error_handling(self, mock_agent):
        """Test that errors in progress callbacks don't break the optimization."""
        def failing_callback(update: ProgressUpdate):
            raise RuntimeError("Callback error")
        
        mock_agent.add_progress_callback(failing_callback)
        
        # This should not raise an exception
        mock_agent._update_progress(OptimizationStatus.ANALYZING, 50.0, "Test step")
        
        assert mock_agent.get_current_status() == OptimizationStatus.ANALYZING
    
    def test_get_status_method(self, mock_agent):
        """Test the get_status method from BaseAgent."""
        status = mock_agent.get_status()
        
        assert status["name"] == "MockOptimizationAgent"
        assert "created_at" in status
        assert status["config"] == {"test_param": "test_value"}
    
    def test_supported_techniques_default(self, mock_agent):
        """Test default implementation of get_supported_techniques."""
        techniques = mock_agent.get_supported_techniques()
        assert isinstance(techniques, list)
        assert len(techniques) == 0  # Default implementation returns empty list


if __name__ == "__main__":
    pytest.main([__file__])