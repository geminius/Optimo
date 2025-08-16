"""
Unit tests for error handling and recovery mechanisms.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import time

from src.utils.exceptions import (
    PlatformError, OptimizationError, ValidationError, ModelLoadingError,
    ErrorCategory, ErrorSeverity
)
from src.utils.retry import (
    RetryConfig, RetryableOperation, retry_with_backoff,
    QUICK_RETRY, STANDARD_RETRY, PERSISTENT_RETRY
)
from src.utils.recovery import (
    RecoveryManager, ModelRecoveryManager, GracefulDegradationManager,
    RecoveryStrategy, RecoveryAction, recovery_manager, model_recovery_manager
)


class TestPlatformExceptions:
    """Test custom exception classes."""
    
    def test_platform_error_creation(self):
        """Test creating platform errors with different parameters."""
        error = PlatformError(
            message="Test error",
            category=ErrorCategory.OPTIMIZATION,
            severity=ErrorSeverity.HIGH,
            error_code="OPT001",
            context={"session_id": "test123"},
            recoverable=True,
            retry_after=30
        )
        
        assert error.message == "Test error"
        assert error.category == ErrorCategory.OPTIMIZATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.error_code == "OPT001"
        assert error.context["session_id"] == "test123"
        assert error.recoverable is True
        assert error.retry_after == 30
    
    def test_optimization_error_context(self):
        """Test optimization error with specific context."""
        error = OptimizationError(
            message="Quantization failed",
            technique="quantization_4bit",
            session_id="sess_123",
            step="model_conversion"
        )
        
        assert error.category == ErrorCategory.OPTIMIZATION
        assert error.context["technique"] == "quantization_4bit"
        assert error.context["session_id"] == "sess_123"
        assert error.context["step"] == "model_conversion"
    
    def test_model_loading_error_context(self):
        """Test model loading error with file context."""
        error = ModelLoadingError(
            message="Invalid model format",
            model_path="/path/to/model.pt",
            model_format="pytorch"
        )
        
        assert error.category == ErrorCategory.MODEL_LOADING
        assert error.context["model_path"] == "/path/to/model.pt"
        assert error.context["model_format"] == "pytorch"
    
    def test_error_serialization(self):
        """Test error serialization to dictionary."""
        error = ValidationError(
            message="Validation failed",
            validation_type="accuracy_check",
            failed_checks=["accuracy_threshold", "performance_regression"]
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["message"] == "Validation failed"
        assert error_dict["category"] == "validation"
        assert error_dict["context"]["validation_type"] == "accuracy_check"
        assert error_dict["context"]["failed_checks"] == ["accuracy_threshold", "performance_regression"]


class TestRetryMechanism:
    """Test retry logic with exponential backoff."""
    
    def test_retry_config_delay_calculation(self):
        """Test delay calculation with exponential backoff."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            max_delay=10.0,
            jitter=False
        )
        
        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 4.0
        assert config.calculate_delay(4) == 8.0
        assert config.calculate_delay(5) == 10.0  # Capped at max_delay
    
    def test_retry_config_with_jitter(self):
        """Test delay calculation with jitter."""
        config = RetryConfig(
            base_delay=1.0,
            exponential_base=2.0,
            jitter=True
        )
        
        delay1 = config.calculate_delay(2)
        delay2 = config.calculate_delay(2)
        
        # With jitter, delays should be different
        # (though there's a small chance they could be the same)
        assert delay1 >= 1.8  # 2.0 - 10% jitter
        assert delay1 <= 2.2  # 2.0 + 10% jitter
    
    def test_should_retry_logic(self):
        """Test retry decision logic."""
        config = RetryConfig(
            max_attempts=3,
            retryable_exceptions=[ConnectionError, TimeoutError],
            non_retryable_exceptions=[ValueError]
        )
        
        # Should retry retryable exceptions
        assert config.should_retry(ConnectionError("Network error"), 1) is True
        assert config.should_retry(TimeoutError("Timeout"), 2) is True
        
        # Should not retry non-retryable exceptions
        assert config.should_retry(ValueError("Invalid value"), 1) is False
        
        # Should not retry after max attempts
        assert config.should_retry(ConnectionError("Network error"), 3) is False
    
    def test_platform_error_retry_logic(self):
        """Test retry logic for platform errors."""
        config = RetryConfig(max_attempts=3)
        
        # Recoverable error should be retried
        recoverable_error = PlatformError(
            "Recoverable error",
            ErrorCategory.NETWORK,
            recoverable=True
        )
        assert config.should_retry(recoverable_error, 1) is True
        
        # Non-recoverable error should not be retried
        non_recoverable_error = PlatformError(
            "Non-recoverable error",
            ErrorCategory.CONFIGURATION,
            recoverable=False
        )
        assert config.should_retry(non_recoverable_error, 1) is False
        
        # Critical error should not be retried
        critical_error = PlatformError(
            "Critical error",
            ErrorCategory.SYSTEM,
            severity=ErrorSeverity.CRITICAL
        )
        assert config.should_retry(critical_error, 1) is False
    
    @patch('time.sleep')
    def test_retry_decorator_success(self, mock_sleep):
        """Test retry decorator with successful retry."""
        call_count = 0
        
        @retry_with_backoff(QUICK_RETRY)
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Network error")
            return "success"
        
        result = flaky_function()
        
        assert result == "success"
        assert call_count == 3
        assert mock_sleep.call_count == 2  # Two retries
    
    @patch('time.sleep')
    def test_retry_decorator_failure(self, mock_sleep):
        """Test retry decorator with ultimate failure."""
        call_count = 0
        
        @retry_with_backoff(QUICK_RETRY)
        def always_failing_function():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent network error")
        
        with pytest.raises(ConnectionError):
            always_failing_function()
        
        assert call_count == 3  # Max attempts
        assert mock_sleep.call_count == 2  # Two retries
    
    @patch('time.sleep')
    def test_retryable_operation(self, mock_sleep):
        """Test RetryableOperation class."""
        call_count = 0
        retry_callback_called = False
        success_callback_called = False
        
        def on_retry(error, attempt):
            nonlocal retry_callback_called
            retry_callback_called = True
        
        def on_success(attempt):
            nonlocal success_callback_called
            success_callback_called = True
        
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise TimeoutError("Timeout")
            return "success"
        
        retry_op = RetryableOperation(
            "test_operation",
            config=QUICK_RETRY,
            on_retry=on_retry,
            on_success=on_success
        )
        
        result = retry_op.execute(flaky_operation)
        
        assert result == "success"
        assert call_count == 2
        assert retry_callback_called is True
        assert success_callback_called is True


class TestRecoveryManager:
    """Test recovery manager functionality."""
    
    def test_recovery_action_creation(self):
        """Test creating recovery actions."""
        action_executed = False
        
        def recovery_action():
            nonlocal action_executed
            action_executed = True
            return True
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.ROLLBACK,
            description="Test rollback",
            action=recovery_action,
            priority=10
        )
        
        assert action.strategy == RecoveryStrategy.ROLLBACK
        assert action.description == "Test rollback"
        assert action.priority == 10
        assert action.can_execute() is True
        
        result = action.execute()
        assert result is True
        assert action_executed is True
    
    def test_recovery_action_with_conditions(self):
        """Test recovery action with execution conditions."""
        condition_met = False
        
        def condition_check():
            return condition_met
        
        def recovery_action():
            return True
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="Conditional recovery",
            action=recovery_action,
            conditions=[condition_check]
        )
        
        # Should not execute when condition is not met
        assert action.can_execute() is False
        
        # Should execute when condition is met
        condition_met = True
        assert action.can_execute() is True
    
    def test_recovery_manager_registration(self):
        """Test registering recovery actions."""
        manager = RecoveryManager()
        
        def recovery_action():
            return True
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            description="Test recovery",
            action=recovery_action,
            priority=5
        )
        
        manager.register_recovery_action(ErrorCategory.OPTIMIZATION, action)
        
        assert ErrorCategory.OPTIMIZATION in manager.recovery_actions
        assert len(manager.recovery_actions[ErrorCategory.OPTIMIZATION]) == 1
        assert manager.recovery_actions[ErrorCategory.OPTIMIZATION][0] == action
    
    def test_recovery_manager_error_handling(self):
        """Test error handling with recovery actions."""
        manager = RecoveryManager()
        recovery_executed = False
        
        def successful_recovery():
            nonlocal recovery_executed
            recovery_executed = True
            return True
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="Successful recovery",
            action=successful_recovery,
            priority=10
        )
        
        manager.register_recovery_action(ErrorCategory.NETWORK, action)
        
        # Test with platform error
        error = PlatformError(
            "Network error",
            ErrorCategory.NETWORK
        )
        
        result = manager.handle_error(error)
        
        assert result is True
        assert recovery_executed is True
        assert len(manager.recovery_history) == 1
        assert manager.recovery_history[0]["success"] is True
    
    def test_recovery_manager_priority_ordering(self):
        """Test that recovery actions are executed in priority order."""
        manager = RecoveryManager()
        execution_order = []
        
        def low_priority_action():
            execution_order.append("low")
            return False  # Fail to test next action
        
        def high_priority_action():
            execution_order.append("high")
            return True  # Success
        
        low_action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="Low priority",
            action=low_priority_action,
            priority=1
        )
        
        high_action = RecoveryAction(
            strategy=RecoveryStrategy.ROLLBACK,
            description="High priority",
            action=high_priority_action,
            priority=10
        )
        
        # Register in reverse priority order
        manager.register_recovery_action(ErrorCategory.SYSTEM, low_action)
        manager.register_recovery_action(ErrorCategory.SYSTEM, high_action)
        
        error = PlatformError("System error", ErrorCategory.SYSTEM)
        result = manager.handle_error(error)
        
        assert result is True
        assert execution_order == ["high"]  # High priority executed first and succeeded


class TestModelRecoveryManager:
    """Test model recovery manager functionality."""
    
    def test_model_snapshot_creation(self):
        """Test creating model snapshots."""
        manager = ModelRecoveryManager()
        
        # Create a simple model
        model = nn.Linear(10, 5)
        model_id = "test_model"
        
        snapshot_id = manager.create_model_snapshot(
            model_id, model, {"test": "metadata"}
        )
        
        assert snapshot_id.startswith(model_id)
        assert model_id in manager.model_snapshots
        assert len(manager.model_snapshots[model_id]) == 1
        
        snapshot = manager.model_snapshots[model_id][0]
        assert snapshot["model_id"] == model_id
        assert snapshot["metadata"]["test"] == "metadata"
        assert "state_dict" in snapshot
    
    def test_model_restoration(self):
        """Test restoring model from snapshot."""
        manager = ModelRecoveryManager()
        
        # Create original model
        original_model = nn.Linear(10, 5)
        model_id = "test_model"
        
        # Create snapshot
        snapshot_id = manager.create_model_snapshot(model_id, original_model)
        
        # Modify model
        modified_model = nn.Linear(10, 5)
        with torch.no_grad():
            modified_model.weight.fill_(999.0)
            modified_model.bias.fill_(999.0)
        
        # Restore from snapshot
        success = manager.restore_model_from_snapshot(modified_model, model_id)
        
        assert success is True
        
        # Check that model was restored
        assert torch.allclose(modified_model.weight, original_model.weight)
        assert torch.allclose(modified_model.bias, original_model.bias)
    
    def test_model_restoration_latest_snapshot(self):
        """Test restoring from latest snapshot when multiple exist."""
        manager = ModelRecoveryManager()
        
        model = nn.Linear(10, 5)
        model_id = "test_model"
        
        # Create multiple snapshots
        snapshot1 = manager.create_model_snapshot(model_id, model, {"version": 1})
        
        # Modify model
        with torch.no_grad():
            model.weight.fill_(1.0)
        
        snapshot2 = manager.create_model_snapshot(model_id, model, {"version": 2})
        
        # Modify model again
        with torch.no_grad():
            model.weight.fill_(2.0)
        
        # Restore from latest (should be version 2)
        success = manager.restore_model_from_snapshot(model, model_id)
        
        assert success is True
        assert torch.allclose(model.weight, torch.ones_like(model.weight))
    
    def test_snapshot_cleanup(self):
        """Test cleaning up old snapshots."""
        manager = ModelRecoveryManager()
        manager.max_snapshots_per_model = 3
        
        model = nn.Linear(10, 5)
        model_id = "test_model"
        
        # Create more snapshots than the limit
        for i in range(5):
            manager.create_model_snapshot(model_id, model, {"version": i})
        
        # Should only keep the last 3
        assert len(manager.model_snapshots[model_id]) == 3
        
        # Check that the latest snapshots are kept
        versions = [s["metadata"]["version"] for s in manager.model_snapshots[model_id]]
        assert versions == [2, 3, 4]
    
    def test_list_snapshots(self):
        """Test listing available snapshots."""
        manager = ModelRecoveryManager()
        
        model = nn.Linear(10, 5)
        model_id = "test_model"
        
        # Create snapshots
        snapshot1 = manager.create_model_snapshot(model_id, model, {"version": 1})
        snapshot2 = manager.create_model_snapshot(model_id, model, {"version": 2})
        
        snapshots = manager.list_snapshots(model_id)
        
        assert len(snapshots) == 2
        assert snapshots[0]["metadata"]["version"] == 1
        assert snapshots[1]["metadata"]["version"] == 2


class TestGracefulDegradationManager:
    """Test graceful degradation manager functionality."""
    
    def test_fallback_strategy_registration(self):
        """Test registering fallback strategies."""
        manager = GracefulDegradationManager()
        
        manager.register_fallback_strategy(
            "quantization_4bit",
            ["quantization_8bit", "pruning_structured"],
            priority=10
        )
        
        assert "quantization_4bit" in manager.fallback_strategies
        assert manager.fallback_strategies["quantization_4bit"] == ["quantization_8bit", "pruning_structured"]
        assert manager.technique_priorities["quantization_4bit"] == 10
    
    def test_get_fallback_techniques(self):
        """Test getting fallback techniques for failed optimization."""
        manager = GracefulDegradationManager()
        
        manager.register_fallback_strategy(
            "quantization_4bit",
            ["quantization_8bit", "pruning_structured", "pruning_unstructured"],
            priority=10
        )
        
        manager.register_fallback_strategy(
            "quantization_8bit",
            ["pruning_structured"],
            priority=8
        )
        
        available_techniques = ["quantization_8bit", "pruning_structured"]
        
        fallbacks = manager.get_fallback_techniques("quantization_4bit", available_techniques)
        
        assert fallbacks == ["quantization_8bit", "pruning_structured"]
    
    def test_create_degraded_plan(self):
        """Test creating degraded optimization plan."""
        manager = GracefulDegradationManager()
        
        # Set up fallback strategies
        manager.register_fallback_strategy(
            "quantization_4bit",
            ["quantization_8bit"],
            priority=10
        )
        
        manager.register_fallback_strategy(
            "pruning_advanced",
            ["pruning_basic"],
            priority=8
        )
        
        original_plan = ["quantization_4bit", "pruning_advanced", "distillation"]
        failed_techniques = ["quantization_4bit", "pruning_advanced"]
        available_techniques = ["quantization_8bit", "pruning_basic", "distillation"]
        
        degraded_plan = manager.create_degraded_plan(
            original_plan, failed_techniques, available_techniques
        )
        
        assert degraded_plan == ["quantization_8bit", "pruning_basic", "distillation"]


class TestIntegratedErrorHandling:
    """Test integrated error handling across components."""
    
    @patch('src.utils.recovery.recovery_manager')
    def test_optimization_agent_error_handling(self, mock_recovery_manager):
        """Test error handling in optimization agent."""
        from src.agents.base import BaseOptimizationAgent, OptimizedModel, ImpactEstimate, ValidationResult
        
        # Mock recovery manager to return success
        mock_recovery_manager.handle_error.return_value = True
        
        class TestOptimizationAgent(BaseOptimizationAgent):
            def initialize(self):
                return True
            
            def cleanup(self):
                pass
            
            def can_optimize(self, model):
                return True
            
            def estimate_impact(self, model):
                return ImpactEstimate(0.1, 0.2, 0.3, 0.8, 10)
            
            def optimize(self, model, config):
                # Simulate failure on first call, success on retry
                if not hasattr(self, '_retry_count'):
                    self._retry_count = 0
                
                self._retry_count += 1
                if self._retry_count == 1:
                    raise OptimizationError("Simulated failure", technique="test")
                
                return OptimizedModel(
                    model=model,
                    optimization_metadata={},
                    performance_metrics={},
                    optimization_time=1.0,
                    technique_used="test"
                )
            
            def validate_result(self, original, optimized):
                return ValidationResult(True, True, {}, [], [])
        
        agent = TestOptimizationAgent({})
        model = nn.Linear(10, 5)
        config = {"model_id": "test"}
        
        result = agent.optimize_with_tracking(model, config)
        
        # Should succeed after recovery and retry
        assert result.success is True
        assert mock_recovery_manager.handle_error.called
    
    def test_error_categorization(self):
        """Test automatic error categorization."""
        manager = RecoveryManager()
        
        # Test different error types
        file_error = FileNotFoundError("Model file not found")
        category = manager._categorize_error(file_error)
        assert category == ErrorCategory.STORAGE
        
        connection_error = ConnectionError("Network connection failed")
        category = manager._categorize_error(connection_error)
        assert category == ErrorCategory.NETWORK
        
        memory_error = MemoryError("Out of memory")
        category = manager._categorize_error(memory_error)
        assert category == ErrorCategory.SYSTEM


if __name__ == "__main__":
    pytest.main([__file__])