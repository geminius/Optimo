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
    
    def test_optimization_agent_error_handling(self):
        """Test error handling in optimization agent."""
        from src.agents.base import BaseOptimizationAgent, OptimizedModel, ImpactEstimate, ValidationResult
        
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
                # Simulate failure - the base class will handle it
                raise OptimizationError("Simulated failure", technique="test")
            
            def validate_result(self, original, optimized):
                return ValidationResult(True, True, {}, [], [])
        
        agent = TestOptimizationAgent({})
        model = nn.Linear(10, 5)
        config = {"model_id": "test"}
        
        result = agent.optimize_with_tracking(model, config)
        
        # Should fail but be handled gracefully
        assert result.success is False
        assert result.error_message is not None
        assert "Simulated failure" in result.error_message
    
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


class TestAdvancedRecoveryScenarios:
    """Test advanced recovery scenarios and edge cases."""
    
    def test_cascading_failures_with_recovery(self):
        """Test handling of cascading failures across multiple components."""
        manager = RecoveryManager()
        recovery_attempts = []
        
        def first_recovery():
            recovery_attempts.append("first")
            raise Exception("First recovery failed")
        
        def second_recovery():
            recovery_attempts.append("second")
            return True
        
        action1 = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="First recovery attempt",
            action=first_recovery,
            priority=10
        )
        
        action2 = RecoveryAction(
            strategy=RecoveryStrategy.ROLLBACK,
            description="Second recovery attempt",
            action=second_recovery,
            priority=5
        )
        
        manager.register_recovery_action(ErrorCategory.OPTIMIZATION, action1)
        manager.register_recovery_action(ErrorCategory.OPTIMIZATION, action2)
        
        error = OptimizationError("Test error", technique="quantization")
        result = manager.handle_error(error)
        
        assert result is True
        assert recovery_attempts == ["first", "second"]
    
    def test_recovery_with_partial_success(self):
        """Test recovery when some operations succeed and others fail."""
        manager = GracefulDegradationManager()
        
        # Register multiple fallback strategies
        manager.register_fallback_strategy(
            "quantization_4bit",
            ["quantization_8bit", "pruning_structured", "pruning_unstructured"],
            priority=10
        )
        
        manager.register_fallback_strategy(
            "distillation_advanced",
            ["distillation_basic", "compression"],
            priority=8
        )
        
        original_plan = ["quantization_4bit", "distillation_advanced", "architecture_search"]
        failed_techniques = ["quantization_4bit", "distillation_advanced"]
        available_techniques = ["quantization_8bit", "distillation_basic", "architecture_search"]
        
        degraded_plan = manager.create_degraded_plan(
            original_plan, failed_techniques, available_techniques
        )
        
        assert "quantization_8bit" in degraded_plan
        assert "distillation_basic" in degraded_plan
        assert "architecture_search" in degraded_plan
        assert len(degraded_plan) == 3
    
    def test_model_snapshot_with_large_model(self):
        """Test model snapshot creation and restoration with larger models."""
        manager = ModelRecoveryManager()
        
        # Create a larger model
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )
        
        model_id = "large_model"
        
        # Initialize with specific weights
        with torch.no_grad():
            for i, layer in enumerate(model):
                if isinstance(layer, nn.Linear):
                    layer.weight.fill_(float(i))
                    layer.bias.fill_(float(i))
        
        # Create snapshot
        snapshot_id = manager.create_model_snapshot(model_id, model, {"size": "large"})
        
        # Modify model
        with torch.no_grad():
            for layer in model:
                if isinstance(layer, nn.Linear):
                    layer.weight.fill_(999.0)
                    layer.bias.fill_(999.0)
        
        # Restore
        success = manager.restore_model_from_snapshot(model, model_id)
        
        assert success is True
        
        # Verify restoration
        with torch.no_grad():
            for i, layer in enumerate(model):
                if isinstance(layer, nn.Linear):
                    assert torch.allclose(layer.weight, torch.full_like(layer.weight, float(i)))
                    assert torch.allclose(layer.bias, torch.full_like(layer.bias, float(i)))
    
    def test_recovery_history_tracking(self):
        """Test that recovery history is properly tracked and maintained."""
        manager = RecoveryManager()
        
        def successful_recovery():
            return True
        
        def failed_recovery():
            return False
        
        success_action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="Successful recovery",
            action=successful_recovery,
            priority=10
        )
        
        fail_action = RecoveryAction(
            strategy=RecoveryStrategy.ROLLBACK,
            description="Failed recovery",
            action=failed_recovery,
            priority=5
        )
        
        manager.register_recovery_action(ErrorCategory.NETWORK, success_action)
        
        # Test successful recovery
        error1 = PlatformError("Network error 1", ErrorCategory.NETWORK)
        result1 = manager.handle_error(error1, {"attempt": 1})
        
        assert result1 is True
        assert len(manager.recovery_history) == 1
        assert manager.recovery_history[0]["success"] is True
        assert manager.recovery_history[0]["context"]["attempt"] == 1
        
        # Test failed recovery
        manager.recovery_actions[ErrorCategory.NETWORK] = [fail_action]
        error2 = PlatformError("Network error 2", ErrorCategory.NETWORK)
        result2 = manager.handle_error(error2, {"attempt": 2})
        
        assert result2 is False
        assert len(manager.recovery_history) == 2
        assert manager.recovery_history[1]["success"] is False
    
    def test_recovery_history_size_limit(self):
        """Test that recovery history respects size limits."""
        manager = RecoveryManager()
        manager.max_history_size = 10
        
        def dummy_recovery():
            return True
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="Dummy recovery",
            action=dummy_recovery,
            priority=1
        )
        
        manager.register_recovery_action(ErrorCategory.SYSTEM, action)
        
        # Generate more records than the limit
        for i in range(15):
            error = PlatformError(f"Error {i}", ErrorCategory.SYSTEM)
            manager.handle_error(error)
        
        # Should only keep the last 10
        assert len(manager.recovery_history) == 10
        assert "Error 5" in manager.recovery_history[0]["error"]
        assert "Error 14" in manager.recovery_history[-1]["error"]
    
    def test_conditional_recovery_actions(self):
        """Test recovery actions with complex conditions."""
        manager = RecoveryManager()
        
        state = {"resources_available": False, "retry_count": 0}
        
        def check_resources():
            return state["resources_available"]
        
        def check_retry_limit():
            return state["retry_count"] < 3
        
        def recovery_action():
            state["retry_count"] += 1
            return True
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="Conditional recovery",
            action=recovery_action,
            conditions=[check_resources, check_retry_limit],
            priority=10
        )
        
        manager.register_recovery_action(ErrorCategory.OPTIMIZATION, action)
        
        # Should not execute when resources unavailable
        error = OptimizationError("Test error", technique="test")
        result = manager.handle_error(error)
        
        assert result is False
        assert state["retry_count"] == 0
        
        # Should execute when resources available
        state["resources_available"] = True
        result = manager.handle_error(error)
        
        assert result is True
        assert state["retry_count"] == 1
    
    def test_snapshot_list_and_metadata(self):
        """Test listing snapshots and accessing metadata."""
        manager = ModelRecoveryManager()
        
        model = nn.Linear(10, 5)
        model_id = "test_model"
        
        # Create multiple snapshots with metadata
        snapshot1_id = manager.create_model_snapshot(model_id, model, {"version": 1, "type": "initial"})
        snapshot2_id = manager.create_model_snapshot(model_id, model, {"version": 2, "type": "optimized"})
        snapshot3_id = manager.create_model_snapshot(model_id, model, {"version": 3, "type": "final"})
        
        # List snapshots
        snapshots = manager.list_snapshots(model_id)
        
        assert len(snapshots) == 3
        assert snapshots[0]["metadata"]["version"] == 1
        assert snapshots[0]["metadata"]["type"] == "initial"
        assert snapshots[1]["metadata"]["version"] == 2
        assert snapshots[1]["metadata"]["type"] == "optimized"
        assert snapshots[2]["metadata"]["version"] == 3
        assert snapshots[2]["metadata"]["type"] == "final"
        
        # Verify snapshot IDs are present
        assert all("snapshot_id" in s for s in snapshots)
        assert all("timestamp" in s for s in snapshots)
    
    def test_snapshot_cleanup_with_custom_retention(self):
        """Test snapshot cleanup with custom retention policy."""
        manager = ModelRecoveryManager()
        
        model = nn.Linear(10, 5)
        model_id = "test_model"
        
        # Create 5 snapshots
        for i in range(5):
            manager.create_model_snapshot(model_id, model, {"version": i})
        
        assert len(manager.model_snapshots[model_id]) == 5
        
        # Cleanup keeping only 2 latest
        manager.cleanup_snapshots(model_id, keep_latest=2)
        
        assert len(manager.model_snapshots[model_id]) == 2
        assert manager.model_snapshots[model_id][0]["metadata"]["version"] == 3
        assert manager.model_snapshots[model_id][1]["metadata"]["version"] == 4


class TestErrorPropagation:
    """Test error propagation and context preservation."""
    
    def test_error_context_enrichment(self):
        """Test that error context is enriched as it propagates."""
        base_error = OptimizationError(
            "Base error",
            technique="quantization",
            session_id="sess_123"
        )
        
        assert base_error.context["technique"] == "quantization"
        assert base_error.context["session_id"] == "sess_123"
        
        # Simulate context enrichment
        base_error.context["step"] = "model_conversion"
        base_error.context["attempt"] = 2
        
        assert base_error.context["step"] == "model_conversion"
        assert base_error.context["attempt"] == 2
    
    def test_error_severity_escalation(self):
        """Test error severity escalation on repeated failures."""
        errors = []
        
        for i in range(3):
            severity = ErrorSeverity.LOW if i == 0 else (
                ErrorSeverity.MEDIUM if i == 1 else ErrorSeverity.HIGH
            )
            
            error = PlatformError(
                f"Error attempt {i}",
                ErrorCategory.NETWORK,
                severity=severity
            )
            errors.append(error)
        
        assert errors[0].severity == ErrorSeverity.LOW
        assert errors[1].severity == ErrorSeverity.MEDIUM
        assert errors[2].severity == ErrorSeverity.HIGH
    
    def test_error_serialization_with_complex_context(self):
        """Test error serialization with complex context data."""
        error = OptimizationError(
            "Complex error",
            technique="quantization",
            session_id="sess_123"
        )
        
        # Add complex context after initialization
        error.context["model_info"] = {
            "layers": 10,
            "parameters": 1000000
        }
        error.context["optimization_config"] = {
            "bits": 4,
            "symmetric": True
        }
        error.context["failed_layers"] = ["layer_1", "layer_5"]
        
        error_dict = error.to_dict()
        
        assert error_dict["context"]["technique"] == "quantization"
        assert error_dict["context"]["model_info"]["layers"] == 10
        assert error_dict["context"]["optimization_config"]["bits"] == 4
        assert "layer_1" in error_dict["context"]["failed_layers"]


class TestRetryEdgeCases:
    """Test edge cases in retry logic."""
    
    @patch('time.sleep')
    def test_retry_with_zero_delay(self, mock_sleep):
        """Test retry with zero delay configuration."""
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.0,
            max_delay=0.0
        )
        
        call_count = 0
        
        @retry_with_backoff(config)
        def quick_retry_function():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("Quick failure")
            return "success"
        
        result = quick_retry_function()
        
        assert result == "success"
        assert call_count == 2
        # Sleep should still be called but with 0 delay
        assert mock_sleep.call_count == 1
    
    @patch('time.sleep')
    def test_retry_with_max_delay_cap(self, mock_sleep):
        """Test that retry delay is capped at max_delay."""
        config = RetryConfig(
            max_attempts=5,
            base_delay=1.0,
            max_delay=5.0,
            exponential_base=10.0,  # Very aggressive exponential
            jitter=False
        )
        
        # Calculate delays for each attempt
        delays = [config.calculate_delay(i) for i in range(1, 6)]
        
        # All delays should be capped at max_delay
        assert all(delay <= 5.0 for delay in delays)
        assert delays[-1] == 5.0  # Last delay should hit the cap
    
    def test_retry_with_mixed_exception_types(self):
        """Test retry behavior with mixed exception types."""
        config = RetryConfig(
            max_attempts=3,
            retryable_exceptions=[ConnectionError, TimeoutError],
            non_retryable_exceptions=[ValueError]
        )
        
        call_count = 0
        
        @retry_with_backoff(config)
        def mixed_exception_function():
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise ConnectionError("Retryable")
            elif call_count == 2:
                raise ValueError("Non-retryable")
            
            return "success"
        
        with pytest.raises(ValueError):
            mixed_exception_function()
        
        # Should have stopped at ValueError
        assert call_count == 2
    
    @patch('time.sleep')
    def test_retryable_operation_callbacks(self, mock_sleep):
        """Test all callbacks in RetryableOperation."""
        retry_calls = []
        success_calls = []
        failure_calls = []
        
        def on_retry(error, attempt):
            retry_calls.append((str(error), attempt))
        
        def on_success(attempt):
            success_calls.append(attempt)
        
        def on_failure(error, attempt):
            failure_calls.append((str(error), attempt))
        
        # Test successful operation after retries
        call_count = 0
        def flaky_operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise TimeoutError(f"Timeout {call_count}")
            return "success"
        
        retry_op = RetryableOperation(
            "test_op",
            config=QUICK_RETRY,
            on_retry=on_retry,
            on_success=on_success,
            on_failure=on_failure
        )
        
        result = retry_op.execute(flaky_operation)
        
        assert result == "success"
        assert len(retry_calls) == 2
        assert len(success_calls) == 1
        assert len(failure_calls) == 0
        assert success_calls[0] == 3  # Succeeded on attempt 3
    
    @patch('time.sleep')
    def test_retryable_operation_ultimate_failure(self, mock_sleep):
        """Test RetryableOperation when all attempts fail."""
        failure_calls = []
        
        def on_failure(error, attempt):
            failure_calls.append((str(error), attempt))
        
        def always_failing():
            raise ConnectionError("Persistent failure")
        
        retry_op = RetryableOperation(
            "failing_op",
            config=QUICK_RETRY,
            on_failure=on_failure
        )
        
        with pytest.raises(ConnectionError):
            retry_op.execute(always_failing)
        
        assert len(failure_calls) == 1
        assert failure_calls[0][1] == 3  # Failed on final attempt


class TestRecoveryManagerIntegration:
    """Test recovery manager integration with other components."""
    
    def test_recovery_with_model_snapshots(self):
        """Test recovery using model snapshots."""
        recovery_mgr = RecoveryManager()
        model_mgr = ModelRecoveryManager()
        
        model = nn.Linear(10, 5)
        model_id = "test_model"
        
        # Create snapshot
        snapshot_id = model_mgr.create_model_snapshot(model_id, model)
        
        # Modify model
        with torch.no_grad():
            model.weight.fill_(999.0)
        
        # Register recovery action that uses snapshot
        def restore_from_snapshot():
            return model_mgr.restore_model_from_snapshot(model, model_id)
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.ROLLBACK,
            description="Restore from snapshot",
            action=restore_from_snapshot,
            priority=10
        )
        
        recovery_mgr.register_recovery_action(ErrorCategory.OPTIMIZATION, action)
        
        # Trigger recovery
        error = OptimizationError("Model corrupted", technique="test")
        result = recovery_mgr.handle_error(error)
        
        assert result is True
        # Model should be restored
        assert not torch.allclose(model.weight, torch.full_like(model.weight, 999.0))
    
    def test_graceful_degradation_with_empty_fallbacks(self):
        """Test graceful degradation when no fallbacks are available."""
        manager = GracefulDegradationManager()
        
        # Register strategy with fallbacks that aren't available
        manager.register_fallback_strategy(
            "advanced_technique",
            ["fallback_1", "fallback_2"],
            priority=10
        )
        
        original_plan = ["advanced_technique", "basic_technique"]
        failed_techniques = ["advanced_technique"]
        available_techniques = ["basic_technique"]  # Fallbacks not available
        
        degraded_plan = manager.create_degraded_plan(
            original_plan, failed_techniques, available_techniques
        )
        
        # Should only include basic_technique
        assert degraded_plan == ["basic_technique"]
        assert "advanced_technique" not in degraded_plan
    
    def test_recovery_manager_clear_history(self):
        """Test clearing recovery history."""
        manager = RecoveryManager()
        
        def dummy_recovery():
            return True
        
        action = RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="Dummy",
            action=dummy_recovery,
            priority=1
        )
        
        manager.register_recovery_action(ErrorCategory.SYSTEM, action)
        
        # Generate some history
        for i in range(5):
            error = PlatformError(f"Error {i}", ErrorCategory.SYSTEM)
            manager.handle_error(error)
        
        assert len(manager.recovery_history) == 5
        
        # Clear history
        manager.clear_history()
        
        assert len(manager.recovery_history) == 0
    
    def test_model_snapshot_nonexistent_model(self):
        """Test restoring from snapshot for nonexistent model."""
        manager = ModelRecoveryManager()
        
        model = nn.Linear(10, 5)
        success = manager.restore_model_from_snapshot(model, "nonexistent_model")
        
        assert success is False
    
    def test_model_snapshot_nonexistent_snapshot_id(self):
        """Test restoring from nonexistent snapshot ID."""
        manager = ModelRecoveryManager()
        
        model = nn.Linear(10, 5)
        model_id = "test_model"
        
        # Create one snapshot
        manager.create_model_snapshot(model_id, model)
        
        # Try to restore from nonexistent snapshot
        success = manager.restore_model_from_snapshot(
            model, model_id, "nonexistent_snapshot"
        )
        
        assert success is False


if __name__ == "__main__":
    pytest.main([__file__])