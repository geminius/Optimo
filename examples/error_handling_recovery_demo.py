"""
Demonstration of error handling and recovery mechanisms in the robotics model optimization platform.

This example shows how the platform handles various types of errors and recovers gracefully.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, Any
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.exceptions import (
    PlatformError, OptimizationError, ModelLoadingError, ValidationError,
    ErrorCategory, ErrorSeverity
)
from src.utils.retry import (
    RetryConfig, RetryableOperation, retry_with_backoff,
    QUICK_RETRY, STANDARD_RETRY, PERSISTENT_RETRY
)
from src.utils.recovery import (
    RecoveryManager, ModelRecoveryManager, GracefulDegradationManager,
    RecoveryStrategy, RecoveryAction,
    recovery_manager, model_recovery_manager, degradation_manager
)
from src.agents.base import BaseOptimizationAgent, OptimizedModel, ImpactEstimate, ValidationResult


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemoOptimizationAgent(BaseOptimizationAgent):
    """Demo optimization agent that can simulate various failure scenarios."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.failure_mode = config.get('failure_mode', 'none')
        self.failure_count = 0
    
    def initialize(self) -> bool:
        """Initialize the agent."""
        logger.info(f"Initializing {self.__class__.__name__}")
        return True
    
    def cleanup(self) -> None:
        """Clean up resources."""
        logger.info(f"Cleaning up {self.__class__.__name__}")
    
    def can_optimize(self, model: torch.nn.Module) -> bool:
        """Check if this agent can optimize the given model."""
        return isinstance(model, nn.Module)
    
    def estimate_impact(self, model: torch.nn.Module) -> ImpactEstimate:
        """Estimate the impact of optimization."""
        if self.failure_mode == 'analysis_failure':
            raise OptimizationError(
                "Analysis failed due to unsupported model architecture",
                technique=self.__class__.__name__,
                step="impact_estimation"
            )
        
        return ImpactEstimate(
            performance_improvement=0.2,
            size_reduction=0.3,
            speed_improvement=0.15,
            confidence=0.8,
            estimated_time_minutes=5
        )
    
    def optimize(self, model: torch.nn.Module, config: Dict[str, Any]) -> OptimizedModel:
        """Execute optimization with potential failures."""
        if self.failure_mode == 'transient_failure':
            self.failure_count += 1
            if self.failure_count <= 2:
                raise OptimizationError(
                    f"Transient optimization failure (attempt {self.failure_count})",
                    technique=self.__class__.__name__,
                    step="optimization_execution"
                )
        
        elif self.failure_mode == 'persistent_failure':
            raise OptimizationError(
                "Persistent optimization failure - technique not compatible",
                technique=self.__class__.__name__,
                step="optimization_execution",
                recoverable=False
            )
        
        elif self.failure_mode == 'resource_exhaustion':
            raise OptimizationError(
                "Insufficient memory for optimization",
                technique=self.__class__.__name__,
                step="optimization_execution",
                context={"required_memory_gb": 16, "available_memory_gb": 8}
            )
        
        # Simulate successful optimization
        optimized_model = model  # In reality, this would be the optimized version
        
        return OptimizedModel(
            model=optimized_model,
            optimization_metadata={
                "technique": self.__class__.__name__,
                "parameters_reduced": 0.3,
                "compression_ratio": 2.5
            },
            performance_metrics={
                "inference_time_ms": 45.2,
                "memory_usage_mb": 128.5,
                "accuracy": 0.94
            },
            optimization_time=3.2,
            technique_used=self.__class__.__name__
        )
    
    def validate_result(self, original: torch.nn.Module, optimized: torch.nn.Module) -> ValidationResult:
        """Validate the optimization result."""
        if self.failure_mode == 'validation_failure':
            return ValidationResult(
                is_valid=False,
                accuracy_preserved=False,
                performance_metrics={"accuracy_drop": 0.15},
                issues=["Accuracy degradation exceeds threshold"],
                recommendations=["Try different optimization parameters", "Use less aggressive compression"]
            )
        
        return ValidationResult(
            is_valid=True,
            accuracy_preserved=True,
            performance_metrics={
                "accuracy": 0.94,
                "inference_speedup": 1.8,
                "size_reduction": 0.3
            },
            issues=[],
            recommendations=[]
        )


def demonstrate_retry_mechanisms():
    """Demonstrate retry mechanisms with different configurations."""
    print("\n" + "="*60)
    print("DEMONSTRATING RETRY MECHANISMS")
    print("="*60)
    
    # Example 1: Simple retry with exponential backoff
    print("\n1. Simple retry with exponential backoff:")
    
    call_count = 0
    
    @retry_with_backoff(QUICK_RETRY)
    def flaky_network_call():
        nonlocal call_count
        call_count += 1
        print(f"  Attempt {call_count}: Making network call...")
        
        if call_count < 3:
            raise ConnectionError("Network temporarily unavailable")
        
        return "Success!"
    
    try:
        result = flaky_network_call()
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed after all retries: {e}")
    
    # Example 2: RetryableOperation with callbacks
    print("\n2. RetryableOperation with progress callbacks:")
    
    def on_retry(error, attempt):
        print(f"  Retry {attempt}: {error}")
    
    def on_success(attempt):
        print(f"  Success on attempt {attempt}")
    
    def on_failure(error, attempt):
        print(f"  Final failure after {attempt} attempts: {error}")
    
    retry_op = RetryableOperation(
        "database_connection",
        config=STANDARD_RETRY,
        on_retry=on_retry,
        on_success=on_success,
        on_failure=on_failure
    )
    
    call_count = 0
    def connect_to_database():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ConnectionError("Database connection timeout")
        return "Connected to database"
    
    try:
        result = retry_op.execute(connect_to_database)
        print(f"  Database result: {result}")
    except Exception as e:
        print(f"  Database connection failed: {e}")


def demonstrate_recovery_mechanisms():
    """Demonstrate recovery mechanisms for different error scenarios."""
    print("\n" + "="*60)
    print("DEMONSTRATING RECOVERY MECHANISMS")
    print("="*60)
    
    # Set up custom recovery manager for demo
    demo_recovery_manager = RecoveryManager()
    
    # Register recovery actions
    def memory_cleanup_action():
        print("  Executing memory cleanup...")
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Memory cleanup completed")
        return True
    
    def fallback_technique_action():
        print("  Switching to fallback optimization technique...")
        return True
    
    def restart_service_action():
        print("  Restarting optimization service...")
        time.sleep(0.1)  # Simulate restart time
        print("  Service restarted")
        return True
    
    # Register recovery actions with different priorities
    demo_recovery_manager.register_recovery_action(
        ErrorCategory.SYSTEM,
        RecoveryAction(
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            description="Free memory and reduce resource usage",
            action=memory_cleanup_action,
            priority=10
        )
    )
    
    demo_recovery_manager.register_recovery_action(
        ErrorCategory.OPTIMIZATION,
        RecoveryAction(
            strategy=RecoveryStrategy.GRACEFUL_DEGRADATION,
            description="Switch to fallback optimization technique",
            action=fallback_technique_action,
            priority=8
        )
    )
    
    demo_recovery_manager.register_recovery_action(
        ErrorCategory.SYSTEM,
        RecoveryAction(
            strategy=RecoveryStrategy.RETRY,
            description="Restart optimization service",
            action=restart_service_action,
            priority=5
        )
    )
    
    # Test recovery scenarios
    print("\n1. System error recovery:")
    system_error = PlatformError(
        "Out of memory during model optimization",
        ErrorCategory.SYSTEM,
        ErrorSeverity.HIGH,
        context={"memory_required_gb": 16, "memory_available_gb": 8}
    )
    
    recovery_success = demo_recovery_manager.handle_error(system_error)
    print(f"  Recovery successful: {recovery_success}")
    
    print("\n2. Optimization error recovery:")
    opt_error = OptimizationError(
        "Quantization technique failed",
        technique="quantization_4bit",
        session_id="demo_session",
        step="model_conversion"
    )
    
    recovery_success = demo_recovery_manager.handle_error(opt_error)
    print(f"  Recovery successful: {recovery_success}")
    
    # Show recovery history
    print("\n3. Recovery history:")
    history = demo_recovery_manager.get_recovery_history()
    for i, record in enumerate(history, 1):
        print(f"  Record {i}:")
        print(f"    Error: {record['error']}")
        print(f"    Category: {record['category']}")
        print(f"    Success: {record['success']}")
        print(f"    Actions attempted: {len(record['actions_attempted'])}")


def demonstrate_model_recovery():
    """Demonstrate model recovery and snapshot functionality."""
    print("\n" + "="*60)
    print("DEMONSTRATING MODEL RECOVERY")
    print("="*60)
    
    # Create a demo model
    original_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    model_id = "demo_model"
    
    print("\n1. Creating model snapshots:")
    
    # Create initial snapshot
    snapshot1_id = model_recovery_manager.create_model_snapshot(
        model_id, original_model, {"version": "1.0", "description": "Initial model"}
    )
    print(f"  Created snapshot: {snapshot1_id}")
    
    # Modify model (simulate optimization)
    with torch.no_grad():
        original_model[0].weight.fill_(1.0)
        original_model[2].weight.fill_(2.0)
    
    # Create second snapshot
    snapshot2_id = model_recovery_manager.create_model_snapshot(
        model_id, original_model, {"version": "1.1", "description": "After first optimization"}
    )
    print(f"  Created snapshot: {snapshot2_id}")
    
    # Simulate a failed optimization that corrupts the model
    print("\n2. Simulating model corruption:")
    with torch.no_grad():
        original_model[0].weight.fill_(float('nan'))
        original_model[2].weight.fill_(float('inf'))
    
    print("  Model corrupted with NaN and Inf values")
    
    # Restore from snapshot
    print("\n3. Restoring model from snapshot:")
    restore_success = model_recovery_manager.restore_model_from_snapshot(
        original_model, model_id
    )
    print(f"  Restoration successful: {restore_success}")
    
    # Verify restoration
    has_nan = torch.isnan(original_model[0].weight).any()
    has_inf = torch.isinf(original_model[2].weight).any()
    print(f"  Model has NaN values: {has_nan}")
    print(f"  Model has Inf values: {has_inf}")
    
    # List available snapshots
    print("\n4. Available snapshots:")
    snapshots = model_recovery_manager.list_snapshots(model_id)
    for snapshot in snapshots:
        print(f"  {snapshot['snapshot_id']}: {snapshot['metadata']['description']}")


def demonstrate_graceful_degradation():
    """Demonstrate graceful degradation when optimization techniques fail."""
    print("\n" + "="*60)
    print("DEMONSTRATING GRACEFUL DEGRADATION")
    print("="*60)
    
    # Set up fallback strategies
    demo_degradation_manager = GracefulDegradationManager()
    
    # Register fallback strategies
    demo_degradation_manager.register_fallback_strategy(
        "quantization_4bit",
        ["quantization_8bit", "pruning_structured"],
        priority=10
    )
    
    demo_degradation_manager.register_fallback_strategy(
        "quantization_8bit",
        ["pruning_structured", "pruning_unstructured"],
        priority=8
    )
    
    demo_degradation_manager.register_fallback_strategy(
        "pruning_structured",
        ["pruning_unstructured"],
        priority=6
    )
    
    print("\n1. Original optimization plan:")
    original_plan = ["quantization_4bit", "pruning_structured", "distillation"]
    print(f"  Techniques: {original_plan}")
    
    print("\n2. Simulating technique failures:")
    failed_techniques = ["quantization_4bit"]
    available_techniques = ["quantization_8bit", "pruning_structured", "pruning_unstructured", "distillation"]
    
    print(f"  Failed techniques: {failed_techniques}")
    print(f"  Available techniques: {available_techniques}")
    
    print("\n3. Creating degraded plan:")
    degraded_plan = demo_degradation_manager.create_degraded_plan(
        original_plan, failed_techniques, available_techniques
    )
    print(f"  Degraded plan: {degraded_plan}")
    
    # Simulate multiple failures
    print("\n4. Handling multiple failures:")
    failed_techniques = ["quantization_4bit", "pruning_structured"]
    
    degraded_plan = demo_degradation_manager.create_degraded_plan(
        original_plan, failed_techniques, available_techniques
    )
    print(f"  Failed techniques: {failed_techniques}")
    print(f"  Final degraded plan: {degraded_plan}")


def demonstrate_integrated_error_handling():
    """Demonstrate integrated error handling in optimization agents."""
    print("\n" + "="*60)
    print("DEMONSTRATING INTEGRATED ERROR HANDLING")
    print("="*60)
    
    # Create test model
    test_model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # Test different failure scenarios
    scenarios = [
        ("none", "Normal operation"),
        ("transient_failure", "Transient failure with retry"),
        ("validation_failure", "Validation failure with rollback"),
        ("analysis_failure", "Analysis failure with recovery")
    ]
    
    for failure_mode, description in scenarios:
        print(f"\n{description}:")
        print("-" * 40)
        
        # Create agent with specific failure mode
        agent = DemoOptimizationAgent({
            'failure_mode': failure_mode,
            'technique': 'demo_optimization'
        })
        
        config = {
            'model_id': 'demo_model',
            'optimization_params': {'compression_ratio': 2.0}
        }
        
        try:
            result = agent.optimize_with_tracking(test_model, config)
            
            print(f"  Optimization result: {'Success' if result.success else 'Failed'}")
            if result.success:
                print(f"  Technique used: {result.technique_used}")
                print(f"  Optimization time: {result.optimization_time:.2f}s")
            else:
                print(f"  Error message: {result.error_message}")
                print(f"  Snapshots available: {len(result.snapshots)}")
            
        except Exception as e:
            print(f"  Unhandled exception: {e}")
        
        finally:
            agent.cleanup()


def main():
    """Run all error handling and recovery demonstrations."""
    print("ROBOTICS MODEL OPTIMIZATION PLATFORM")
    print("Error Handling and Recovery Mechanisms Demo")
    print("=" * 80)
    
    try:
        demonstrate_retry_mechanisms()
        demonstrate_recovery_mechanisms()
        demonstrate_model_recovery()
        demonstrate_graceful_degradation()
        demonstrate_integrated_error_handling()
        
        print("\n" + "="*80)
        print("DEMO COMPLETED SUCCESSFULLY")
        print("All error handling and recovery mechanisms demonstrated.")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        raise


if __name__ == "__main__":
    main()