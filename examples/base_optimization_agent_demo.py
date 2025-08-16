"""
Demo script showing how to use the BaseOptimizationAgent framework.
"""

import torch
import torch.nn as nn
from typing import Dict, Any
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.base import (
    BaseOptimizationAgent,
    OptimizedModel,
    ImpactEstimate,
    ValidationResult,
    ProgressUpdate,
    OptimizationStatus
)


class DemoOptimizationAgent(BaseOptimizationAgent):
    """
    Demo optimization agent that simulates a simple optimization process.
    This agent demonstrates all the features of the base framework.
    """
    
    def initialize(self) -> bool:
        """Initialize the demo agent."""
        self.logger.info("Demo optimization agent initialized")
        return True
    
    def cleanup(self) -> None:
        """Clean up demo agent resources."""
        self.logger.info("Demo optimization agent cleaned up")
    
    def can_optimize(self, model: torch.nn.Module) -> bool:
        """Check if we can optimize the model (always true for demo)."""
        return isinstance(model, nn.Module)
    
    def estimate_impact(self, model: torch.nn.Module) -> ImpactEstimate:
        """Estimate optimization impact based on model size."""
        param_count = sum(p.numel() for p in model.parameters())
        
        # Simulate impact estimation based on model complexity
        if param_count < 1000:
            performance_gain = 0.1
            size_reduction = 0.05
            speed_improvement = 0.15
            confidence = 0.9
        elif param_count < 10000:
            performance_gain = 0.2
            size_reduction = 0.15
            speed_improvement = 0.25
            confidence = 0.8
        else:
            performance_gain = 0.3
            size_reduction = 0.25
            speed_improvement = 0.4
            confidence = 0.7
        
        return ImpactEstimate(
            performance_improvement=performance_gain,
            size_reduction=size_reduction,
            speed_improvement=speed_improvement,
            confidence=confidence,
            estimated_time_minutes=max(1, param_count // 1000)
        )
    
    def optimize(self, model: torch.nn.Module, config: Dict[str, Any]) -> OptimizedModel:
        """
        Simulate optimization process with progress updates and cancellation checks.
        """
        self.logger.info(f"Starting optimization with config: {config}")
        
        # Simulate multi-step optimization process
        steps = [
            ("Analyzing model structure", 10),
            ("Preparing optimization", 20),
            ("Applying transformations", 60),
            ("Finalizing optimization", 90)
        ]
        
        for step_name, progress in steps:
            # Check for cancellation at each step
            if self.is_cancelled():
                self.logger.info("Optimization cancelled during step: " + step_name)
                raise RuntimeError("Optimization was cancelled")
            
            # Update progress
            self._update_progress(
                OptimizationStatus.OPTIMIZING,
                progress,
                step_name,
                estimated_remaining=max(0, (100 - progress) // 20),
                message=f"Processing {step_name.lower()}"
            )
            
            # Simulate work
            time.sleep(0.1)
        
        # Create optimized model (for demo, create a copy and modify slightly)
        import copy
        optimized_model = copy.deepcopy(model)
        
        # Simulate optimization by slightly modifying all weights
        with torch.no_grad():
            for param in optimized_model.parameters():
                param.data *= 0.95  # Simulate 5% reduction
        
        return OptimizedModel(
            model=optimized_model,
            optimization_metadata={
                "technique": "demo_optimization",
                "config": config,
                "original_params": sum(p.numel() for p in model.parameters()),
                "optimized_params": sum(p.numel() for p in optimized_model.parameters())
            },
            performance_metrics={
                "accuracy": 0.95,
                "inference_speed": 1.25,
                "memory_usage": 0.85
            },
            optimization_time=0.4,  # Simulated time
            technique_used="demo_optimization"
        )
    
    def validate_result(self, original: torch.nn.Module, optimized: torch.nn.Module) -> ValidationResult:
        """Validate the optimization result."""
        self.logger.info("Validating optimization result")
        
        # Simulate validation checks
        original_params = sum(p.numel() for p in original.parameters())
        optimized_params = sum(p.numel() for p in optimized.parameters())
        
        # Check if parameter count is preserved (for this demo)
        params_preserved = original_params == optimized_params
        
        # Simulate accuracy check
        accuracy_preserved = True  # In real implementation, would run inference tests
        
        issues = []
        recommendations = []
        
        if not params_preserved:
            issues.append("Parameter count changed during optimization")
            recommendations.append("Consider using parameter-preserving optimization")
        
        if accuracy_preserved:
            recommendations.append("Optimization successful - consider deploying")
        
        return ValidationResult(
            is_valid=params_preserved and accuracy_preserved,
            accuracy_preserved=accuracy_preserved,
            performance_metrics={
                "accuracy": 0.95 if accuracy_preserved else 0.85,
                "parameter_preservation": 1.0 if params_preserved else 0.0
            },
            issues=issues,
            recommendations=recommendations
        )
    
    def get_supported_techniques(self) -> list[str]:
        """Return list of supported optimization techniques."""
        return ["demo_optimization", "weight_scaling", "parameter_adjustment"]


def progress_callback(update: ProgressUpdate):
    """Callback function to handle progress updates."""
    print(f"[{update.timestamp.strftime('%H:%M:%S')}] "
          f"{update.status.value}: {update.progress_percentage:.1f}% - {update.current_step}")
    if update.message:
        print(f"  Message: {update.message}")
    if update.estimated_remaining_minutes:
        print(f"  Estimated remaining: {update.estimated_remaining_minutes} minutes")


def main():
    """Demonstrate the BaseOptimizationAgent framework."""
    print("=== BaseOptimizationAgent Framework Demo ===\n")
    
    # Create a simple model for testing
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    print(f"Original model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create demo optimization agent
    config = {
        "optimization_level": "moderate",
        "preserve_accuracy": True,
        "target_speedup": 1.2
    }
    
    agent = DemoOptimizationAgent(config)
    agent.initialize()
    
    # Add progress callback
    agent.add_progress_callback(progress_callback)
    
    print("\n1. Testing successful optimization:")
    print("-" * 40)
    
    # Run optimization
    result = agent.optimize_with_tracking(model, config, operation_id="demo_001")
    
    print(f"\nOptimization Result:")
    print(f"  Success: {result.success}")
    print(f"  Technique: {result.technique_used}")
    print(f"  Time: {result.optimization_time:.2f}s")
    print(f"  Performance metrics: {result.performance_metrics}")
    print(f"  Snapshots created: {len(result.snapshots)}")
    
    if result.validation_result:
        print(f"  Validation: {'PASSED' if result.validation_result.is_valid else 'FAILED'}")
        if result.validation_result.recommendations:
            print(f"  Recommendations: {result.validation_result.recommendations}")
    
    print(f"\n2. Testing optimization cancellation:")
    print("-" * 40)
    
    # Test cancellation
    agent2 = DemoOptimizationAgent(config)
    agent2.initialize()
    agent2.add_progress_callback(progress_callback)
    
    # Cancel before starting
    agent2.cancel_optimization()
    result2 = agent2.optimize_with_tracking(model, config, operation_id="demo_002")
    
    print(f"\nCancellation Result:")
    print(f"  Success: {result2.success}")
    print(f"  Error: {result2.error_message}")
    
    print(f"\n3. Testing snapshot and rollback functionality:")
    print("-" * 40)
    
    # Demonstrate snapshot functionality
    agent3 = DemoOptimizationAgent(config)
    agent3.initialize()
    
    # Create snapshots manually
    snapshot1 = agent3._create_snapshot(model, "before_modification")
    
    # Modify model
    with torch.no_grad():
        model[0].weight.fill_(999.0)
    
    print(f"Model modified - first weight value: {model[0].weight[0, 0].item()}")
    
    # Rollback
    success = agent3._rollback_to_snapshot(model, snapshot1)
    print(f"Rollback successful: {success}")
    print(f"After rollback - first weight value: {model[0].weight[0, 0].item()}")
    
    # Cleanup
    agent.cleanup()
    agent2.cleanup()
    agent3.cleanup()
    
    print("\n=== Demo completed successfully! ===")


if __name__ == "__main__":
    main()