"""
Demo script for the Evaluation Agent.

This script demonstrates how to use the EvaluationAgent to evaluate models,
compare original vs optimized models, and validate performance against thresholds.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any

from src.agents.evaluation.agent import EvaluationAgent
from src.models.core import ValidationStatus


class DemoModel(nn.Module):
    """Demo model for evaluation testing."""
    
    def __init__(self, size="medium"):
        super().__init__()
        
        if size == "small":
            self.layers = nn.Sequential(
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 10)
            )
        elif size == "medium":
            self.layers = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        else:  # large
            self.layers = nn.Sequential(
                nn.Conv2d(3, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 256, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((8, 8)),
                nn.Flatten(),
                nn.Linear(256 * 8 * 8, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            )
    
    def forward(self, x):
        return self.layers(x)


def demo_basic_evaluation():
    """Demonstrate basic model evaluation."""
    print("=" * 60)
    print("DEMO: Basic Model Evaluation")
    print("=" * 60)
    
    # Create evaluation agent
    config = {
        "benchmark_samples": 20,
        "warmup_samples": 5,
        "accuracy_threshold": 0.95,
        "performance_threshold": 1.1
    }
    
    agent = EvaluationAgent(config)
    agent.initialize()
    
    # Create a demo model
    model = DemoModel("medium")
    print(f"Created demo model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Define benchmarks to run
    benchmarks = [
        {"name": "inference_speed", "type": "inference_speed", "input_shape": (1, 3, 224, 224)},
        {"name": "memory_usage", "type": "memory_usage", "input_shape": (1, 3, 224, 224)},
        {"name": "throughput", "type": "throughput", "input_shape": (1, 3, 224, 224)},
        {"name": "model_size", "type": "model_size"},
        {"name": "accuracy", "type": "accuracy", "input_shape": (1, 3, 224, 224)},
        {"name": "flops", "type": "flops", "input_shape": (1, 3, 224, 224)}
    ]
    
    print(f"Running {len(benchmarks)} benchmarks...")
    
    # Evaluate the model
    report = agent.evaluate_model(model, benchmarks)
    
    # Display results
    print(f"\nEvaluation completed in {report.evaluation_duration_seconds:.2f} seconds")
    print(f"Validation Status: {report.validation_status.value}")
    
    print("\nBenchmark Results:")
    print("-" * 50)
    for benchmark in report.benchmarks:
        direction = "↓" if not benchmark.higher_is_better else "↑"
        print(f"  {benchmark.benchmark_name:15s}: {benchmark.score:8.2f} {benchmark.unit:10s} {direction}")
        print(f"    Execution time: {benchmark.execution_time_seconds:.3f}s")
    
    print("\nPerformance Metrics Summary:")
    print("-" * 50)
    metrics = report.performance_metrics
    if metrics.inference_time_ms > 0:
        print(f"  Inference Time: {metrics.inference_time_ms:.2f} ms")
    if metrics.memory_usage_mb > 0:
        print(f"  Memory Usage: {metrics.memory_usage_mb:.2f} MB")
    if metrics.model_size_mb > 0:
        print(f"  Model Size: {metrics.model_size_mb:.2f} MB")
    if metrics.throughput_samples_per_sec > 0:
        print(f"  Throughput: {metrics.throughput_samples_per_sec:.2f} samples/sec")
    if metrics.accuracy is not None:
        print(f"  Accuracy: {metrics.accuracy:.4f}")
    
    print("\nRecommendations:")
    print("-" * 50)
    for i, rec in enumerate(report.recommendations, 1):
        print(f"  {i}. {rec}")
    
    agent.cleanup()
    return report


def demo_model_comparison():
    """Demonstrate model comparison between original and optimized versions."""
    print("\n" + "=" * 60)
    print("DEMO: Model Comparison")
    print("=" * 60)
    
    # Create evaluation agent
    config = {"benchmark_samples": 15, "warmup_samples": 3}
    agent = EvaluationAgent(config)
    agent.initialize()
    
    # Create original and "optimized" models
    original_model = DemoModel("large")
    optimized_model = DemoModel("medium")  # Smaller model simulates optimization
    
    print(f"Original model parameters: {sum(p.numel() for p in original_model.parameters()):,}")
    print(f"Optimized model parameters: {sum(p.numel() for p in optimized_model.parameters()):,}")
    
    print("\nComparing models...")
    
    # Compare the models
    comparison = agent.compare_models(original_model, optimized_model)
    
    # Display comparison results
    print("\nComparison Results:")
    print("-" * 50)
    
    print("Improvements:")
    if comparison.improvements:
        for metric, improvement in comparison.improvements.items():
            print(f"  {metric:25s}: +{improvement:6.2f}%")
    else:
        print("  No improvements detected")
    
    print("\nRegressions:")
    if comparison.regressions:
        for metric, regression in comparison.regressions.items():
            print(f"  {metric:25s}: -{regression:6.2f}%")
    else:
        print("  No regressions detected")
    
    print(f"\nOverall Score: {comparison.overall_score:.2f}")
    print(f"Recommendation: {comparison.recommendation}")
    
    agent.cleanup()
    return comparison


def demo_performance_validation():
    """Demonstrate performance validation against thresholds."""
    print("\n" + "=" * 60)
    print("DEMO: Performance Validation")
    print("=" * 60)
    
    # Create evaluation agent
    config = {"benchmark_samples": 10, "warmup_samples": 2}
    agent = EvaluationAgent(config)
    agent.initialize()
    
    # Create a model to validate
    model = DemoModel("medium")
    print(f"Validating model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Define performance thresholds
    thresholds = {
        "inference_time_ms": 100.0,      # Max 100ms inference time
        "memory_usage_mb": 200.0,        # Max 200MB memory usage
        "model_size_mb": 50.0,           # Max 50MB model size
        "accuracy": 0.8,                 # Min 80% accuracy
        "throughput_samples_per_sec": 5.0  # Min 5 samples/sec throughput
    }
    
    print("\nPerformance Thresholds:")
    print("-" * 50)
    for metric, threshold in thresholds.items():
        print(f"  {metric:25s}: {threshold}")
    
    print("\nValidating performance...")
    
    # Validate performance
    validation_result = agent.validate_performance(model, thresholds)
    
    # Display validation results
    print(f"\nValidation Result: {'PASSED' if validation_result['passed'] else 'FAILED'}")
    
    if validation_result["failures"]:
        print("\nFailures:")
        print("-" * 50)
        for failure in validation_result["failures"]:
            print(f"  ❌ {failure}")
    
    if validation_result["warnings"]:
        print("\nWarnings:")
        print("-" * 50)
        for warning in validation_result["warnings"]:
            print(f"  ⚠️  {warning}")
    
    print("\nActual Metrics:")
    print("-" * 50)
    for metric, value in validation_result["metrics"].items():
        if value is not None and value != 0:
            threshold = thresholds.get(metric, "N/A")
            status = "✅" if metric not in [f.split()[1] for f in validation_result["failures"]] else "❌"
            if isinstance(value, (int, float)):
                print(f"  {status} {metric:25s}: {value:8.4f} (threshold: {threshold})")
            else:
                print(f"  {status} {metric:25s}: {value} (threshold: {threshold})")
    
    agent.cleanup()
    return validation_result


def demo_custom_benchmarks():
    """Demonstrate using custom benchmarks."""
    print("\n" + "=" * 60)
    print("DEMO: Custom Benchmarks")
    print("=" * 60)
    
    # Create evaluation agent
    config = {"benchmark_samples": 10, "warmup_samples": 2}
    agent = EvaluationAgent(config)
    agent.initialize()
    
    # Create a model
    model = DemoModel("small")
    
    # Define custom benchmark configurations
    custom_benchmarks = [
        {
            "name": "speed_test",
            "type": "inference_speed",
            "input_shape": (1, 100)  # 1D input for small model
        },
        {
            "name": "memory_test", 
            "type": "memory_usage",
            "input_shape": (1, 100)
        },
        {
            "name": "size_analysis",
            "type": "model_size"
        },
        {
            "name": "throughput_analysis",
            "type": "throughput",
            "input_shape": (1, 100)
        }
    ]
    
    print(f"Running {len(custom_benchmarks)} custom benchmarks on small model...")
    
    # Run evaluation with custom benchmarks
    report = agent.evaluate_model(model, custom_benchmarks)
    
    # Display results
    print(f"\nCustom Evaluation Results:")
    print("-" * 50)
    
    for benchmark in report.benchmarks:
        print(f"\n{benchmark.benchmark_name}:")
        print(f"  Score: {benchmark.score:.4f} {benchmark.unit}")
        print(f"  Execution Time: {benchmark.execution_time_seconds:.3f}s")
        print(f"  Better Direction: {'Higher' if benchmark.higher_is_better else 'Lower'}")
        
        if benchmark.metadata:
            print("  Metadata:")
            for key, value in benchmark.metadata.items():
                print(f"    {key}: {value}")
    
    agent.cleanup()
    return report


def demo_error_handling():
    """Demonstrate error handling in evaluation."""
    print("\n" + "=" * 60)
    print("DEMO: Error Handling")
    print("=" * 60)
    
    # Create evaluation agent
    config = {"benchmark_samples": 5, "warmup_samples": 1}
    agent = EvaluationAgent(config)
    agent.initialize()
    
    # Create a model that will cause errors
    class ErrorModel(nn.Module):
        def forward(self, x):
            if x.shape[0] > 0:  # Always true, but creates an error condition
                raise RuntimeError("Simulated model error for demo")
            return x
    
    error_model = ErrorModel()
    
    # Try to evaluate the problematic model
    benchmarks = [
        {"name": "inference_speed", "type": "inference_speed"},
        {"name": "model_size", "type": "model_size"}  # This should still work
    ]
    
    print("Evaluating model that will cause errors...")
    
    report = agent.evaluate_model(error_model, benchmarks)
    
    print(f"\nEvaluation Status: {report.validation_status.value}")
    print(f"Successful Benchmarks: {len(report.benchmarks)}")
    print(f"Validation Errors: {len(report.validation_errors)}")
    
    if report.validation_errors:
        print("\nErrors Encountered:")
        print("-" * 50)
        for error in report.validation_errors:
            print(f"  ❌ {error}")
    
    if report.benchmarks:
        print("\nSuccessful Benchmarks:")
        print("-" * 50)
        for benchmark in report.benchmarks:
            print(f"  ✅ {benchmark.benchmark_name}: {benchmark.score:.4f} {benchmark.unit}")
    
    # Test invalid benchmark type
    print("\nTesting invalid benchmark type...")
    invalid_benchmarks = [{"name": "invalid", "type": "nonexistent_benchmark"}]
    
    invalid_report = agent.evaluate_model(DemoModel("small"), invalid_benchmarks)
    print(f"Invalid benchmark result: {invalid_report.validation_status.value}")
    print(f"Errors: {len(invalid_report.validation_errors)}")
    
    agent.cleanup()
    return report


def main():
    """Run all evaluation agent demos."""
    print("Evaluation Agent Demo")
    print("=" * 60)
    print("This demo showcases the capabilities of the EvaluationAgent")
    print("for comprehensive model evaluation, comparison, and validation.")
    
    try:
        # Run all demos
        basic_report = demo_basic_evaluation()
        comparison_result = demo_model_comparison()
        validation_result = demo_performance_validation()
        custom_report = demo_custom_benchmarks()
        error_report = demo_error_handling()
        
        print("\n" + "=" * 60)
        print("DEMO SUMMARY")
        print("=" * 60)
        print("✅ Basic Evaluation: Completed successfully")
        print("✅ Model Comparison: Completed successfully")
        print("✅ Performance Validation: Completed successfully")
        print("✅ Custom Benchmarks: Completed successfully")
        print("✅ Error Handling: Demonstrated graceful error handling")
        
        print(f"\nKey Insights:")
        print(f"- Basic evaluation identified {len(basic_report.recommendations)} recommendations")
        print(f"- Model comparison overall score: {comparison_result.overall_score:.2f}")
        print(f"- Performance validation: {'PASSED' if validation_result['passed'] else 'FAILED'}")
        print(f"- Custom benchmarks: {len(custom_report.benchmarks)} benchmarks executed")
        print(f"- Error handling: {len(error_report.validation_errors)} errors handled gracefully")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()