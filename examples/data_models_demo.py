#!/usr/bin/env python3
"""
Demonstration of the core data models and validation functionality.
"""

import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models import (
    ModelMetadata, ModelType, ModelFramework,
    OptimizationSession, OptimizationStatus, OptimizationStep,
    AnalysisReport, ArchitectureSummary, PerformanceProfile,
    OptimizationOpportunity, Recommendation,
    EvaluationReport, BenchmarkResult, PerformanceMetrics,
    ValidationStatus,
    validate_model_metadata, validate_optimization_session,
    validate_analysis_report, validate_evaluation_report,
    OptimizationParameterValidator
)


def demo_model_metadata():
    """Demonstrate ModelMetadata usage."""
    print("=== ModelMetadata Demo ===")
    
    # Create a model metadata instance
    metadata = ModelMetadata(
        name="OpenVLA-7B",
        version="1.2.0",
        model_type=ModelType.OPENVLA,
        framework=ModelFramework.PYTORCH,
        size_mb=13500.0,
        parameters=7000000000,
        tags=["robotics", "vision-language", "manipulation"],
        description="OpenVLA model for robotic manipulation tasks",
        author="OpenVLA Team"
    )
    
    print(f"Model: {metadata.name} v{metadata.version}")
    print(f"Type: {metadata.model_type.value}")
    print(f"Framework: {metadata.framework.value}")
    print(f"Size: {metadata.size_mb:.1f} MB")
    print(f"Parameters: {metadata.parameters:,}")
    print(f"Tags: {', '.join(metadata.tags)}")
    print(f"Created: {metadata.created_at}")
    
    # Validate the metadata
    is_valid, issues = validate_model_metadata(metadata)
    print(f"Validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    print()


def demo_optimization_session():
    """Demonstrate OptimizationSession usage."""
    print("=== OptimizationSession Demo ===")
    
    # Create optimization steps
    quantization_step = OptimizationStep(
        technique="quantization",
        status=OptimizationStatus.COMPLETED,
        parameters={
            "quantization_bits": 8,
            "quantization_type": "dynamic"
        },
        results={
            "size_reduction_percent": 75.0,
            "accuracy_retention": 0.98
        }
    )
    
    pruning_step = OptimizationStep(
        technique="pruning",
        status=OptimizationStatus.PENDING,
        parameters={
            "pruning_ratio": 0.5,
            "pruning_type": "structured"
        }
    )
    
    # Create optimization session
    session = OptimizationSession(
        model_id="model_123",
        criteria_name="edge_deployment",
        status=OptimizationStatus.RUNNING,
        steps=[quantization_step, pruning_step],
        priority=2,
        tags=["edge", "mobile"],
        notes="Optimizing for mobile deployment"
    )
    
    print(f"Session ID: {session.id}")
    print(f"Model ID: {session.model_id}")
    print(f"Status: {session.status.value}")
    print(f"Priority: {session.priority}")
    print(f"Steps: {len(session.steps)}")
    print(f"Active: {session.is_active}")
    
    for i, step in enumerate(session.steps, 1):
        print(f"  Step {i}: {step.technique} ({step.status.value})")
        if step.parameters:
            for param, value in step.parameters.items():
                print(f"    {param}: {value}")
    
    # Validate the session
    is_valid, issues = validate_optimization_session(session)
    print(f"Validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    print()


def demo_analysis_report():
    """Demonstrate AnalysisReport usage."""
    print("=== AnalysisReport Demo ===")
    
    # Create architecture summary
    arch_summary = ArchitectureSummary(
        total_layers=48,
        layer_types={"Linear": 32, "LayerNorm": 48, "Attention": 24},
        input_shape=[1, 224, 224, 3],
        output_shape=[1, 512],
        total_parameters=7000000000,
        trainable_parameters=7000000000,
        model_depth=48,
        computational_complexity=1.2e12,  # FLOPs
        memory_footprint_mb=13500.0
    )
    
    # Create performance profile
    perf_profile = PerformanceProfile(
        inference_time_ms=150.0,
        memory_usage_mb=8500.0,
        throughput_samples_per_sec=6.67,
        gpu_utilization_percent=85.0,
        cpu_utilization_percent=45.0,
        benchmark_scores={"accuracy": 0.92, "success_rate": 0.88}
    )
    
    # Create optimization opportunities
    opportunities = [
        OptimizationOpportunity(
            technique="quantization",
            estimated_size_reduction_percent=75.0,
            estimated_speed_improvement_percent=40.0,
            estimated_accuracy_impact_percent=-2.0,
            confidence_score=0.9,
            complexity="low",
            description="8-bit quantization with minimal accuracy loss"
        ),
        OptimizationOpportunity(
            technique="pruning",
            estimated_size_reduction_percent=50.0,
            estimated_speed_improvement_percent=30.0,
            estimated_accuracy_impact_percent=-5.0,
            confidence_score=0.7,
            complexity="medium",
            description="Structured pruning of attention layers"
        )
    ]
    
    # Create recommendations
    recommendations = [
        Recommendation(
            technique="quantization",
            priority=1,
            rationale="High impact, low risk optimization",
            expected_benefits=["Significant size reduction", "Faster inference"],
            potential_risks=["Minor accuracy loss"],
            estimated_effort="low"
        )
    ]
    
    # Create analysis report
    report = AnalysisReport(
        model_id="model_123",
        architecture_summary=arch_summary,
        performance_profile=perf_profile,
        optimization_opportunities=opportunities,
        compatibility_matrix={
            "quantization": True,
            "pruning": True,
            "distillation": False,
            "architecture_search": False
        },
        recommendations=recommendations,
        analysis_duration_seconds=45.5
    )
    
    print(f"Analysis ID: {report.analysis_id}")
    print(f"Model ID: {report.model_id}")
    print(f"Duration: {report.analysis_duration_seconds}s")
    print(f"Architecture: {arch_summary.total_layers} layers, {arch_summary.total_parameters:,} params")
    print(f"Performance: {perf_profile.inference_time_ms}ms inference, {perf_profile.throughput_samples_per_sec:.1f} samples/sec")
    print(f"Opportunities: {len(opportunities)}")
    
    for opp in opportunities:
        print(f"  - {opp.technique}: {opp.estimated_size_reduction_percent}% size reduction (confidence: {opp.confidence_score})")
    
    print(f"Recommendations: {len(recommendations)}")
    for rec in recommendations:
        print(f"  - {rec.technique} (priority {rec.priority}): {rec.rationale}")
    
    # Validate the report
    is_valid, issues = validate_analysis_report(report)
    print(f"Validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    print()


def demo_evaluation_report():
    """Demonstrate EvaluationReport usage."""
    print("=== EvaluationReport Demo ===")
    
    # Create benchmark results
    benchmarks = [
        BenchmarkResult(
            benchmark_name="RLBench Manipulation",
            score=0.88,
            unit="success_rate",
            higher_is_better=True,
            baseline_score=0.92,
            improvement_percent=-4.3,
            execution_time_seconds=120.0
        ),
        BenchmarkResult(
            benchmark_name="Inference Speed",
            score=150.0,
            unit="ms",
            higher_is_better=False,
            baseline_score=250.0,
            improvement_percent=40.0,
            execution_time_seconds=30.0
        )
    ]
    
    # Create performance metrics
    metrics = PerformanceMetrics(
        accuracy=0.88,
        inference_time_ms=150.0,
        memory_usage_mb=3500.0,  # After optimization
        model_size_mb=3375.0,    # After optimization
        throughput_samples_per_sec=6.67,
        flops=300000000000,      # Reduced FLOPs
        custom_metrics={"energy_efficiency": 0.75}
    )
    
    # Create evaluation report
    report = EvaluationReport(
        model_id="model_123_optimized",
        session_id="session_456",
        benchmarks=benchmarks,
        performance_metrics=metrics,
        validation_status=ValidationStatus.PASSED,
        recommendations=[
            "Model optimization successful",
            "Consider further pruning for edge deployment",
            "Monitor accuracy in production"
        ],
        evaluation_duration_seconds=180.0
    )
    
    print(f"Evaluation ID: {report.evaluation_id}")
    print(f"Model ID: {report.model_id}")
    print(f"Session ID: {report.session_id}")
    print(f"Duration: {report.evaluation_duration_seconds}s")
    print(f"Status: {report.validation_status.value}")
    print(f"Overall Success: {report.overall_success}")
    
    print(f"Benchmarks: {len(benchmarks)}")
    for bench in benchmarks:
        improvement = f"{bench.improvement_percent:+.1f}%" if bench.improvement_percent else "N/A"
        print(f"  - {bench.benchmark_name}: {bench.score} {bench.unit} ({improvement})")
    
    print(f"Performance Metrics:")
    print(f"  - Accuracy: {metrics.accuracy}")
    print(f"  - Inference Time: {metrics.inference_time_ms}ms")
    print(f"  - Memory Usage: {metrics.memory_usage_mb}MB")
    print(f"  - Model Size: {metrics.model_size_mb}MB")
    
    # Validate the report
    is_valid, issues = validate_evaluation_report(report)
    print(f"Validation: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    print()


def demo_parameter_validation():
    """Demonstrate parameter validation."""
    print("=== Parameter Validation Demo ===")
    
    # Test valid quantization parameters
    valid_params = {
        "quantization_bits": 8,
        "quantization_type": "dynamic"
    }
    
    is_valid, issues = OptimizationParameterValidator.validate_optimization_parameters(
        "quantization", valid_params
    )
    print(f"Valid quantization params: {'✓ Valid' if is_valid else '✗ Invalid'}")
    
    # Test invalid parameters
    invalid_params = {
        "quantization_bits": 64,  # Too high
        "pruning_ratio": 1.5      # Invalid ratio
    }
    
    is_valid, issues = OptimizationParameterValidator.validate_optimization_parameters(
        "quantization", invalid_params
    )
    print(f"Invalid quantization params: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    # Test distillation parameters
    distillation_params = {
        "temperature": 4.0,
        "alpha": 0.7,
        "beta": 0.3
    }
    
    is_valid, issues = OptimizationParameterValidator.validate_optimization_parameters(
        "distillation", distillation_params
    )
    print(f"Distillation params: {'✓ Valid' if is_valid else '✗ Invalid'}")
    if issues:
        for issue in issues:
            print(f"  - {issue}")
    
    print()


def main():
    """Run all demonstrations."""
    print("🤖 Robotics Model Optimization Platform - Data Models Demo\n")
    
    demo_model_metadata()
    demo_optimization_session()
    demo_analysis_report()
    demo_evaluation_report()
    demo_parameter_validation()
    
    print("✅ All demonstrations completed successfully!")


if __name__ == "__main__":
    main()