"""
Demo script for Planning Agent functionality.

This script demonstrates how to use the PlanningAgent to create optimization plans
based on analysis reports and configurable criteria.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.agents.planning.agent import PlanningAgent
from src.models.core import (
    AnalysisReport, ArchitectureSummary, PerformanceProfile,
    OptimizationOpportunity, Recommendation
)
from src.config.optimization_criteria import (
    OptimizationCriteria, OptimizationConstraints, PerformanceThreshold,
    OptimizationTechnique, PerformanceMetric
)


def create_sample_analysis_report():
    """Create a sample analysis report for demonstration."""
    print("📊 Creating sample analysis report...")
    
    # Architecture summary for a large robotics model
    architecture_summary = ArchitectureSummary(
        total_layers=120,
        layer_types={
            "Linear": 25,
            "Conv2d": 40,
            "BatchNorm2d": 30,
            "MultiheadAttention": 12,
            "LayerNorm": 13
        },
        total_parameters=50_000_000,  # 50M parameters
        trainable_parameters=48_000_000,
        model_depth=24,
        memory_footprint_mb=200.0
    )
    
    # Performance profile showing current performance
    performance_profile = PerformanceProfile(
        inference_time_ms=250.0,
        memory_usage_mb=1024.0,
        throughput_samples_per_sec=4.0,
        gpu_utilization_percent=85.0,
        cpu_utilization_percent=60.0
    )
    
    # Optimization opportunities identified by analysis
    opportunities = [
        OptimizationOpportunity(
            technique="quantization",
            estimated_size_reduction_percent=50.0,
            estimated_speed_improvement_percent=35.0,
            estimated_accuracy_impact_percent=-1.5,
            confidence_score=0.85,
            complexity="medium",
            description="Model has many quantizable layers (Linear, Conv2d)"
        ),
        OptimizationOpportunity(
            technique="pruning",
            estimated_size_reduction_percent=40.0,
            estimated_speed_improvement_percent=25.0,
            estimated_accuracy_impact_percent=-3.0,
            confidence_score=0.75,
            complexity="medium",
            description="Model shows high parameter redundancy in attention layers"
        ),
        OptimizationOpportunity(
            technique="distillation",
            estimated_size_reduction_percent=75.0,
            estimated_speed_improvement_percent=70.0,
            estimated_accuracy_impact_percent=-8.0,
            confidence_score=0.65,
            complexity="high",
            description="Large model suitable for knowledge distillation to smaller architecture"
        ),
        OptimizationOpportunity(
            technique="compression",
            estimated_size_reduction_percent=25.0,
            estimated_speed_improvement_percent=15.0,
            estimated_accuracy_impact_percent=-2.0,
            confidence_score=0.70,
            complexity="low",
            description="Linear layers can benefit from tensor decomposition"
        )
    ]
    
    # Compatibility assessment
    compatibility_matrix = {
        "quantization": True,
        "pruning": True,
        "distillation": True,
        "compression": True,
        "architecture_search": False  # Too complex for this demo
    }
    
    # Recommendations from analysis
    recommendations = [
        Recommendation(
            technique="quantization",
            priority=1,
            rationale="High impact with low accuracy loss and good confidence",
            expected_benefits=["50% size reduction", "35% speed improvement"],
            potential_risks=["Minor accuracy degradation"],
            estimated_effort="medium"
        ),
        Recommendation(
            technique="pruning",
            priority=2,
            rationale="Good complementary technique to quantization",
            expected_benefits=["40% size reduction", "25% speed improvement"],
            potential_risks=["Moderate accuracy impact"],
            estimated_effort="medium"
        )
    ]
    
    return AnalysisReport(
        model_id="openvla_robotics_model_v2",
        architecture_summary=architecture_summary,
        performance_profile=performance_profile,
        optimization_opportunities=opportunities,
        compatibility_matrix=compatibility_matrix,
        recommendations=recommendations,
        analysis_duration_seconds=45.2
    )


def create_optimization_criteria():
    """Create different optimization criteria for demonstration."""
    criteria_configs = {}
    
    # 1. Edge deployment criteria - aggressive optimization
    edge_constraints = OptimizationConstraints(
        max_optimization_time_minutes=180,
        preserve_accuracy_threshold=0.92,  # Allow more accuracy loss for edge
        allowed_techniques=[
            OptimizationTechnique.QUANTIZATION,
            OptimizationTechnique.PRUNING,
            OptimizationTechnique.COMPRESSION
        ],
        max_memory_usage_gb=8.0
    )
    
    edge_priority_weights = {
        PerformanceMetric.MODEL_SIZE: 0.5,      # Size is critical for edge
        PerformanceMetric.INFERENCE_TIME: 0.3,
        PerformanceMetric.MEMORY_USAGE: 0.2
    }
    
    criteria_configs["edge"] = OptimizationCriteria(
        name="edge_deployment",
        description="Aggressive optimization for edge device deployment",
        constraints=edge_constraints,
        priority_weights=edge_priority_weights,
        target_deployment="edge"
    )
    
    # 2. Cloud deployment criteria - balanced approach
    cloud_constraints = OptimizationConstraints(
        max_optimization_time_minutes=120,
        preserve_accuracy_threshold=0.97,  # Higher accuracy preservation
        allowed_techniques=[
            OptimizationTechnique.QUANTIZATION,
            OptimizationTechnique.PRUNING,
            OptimizationTechnique.DISTILLATION
        ]
    )
    
    cloud_priority_weights = {
        PerformanceMetric.INFERENCE_TIME: 0.4,
        PerformanceMetric.ACCURACY: 0.3,
        PerformanceMetric.MODEL_SIZE: 0.3
    }
    
    criteria_configs["cloud"] = OptimizationCriteria(
        name="cloud_deployment",
        description="Balanced optimization for cloud deployment",
        constraints=cloud_constraints,
        priority_weights=cloud_priority_weights,
        target_deployment="cloud"
    )
    
    # 3. Research criteria - conservative approach
    research_constraints = OptimizationConstraints(
        max_optimization_time_minutes=60,
        preserve_accuracy_threshold=0.99,  # Minimal accuracy loss
        allowed_techniques=[OptimizationTechnique.QUANTIZATION]  # Only safe techniques
    )
    
    research_priority_weights = {
        PerformanceMetric.ACCURACY: 0.6,
        PerformanceMetric.INFERENCE_TIME: 0.25,
        PerformanceMetric.MODEL_SIZE: 0.15
    }
    
    criteria_configs["research"] = OptimizationCriteria(
        name="research_deployment",
        description="Conservative optimization preserving model accuracy",
        constraints=research_constraints,
        priority_weights=research_priority_weights,
        target_deployment="general"
    )
    
    return criteria_configs


def demonstrate_technique_prioritization(agent, analysis_report):
    """Demonstrate technique prioritization functionality."""
    print("\n🎯 Demonstrating technique prioritization...")
    
    opportunities = analysis_report.optimization_opportunities
    prioritized = agent.prioritize_techniques(opportunities)
    
    print(f"\nFound {len(prioritized)} optimization techniques:")
    print("-" * 80)
    
    for i, tech in enumerate(prioritized, 1):
        print(f"{i}. {tech['technique'].upper()}")
        print(f"   Overall Score: {tech['overall_score']:.3f}")
        print(f"   Impact Score: {tech['impact_score']:.3f}")
        print(f"   Feasibility Score: {tech['feasibility_score']:.3f}")
        print(f"   Risk Score: {tech['risk_score']:.3f}")
        print(f"   Expected Benefits:")
        print(f"     - Size Reduction: {tech['estimated_size_reduction']:.1f}%")
        print(f"     - Speed Improvement: {tech['estimated_speed_improvement']:.1f}%")
        print(f"     - Accuracy Impact: {tech['estimated_accuracy_impact']:.1f}%")
        print(f"   Confidence: {tech['confidence']:.1%}")
        print(f"   Complexity: {tech['complexity']}")
        print()


def demonstrate_plan_creation(agent, analysis_report, criteria, criteria_name):
    """Demonstrate optimization plan creation."""
    print(f"\n📋 Creating optimization plan for {criteria_name} deployment...")
    
    plan = agent.plan_optimization(analysis_report, criteria)
    
    print(f"\nOptimization Plan: {plan.plan_id}")
    print(f"Model ID: {plan.model_id}")
    print(f"Criteria: {plan.criteria_name}")
    print(f"Total Steps: {len(plan.steps)}")
    print(f"Estimated Duration: {plan.total_estimated_duration_minutes} minutes")
    print(f"Created: {plan.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nExpected Improvements:")
    for metric, value in plan.expected_improvements.items():
        print(f"  - {metric.replace('_', ' ').title()}: {value:.1f}%")
    
    print(f"\nRisk Assessment:")
    print(f"  {plan.risk_assessment}")
    
    print(f"\nOptimization Steps:")
    print("-" * 60)
    
    for i, step in enumerate(plan.steps, 1):
        print(f"{i}. {step.technique.upper()} (Priority {step.priority})")
        print(f"   Duration: {step.estimated_duration_minutes} minutes")
        print(f"   Risk Level: {step.risk_level}")
        
        if step.prerequisites:
            print(f"   Prerequisites: {', '.join(step.prerequisites)}")
        
        print(f"   Expected Impact:")
        for metric, value in step.expected_impact.items():
            print(f"     - {metric.replace('_', ' ').title()}: {value:.1f}%")
        
        print(f"   Parameters:")
        for param, value in step.parameters.items():
            print(f"     - {param}: {value}")
        
        print(f"   Rollback Plan: {step.rollback_plan}")
        print()
    
    print(f"Validation Strategy:")
    print(f"  {plan.validation_strategy}")
    
    print(f"\nRollback Strategy:")
    print(f"  {plan.rollback_strategy}")
    
    return plan


def demonstrate_plan_validation(agent, plan):
    """Demonstrate plan validation functionality."""
    print(f"\n✅ Validating optimization plan...")
    
    validation_result = agent.validate_plan(plan)
    
    print(f"Validation Result: {'VALID' if validation_result.is_valid else 'INVALID'}")
    
    if validation_result.issues:
        print(f"\nIssues Found:")
        for issue in validation_result.issues:
            print(f"  ❌ {issue}")
    
    if validation_result.warnings:
        print(f"\nWarnings:")
        for warning in validation_result.warnings:
            print(f"  ⚠️  {warning}")
    
    if validation_result.recommendations:
        print(f"\nRecommendations:")
        for rec in validation_result.recommendations:
            print(f"  💡 {rec}")
    
    if validation_result.is_valid and not validation_result.warnings:
        print("  ✅ Plan is valid and ready for execution!")


def compare_deployment_strategies(agent, analysis_report, criteria_configs):
    """Compare optimization strategies for different deployment targets."""
    print("\n🔄 Comparing optimization strategies across deployment targets...")
    print("=" * 80)
    
    plans = {}
    
    for deployment_type, criteria in criteria_configs.items():
        print(f"\n{deployment_type.upper()} DEPLOYMENT STRATEGY")
        print("-" * 40)
        
        plan = agent.plan_optimization(analysis_report, criteria)
        plans[deployment_type] = plan
        
        print(f"Steps: {len(plan.steps)}")
        print(f"Duration: {plan.total_estimated_duration_minutes} min")
        print(f"Techniques: {', '.join([step.technique for step in plan.steps])}")
        
        improvements = plan.expected_improvements
        print(f"Expected Size Reduction: {improvements.get('size_reduction_percent', 0):.1f}%")
        print(f"Expected Speed Improvement: {improvements.get('speed_improvement_percent', 0):.1f}%")
        print(f"Expected Accuracy Impact: {improvements.get('accuracy_impact_percent', 0):.1f}%")
    
    # Summary comparison
    print(f"\n📊 STRATEGY COMPARISON SUMMARY")
    print("-" * 40)
    
    for deployment_type, plan in plans.items():
        improvements = plan.expected_improvements
        size_reduction = improvements.get('size_reduction_percent', 0)
        speed_improvement = improvements.get('speed_improvement_percent', 0)
        accuracy_impact = abs(improvements.get('accuracy_impact_percent', 0))
        
        print(f"{deployment_type.capitalize():12} | "
              f"Size: {size_reduction:5.1f}% | "
              f"Speed: {speed_improvement:5.1f}% | "
              f"Acc Loss: {accuracy_impact:4.1f}% | "
              f"Steps: {len(plan.steps)}")


def main():
    """Main demonstration function."""
    print("🤖 Planning Agent Demonstration")
    print("=" * 50)
    
    # Create planning agent with balanced configuration
    config = {
        "max_plan_steps": 4,
        "risk_tolerance": 0.7,
        "performance_weight": 0.4,
        "feasibility_weight": 0.3,
        "risk_weight": 0.2,
        "cost_weight": 0.1,
        "min_impact_threshold": 0.1,
        "max_accuracy_loss": 0.05
    }
    
    agent = PlanningAgent(config)
    
    # Initialize agent
    if not agent.initialize():
        print("❌ Failed to initialize Planning Agent")
        return
    
    print("✅ Planning Agent initialized successfully")
    
    try:
        # Create sample data
        analysis_report = create_sample_analysis_report()
        criteria_configs = create_optimization_criteria()
        
        print(f"\n📈 Analysis Report Summary:")
        print(f"Model: {analysis_report.model_id}")
        print(f"Parameters: {analysis_report.architecture_summary.total_parameters:,}")
        print(f"Current Inference Time: {analysis_report.performance_profile.inference_time_ms:.1f}ms")
        print(f"Current Memory Usage: {analysis_report.performance_profile.memory_usage_mb:.1f}MB")
        print(f"Optimization Opportunities: {len(analysis_report.optimization_opportunities)}")
        
        # Demonstrate technique prioritization
        demonstrate_technique_prioritization(agent, analysis_report)
        
        # Demonstrate plan creation for different deployment scenarios
        for deployment_type, criteria in criteria_configs.items():
            plan = demonstrate_plan_creation(agent, analysis_report, criteria, deployment_type)
            demonstrate_plan_validation(agent, plan)
            print("\n" + "="*80)
        
        # Compare strategies
        compare_deployment_strategies(agent, analysis_report, criteria_configs)
        
        print(f"\n🎉 Planning Agent demonstration completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        agent.cleanup()
        print("🧹 Planning Agent cleanup completed")


if __name__ == "__main__":
    main()