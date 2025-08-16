"""
Unit tests for Planning Agent.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from typing import Dict, List, Any

from src.agents.planning.agent import (
    PlanningAgent, OptimizationPlan, OptimizationStep, 
    TechniqueScore, ValidationResult
)
from src.models.core import (
    AnalysisReport, ArchitectureSummary, PerformanceProfile,
    OptimizationOpportunity, Recommendation
)
from src.config.optimization_criteria import (
    OptimizationCriteria, OptimizationConstraints, PerformanceThreshold,
    OptimizationTechnique, PerformanceMetric
)


class TestPlanningAgent:
    """Test cases for PlanningAgent class."""
    
    @pytest.fixture
    def planning_agent(self):
        """Create a PlanningAgent instance for testing."""
        config = {
            "max_plan_steps": 3,
            "risk_tolerance": 0.7,
            "performance_weight": 0.4,
            "feasibility_weight": 0.3,
            "risk_weight": 0.2,
            "cost_weight": 0.1,
            "min_impact_threshold": 0.1,
            "max_accuracy_loss": 0.05
        }
        return PlanningAgent(config)
    
    @pytest.fixture
    def sample_analysis_report(self):
        """Create a sample analysis report for testing."""
        architecture_summary = ArchitectureSummary(
            total_layers=50,
            layer_types={"Linear": 10, "Conv2d": 20, "BatchNorm2d": 15},
            total_parameters=10_000_000,
            trainable_parameters=9_500_000,
            model_depth=12,
            memory_footprint_mb=40.0
        )
        
        performance_profile = PerformanceProfile(
            inference_time_ms=150.0,
            memory_usage_mb=512.0,
            throughput_samples_per_sec=6.67,
            gpu_utilization_percent=75.0,
            cpu_utilization_percent=45.0
        )
        
        opportunities = [
            OptimizationOpportunity(
                technique="quantization",
                estimated_size_reduction_percent=50.0,
                estimated_speed_improvement_percent=30.0,
                estimated_accuracy_impact_percent=-2.0,
                confidence_score=0.8,
                complexity="medium",
                description="Model suitable for quantization"
            ),
            OptimizationOpportunity(
                technique="pruning",
                estimated_size_reduction_percent=30.0,
                estimated_speed_improvement_percent=20.0,
                estimated_accuracy_impact_percent=-3.0,
                confidence_score=0.7,
                complexity="medium",
                description="Model has redundant parameters"
            ),
            OptimizationOpportunity(
                technique="distillation",
                estimated_size_reduction_percent=70.0,
                estimated_speed_improvement_percent=60.0,
                estimated_accuracy_impact_percent=-8.0,
                confidence_score=0.6,
                complexity="high",
                description="Large model suitable for distillation"
            )
        ]
        
        return AnalysisReport(
            model_id="test_model_123",
            architecture_summary=architecture_summary,
            performance_profile=performance_profile,
            optimization_opportunities=opportunities,
            compatibility_matrix={
                "quantization": True,
                "pruning": True,
                "distillation": True,
                "compression": False
            },
            recommendations=[
                Recommendation(
                    technique="quantization",
                    priority=1,
                    rationale="High impact, low risk"
                )
            ]
        )
    
    @pytest.fixture
    def sample_criteria(self):
        """Create sample optimization criteria for testing."""
        constraints = OptimizationConstraints(
            max_optimization_time_minutes=120,
            preserve_accuracy_threshold=0.95,
            allowed_techniques=[
                OptimizationTechnique.QUANTIZATION,
                OptimizationTechnique.PRUNING,
                OptimizationTechnique.DISTILLATION
            ]
        )
        
        priority_weights = {
            PerformanceMetric.MODEL_SIZE: 0.4,
            PerformanceMetric.INFERENCE_TIME: 0.4,
            PerformanceMetric.ACCURACY: 0.2
        }
        
        return OptimizationCriteria(
            name="test_criteria",
            description="Test optimization criteria",
            constraints=constraints,
            priority_weights=priority_weights,
            target_deployment="general"
        )
    
    def test_initialization(self, planning_agent):
        """Test PlanningAgent initialization."""
        assert planning_agent.max_plan_steps == 3
        assert planning_agent.risk_tolerance == 0.7
        assert planning_agent.performance_weight == 0.4
        assert planning_agent.feasibility_weight == 0.3
        assert planning_agent.risk_weight == 0.2
        assert planning_agent.cost_weight == 0.1
        
        # Test initialization method
        assert planning_agent.initialize() is True
    
    def test_cleanup(self, planning_agent):
        """Test PlanningAgent cleanup."""
        # Should not raise any exceptions
        planning_agent.cleanup()
    
    def test_plan_optimization(self, planning_agent, sample_analysis_report, sample_criteria):
        """Test optimization plan creation."""
        plan = planning_agent.plan_optimization(sample_analysis_report, sample_criteria)
        
        assert isinstance(plan, OptimizationPlan)
        assert plan.model_id == "test_model_123"
        assert plan.criteria_name == "test_criteria"
        assert len(plan.steps) > 0
        assert len(plan.steps) <= planning_agent.max_plan_steps
        assert plan.total_estimated_duration_minutes > 0
        assert isinstance(plan.expected_improvements, dict)
        assert plan.risk_assessment != ""
        assert plan.validation_strategy != ""
        assert plan.rollback_strategy != ""
    
    def test_prioritize_techniques(self, planning_agent, sample_analysis_report):
        """Test technique prioritization."""
        opportunities = sample_analysis_report.optimization_opportunities
        prioritized = planning_agent.prioritize_techniques(opportunities)
        
        assert len(prioritized) == len(opportunities)
        
        # Check that results are sorted by overall_score (descending)
        scores = [tech["overall_score"] for tech in prioritized]
        assert scores == sorted(scores, reverse=True)
        
        # Check required fields
        for tech in prioritized:
            assert "technique" in tech
            assert "impact_score" in tech
            assert "feasibility_score" in tech
            assert "risk_score" in tech
            assert "overall_score" in tech
            assert "confidence" in tech
            assert "complexity" in tech
    
    def test_validate_plan_valid(self, planning_agent, sample_analysis_report, sample_criteria):
        """Test validation of a valid plan."""
        plan = planning_agent.plan_optimization(sample_analysis_report, sample_criteria)
        validation_result = planning_agent.validate_plan(plan)
        
        assert isinstance(validation_result, ValidationResult)
        assert validation_result.is_valid is True
        assert len(validation_result.issues) == 0
    
    def test_validate_plan_empty(self, planning_agent):
        """Test validation of an empty plan."""
        empty_plan = OptimizationPlan(
            model_id="test_model",
            criteria_name="test_criteria",
            steps=[],
            total_estimated_duration_minutes=0
        )
        
        validation_result = planning_agent.validate_plan(empty_plan)
        
        assert validation_result.is_valid is False
        assert "no optimization steps" in " ".join(validation_result.issues).lower()
        assert "invalid duration" in " ".join(validation_result.issues).lower()
    
    def test_validate_plan_unmet_prerequisites(self, planning_agent):
        """Test validation of plan with unmet prerequisites."""
        step_with_prereq = OptimizationStep(
            technique="quantization",
            prerequisites=["nonexistent_technique"],
            estimated_duration_minutes=30
        )
        
        plan = OptimizationPlan(
            model_id="test_model",
            criteria_name="test_criteria",
            steps=[step_with_prereq],
            total_estimated_duration_minutes=30
        )
        
        validation_result = planning_agent.validate_plan(plan)
        
        assert validation_result.is_valid is False
        assert any("unmet prerequisite" in issue.lower() for issue in validation_result.issues)
    
    def test_score_techniques(self, planning_agent, sample_analysis_report, sample_criteria):
        """Test technique scoring logic."""
        scores = planning_agent._score_techniques(sample_analysis_report, sample_criteria)
        
        assert len(scores) == len(sample_analysis_report.optimization_opportunities)
        
        for score in scores:
            assert isinstance(score, TechniqueScore)
            assert 0.0 <= score.impact_score <= 1.0
            assert 0.0 <= score.feasibility_score <= 1.0
            assert 0.0 <= score.risk_score <= 1.0
            assert 0.0 <= score.cost_score <= 1.0
            assert score.rationale != ""
            assert isinstance(score.constraints_satisfied, bool)
    
    def test_calculate_impact_score(self, planning_agent, sample_criteria):
        """Test impact score calculation."""
        opportunity = OptimizationOpportunity(
            technique="quantization",
            estimated_size_reduction_percent=50.0,
            estimated_speed_improvement_percent=30.0,
            confidence_score=0.8
        )
        
        impact_score = planning_agent._calculate_impact_score(opportunity, sample_criteria)
        
        assert 0.0 <= impact_score <= 1.0
        assert impact_score > 0  # Should have positive impact
    
    def test_calculate_risk_score(self, planning_agent, sample_criteria):
        """Test risk score calculation."""
        low_risk_opportunity = OptimizationOpportunity(
            technique="quantization",
            estimated_accuracy_impact_percent=-1.0,
            confidence_score=0.9,
            complexity="low"
        )
        
        high_risk_opportunity = OptimizationOpportunity(
            technique="distillation",
            estimated_accuracy_impact_percent=-10.0,
            confidence_score=0.5,
            complexity="high"
        )
        
        low_risk_score = planning_agent._calculate_risk_score(low_risk_opportunity, sample_criteria)
        high_risk_score = planning_agent._calculate_risk_score(high_risk_opportunity, sample_criteria)
        
        assert 0.0 <= low_risk_score <= 1.0
        assert 0.0 <= high_risk_score <= 1.0
        assert high_risk_score > low_risk_score
    
    def test_check_constraints_allowed_techniques(self, planning_agent, sample_criteria):
        """Test constraint checking for allowed techniques."""
        allowed_opportunity = OptimizationOpportunity(
            technique="quantization",
            estimated_accuracy_impact_percent=-2.0
        )
        
        forbidden_opportunity = OptimizationOpportunity(
            technique="compression",
            estimated_accuracy_impact_percent=-2.0
        )
        
        # Modify criteria to only allow quantization
        sample_criteria.constraints.allowed_techniques = [OptimizationTechnique.QUANTIZATION]
        
        assert planning_agent._check_constraints(allowed_opportunity, sample_criteria) is True
        assert planning_agent._check_constraints(forbidden_opportunity, sample_criteria) is False
    
    def test_check_constraints_accuracy_threshold(self, planning_agent, sample_criteria):
        """Test constraint checking for accuracy preservation."""
        good_opportunity = OptimizationOpportunity(
            technique="quantization",
            estimated_accuracy_impact_percent=-2.0  # Within 5% threshold
        )
        
        bad_opportunity = OptimizationOpportunity(
            technique="aggressive_pruning",
            estimated_accuracy_impact_percent=-10.0  # Exceeds 5% threshold
        )
        
        assert planning_agent._check_constraints(good_opportunity, sample_criteria) is True
        assert planning_agent._check_constraints(bad_opportunity, sample_criteria) is False
    
    def test_filter_by_constraints(self, planning_agent, sample_criteria):
        """Test filtering techniques by constraints."""
        # Create technique scores with different constraint satisfaction
        scores = [
            TechniqueScore(
                technique="quantization",
                impact_score=0.8,
                feasibility_score=0.9,
                risk_score=0.2,
                cost_score=0.3,
                overall_score=0.7,
                rationale="Good technique",
                constraints_satisfied=True
            ),
            TechniqueScore(
                technique="bad_technique",
                impact_score=0.9,
                feasibility_score=0.8,
                risk_score=0.9,  # High risk
                cost_score=0.2,
                overall_score=0.8,
                rationale="High risk technique",
                constraints_satisfied=False
            )
        ]
        
        filtered = planning_agent._filter_by_constraints(scores, sample_criteria)
        
        assert len(filtered) == 1
        assert filtered[0].technique == "quantization"
    
    def test_generate_optimization_steps(self, planning_agent, sample_analysis_report, sample_criteria):
        """Test optimization step generation."""
        # Create valid technique scores
        scores = [
            TechniqueScore(
                technique="quantization",
                impact_score=0.8,
                feasibility_score=0.9,
                risk_score=0.2,
                cost_score=0.3,
                overall_score=0.7,
                rationale="Good technique",
                constraints_satisfied=True
            )
        ]
        
        steps = planning_agent._generate_optimization_steps(
            scores, sample_analysis_report, sample_criteria
        )
        
        assert len(steps) == 1
        step = steps[0]
        assert isinstance(step, OptimizationStep)
        assert step.technique == "quantization"
        assert step.priority == 1
        assert step.estimated_duration_minutes > 0
        assert isinstance(step.parameters, dict)
        assert isinstance(step.expected_impact, dict)
        assert step.risk_level in ["low", "medium", "high"]
    
    def test_determine_prerequisites(self, planning_agent):
        """Test prerequisite determination."""
        all_techniques = ["pruning", "quantization", "distillation"]
        
        # Quantization should have pruning as prerequisite if both are present
        quantization_prereqs = planning_agent._determine_prerequisites("quantization", all_techniques)
        assert "pruning" in quantization_prereqs
        
        # Pruning should have no prerequisites
        pruning_prereqs = planning_agent._determine_prerequisites("pruning", all_techniques)
        assert len(pruning_prereqs) == 0
        
        # Distillation should be independent
        distillation_prereqs = planning_agent._determine_prerequisites("distillation", all_techniques)
        assert len(distillation_prereqs) == 0
    
    def test_estimate_step_duration(self, planning_agent, sample_analysis_report):
        """Test step duration estimation."""
        arch_summary = sample_analysis_report.architecture_summary
        
        # Test different techniques
        quantization_duration = planning_agent._estimate_step_duration("quantization", arch_summary)
        pruning_duration = planning_agent._estimate_step_duration("pruning", arch_summary)
        distillation_duration = planning_agent._estimate_step_duration("distillation", arch_summary)
        
        assert quantization_duration > 0
        assert pruning_duration > 0
        assert distillation_duration > 0
        
        # Distillation should take longer than quantization
        assert distillation_duration > quantization_duration
    
    def test_create_step_parameters(self, planning_agent, sample_criteria):
        """Test step parameter creation."""
        # Test quantization parameters
        quant_params = planning_agent._create_step_parameters("quantization", sample_criteria)
        assert "bits" in quant_params
        assert "calibration_samples" in quant_params
        
        # Test pruning parameters
        prune_params = planning_agent._create_step_parameters("pruning", sample_criteria)
        assert "sparsity" in prune_params
        assert "structured" in prune_params
        
        # Test edge deployment adjustments
        edge_criteria = sample_criteria
        edge_criteria.target_deployment = "edge"
        
        edge_quant_params = planning_agent._create_step_parameters("quantization", edge_criteria)
        assert edge_quant_params["bits"] <= quant_params["bits"]  # More aggressive for edge
    
    def test_calculate_expected_improvements(self, planning_agent):
        """Test expected improvements calculation."""
        steps = [
            OptimizationStep(
                technique="quantization",
                expected_impact={
                    "size_reduction_percent": 50.0,
                    "speed_improvement_percent": 30.0,
                    "accuracy_impact_percent": -2.0
                }
            ),
            OptimizationStep(
                technique="pruning",
                expected_impact={
                    "size_reduction_percent": 30.0,
                    "speed_improvement_percent": 20.0,
                    "accuracy_impact_percent": -3.0
                }
            )
        ]
        
        improvements = planning_agent._calculate_expected_improvements(steps)
        
        assert "size_reduction_percent" in improvements
        assert "speed_improvement_percent" in improvements
        assert "accuracy_impact_percent" in improvements
        
        # Should apply diminishing returns for multiple steps
        assert improvements["size_reduction_percent"] < 80.0  # Less than sum due to diminishing returns
        assert improvements["accuracy_impact_percent"] == -5.0  # Accuracy impacts are additive
    
    def test_assess_plan_risk(self, planning_agent):
        """Test plan risk assessment."""
        low_risk_steps = [
            OptimizationStep(technique="quantization", risk_level="low"),
            OptimizationStep(technique="pruning", risk_level="low")
        ]
        
        high_risk_steps = [
            OptimizationStep(technique="distillation", risk_level="high"),
            OptimizationStep(technique="aggressive_pruning", risk_level="high")
        ]
        
        mixed_risk_steps = [
            OptimizationStep(technique="quantization", risk_level="low"),
            OptimizationStep(technique="distillation", risk_level="high")
        ]
        
        low_risk_assessment = planning_agent._assess_plan_risk(low_risk_steps)
        high_risk_assessment = planning_agent._assess_plan_risk(high_risk_steps)
        mixed_risk_assessment = planning_agent._assess_plan_risk(mixed_risk_steps)
        
        assert "low risk" in low_risk_assessment.lower()
        assert "very high risk" in high_risk_assessment.lower()
        assert "high risk" in mixed_risk_assessment.lower()
    
    def test_create_validation_strategy(self, planning_agent, sample_criteria):
        """Test validation strategy creation."""
        strategy = planning_agent._create_validation_strategy(sample_criteria)
        
        assert "accuracy preservation" in strategy.lower()
        assert "95" in strategy  # Accuracy threshold (could be 0.95, 95%, or 95.0%)
        
        # Test with performance thresholds
        sample_criteria.performance_thresholds = [
            PerformanceThreshold(
                metric=PerformanceMetric.INFERENCE_TIME,
                max_value=100.0
            )
        ]
        
        strategy_with_thresholds = planning_agent._create_validation_strategy(sample_criteria)
        assert "inference_time" in strategy_with_thresholds.lower()
    
    def test_create_rollback_strategy(self, planning_agent):
        """Test rollback strategy creation."""
        steps = [
            OptimizationStep(technique="quantization", risk_level="low"),
            OptimizationStep(technique="pruning", risk_level="medium"),
            OptimizationStep(technique="distillation", risk_level="high")
        ]
        
        strategy = planning_agent._create_rollback_strategy(steps)
        
        assert "checkpoint" in strategy.lower()
        assert "backup" in strategy.lower()
        assert "high-risk" in strategy.lower()  # Should mention high-risk steps
    
    def test_edge_cases(self, planning_agent):
        """Test edge cases and error handling."""
        # Test with empty analysis report
        empty_report = AnalysisReport(
            model_id="empty_model",
            optimization_opportunities=[],
            compatibility_matrix={}
        )
        
        empty_criteria = OptimizationCriteria(
            name="empty_criteria",
            description="Empty criteria"
        )
        
        # Should handle empty opportunities gracefully
        plan = planning_agent.plan_optimization(empty_report, empty_criteria)
        assert isinstance(plan, OptimizationPlan)
        assert len(plan.steps) == 0
        
        # Test prioritize_techniques with empty list
        prioritized = planning_agent.prioritize_techniques([])
        assert len(prioritized) == 0
    
    def test_configuration_impact(self):
        """Test how different configurations affect planning behavior."""
        # Conservative configuration
        conservative_config = {
            "risk_tolerance": 0.3,  # Low risk tolerance
            "min_impact_threshold": 0.2,  # High impact threshold
            "max_plan_steps": 2
        }
        
        # Aggressive configuration
        aggressive_config = {
            "risk_tolerance": 0.9,  # High risk tolerance
            "min_impact_threshold": 0.05,  # Low impact threshold
            "max_plan_steps": 5
        }
        
        conservative_agent = PlanningAgent(conservative_config)
        aggressive_agent = PlanningAgent(aggressive_config)
        
        assert conservative_agent.risk_tolerance < aggressive_agent.risk_tolerance
        assert conservative_agent.min_impact_threshold > aggressive_agent.min_impact_threshold
        assert conservative_agent.max_plan_steps < aggressive_agent.max_plan_steps


if __name__ == "__main__":
    pytest.main([__file__])