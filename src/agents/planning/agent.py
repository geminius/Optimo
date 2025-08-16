"""
Planning Agent implementation for robotics model optimization platform.

This module provides the PlanningAgent class that makes intelligent decisions
about optimization strategies based on analysis reports and configurable criteria.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from ..base import BasePlanningAgent
from ...models.core import (
    AnalysisReport, OptimizationOpportunity, Recommendation,
    ArchitectureSummary, PerformanceProfile
)
from ...config.optimization_criteria import (
    OptimizationCriteria, OptimizationTechnique, PerformanceMetric,
    PerformanceThreshold, OptimizationConstraints
)


logger = logging.getLogger(__name__)


@dataclass
class TechniqueScore:
    """Score for an optimization technique."""
    technique: str
    impact_score: float  # 0.0 to 1.0
    feasibility_score: float  # 0.0 to 1.0
    risk_score: float  # 0.0 to 1.0 (lower is better)
    cost_score: float  # 0.0 to 1.0 (lower is better)
    overall_score: float  # Weighted combination
    rationale: str
    constraints_satisfied: bool


@dataclass
class OptimizationStep:
    """Individual step in optimization plan."""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    technique: str = ""
    priority: int = 1  # 1 (highest) to 5 (lowest)
    estimated_duration_minutes: int = 30
    prerequisites: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_impact: Dict[str, float] = field(default_factory=dict)
    risk_level: str = "medium"  # "low", "medium", "high"
    rollback_plan: str = ""


@dataclass
class OptimizationPlan:
    """Complete optimization plan."""
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_id: str = ""
    criteria_name: str = ""
    steps: List[OptimizationStep] = field(default_factory=list)
    total_estimated_duration_minutes: int = 0
    expected_improvements: Dict[str, float] = field(default_factory=dict)
    risk_assessment: str = ""
    validation_strategy: str = ""
    rollback_strategy: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "PlanningAgent"


@dataclass
class ValidationResult:
    """Result of plan validation."""
    is_valid: bool
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class PlanningAgent(BasePlanningAgent):
    """
    Planning agent that makes intelligent decisions about optimization strategies
    based on analysis reports and configurable criteria.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        
        # Configuration parameters
        self.max_plan_steps = config.get("max_plan_steps", 5)
        self.risk_tolerance = config.get("risk_tolerance", 0.7)  # 0.0 to 1.0
        self.performance_weight = config.get("performance_weight", 0.4)
        self.feasibility_weight = config.get("feasibility_weight", 0.3)
        self.risk_weight = config.get("risk_weight", 0.2)
        self.cost_weight = config.get("cost_weight", 0.1)
        
        # Rule-based decision parameters
        self.min_impact_threshold = config.get("min_impact_threshold", 0.1)  # 10% minimum improvement
        self.max_accuracy_loss = config.get("max_accuracy_loss", 0.05)  # 5% max accuracy loss
        self.parallel_optimization_limit = config.get("parallel_optimization_limit", 2)
        
        logger.info(f"PlanningAgent initialized with risk tolerance: {self.risk_tolerance}")
    
    def initialize(self) -> bool:
        """Initialize the planning agent."""
        try:
            logger.info("Initializing PlanningAgent")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize PlanningAgent: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources used by the agent."""
        logger.info("PlanningAgent cleanup completed")
    
    def plan_optimization(
        self, 
        analysis_report: AnalysisReport, 
        criteria: OptimizationCriteria
    ) -> OptimizationPlan:
        """
        Create an optimization plan based on analysis and criteria.
        
        Args:
            analysis_report: Analysis report from AnalysisAgent
            criteria: Optimization criteria and constraints
            
        Returns:
            OptimizationPlan with prioritized steps
        """
        logger.info(f"Creating optimization plan for model: {analysis_report.model_id}")
        
        try:
            # Score and prioritize techniques
            technique_scores = self._score_techniques(analysis_report, criteria)
            
            # Filter techniques based on constraints
            valid_techniques = self._filter_by_constraints(technique_scores, criteria)
            
            # Generate optimization steps
            optimization_steps = self._generate_optimization_steps(
                valid_techniques, analysis_report, criteria
            )
            
            # Calculate expected improvements
            expected_improvements = self._calculate_expected_improvements(optimization_steps)
            
            # Create plan
            plan = OptimizationPlan(
                model_id=analysis_report.model_id,
                criteria_name=criteria.name,
                steps=optimization_steps,
                total_estimated_duration_minutes=sum(step.estimated_duration_minutes for step in optimization_steps),
                expected_improvements=expected_improvements,
                risk_assessment=self._assess_plan_risk(optimization_steps),
                validation_strategy=self._create_validation_strategy(criteria),
                rollback_strategy=self._create_rollback_strategy(optimization_steps)
            )
            
            logger.info(f"Created optimization plan with {len(optimization_steps)} steps")
            return plan
            
        except Exception as e:
            logger.error(f"Failed to create optimization plan: {e}")
            raise
    
    def prioritize_techniques(
        self, 
        opportunities: List[OptimizationOpportunity]
    ) -> List[Dict[str, Any]]:
        """
        Prioritize optimization techniques based on impact and feasibility.
        
        Args:
            opportunities: List of optimization opportunities
            
        Returns:
            List of prioritized techniques with scores
        """
        prioritized = []
        
        for opp in opportunities:
            # Calculate composite score
            impact_score = (
                opp.estimated_size_reduction_percent + 
                opp.estimated_speed_improvement_percent
            ) / 200.0  # Normalize to 0-1
            
            feasibility_score = opp.confidence_score
            
            # Risk score based on accuracy impact and complexity
            accuracy_risk = abs(opp.estimated_accuracy_impact_percent) / 100.0
            complexity_risk = {"low": 0.1, "medium": 0.5, "high": 0.9}.get(opp.complexity, 0.5)
            risk_score = (accuracy_risk + complexity_risk) / 2.0
            
            # Overall score (lower risk is better)
            overall_score = (
                impact_score * self.performance_weight +
                feasibility_score * self.feasibility_weight -
                risk_score * self.risk_weight
            )
            
            prioritized_technique = {
                "technique": opp.technique,
                "impact_score": impact_score,
                "feasibility_score": feasibility_score,
                "risk_score": risk_score,
                "overall_score": overall_score,
                "estimated_size_reduction": opp.estimated_size_reduction_percent,
                "estimated_speed_improvement": opp.estimated_speed_improvement_percent,
                "estimated_accuracy_impact": opp.estimated_accuracy_impact_percent,
                "confidence": opp.confidence_score,
                "complexity": opp.complexity,
                "description": opp.description
            }
            
            prioritized.append(prioritized_technique)
        
        # Sort by overall score (descending)
        prioritized.sort(key=lambda x: x["overall_score"], reverse=True)
        
        logger.info(f"Prioritized {len(prioritized)} optimization techniques")
        return prioritized
    
    def validate_plan(self, plan: OptimizationPlan) -> ValidationResult:
        """
        Validate an optimization plan for feasibility and consistency.
        
        Args:
            plan: Optimization plan to validate
            
        Returns:
            ValidationResult with validation status and issues
        """
        issues = []
        warnings = []
        recommendations = []
        
        # Check plan structure
        if not plan.steps:
            issues.append("Plan contains no optimization steps")
        
        if plan.total_estimated_duration_minutes <= 0:
            issues.append("Plan has invalid duration estimate")
        
        # Check step dependencies
        step_techniques = [step.technique for step in plan.steps]
        for step in plan.steps:
            for prereq in step.prerequisites:
                if prereq not in step_techniques:
                    issues.append(f"Step {step.technique} has unmet prerequisite: {prereq}")
        
        # Check for conflicting techniques
        conflicting_pairs = [
            ("quantization", "pruning"),  # Order matters for these
        ]
        
        for tech1, tech2 in conflicting_pairs:
            if tech1 in step_techniques and tech2 in step_techniques:
                tech1_idx = next(i for i, step in enumerate(plan.steps) if step.technique == tech1)
                tech2_idx = next(i for i, step in enumerate(plan.steps) if step.technique == tech2)
                
                if tech1_idx > tech2_idx:
                    warnings.append(f"Consider applying {tech2} before {tech1} for better results")
        
        # Check resource constraints
        if plan.total_estimated_duration_minutes > 240:  # 4 hours
            warnings.append("Plan duration exceeds 4 hours - consider breaking into phases")
        
        # Check risk levels
        high_risk_steps = [step for step in plan.steps if step.risk_level == "high"]
        if len(high_risk_steps) > 1:
            warnings.append("Multiple high-risk steps detected - consider sequential execution")
        
        # Generate recommendations
        if len(plan.steps) > self.max_plan_steps:
            recommendations.append(f"Consider reducing steps to {self.max_plan_steps} or fewer")
        
        if not any(step.technique == "quantization" for step in plan.steps):
            recommendations.append("Consider adding quantization for better size reduction")
        
        is_valid = len(issues) == 0
        
        logger.info(f"Plan validation completed: {'VALID' if is_valid else 'INVALID'}")
        if issues:
            logger.warning(f"Validation issues: {issues}")
        
        return ValidationResult(
            is_valid=is_valid,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _score_techniques(
        self, 
        analysis_report: AnalysisReport, 
        criteria: OptimizationCriteria
    ) -> List[TechniqueScore]:
        """Score optimization techniques based on analysis and criteria."""
        scores = []
        
        for opportunity in analysis_report.optimization_opportunities:
            # Calculate impact score based on criteria priorities
            impact_score = self._calculate_impact_score(opportunity, criteria)
            
            # Calculate feasibility score
            feasibility_score = opportunity.confidence_score
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(opportunity, criteria)
            
            # Calculate cost score (based on complexity and time)
            cost_score = self._calculate_cost_score(opportunity)
            
            # Check constraint satisfaction
            constraints_satisfied = self._check_constraints(opportunity, criteria)
            
            # Calculate overall score
            overall_score = (
                impact_score * self.performance_weight +
                feasibility_score * self.feasibility_weight -
                risk_score * self.risk_weight -
                cost_score * self.cost_weight
            )
            
            # Apply constraint penalty
            if not constraints_satisfied:
                overall_score *= 0.5  # 50% penalty for constraint violations
            
            technique_score = TechniqueScore(
                technique=opportunity.technique,
                impact_score=impact_score,
                feasibility_score=feasibility_score,
                risk_score=risk_score,
                cost_score=cost_score,
                overall_score=overall_score,
                rationale=self._generate_score_rationale(opportunity, impact_score, risk_score),
                constraints_satisfied=constraints_satisfied
            )
            
            scores.append(technique_score)
        
        # Sort by overall score
        scores.sort(key=lambda x: x.overall_score, reverse=True)
        
        return scores
    
    def _calculate_impact_score(
        self, 
        opportunity: OptimizationOpportunity, 
        criteria: OptimizationCriteria
    ) -> float:
        """Calculate impact score based on criteria priorities."""
        impact_score = 0.0
        total_weight = 0.0
        
        # Size reduction impact
        if PerformanceMetric.MODEL_SIZE in criteria.priority_weights:
            weight = criteria.priority_weights[PerformanceMetric.MODEL_SIZE]
            impact = opportunity.estimated_size_reduction_percent / 100.0
            impact_score += impact * weight
            total_weight += weight
        
        # Speed improvement impact
        if PerformanceMetric.INFERENCE_TIME in criteria.priority_weights:
            weight = criteria.priority_weights[PerformanceMetric.INFERENCE_TIME]
            impact = opportunity.estimated_speed_improvement_percent / 100.0
            impact_score += impact * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            impact_score /= total_weight
        else:
            # Default equal weighting
            impact_score = (
                opportunity.estimated_size_reduction_percent + 
                opportunity.estimated_speed_improvement_percent
            ) / 200.0
        
        return min(1.0, max(0.0, impact_score))
    
    def _calculate_risk_score(
        self, 
        opportunity: OptimizationOpportunity, 
        criteria: OptimizationCriteria
    ) -> float:
        """Calculate risk score for an optimization technique."""
        risk_factors = []
        
        # Accuracy impact risk
        accuracy_impact = abs(opportunity.estimated_accuracy_impact_percent)
        accuracy_risk = min(1.0, accuracy_impact / 20.0)  # Normalize to 20% max
        risk_factors.append(accuracy_risk)
        
        # Complexity risk
        complexity_risks = {"low": 0.1, "medium": 0.5, "high": 0.9}
        complexity_risk = complexity_risks.get(opportunity.complexity, 0.5)
        risk_factors.append(complexity_risk)
        
        # Confidence risk (inverse of confidence)
        confidence_risk = 1.0 - opportunity.confidence_score
        risk_factors.append(confidence_risk)
        
        # Check against accuracy preservation threshold
        if accuracy_impact > (criteria.constraints.preserve_accuracy_threshold * 100):
            risk_factors.append(1.0)  # Maximum risk for exceeding threshold
        
        # Average risk factors
        return sum(risk_factors) / len(risk_factors)
    
    def _calculate_cost_score(self, opportunity: OptimizationOpportunity) -> float:
        """Calculate cost score based on complexity and estimated effort."""
        complexity_costs = {"low": 0.2, "medium": 0.5, "high": 0.9}
        return complexity_costs.get(opportunity.complexity, 0.5)
    
    def _check_constraints(
        self, 
        opportunity: OptimizationOpportunity, 
        criteria: OptimizationCriteria
    ) -> bool:
        """Check if technique satisfies constraints."""
        constraints = criteria.constraints
        
        # Check if technique is allowed
        try:
            technique_enum = OptimizationTechnique(opportunity.technique)
            if technique_enum in constraints.forbidden_techniques:
                return False
            
            if (constraints.allowed_techniques and 
                technique_enum not in constraints.allowed_techniques):
                return False
        except ValueError:
            # Unknown technique - allow by default but log warning
            logger.warning(f"Unknown optimization technique: {opportunity.technique}")
        
        # Check accuracy preservation
        accuracy_impact = abs(opportunity.estimated_accuracy_impact_percent)
        if accuracy_impact > ((1.0 - constraints.preserve_accuracy_threshold) * 100):
            return False
        
        return True
    
    def _generate_score_rationale(
        self, 
        opportunity: OptimizationOpportunity, 
        impact_score: float, 
        risk_score: float
    ) -> str:
        """Generate human-readable rationale for technique score."""
        rationale_parts = []
        
        # Impact assessment
        if impact_score > 0.7:
            rationale_parts.append("High expected performance improvement")
        elif impact_score > 0.4:
            rationale_parts.append("Moderate expected performance improvement")
        else:
            rationale_parts.append("Low expected performance improvement")
        
        # Risk assessment
        if risk_score > 0.7:
            rationale_parts.append("high risk of accuracy degradation")
        elif risk_score > 0.4:
            rationale_parts.append("moderate risk profile")
        else:
            rationale_parts.append("low risk profile")
        
        # Confidence
        if opportunity.confidence_score > 0.8:
            rationale_parts.append("high confidence in estimates")
        elif opportunity.confidence_score > 0.5:
            rationale_parts.append("moderate confidence in estimates")
        else:
            rationale_parts.append("low confidence in estimates")
        
        return "; ".join(rationale_parts)
    
    def _filter_by_constraints(
        self, 
        technique_scores: List[TechniqueScore], 
        criteria: OptimizationCriteria
    ) -> List[TechniqueScore]:
        """Filter techniques that satisfy constraints."""
        valid_techniques = []
        
        for score in technique_scores:
            # Check constraint satisfaction
            if not score.constraints_satisfied:
                logger.info(f"Filtering out {score.technique} due to constraint violations")
                continue
            
            # Check minimum impact threshold
            if score.impact_score < self.min_impact_threshold:
                logger.info(f"Filtering out {score.technique} due to low impact score")
                continue
            
            # Check risk tolerance
            if score.risk_score > self.risk_tolerance:
                logger.info(f"Filtering out {score.technique} due to high risk")
                continue
            
            valid_techniques.append(score)
        
        return valid_techniques
    
    def _generate_optimization_steps(
        self, 
        technique_scores: List[TechniqueScore], 
        analysis_report: AnalysisReport, 
        criteria: OptimizationCriteria
    ) -> List[OptimizationStep]:
        """Generate optimization steps from scored techniques."""
        steps = []
        
        # Limit number of steps
        selected_techniques = technique_scores[:self.max_plan_steps]
        
        for i, score in enumerate(selected_techniques):
            # Determine prerequisites and ordering
            prerequisites = self._determine_prerequisites(score.technique, [s.technique for s in selected_techniques])
            
            # Estimate duration based on complexity and model size
            duration = self._estimate_step_duration(score.technique, analysis_report.architecture_summary)
            
            # Create step parameters
            parameters = self._create_step_parameters(score.technique, criteria)
            
            # Calculate expected impact
            opportunity = next(
                (opp for opp in analysis_report.optimization_opportunities 
                 if opp.technique == score.technique), 
                None
            )
            
            expected_impact = {}
            if opportunity:
                expected_impact = {
                    "size_reduction_percent": opportunity.estimated_size_reduction_percent,
                    "speed_improvement_percent": opportunity.estimated_speed_improvement_percent,
                    "accuracy_impact_percent": opportunity.estimated_accuracy_impact_percent
                }
            
            step = OptimizationStep(
                technique=score.technique,
                priority=i + 1,
                estimated_duration_minutes=duration,
                prerequisites=prerequisites,
                parameters=parameters,
                expected_impact=expected_impact,
                risk_level=self._determine_risk_level(score.risk_score),
                rollback_plan=f"Restore model checkpoint before {score.technique} optimization"
            )
            
            steps.append(step)
        
        return steps
    
    def _determine_prerequisites(self, technique: str, all_techniques: List[str]) -> List[str]:
        """Determine prerequisites for a technique based on best practices."""
        prerequisites = []
        
        # Define technique dependencies
        dependencies = {
            "pruning": [],  # Pruning should come first
            "quantization": ["pruning"],  # Quantization after pruning for better results
            "distillation": [],  # Can be independent
            "compression": ["pruning"],  # Compression after pruning
            "architecture_search": []  # Independent technique
        }
        
        technique_deps = dependencies.get(technique, [])
        
        # Only include prerequisites that are in the current plan
        for dep in technique_deps:
            if dep in all_techniques:
                prerequisites.append(dep)
        
        return prerequisites
    
    def _estimate_step_duration(self, technique: str, arch_summary: ArchitectureSummary) -> int:
        """Estimate duration for optimization step based on technique and model size."""
        base_durations = {
            "quantization": 15,  # minutes
            "pruning": 30,
            "distillation": 120,
            "compression": 45,
            "architecture_search": 180
        }
        
        base_duration = base_durations.get(technique, 30)
        
        # Scale based on model size
        param_millions = arch_summary.total_parameters / 1_000_000
        
        if param_millions > 100:  # Very large model
            scale_factor = 3.0
        elif param_millions > 10:  # Large model
            scale_factor = 2.0
        elif param_millions > 1:  # Medium model
            scale_factor = 1.5
        else:  # Small model
            scale_factor = 1.0
        
        return int(base_duration * scale_factor)
    
    def _create_step_parameters(self, technique: str, criteria: OptimizationCriteria) -> Dict[str, Any]:
        """Create parameters for optimization step based on technique and criteria."""
        parameters = {}
        
        if technique == "quantization":
            parameters = {
                "bits": 8,  # Default to 8-bit quantization
                "calibration_samples": 100,
                "preserve_accuracy": True
            }
        elif technique == "pruning":
            parameters = {
                "sparsity": 0.5,  # 50% sparsity
                "structured": False,  # Unstructured pruning by default
                "gradual": True
            }
        elif technique == "distillation":
            parameters = {
                "temperature": 4.0,
                "alpha": 0.7,
                "student_size_ratio": 0.5
            }
        elif technique == "compression":
            parameters = {
                "compression_ratio": 0.3,
                "method": "svd"
            }
        
        # Apply criteria-specific adjustments
        if criteria.target_deployment == "edge":
            # More aggressive optimization for edge deployment
            if technique == "quantization":
                parameters["bits"] = 4
            elif technique == "pruning":
                parameters["sparsity"] = 0.7
        
        return parameters
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """Determine risk level category from risk score."""
        if risk_score > 0.7:
            return "high"
        elif risk_score > 0.4:
            return "medium"
        else:
            return "low"
    
    def _calculate_expected_improvements(self, steps: List[OptimizationStep]) -> Dict[str, float]:
        """Calculate expected cumulative improvements from all steps."""
        total_size_reduction = 0.0
        total_speed_improvement = 0.0
        total_accuracy_impact = 0.0
        
        for step in steps:
            impact = step.expected_impact
            total_size_reduction += impact.get("size_reduction_percent", 0.0)
            total_speed_improvement += impact.get("speed_improvement_percent", 0.0)
            total_accuracy_impact += impact.get("accuracy_impact_percent", 0.0)
        
        # Apply diminishing returns for multiple optimizations
        size_reduction_factor = 0.8 if len(steps) > 1 else 1.0
        speed_improvement_factor = 0.9 if len(steps) > 1 else 1.0
        
        return {
            "size_reduction_percent": total_size_reduction * size_reduction_factor,
            "speed_improvement_percent": total_speed_improvement * speed_improvement_factor,
            "accuracy_impact_percent": total_accuracy_impact
        }
    
    def _assess_plan_risk(self, steps: List[OptimizationStep]) -> str:
        """Assess overall risk level of the optimization plan."""
        risk_levels = [step.risk_level for step in steps]
        
        if "high" in risk_levels:
            if risk_levels.count("high") > 1:
                return "Very high risk due to multiple high-risk optimizations"
            else:
                return "High risk due to aggressive optimization technique"
        elif "medium" in risk_levels:
            return "Medium risk with potential for accuracy degradation"
        else:
            return "Low risk with conservative optimization approach"
    
    def _create_validation_strategy(self, criteria: OptimizationCriteria) -> str:
        """Create validation strategy based on criteria."""
        strategies = []
        
        # Always validate accuracy preservation
        threshold = criteria.constraints.preserve_accuracy_threshold
        strategies.append(f"Validate accuracy preservation above {threshold:.1%}")
        
        # Add performance threshold validations
        for threshold in criteria.performance_thresholds:
            if threshold.min_value is not None:
                strategies.append(f"Ensure {threshold.metric.value} >= {threshold.min_value}")
            if threshold.max_value is not None:
                strategies.append(f"Ensure {threshold.metric.value} <= {threshold.max_value}")
        
        # Add deployment-specific validations
        if criteria.target_deployment == "edge":
            strategies.append("Validate edge device compatibility")
        elif criteria.target_deployment == "mobile":
            strategies.append("Validate mobile deployment constraints")
        
        return "; ".join(strategies)
    
    def _create_rollback_strategy(self, steps: List[OptimizationStep]) -> str:
        """Create rollback strategy for the optimization plan."""
        if not steps:
            return "No rollback needed for empty plan"
        
        strategies = [
            "Create checkpoint before each optimization step",
            "Validate results after each step before proceeding",
            "Maintain original model backup throughout process"
        ]
        
        # Add step-specific rollback considerations
        high_risk_steps = [step for step in steps if step.risk_level == "high"]
        if high_risk_steps:
            strategies.append("Extra validation checkpoints for high-risk steps")
        
        if len(steps) > 3:
            strategies.append("Consider phased rollback for complex plans")
        
        return "; ".join(strategies)