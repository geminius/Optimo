"""
Unit tests for base agent classes (BaseAgent, BaseAnalysisAgent, BasePlanningAgent, BaseEvaluationAgent).
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List
from abc import ABC

from src.agents.base import (
    BaseAgent,
    BaseAnalysisAgent,
    BasePlanningAgent,
    BaseEvaluationAgent,
    ImpactEstimate,
    ValidationResult,
    OptimizationStatus,
    ProgressUpdate,
    OptimizationSnapshot
)


class MockBaseAgent(BaseAgent):
    """Mock implementation of BaseAgent for testing."""
    
    def __init__(self, config: Dict[str, Any], init_success: bool = True):
        super().__init__(config)
        self.init_success = init_success
        self.initialized = False
        self.cleaned_up = False
    
    def initialize(self) -> bool:
        self.initialized = True
        return self.init_success
    
    def cleanup(self) -> None:
        self.cleaned_up = True


class MockAnalysisAgent(BaseAnalysisAgent):
    """Mock implementation of BaseAnalysisAgent for testing."""
    
    def __init__(self, config: Dict[str, Any], should_fail: bool = False):
        super().__init__(config)
        self.should_fail = should_fail
        self.analyze_called = False
        self.bottlenecks_called = False
    
    def initialize(self) -> bool:
        return True
    
    def cleanup(self) -> None:
        pass
    
    def analyze_model(self, model_path: str) -> Dict[str, Any]:
        self.analyze_called = True
        if self.should_fail:
            raise RuntimeError("Analysis failed")
        
        return {
            "model_type": "cnn",
            "parameters": 1000000,
            "layers": 10,
            "input_shape": [224, 224, 3],
            "output_shape": [1000],
            "analysis_timestamp": datetime.now().isoformat(),
            "optimization_opportunities": [
                {"technique": "quantization", "potential_reduction": 0.75},
                {"technique": "pruning", "potential_reduction": 0.5}
            ]
        }
    
    def identify_bottlenecks(self, model: torch.nn.Module) -> List[Dict[str, Any]]:
        self.bottlenecks_called = True
        if self.should_fail:
            raise RuntimeError("Bottleneck identification failed")
        
        return [
            {
                "layer_name": "conv1",
                "layer_type": "Conv2d",
                "bottleneck_type": "memory",
                "severity": "high",
                "impact": 0.8,
                "recommendations": ["Consider reducing kernel size", "Apply pruning"]
            },
            {
                "layer_name": "fc1",
                "layer_type": "Linear",
                "bottleneck_type": "computation",
                "severity": "medium",
                "impact": 0.6,
                "recommendations": ["Apply quantization", "Reduce hidden dimensions"]
            }
        ]


class MockPlanningAgent(BasePlanningAgent):
    """Mock implementation of BasePlanningAgent for testing."""
    
    def __init__(self, config: Dict[str, Any], should_fail: bool = False):
        super().__init__(config)
        self.should_fail = should_fail
        self.plan_called = False
        self.prioritize_called = False
    
    def initialize(self) -> bool:
        return True
    
    def cleanup(self) -> None:
        pass
    
    def plan_optimization(self, analysis_report: Dict[str, Any], criteria: Dict[str, Any]) -> Dict[str, Any]:
        self.plan_called = True
        if self.should_fail:
            raise RuntimeError("Planning failed")
        
        return {
            "plan_id": "test_plan_123",
            "optimization_sequence": [
                {
                    "technique": "quantization",
                    "priority": 1,
                    "expected_impact": 0.7,
                    "estimated_time": 300,
                    "config": {"bits": 8, "method": "dynamic"}
                },
                {
                    "technique": "pruning",
                    "priority": 2,
                    "expected_impact": 0.5,
                    "estimated_time": 600,
                    "config": {"sparsity": 0.5, "structured": False}
                }
            ],
            "total_estimated_time": 900,
            "expected_performance_gain": 0.85,
            "risk_assessment": "low",
            "fallback_strategies": ["rollback", "partial_optimization"]
        }
    
    def prioritize_techniques(self, opportunities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        self.prioritize_called = True
        if self.should_fail:
            raise RuntimeError("Prioritization failed")
        
        # Sort by potential impact (descending)
        sorted_opportunities = sorted(
            opportunities, 
            key=lambda x: x.get("potential_reduction", 0), 
            reverse=True
        )
        
        # Add priority scores
        for i, opportunity in enumerate(sorted_opportunities):
            opportunity["priority"] = i + 1
            opportunity["feasibility_score"] = 0.9 - (i * 0.1)
        
        return sorted_opportunities


class MockEvaluationAgent(BaseEvaluationAgent):
    """Mock implementation of BaseEvaluationAgent for testing."""
    
    def __init__(self, config: Dict[str, Any], should_fail: bool = False):
        super().__init__(config)
        self.should_fail = should_fail
        self.evaluate_called = False
        self.compare_called = False
    
    def initialize(self) -> bool:
        return True
    
    def cleanup(self) -> None:
        pass
    
    def evaluate_model(self, model: torch.nn.Module, benchmarks: List[Dict[str, Any]]) -> Dict[str, Any]:
        self.evaluate_called = True
        if self.should_fail:
            raise RuntimeError("Evaluation failed")
        
        return {
            "evaluation_id": "eval_123",
            "model_id": "model_456",
            "benchmark_results": [
                {
                    "benchmark_name": "accuracy_test",
                    "score": 0.95,
                    "metric": "accuracy",
                    "passed": True
                },
                {
                    "benchmark_name": "speed_test",
                    "score": 1.2,
                    "metric": "inference_time_ms",
                    "passed": True
                }
            ],
            "overall_score": 0.92,
            "performance_metrics": {
                "accuracy": 0.95,
                "inference_time": 1.2,
                "memory_usage": 512.0,
                "throughput": 100.0
            },
            "evaluation_timestamp": datetime.now().isoformat()
        }
    
    def compare_models(self, original: torch.nn.Module, optimized: torch.nn.Module) -> Dict[str, Any]:
        self.compare_called = True
        if self.should_fail:
            raise RuntimeError("Comparison failed")
        
        return {
            "comparison_id": "comp_789",
            "original_metrics": {
                "accuracy": 0.94,
                "inference_time": 2.5,
                "memory_usage": 1024.0,
                "model_size": 100.0
            },
            "optimized_metrics": {
                "accuracy": 0.95,
                "inference_time": 1.2,
                "memory_usage": 512.0,
                "model_size": 25.0
            },
            "improvements": {
                "accuracy": 0.01,
                "inference_time": -1.3,  # Negative means faster
                "memory_usage": -512.0,  # Negative means less memory
                "model_size": -75.0      # Negative means smaller
            },
            "improvement_percentages": {
                "accuracy": 1.06,
                "inference_time": 52.0,
                "memory_usage": 50.0,
                "model_size": 75.0
            },
            "overall_improvement": 0.44,
            "recommendation": "optimization_successful"
        }


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(16, 10)
    )


@pytest.fixture
def mock_base_agent():
    """Create a mock base agent for testing."""
    config = {"test_param": "test_value", "timeout": 30}
    return MockBaseAgent(config)


@pytest.fixture
def failing_base_agent():
    """Create a mock base agent that fails initialization."""
    config = {"test_param": "test_value"}
    return MockBaseAgent(config, init_success=False)


@pytest.fixture
def mock_analysis_agent():
    """Create a mock analysis agent for testing."""
    config = {"analysis_depth": "full", "timeout": 60}
    return MockAnalysisAgent(config)


@pytest.fixture
def failing_analysis_agent():
    """Create a mock analysis agent that fails operations."""
    config = {"analysis_depth": "full"}
    return MockAnalysisAgent(config, should_fail=True)


@pytest.fixture
def mock_planning_agent():
    """Create a mock planning agent for testing."""
    config = {"planning_strategy": "greedy", "max_techniques": 3}
    return MockPlanningAgent(config)


@pytest.fixture
def failing_planning_agent():
    """Create a mock planning agent that fails operations."""
    config = {"planning_strategy": "greedy"}
    return MockPlanningAgent(config, should_fail=True)


@pytest.fixture
def mock_evaluation_agent():
    """Create a mock evaluation agent for testing."""
    config = {"evaluation_mode": "comprehensive", "benchmarks": ["accuracy", "speed"]}
    return MockEvaluationAgent(config)


@pytest.fixture
def failing_evaluation_agent():
    """Create a mock evaluation agent that fails operations."""
    config = {"evaluation_mode": "comprehensive"}
    return MockEvaluationAgent(config, should_fail=True)


class TestBaseAgent:
    """Test cases for BaseAgent abstract class."""
    
    def test_initialization(self, mock_base_agent):
        """Test base agent initialization."""
        assert mock_base_agent.name == "MockBaseAgent"
        assert mock_base_agent.config == {"test_param": "test_value", "timeout": 30}
        assert isinstance(mock_base_agent.created_at, datetime)
        assert not mock_base_agent.initialized
        assert not mock_base_agent.cleaned_up
    
    def test_successful_initialization(self, mock_base_agent):
        """Test successful agent initialization."""
        result = mock_base_agent.initialize()
        assert result is True
        assert mock_base_agent.initialized
    
    def test_failed_initialization(self, failing_base_agent):
        """Test failed agent initialization."""
        result = failing_base_agent.initialize()
        assert result is False
        assert failing_base_agent.initialized  # Still called, but returned False
    
    def test_cleanup(self, mock_base_agent):
        """Test agent cleanup."""
        mock_base_agent.cleanup()
        assert mock_base_agent.cleaned_up
    
    def test_get_status(self, mock_base_agent):
        """Test get_status method."""
        status = mock_base_agent.get_status()
        
        assert isinstance(status, dict)
        assert status["name"] == "MockBaseAgent"
        assert status["config"] == {"test_param": "test_value", "timeout": 30}
        assert "created_at" in status
        assert isinstance(status["created_at"], str)  # Should be ISO format
    
    def test_abstract_methods_enforcement(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAgent({"test": "config"})


class TestBaseAnalysisAgent:
    """Test cases for BaseAnalysisAgent abstract class."""
    
    def test_inheritance(self, mock_analysis_agent):
        """Test that BaseAnalysisAgent inherits from BaseAgent."""
        assert isinstance(mock_analysis_agent, BaseAgent)
        assert isinstance(mock_analysis_agent, BaseAnalysisAgent)
    
    def test_analyze_model_success(self, mock_analysis_agent):
        """Test successful model analysis."""
        model_path = "/path/to/test/model.pth"
        result = mock_analysis_agent.analyze_model(model_path)
        
        assert mock_analysis_agent.analyze_called
        assert isinstance(result, dict)
        assert "model_type" in result
        assert "parameters" in result
        assert "optimization_opportunities" in result
        assert result["model_type"] == "cnn"
        assert result["parameters"] == 1000000
    
    def test_analyze_model_failure(self, failing_analysis_agent):
        """Test model analysis failure."""
        model_path = "/path/to/test/model.pth"
        
        with pytest.raises(RuntimeError, match="Analysis failed"):
            failing_analysis_agent.analyze_model(model_path)
        
        assert failing_analysis_agent.analyze_called
    
    def test_identify_bottlenecks_success(self, mock_analysis_agent, simple_model):
        """Test successful bottleneck identification."""
        bottlenecks = mock_analysis_agent.identify_bottlenecks(simple_model)
        
        assert mock_analysis_agent.bottlenecks_called
        assert isinstance(bottlenecks, list)
        assert len(bottlenecks) == 2
        
        # Check first bottleneck
        assert bottlenecks[0]["layer_name"] == "conv1"
        assert bottlenecks[0]["bottleneck_type"] == "memory"
        assert bottlenecks[0]["severity"] == "high"
        assert "recommendations" in bottlenecks[0]
        
        # Check second bottleneck
        assert bottlenecks[1]["layer_name"] == "fc1"
        assert bottlenecks[1]["bottleneck_type"] == "computation"
        assert bottlenecks[1]["severity"] == "medium"
    
    def test_identify_bottlenecks_failure(self, failing_analysis_agent, simple_model):
        """Test bottleneck identification failure."""
        with pytest.raises(RuntimeError, match="Bottleneck identification failed"):
            failing_analysis_agent.identify_bottlenecks(simple_model)
        
        assert failing_analysis_agent.bottlenecks_called
    
    def test_abstract_methods_enforcement(self):
        """Test that BaseAnalysisAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAnalysisAgent({"test": "config"})


class TestBasePlanningAgent:
    """Test cases for BasePlanningAgent abstract class."""
    
    def test_inheritance(self, mock_planning_agent):
        """Test that BasePlanningAgent inherits from BaseAgent."""
        assert isinstance(mock_planning_agent, BaseAgent)
        assert isinstance(mock_planning_agent, BasePlanningAgent)
    
    def test_plan_optimization_success(self, mock_planning_agent):
        """Test successful optimization planning."""
        analysis_report = {
            "model_type": "cnn",
            "optimization_opportunities": [
                {"technique": "quantization", "potential_reduction": 0.75},
                {"technique": "pruning", "potential_reduction": 0.5}
            ]
        }
        criteria = {
            "max_accuracy_loss": 0.02,
            "target_size_reduction": 0.5,
            "max_optimization_time": 3600
        }
        
        plan = mock_planning_agent.plan_optimization(analysis_report, criteria)
        
        assert mock_planning_agent.plan_called
        assert isinstance(plan, dict)
        assert "plan_id" in plan
        assert "optimization_sequence" in plan
        assert len(plan["optimization_sequence"]) == 2
        
        # Check first optimization step
        first_step = plan["optimization_sequence"][0]
        assert first_step["technique"] == "quantization"
        assert first_step["priority"] == 1
        assert "config" in first_step
    
    def test_plan_optimization_failure(self, failing_planning_agent):
        """Test optimization planning failure."""
        analysis_report = {"model_type": "cnn"}
        criteria = {"max_accuracy_loss": 0.02}
        
        with pytest.raises(RuntimeError, match="Planning failed"):
            failing_planning_agent.plan_optimization(analysis_report, criteria)
        
        assert failing_planning_agent.plan_called
    
    def test_prioritize_techniques_success(self, mock_planning_agent):
        """Test successful technique prioritization."""
        opportunities = [
            {"technique": "quantization", "potential_reduction": 0.75},
            {"technique": "pruning", "potential_reduction": 0.5},
            {"technique": "distillation", "potential_reduction": 0.9}
        ]
        
        prioritized = mock_planning_agent.prioritize_techniques(opportunities)
        
        assert mock_planning_agent.prioritize_called
        assert isinstance(prioritized, list)
        assert len(prioritized) == 3
        
        # Should be sorted by potential_reduction (descending)
        assert prioritized[0]["technique"] == "distillation"  # 0.9
        assert prioritized[1]["technique"] == "quantization"  # 0.75
        assert prioritized[2]["technique"] == "pruning"       # 0.5
        
        # Check priority scores were added
        assert prioritized[0]["priority"] == 1
        assert prioritized[1]["priority"] == 2
        assert prioritized[2]["priority"] == 3
        
        # Check feasibility scores
        assert prioritized[0]["feasibility_score"] == 0.9
        assert prioritized[1]["feasibility_score"] == 0.8
        assert prioritized[2]["feasibility_score"] == 0.7
    
    def test_prioritize_techniques_failure(self, failing_planning_agent):
        """Test technique prioritization failure."""
        opportunities = [{"technique": "quantization", "potential_reduction": 0.75}]
        
        with pytest.raises(RuntimeError, match="Prioritization failed"):
            failing_planning_agent.prioritize_techniques(opportunities)
        
        assert failing_planning_agent.prioritize_called
    
    def test_abstract_methods_enforcement(self):
        """Test that BasePlanningAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePlanningAgent({"test": "config"})


class TestBaseEvaluationAgent:
    """Test cases for BaseEvaluationAgent abstract class."""
    
    def test_inheritance(self, mock_evaluation_agent):
        """Test that BaseEvaluationAgent inherits from BaseAgent."""
        assert isinstance(mock_evaluation_agent, BaseAgent)
        assert isinstance(mock_evaluation_agent, BaseEvaluationAgent)
    
    def test_evaluate_model_success(self, mock_evaluation_agent, simple_model):
        """Test successful model evaluation."""
        benchmarks = [
            {"name": "accuracy_test", "type": "classification"},
            {"name": "speed_test", "type": "performance"}
        ]
        
        result = mock_evaluation_agent.evaluate_model(simple_model, benchmarks)
        
        assert mock_evaluation_agent.evaluate_called
        assert isinstance(result, dict)
        assert "evaluation_id" in result
        assert "benchmark_results" in result
        assert "overall_score" in result
        assert "performance_metrics" in result
        
        # Check benchmark results
        assert len(result["benchmark_results"]) == 2
        assert result["benchmark_results"][0]["benchmark_name"] == "accuracy_test"
        assert result["benchmark_results"][0]["passed"] is True
        
        # Check performance metrics
        metrics = result["performance_metrics"]
        assert "accuracy" in metrics
        assert "inference_time" in metrics
        assert metrics["accuracy"] == 0.95
    
    def test_evaluate_model_failure(self, failing_evaluation_agent, simple_model):
        """Test model evaluation failure."""
        benchmarks = [{"name": "accuracy_test", "type": "classification"}]
        
        with pytest.raises(RuntimeError, match="Evaluation failed"):
            failing_evaluation_agent.evaluate_model(simple_model, benchmarks)
        
        assert failing_evaluation_agent.evaluate_called
    
    def test_compare_models_success(self, mock_evaluation_agent, simple_model):
        """Test successful model comparison."""
        # Create a slightly different model for comparison
        optimized_model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),  # Fewer channels
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(8, 10)
        )
        
        result = mock_evaluation_agent.compare_models(simple_model, optimized_model)
        
        assert mock_evaluation_agent.compare_called
        assert isinstance(result, dict)
        assert "comparison_id" in result
        assert "original_metrics" in result
        assert "optimized_metrics" in result
        assert "improvements" in result
        assert "improvement_percentages" in result
        
        # Check metrics structure
        original_metrics = result["original_metrics"]
        optimized_metrics = result["optimized_metrics"]
        improvements = result["improvements"]
        
        assert "accuracy" in original_metrics
        assert "inference_time" in original_metrics
        assert "memory_usage" in original_metrics
        assert "model_size" in original_metrics
        
        # Check improvements (negative values indicate better performance)
        assert improvements["inference_time"] < 0  # Faster
        assert improvements["memory_usage"] < 0    # Less memory
        assert improvements["model_size"] < 0      # Smaller
        assert improvements["accuracy"] > 0        # Better accuracy
    
    def test_compare_models_failure(self, failing_evaluation_agent, simple_model):
        """Test model comparison failure."""
        optimized_model = simple_model  # Same model for simplicity
        
        with pytest.raises(RuntimeError, match="Comparison failed"):
            failing_evaluation_agent.compare_models(simple_model, optimized_model)
        
        assert failing_evaluation_agent.compare_called
    
    def test_abstract_methods_enforcement(self):
        """Test that BaseEvaluationAgent cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvaluationAgent({"test": "config"})


class TestDataClassesAndEnums:
    """Test cases for data classes and enums used by base agents."""
    
    def test_impact_estimate_creation(self):
        """Test ImpactEstimate data class creation."""
        estimate = ImpactEstimate(
            performance_improvement=0.2,
            size_reduction=0.5,
            speed_improvement=0.3,
            confidence=0.8,
            estimated_time_minutes=10
        )
        
        assert estimate.performance_improvement == 0.2
        assert estimate.size_reduction == 0.5
        assert estimate.speed_improvement == 0.3
        assert estimate.confidence == 0.8
        assert estimate.estimated_time_minutes == 10
    
    def test_validation_result_creation(self):
        """Test ValidationResult data class creation."""
        result = ValidationResult(
            is_valid=True,
            accuracy_preserved=True,
            performance_metrics={"accuracy": 0.95, "speed": 1.2},
            issues=[],
            recommendations=["Good optimization"]
        )
        
        assert result.is_valid is True
        assert result.accuracy_preserved is True
        assert result.performance_metrics["accuracy"] == 0.95
        assert len(result.issues) == 0
        assert len(result.recommendations) == 1
    
    def test_optimization_status_enum(self):
        """Test OptimizationStatus enum values."""
        assert OptimizationStatus.NOT_STARTED.value == "not_started"
        assert OptimizationStatus.INITIALIZING.value == "initializing"
        assert OptimizationStatus.ANALYZING.value == "analyzing"
        assert OptimizationStatus.OPTIMIZING.value == "optimizing"
        assert OptimizationStatus.VALIDATING.value == "validating"
        assert OptimizationStatus.COMPLETED.value == "completed"
        assert OptimizationStatus.FAILED.value == "failed"
        assert OptimizationStatus.CANCELLED.value == "cancelled"
        assert OptimizationStatus.ROLLED_BACK.value == "rolled_back"
    
    def test_progress_update_creation(self):
        """Test ProgressUpdate data class creation."""
        update = ProgressUpdate(
            status=OptimizationStatus.ANALYZING,
            progress_percentage=50.0,
            current_step="Analyzing model architecture",
            estimated_remaining_minutes=5,
            message="Processing layer 5 of 10"
        )
        
        assert update.status == OptimizationStatus.ANALYZING
        assert update.progress_percentage == 50.0
        assert update.current_step == "Analyzing model architecture"
        assert update.estimated_remaining_minutes == 5
        assert update.message == "Processing layer 5 of 10"
        assert isinstance(update.timestamp, datetime)
    
    def test_optimization_snapshot_creation(self):
        """Test OptimizationSnapshot data class creation."""
        model_state = {"layer1.weight": torch.randn(10, 5)}
        metadata = {"checkpoint_type": "pre_optimization"}
        
        snapshot = OptimizationSnapshot(
            model_state_dict=model_state,
            metadata=metadata,
            timestamp=datetime.now(),
            checkpoint_name="test_checkpoint"
        )
        
        assert snapshot.model_state_dict == model_state
        assert snapshot.metadata == metadata
        assert snapshot.checkpoint_name == "test_checkpoint"
        assert isinstance(snapshot.timestamp, datetime)


if __name__ == "__main__":
    pytest.main([__file__])