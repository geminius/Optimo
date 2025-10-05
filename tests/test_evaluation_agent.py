"""
Unit tests for the Evaluation Agent.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from src.agents.evaluation.agent import EvaluationAgent
from src.models.core import (
    EvaluationReport, BenchmarkResult, PerformanceMetrics,
    ComparisonResult, ValidationStatus
)


class SimpleTestModel(nn.Module):
    """Simple test model for evaluation."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class LargeTestModel(nn.Module):
    """Large test model for comparison testing."""
    
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1000, 2048),
            nn.ReLU(),
            nn.Linear(2048, 4096),
            nn.ReLU(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1000),
            nn.ReLU(),
            nn.Linear(1000, 100)
        )
    
    def forward(self, x):
        return self.layers(x)


class OptimizedTestModel(nn.Module):
    """Smaller optimized version of test model."""
    
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 100)
        )
    
    def forward(self, x):
        return self.layers(x)


@pytest.fixture
def evaluation_agent():
    """Create an EvaluationAgent instance for testing."""
    config = {
        "benchmark_samples": 10,  # Reduced for faster testing
        "warmup_samples": 2,
        "accuracy_threshold": 0.95,
        "performance_threshold": 1.1,
        "timeout_seconds": 30
    }
    agent = EvaluationAgent(config)
    agent.initialize()
    return agent


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    return SimpleTestModel()


@pytest.fixture
def large_model():
    """Create a large test model."""
    return LargeTestModel()


@pytest.fixture
def optimized_model():
    """Create an optimized test model."""
    return OptimizedTestModel()


@pytest.fixture
def standard_benchmarks():
    """Create standard benchmark configurations."""
    return [
        {"name": "inference_speed", "type": "inference_speed"},
        {"name": "memory_usage", "type": "memory_usage"},
        {"name": "throughput", "type": "throughput"},
        {"name": "model_size", "type": "model_size"}
    ]


class TestEvaluationAgent:
    """Test cases for EvaluationAgent."""
    
    def test_initialization(self, evaluation_agent):
        """Test agent initialization."""
        assert evaluation_agent.name == "EvaluationAgent"
        assert evaluation_agent.benchmark_samples == 10
        assert evaluation_agent.warmup_samples == 2
        assert evaluation_agent.accuracy_threshold == 0.95
        assert evaluation_agent.device is not None
        assert len(evaluation_agent._benchmark_registry) > 0
    
    def test_cleanup(self, evaluation_agent):
        """Test agent cleanup."""
        # Should not raise any exceptions
        evaluation_agent.cleanup()
    
    def test_evaluate_model_success(self, evaluation_agent, simple_model, standard_benchmarks):
        """Test successful model evaluation."""
        report = evaluation_agent.evaluate_model(simple_model, standard_benchmarks)
        
        assert isinstance(report, EvaluationReport)
        assert len(report.benchmarks) > 0
        assert isinstance(report.performance_metrics, PerformanceMetrics)
        assert report.validation_status in [ValidationStatus.PASSED, ValidationStatus.WARNING, ValidationStatus.FAILED]
        assert report.evaluation_duration_seconds > 0
        assert isinstance(report.recommendations, list)
    
    def test_evaluate_model_empty_benchmarks(self, evaluation_agent, simple_model):
        """Test evaluation with empty benchmark list."""
        report = evaluation_agent.evaluate_model(simple_model, [])
        
        assert isinstance(report, EvaluationReport)
        assert len(report.benchmarks) == 0
        assert report.validation_status == ValidationStatus.FAILED
    
    def test_evaluate_model_invalid_benchmark(self, evaluation_agent, simple_model):
        """Test evaluation with invalid benchmark."""
        invalid_benchmarks = [{"name": "invalid", "type": "invalid_type"}]
        
        report = evaluation_agent.evaluate_model(simple_model, invalid_benchmarks)
        
        assert isinstance(report, EvaluationReport)
        assert len(report.validation_errors) > 0
        assert report.validation_status == ValidationStatus.FAILED
    
    def test_compare_models_success(self, evaluation_agent, large_model, optimized_model):
        """Test successful model comparison."""
        comparison = evaluation_agent.compare_models(large_model, optimized_model)
        
        assert isinstance(comparison, ComparisonResult)
        assert isinstance(comparison.original_metrics, PerformanceMetrics)
        assert isinstance(comparison.optimized_metrics, PerformanceMetrics)
        assert isinstance(comparison.improvements, dict)
        assert isinstance(comparison.regressions, dict)
        assert isinstance(comparison.overall_score, float)
        assert isinstance(comparison.recommendation, str)
    
    def test_validate_performance_success(self, evaluation_agent, simple_model):
        """Test performance validation with reasonable thresholds."""
        thresholds = {
            "inference_time_ms": 1000.0,  # 1 second max
            "memory_usage_mb": 1000.0,    # 1GB max
            "model_size_mb": 100.0        # 100MB max
        }
        
        result = evaluation_agent.validate_performance(simple_model, thresholds)
        
        assert isinstance(result, dict)
        assert "passed" in result
        assert "failures" in result
        assert "warnings" in result
        assert "metrics" in result
        assert isinstance(result["passed"], bool)
        assert isinstance(result["failures"], list)
        assert isinstance(result["warnings"], list)
    
    def test_validate_performance_strict_thresholds(self, evaluation_agent, simple_model):
        """Test performance validation with very strict thresholds."""
        thresholds = {
            "inference_time_ms": 0.001,   # Very strict
            "memory_usage_mb": 0.001,     # Very strict
            "model_size_mb": 0.001        # Very strict
        }
        
        result = evaluation_agent.validate_performance(simple_model, thresholds)
        
        assert isinstance(result, dict)
        assert result["passed"] is False  # Should fail with strict thresholds
        assert len(result["failures"]) > 0
    
    def test_benchmark_inference_speed(self, evaluation_agent, simple_model):
        """Test inference speed benchmark."""
        config = {"input_shape": (1, 3, 224, 224)}
        
        score, unit, higher_is_better, metadata = evaluation_agent._benchmark_inference_speed(
            simple_model, config
        )
        
        assert isinstance(score, float)
        assert score > 0
        assert unit == "ms"
        assert higher_is_better is False  # Lower is better for inference time
        assert isinstance(metadata, dict)
        assert "samples" in metadata
        assert "std_dev_ms" in metadata
    
    def test_benchmark_memory_usage(self, evaluation_agent, simple_model):
        """Test memory usage benchmark."""
        config = {"input_shape": (1, 3, 224, 224)}
        
        score, unit, higher_is_better, metadata = evaluation_agent._benchmark_memory_usage(
            simple_model, config
        )
        
        assert isinstance(score, float)
        assert score >= 0
        assert unit == "MB"
        assert higher_is_better is False  # Lower is better for memory usage
        assert isinstance(metadata, dict)
        assert "baseline_memory_mb" in metadata
        assert "peak_memory_mb" in metadata
    
    def test_benchmark_throughput(self, evaluation_agent, simple_model):
        """Test throughput benchmark."""
        config = {"input_shape": (1, 3, 224, 224)}
        
        score, unit, higher_is_better, metadata = evaluation_agent._benchmark_throughput(
            simple_model, config
        )
        
        assert isinstance(score, float)
        assert score > 0
        assert unit == "samples/sec"
        assert higher_is_better is True  # Higher is better for throughput
        assert isinstance(metadata, dict)
        assert "batch_size" in metadata
        assert "total_samples" in metadata
    
    def test_benchmark_accuracy(self, evaluation_agent, simple_model):
        """Test accuracy benchmark."""
        config = {"input_shape": (1, 3, 224, 224)}
        
        score, unit, higher_is_better, metadata = evaluation_agent._benchmark_accuracy(
            simple_model, config
        )
        
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        assert unit == "score"
        assert higher_is_better is True  # Higher is better for accuracy
        assert isinstance(metadata, dict)
        assert "method" in metadata
    
    def test_benchmark_model_size(self, evaluation_agent, simple_model):
        """Test model size benchmark."""
        config = {}
        
        score, unit, higher_is_better, metadata = evaluation_agent._benchmark_model_size(
            simple_model, config
        )
        
        assert isinstance(score, float)
        assert score > 0
        assert unit == "MB"
        assert higher_is_better is False  # Lower is better for model size
        assert isinstance(metadata, dict)
        assert "parameter_size_mb" in metadata
        assert "total_parameters" in metadata
    
    def test_benchmark_flops(self, evaluation_agent, simple_model):
        """Test FLOPs benchmark."""
        config = {"input_shape": (1, 3, 224, 224)}
        
        score, unit, higher_is_better, metadata = evaluation_agent._benchmark_flops(
            simple_model, config
        )
        
        assert isinstance(score, float)
        assert score >= 0
        assert unit == "MFLOPs"
        assert higher_is_better is False  # Lower is better for efficiency
        assert isinstance(metadata, dict)
        assert "estimation_method" in metadata
        assert "total_flops" in metadata
    
    def test_benchmark_energy_efficiency(self, evaluation_agent, simple_model):
        """Test energy efficiency benchmark."""
        config = {"input_shape": (1, 3, 224, 224)}
        
        score, unit, higher_is_better, metadata = evaluation_agent._benchmark_energy_efficiency(
            simple_model, config
        )
        
        assert isinstance(score, float)
        assert score >= 0
        assert unit == "energy_units"
        assert higher_is_better is False  # Lower is better for energy
        assert isinstance(metadata, dict)
        assert "duration_seconds" in metadata
        assert "estimation_method" in metadata
    
    def test_run_benchmark_success(self, evaluation_agent, simple_model):
        """Test running a single benchmark."""
        config = {"name": "test_inference", "type": "inference_speed", "input_shape": (1, 3, 224, 224)}
        
        result = evaluation_agent._run_benchmark(simple_model, config)
        
        assert isinstance(result, BenchmarkResult)
        assert result.benchmark_name == "test_inference"
        assert result.score > 0
        assert result.execution_time_seconds > 0
        assert isinstance(result.metadata, dict)
    
    def test_run_benchmark_invalid_type(self, evaluation_agent, simple_model):
        """Test running benchmark with invalid type."""
        config = {"name": "invalid", "type": "invalid_benchmark_type"}
        
        with pytest.raises(ValueError, match="Unknown benchmark type"):
            evaluation_agent._run_benchmark(simple_model, config)
    
    def test_get_compatible_input_with_config(self, evaluation_agent, simple_model):
        """Test getting compatible input with specified shape."""
        config = {"input_shape": (1, 3, 224, 224)}
        
        input_tensor = evaluation_agent._get_compatible_input(simple_model, config)
        
        assert isinstance(input_tensor, torch.Tensor)
        assert input_tensor.shape == (1, 3, 224, 224)
        assert input_tensor.device == evaluation_agent.device
    
    def test_get_compatible_input_auto_detect(self, evaluation_agent, simple_model):
        """Test getting compatible input with auto-detection."""
        config = {}
        
        input_tensor = evaluation_agent._get_compatible_input(simple_model, config)
        
        assert isinstance(input_tensor, torch.Tensor)
        assert input_tensor.device == evaluation_agent.device
        
        # Should be able to run forward pass
        with torch.no_grad():
            output = simple_model(input_tensor)
            assert output is not None
    
    def test_find_compatible_input(self, evaluation_agent, simple_model):
        """Test finding compatible input shape."""
        compatible_input = evaluation_agent._find_compatible_input(simple_model)
        
        assert isinstance(compatible_input, torch.Tensor)
        assert compatible_input.device == evaluation_agent.device
        
        # Should be able to run forward pass
        with torch.no_grad():
            output = simple_model(compatible_input)
            assert output is not None
    
    def test_update_performance_metrics(self, evaluation_agent):
        """Test updating performance metrics with benchmark results."""
        metrics = PerformanceMetrics()
        
        # Test inference speed update
        inference_result = BenchmarkResult(
            benchmark_name="inference_speed",
            score=50.0,
            execution_time_seconds=0.1,
            unit="ms",
            higher_is_better=False
        )
        evaluation_agent._update_performance_metrics(metrics, inference_result)
        assert metrics.inference_time_ms == 50.0
        
        # Test memory usage update
        memory_result = BenchmarkResult(
            benchmark_name="memory_usage",
            score=100.0,
            execution_time_seconds=0.1,
            unit="MB",
            higher_is_better=False
        )
        evaluation_agent._update_performance_metrics(metrics, memory_result)
        assert metrics.memory_usage_mb == 100.0
        
        # Test custom metric update
        custom_result = BenchmarkResult(
            benchmark_name="custom_metric",
            score=75.0,
            execution_time_seconds=0.1,
            unit="score",
            higher_is_better=True
        )
        evaluation_agent._update_performance_metrics(metrics, custom_result)
        assert "custom_metric" in metrics.custom_metrics
        assert metrics.custom_metrics["custom_metric"] == 75.0
    
    def test_determine_validation_status_passed(self, evaluation_agent):
        """Test validation status determination - passed case."""
        benchmark_results = [
            BenchmarkResult(benchmark_name="inference_speed", score=50.0, execution_time_seconds=0.1, unit="ms", higher_is_better=False),
            BenchmarkResult(benchmark_name="accuracy", score=0.95, execution_time_seconds=0.1, unit="score", higher_is_better=True)
        ]
        validation_errors = []
        
        status = evaluation_agent._determine_validation_status(benchmark_results, validation_errors)
        assert status == ValidationStatus.PASSED
    
    def test_determine_validation_status_failed_with_errors(self, evaluation_agent):
        """Test validation status determination - failed with errors."""
        benchmark_results = []
        validation_errors = ["Some error occurred"]
        
        status = evaluation_agent._determine_validation_status(benchmark_results, validation_errors)
        assert status == ValidationStatus.FAILED
    
    def test_determine_validation_status_failed_no_results(self, evaluation_agent):
        """Test validation status determination - failed with no results."""
        benchmark_results = []
        validation_errors = []
        
        status = evaluation_agent._determine_validation_status(benchmark_results, validation_errors)
        assert status == ValidationStatus.FAILED
    
    def test_determine_validation_status_warning(self, evaluation_agent):
        """Test validation status determination - warning case."""
        benchmark_results = [
            BenchmarkResult(benchmark_name="inference_speed", score=0.0, execution_time_seconds=0.1, unit="ms", higher_is_better=False),  # Failed critical benchmark
            BenchmarkResult(benchmark_name="memory_usage", score=100.0, execution_time_seconds=0.1, unit="MB", higher_is_better=False)
        ]
        validation_errors = []
        
        status = evaluation_agent._determine_validation_status(benchmark_results, validation_errors)
        assert status == ValidationStatus.WARNING
    
    def test_generate_evaluation_recommendations(self, evaluation_agent):
        """Test evaluation recommendation generation."""
        benchmark_results = [
            BenchmarkResult(benchmark_name="inference_speed", score=150.0, execution_time_seconds=0.1, unit="ms", higher_is_better=False),  # Slow
            BenchmarkResult(benchmark_name="memory_usage", score=1500.0, execution_time_seconds=0.1, unit="MB", higher_is_better=False),   # High memory
            BenchmarkResult(benchmark_name="accuracy", score=0.7, execution_time_seconds=0.1, unit="score", higher_is_better=True),        # Low accuracy
            BenchmarkResult(benchmark_name="model_size", score=600.0, execution_time_seconds=0.1, unit="MB", higher_is_better=False)       # Large size
        ]
        
        metrics = PerformanceMetrics()  # Use default metrics
        
        recommendations = evaluation_agent._generate_evaluation_recommendations(
            benchmark_results, metrics
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Should contain specific recommendations for detected issues
        rec_text = " ".join(recommendations).lower()
        assert "inference time" in rec_text or "optimization" in rec_text
        assert "memory" in rec_text or "compression" in rec_text
        assert "accuracy" in rec_text
        assert "size" in rec_text or "quantization" in rec_text or "pruning" in rec_text
    
    def test_calculate_overall_score_improvements(self, evaluation_agent):
        """Test overall score calculation with improvements."""
        improvements = {
            "inference_time_ms": 20.0,  # 20% improvement
            "memory_usage_mb": 15.0,    # 15% improvement
            "model_size_mb": 30.0       # 30% improvement
        }
        regressions = {}
        
        score = evaluation_agent._calculate_overall_score(improvements, regressions)
        
        assert isinstance(score, float)
        assert score > 0  # Should be positive with improvements
    
    def test_calculate_overall_score_regressions(self, evaluation_agent):
        """Test overall score calculation with regressions."""
        improvements = {}
        regressions = {
            "accuracy": 10.0,           # 10% accuracy loss
            "throughput_samples_per_sec": 5.0  # 5% throughput loss
        }
        
        score = evaluation_agent._calculate_overall_score(improvements, regressions)
        
        assert isinstance(score, float)
        assert score < 0  # Should be negative with regressions
    
    def test_calculate_overall_score_mixed(self, evaluation_agent):
        """Test overall score calculation with mixed results."""
        improvements = {
            "model_size_mb": 50.0,      # 50% size reduction
            "memory_usage_mb": 30.0     # 30% memory reduction
        }
        regressions = {
            "accuracy": 5.0             # 5% accuracy loss
        }
        
        score = evaluation_agent._calculate_overall_score(improvements, regressions)
        
        assert isinstance(score, float)
        # Should be positive overall due to significant size/memory improvements
    
    def test_generate_comparison_recommendation(self, evaluation_agent):
        """Test comparison recommendation generation."""
        # Test significant improvements
        improvements = {"inference_time_ms": 20.0}
        regressions = {}
        
        rec1 = evaluation_agent._generate_comparison_recommendation(improvements, regressions, 15.0)
        assert "significant improvements" in rec1.lower()
        assert "recommended" in rec1.lower()
        
        # Test moderate improvements
        rec2 = evaluation_agent._generate_comparison_recommendation(improvements, regressions, 7.0)
        assert "moderate improvements" in rec2.lower()
        
        # Test minor improvements
        rec3 = evaluation_agent._generate_comparison_recommendation(improvements, regressions, 2.0)
        assert "minor improvements" in rec3.lower()
        
        # Test mixed results
        rec4 = evaluation_agent._generate_comparison_recommendation(improvements, regressions, -2.0)
        assert "mixed results" in rec4.lower()
        
        # Test significant regressions
        rec5 = evaluation_agent._generate_comparison_recommendation(improvements, regressions, -10.0)
        assert "significant regressions" in rec5.lower()
        assert "not recommended" in rec5.lower()
    
    def test_check_threshold_higher_is_better(self, evaluation_agent):
        """Test threshold checking for higher-is-better metrics."""
        # Accuracy should be higher than threshold
        assert evaluation_agent._check_threshold("accuracy", 0.95, 0.90) is True
        assert evaluation_agent._check_threshold("accuracy", 0.85, 0.90) is False
        
        # Throughput should be higher than threshold
        assert evaluation_agent._check_threshold("throughput_samples_per_sec", 100.0, 50.0) is True
        assert evaluation_agent._check_threshold("throughput_samples_per_sec", 30.0, 50.0) is False
    
    def test_check_threshold_lower_is_better(self, evaluation_agent):
        """Test threshold checking for lower-is-better metrics."""
        # Inference time should be lower than threshold
        assert evaluation_agent._check_threshold("inference_time_ms", 50.0, 100.0) is True
        assert evaluation_agent._check_threshold("inference_time_ms", 150.0, 100.0) is False
        
        # Memory usage should be lower than threshold
        assert evaluation_agent._check_threshold("memory_usage_mb", 500.0, 1000.0) is True
        assert evaluation_agent._check_threshold("memory_usage_mb", 1500.0, 1000.0) is False
    
    def test_get_standard_benchmarks(self, evaluation_agent):
        """Test getting standard benchmark configurations."""
        benchmarks = evaluation_agent._get_standard_benchmarks()
        
        assert isinstance(benchmarks, list)
        assert len(benchmarks) > 0
        
        # Check that all standard benchmarks are included
        benchmark_names = [b["name"] for b in benchmarks]
        expected_benchmarks = [
            "inference_speed", "memory_usage", "throughput", 
            "accuracy", "model_size", "flops", "energy_efficiency"
        ]
        
        for expected in expected_benchmarks:
            assert expected in benchmark_names
    
    @patch('psutil.Process')
    def test_get_memory_usage_cpu(self, mock_process, evaluation_agent):
        """Test memory usage measurement on CPU."""
        # Mock CPU memory usage
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 1024 * 1024 * 100  # 100MB
        mock_process.return_value = mock_process_instance
        
        with patch('torch.cuda.is_available', return_value=False):
            memory_usage = evaluation_agent._get_memory_usage()
            
        assert isinstance(memory_usage, float)
        assert memory_usage == 100.0  # 100MB
    
    @patch('torch.cuda.memory_allocated')
    def test_get_memory_usage_gpu(self, mock_cuda_memory, evaluation_agent):
        """Test memory usage measurement on GPU."""
        mock_cuda_memory.return_value = 1024 * 1024 * 200  # 200MB
        
        with patch('torch.cuda.is_available', return_value=True):
            memory_usage = evaluation_agent._get_memory_usage()
            
        assert isinstance(memory_usage, float)
        assert memory_usage == 200.0  # 200MB


class TestEvaluationAgentIntegration:
    """Integration tests for EvaluationAgent."""
    
    def test_full_evaluation_workflow(self, evaluation_agent, simple_model):
        """Test complete evaluation workflow."""
        benchmarks = evaluation_agent._get_standard_benchmarks()
        
        report = evaluation_agent.evaluate_model(simple_model, benchmarks)
        
        # Verify all components are present and valid
        assert report.model_id is not None
        assert len(report.benchmarks) > 0
        assert report.performance_metrics is not None
        assert report.validation_status in [ValidationStatus.PASSED, ValidationStatus.WARNING, ValidationStatus.FAILED]
        assert len(report.recommendations) > 0
        assert report.evaluation_duration_seconds > 0
        
        # Verify benchmark results are reasonable
        for benchmark in report.benchmarks:
            assert benchmark.score >= 0
            assert benchmark.execution_time_seconds >= 0
            assert isinstance(benchmark.metadata, dict)
    
    def test_model_comparison_workflow(self, evaluation_agent, large_model, optimized_model):
        """Test complete model comparison workflow."""
        comparison = evaluation_agent.compare_models(large_model, optimized_model)
        
        # Verify comparison structure
        assert isinstance(comparison.original_metrics, PerformanceMetrics)
        assert isinstance(comparison.optimized_metrics, PerformanceMetrics)
        assert isinstance(comparison.improvements, dict)
        assert isinstance(comparison.regressions, dict)
        assert isinstance(comparison.overall_score, float)
        assert isinstance(comparison.recommendation, str)
        
        # Optimized model should show some improvements (likely in size)
        # Note: This might not always be true depending on the models, but it's a reasonable expectation
        total_changes = len(comparison.improvements) + len(comparison.regressions)
        assert total_changes > 0  # Should detect some differences
    
    def test_performance_validation_workflow(self, evaluation_agent, simple_model):
        """Test complete performance validation workflow."""
        # Set reasonable thresholds
        thresholds = {
            "inference_time_ms": 500.0,
            "memory_usage_mb": 500.0,
            "model_size_mb": 50.0,
            "accuracy": 0.5
        }
        
        result = evaluation_agent.validate_performance(simple_model, thresholds)
        
        assert isinstance(result, dict)
        assert "passed" in result
        assert "failures" in result
        assert "warnings" in result
        assert "metrics" in result
        
        # Should have some metrics
        assert len(result["metrics"]) > 0
    
    def test_evaluation_with_different_input_shapes(self, evaluation_agent):
        """Test evaluation with models requiring different input shapes."""
        # Test with 1D input model
        linear_model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        benchmarks = [
            {"name": "inference_speed", "type": "inference_speed", "input_shape": (1, 100)},
            {"name": "model_size", "type": "model_size"}
        ]
        
        report = evaluation_agent.evaluate_model(linear_model, benchmarks)
        
        assert isinstance(report, EvaluationReport)
        assert len(report.benchmarks) > 0
        assert report.validation_status != ValidationStatus.FAILED
    
    @patch('torch.cuda.is_available')
    def test_evaluation_cpu_only(self, mock_cuda_available):
        """Test evaluation when CUDA is not available."""
        mock_cuda_available.return_value = False
        
        config = {"benchmark_samples": 5, "warmup_samples": 1}
        agent = EvaluationAgent(config)
        agent.initialize()
        
        model = SimpleTestModel()
        benchmarks = agent._get_standard_benchmarks()[:3]  # Reduced for faster testing
        
        report = agent.evaluate_model(model, benchmarks)
        
        assert report is not None
        assert agent.device.type == "cpu"
        assert len(report.benchmarks) > 0
        
        agent.cleanup()
    
    def test_benchmark_error_handling(self, evaluation_agent):
        """Test error handling in benchmark execution."""
        # Create a model that will cause errors
        class ErrorModel(nn.Module):
            def forward(self, x):
                raise RuntimeError("Intentional error for testing")
        
        error_model = ErrorModel()
        benchmarks = [{"name": "inference_speed", "type": "inference_speed"}]
        
        report = evaluation_agent.evaluate_model(error_model, benchmarks)
        
        # Should handle errors gracefully
        assert isinstance(report, EvaluationReport)
        assert report.validation_status == ValidationStatus.FAILED
        assert len(report.validation_errors) > 0


if __name__ == "__main__":
    pytest.main([__file__])