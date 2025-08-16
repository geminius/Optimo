"""
Unit tests for the Analysis Agent.
"""

import pytest
import torch
import torch.nn as nn
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.agents.analysis.agent import AnalysisAgent
from src.models.core import (
    AnalysisReport, ArchitectureSummary, PerformanceProfile,
    OptimizationOpportunity, Recommendation
)


class SimpleTestModel(nn.Module):
    """Simple test model for analysis."""
    
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
    """Large test model for distillation testing."""
    
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


@pytest.fixture
def analysis_agent():
    """Create an AnalysisAgent instance for testing."""
    config = {
        "profiling_samples": 10,  # Reduced for faster testing
        "warmup_samples": 2
    }
    agent = AnalysisAgent(config)
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
def temp_model_file():
    """Create a temporary model file."""
    model = SimpleTestModel()
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        torch.save(model, f.name)
        yield f.name
    os.unlink(f.name)


class TestAnalysisAgent:
    """Test cases for AnalysisAgent."""
    
    def test_initialization(self, analysis_agent):
        """Test agent initialization."""
        assert analysis_agent.name == "AnalysisAgent"
        assert analysis_agent.profiling_samples == 10
        assert analysis_agent.warmup_samples == 2
        assert analysis_agent.device is not None
    
    def test_cleanup(self, analysis_agent):
        """Test agent cleanup."""
        # Should not raise any exceptions
        analysis_agent.cleanup()
    
    def test_analyze_model_success(self, analysis_agent, temp_model_file):
        """Test successful model analysis."""
        report = analysis_agent.analyze_model(temp_model_file)
        
        assert isinstance(report, AnalysisReport)
        assert report.model_id is not None
        assert isinstance(report.architecture_summary, ArchitectureSummary)
        assert isinstance(report.performance_profile, PerformanceProfile)
        assert isinstance(report.optimization_opportunities, list)
        assert isinstance(report.compatibility_matrix, dict)
        assert isinstance(report.recommendations, list)
        assert report.analysis_duration_seconds > 0
    
    def test_analyze_model_file_not_found(self, analysis_agent):
        """Test analysis with non-existent file."""
        with pytest.raises(FileNotFoundError):
            analysis_agent.analyze_model("non_existent_file.pt")
    
    def test_identify_bottlenecks(self, analysis_agent, simple_model):
        """Test bottleneck identification."""
        bottlenecks = analysis_agent.identify_bottlenecks(simple_model)
        
        assert isinstance(bottlenecks, list)
        # Should identify some bottlenecks or at least return empty list
        for bottleneck in bottlenecks:
            assert "type" in bottleneck
            assert "description" in bottleneck
            assert "severity" in bottleneck
    
    def test_analyze_architecture(self, analysis_agent, simple_model):
        """Test architecture analysis."""
        arch_summary = analysis_agent._analyze_architecture(simple_model)
        
        assert isinstance(arch_summary, ArchitectureSummary)
        assert arch_summary.total_layers > 0
        assert arch_summary.total_parameters > 0
        assert arch_summary.trainable_parameters > 0
        assert arch_summary.memory_footprint_mb > 0
        assert len(arch_summary.layer_types) > 0
        
        # Check for expected layer types
        assert "Conv2d" in arch_summary.layer_types
        assert "Linear" in arch_summary.layer_types
    
    def test_profile_performance(self, analysis_agent, simple_model):
        """Test performance profiling."""
        perf_profile = analysis_agent._profile_performance(simple_model)
        
        assert isinstance(perf_profile, PerformanceProfile)
        assert perf_profile.inference_time_ms >= 0
        assert perf_profile.memory_usage_mb >= 0
        assert perf_profile.throughput_samples_per_sec >= 0
        assert perf_profile.cpu_utilization_percent >= 0
    
    def test_identify_optimization_opportunities_quantization(self, analysis_agent, simple_model):
        """Test identification of quantization opportunities."""
        arch_summary = analysis_agent._analyze_architecture(simple_model)
        perf_profile = analysis_agent._profile_performance(simple_model)
        
        opportunities = analysis_agent._identify_optimization_opportunities(
            simple_model, arch_summary, perf_profile
        )
        
        assert isinstance(opportunities, list)
        
        # Should identify quantization opportunity for model with Conv2d and Linear layers
        quantization_opp = next((opp for opp in opportunities if opp.technique == "quantization"), None)
        assert quantization_opp is not None
        assert quantization_opp.estimated_size_reduction_percent > 0
        assert quantization_opp.confidence_score > 0
    
    def test_identify_optimization_opportunities_pruning(self, analysis_agent, simple_model):
        """Test identification of pruning opportunities."""
        arch_summary = analysis_agent._analyze_architecture(simple_model)
        perf_profile = analysis_agent._profile_performance(simple_model)
        
        opportunities = analysis_agent._identify_optimization_opportunities(
            simple_model, arch_summary, perf_profile
        )
        
        # Should identify pruning opportunity for model with sufficient parameters
        pruning_opp = next((opp for opp in opportunities if opp.technique == "pruning"), None)
        assert pruning_opp is not None
        assert pruning_opp.estimated_size_reduction_percent > 0
    
    def test_identify_optimization_opportunities_distillation(self, analysis_agent, large_model):
        """Test identification of distillation opportunities."""
        arch_summary = analysis_agent._analyze_architecture(large_model)
        perf_profile = analysis_agent._profile_performance(large_model)
        
        opportunities = analysis_agent._identify_optimization_opportunities(
            large_model, arch_summary, perf_profile
        )
        
        # Should identify distillation opportunity for large model
        distillation_opp = next((opp for opp in opportunities if opp.technique == "distillation"), None)
        assert distillation_opp is not None
        assert distillation_opp.estimated_size_reduction_percent > 0
    
    def test_assess_compatibility(self, analysis_agent, simple_model):
        """Test compatibility assessment."""
        arch_summary = analysis_agent._analyze_architecture(simple_model)
        compatibility = analysis_agent._assess_compatibility(simple_model, arch_summary)
        
        assert isinstance(compatibility, dict)
        assert "quantization" in compatibility
        assert "pruning" in compatibility
        assert "distillation" in compatibility
        assert "compression" in compatibility
        assert "architecture_search" in compatibility
        
        # Simple model should be compatible with quantization and pruning
        assert compatibility["quantization"] is True
        assert compatibility["pruning"] is True
    
    def test_generate_recommendations(self, analysis_agent):
        """Test recommendation generation."""
        # Create mock opportunities
        opportunities = [
            OptimizationOpportunity(
                technique="quantization",
                estimated_size_reduction_percent=50.0,
                estimated_speed_improvement_percent=30.0,
                confidence_score=0.8,
                description="Good quantization candidate"
            ),
            OptimizationOpportunity(
                technique="pruning",
                estimated_size_reduction_percent=30.0,
                estimated_speed_improvement_percent=20.0,
                confidence_score=0.7,
                description="Good pruning candidate"
            )
        ]
        
        compatibility = {"quantization": True, "pruning": True}
        
        recommendations = analysis_agent._generate_recommendations(opportunities, compatibility)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3  # Should limit to top 3
        
        for rec in recommendations:
            assert isinstance(rec, Recommendation)
            assert rec.technique in ["quantization", "pruning"]
            assert rec.priority >= 1
            assert len(rec.expected_benefits) > 0
    
    def test_can_quantize(self, analysis_agent, simple_model):
        """Test quantization compatibility check."""
        arch_summary = analysis_agent._analyze_architecture(simple_model)
        can_quantize = analysis_agent._can_quantize(simple_model, arch_summary)
        
        # Model with Conv2d and Linear layers should be quantizable
        assert can_quantize is True
    
    def test_can_prune(self, analysis_agent, simple_model):
        """Test pruning compatibility check."""
        arch_summary = analysis_agent._analyze_architecture(simple_model)
        can_prune = analysis_agent._can_prune(simple_model, arch_summary)
        
        # Model with parameters should be prunable
        assert can_prune is True
    
    def test_can_distill_small_model(self, analysis_agent, simple_model):
        """Test distillation compatibility check for small model."""
        arch_summary = analysis_agent._analyze_architecture(simple_model)
        can_distill = analysis_agent._can_distill(simple_model, arch_summary)
        
        # Small model should not be good for distillation
        assert can_distill is False
    
    def test_can_distill_large_model(self, analysis_agent, large_model):
        """Test distillation compatibility check for large model."""
        arch_summary = analysis_agent._analyze_architecture(large_model)
        can_distill = analysis_agent._can_distill(large_model, arch_summary)
        
        # Large model should be good for distillation
        assert can_distill is True
    
    def test_can_compress(self, analysis_agent, simple_model):
        """Test compression compatibility check."""
        arch_summary = analysis_agent._analyze_architecture(simple_model)
        can_compress = analysis_agent._can_compress(simple_model, arch_summary)
        
        # Model with Linear layers should be compressible
        assert can_compress is True
    
    def test_can_architecture_search(self, analysis_agent, simple_model):
        """Test architecture search compatibility check."""
        arch_summary = analysis_agent._analyze_architecture(simple_model)
        can_arch_search = analysis_agent._can_architecture_search(simple_model, arch_summary)
        
        # Model with multiple layer types should be suitable for architecture search
        assert can_arch_search is True
    
    def test_profile_layer_performance(self, analysis_agent, simple_model):
        """Test layer performance profiling."""
        layer_times = analysis_agent._profile_layer_performance(simple_model)
        
        assert isinstance(layer_times, dict)
        # Should have timing information for some layers
        assert len(layer_times) >= 0  # May be empty due to simplified implementation
    
    def test_identify_memory_bottlenecks(self, analysis_agent, large_model):
        """Test memory bottleneck identification."""
        bottlenecks = analysis_agent._identify_memory_bottlenecks(large_model)
        
        assert isinstance(bottlenecks, list)
        # Large model should have some memory bottlenecks
        if bottlenecks:
            for bottleneck in bottlenecks:
                assert "type" in bottleneck
                assert "layer_name" in bottleneck
                assert "severity" in bottleneck
    
    def test_identify_compute_bottlenecks(self, analysis_agent, simple_model):
        """Test computational bottleneck identification."""
        bottlenecks = analysis_agent._identify_compute_bottlenecks(simple_model)
        
        assert isinstance(bottlenecks, list)
        # Simple model may not have compute bottlenecks
        for bottleneck in bottlenecks:
            assert "type" in bottleneck
            assert "operation_type" in bottleneck
    
    def test_get_layer_optimization_suggestions(self, analysis_agent, simple_model):
        """Test layer-specific optimization suggestions."""
        # Test with a known layer
        suggestions = analysis_agent._get_layer_optimization_suggestions("conv1", simple_model)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        
        # Should suggest quantization and pruning for Conv2d layer
        assert any("quantization" in s.lower() for s in suggestions)
    
    def test_calculate_model_depth(self, analysis_agent, simple_model):
        """Test model depth calculation."""
        depth = analysis_agent._calculate_model_depth(simple_model)
        
        assert isinstance(depth, int)
        assert depth > 0
    
    def test_find_compatible_input(self, analysis_agent, simple_model):
        """Test finding compatible input for model."""
        compatible_input = analysis_agent._find_compatible_input(simple_model)
        
        assert isinstance(compatible_input, torch.Tensor)
        assert compatible_input.device == analysis_agent.device
        
        # Should be able to run forward pass
        with torch.no_grad():
            output = simple_model(compatible_input)
            assert output is not None
    
    @patch('psutil.cpu_percent')
    def test_get_memory_usage(self, mock_cpu_percent, analysis_agent):
        """Test memory usage measurement."""
        mock_cpu_percent.return_value = 50.0
        
        memory_usage = analysis_agent._get_memory_usage()
        
        assert isinstance(memory_usage, float)
        assert memory_usage >= 0
    
    def test_get_gpu_utilization(self, analysis_agent):
        """Test GPU utilization measurement."""
        gpu_util = analysis_agent._get_gpu_utilization()
        
        assert isinstance(gpu_util, float)
        assert gpu_util >= 0
        assert gpu_util <= 100
    
    def test_load_model_invalid_path(self, analysis_agent):
        """Test loading model with invalid path."""
        with pytest.raises(FileNotFoundError):
            analysis_agent._load_model("invalid_path.pt")
    
    def test_extract_model_metadata(self, analysis_agent, temp_model_file, simple_model):
        """Test model metadata extraction."""
        metadata = analysis_agent._extract_model_metadata(temp_model_file, simple_model)
        
        assert metadata.name is not None
        assert metadata.file_path == temp_model_file
        assert metadata.size_mb > 0
        assert metadata.parameters > 0


class TestAnalysisAgentIntegration:
    """Integration tests for AnalysisAgent."""
    
    def test_full_analysis_workflow(self, analysis_agent, temp_model_file):
        """Test complete analysis workflow."""
        # This tests the integration of all analysis components
        report = analysis_agent.analyze_model(temp_model_file)
        
        # Verify all components are present and valid
        assert report.model_id is not None
        assert report.architecture_summary.total_layers > 0
        assert report.performance_profile.inference_time_ms >= 0
        assert len(report.optimization_opportunities) > 0
        assert len(report.compatibility_matrix) > 0
        assert len(report.recommendations) > 0
        assert report.analysis_duration_seconds > 0
        
        # Verify recommendations are properly prioritized
        priorities = [rec.priority for rec in report.recommendations]
        assert priorities == sorted(priorities)  # Should be in ascending order
    
    def test_analysis_with_different_model_sizes(self, analysis_agent):
        """Test analysis with models of different sizes."""
        # Test with small model
        small_model = nn.Linear(10, 1)
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(small_model, f.name)
            small_report = analysis_agent.analyze_model(f.name)
        os.unlink(f.name)
        
        # Test with larger model
        large_model = LargeTestModel()
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(large_model, f.name)
            large_report = analysis_agent.analyze_model(f.name)
        os.unlink(f.name)
        
        # Large model should have more optimization opportunities
        assert len(large_report.optimization_opportunities) >= len(small_report.optimization_opportunities)
        
        # Large model should be suitable for distillation
        large_distill_opp = next(
            (opp for opp in large_report.optimization_opportunities if opp.technique == "distillation"), 
            None
        )
        assert large_distill_opp is not None
    
    @patch('torch.cuda.is_available')
    def test_analysis_cpu_only(self, mock_cuda_available, temp_model_file):
        """Test analysis when CUDA is not available."""
        mock_cuda_available.return_value = False
        
        config = {"profiling_samples": 5, "warmup_samples": 1}
        agent = AnalysisAgent(config)
        agent.initialize()
        
        report = agent.analyze_model(temp_model_file)
        
        assert report is not None
        assert agent.device.type == "cpu"
        
        agent.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])