"""
Unit tests for the DistillationAgent.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import tempfile
import os

from src.agents.optimization.distillation import (
    DistillationAgent, DistillationType, StudentArchitecture, DistillationConfig,
    DistillationLoss, FeatureDistillationLoss
)
from src.agents.base import ImpactEstimate, ValidationResult, OptimizedModel


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=784, hidden_size=1024, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)


class TestDistillationConfig:
    """Test DistillationConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = DistillationConfig(
            distillation_type=DistillationType.RESPONSE,
            student_architecture=StudentArchitecture.SMALLER_SAME
        )
        
        assert config.distillation_type == DistillationType.RESPONSE
        assert config.student_architecture == StudentArchitecture.SMALLER_SAME
        assert config.temperature == 4.0
        assert config.alpha == 0.7
        assert config.beta == 0.3
        assert config.feature_layers == []
    
    def test_invalid_alpha_beta(self):
        """Test validation of alpha and beta parameters."""
        with pytest.raises(ValueError, match="Alpha must be between 0 and 1"):
            DistillationConfig(
                distillation_type=DistillationType.RESPONSE,
                student_architecture=StudentArchitecture.SMALLER_SAME,
                alpha=1.5
            )
        
        with pytest.raises(ValueError, match="Alpha \\+ Beta must equal 1.0"):
            DistillationConfig(
                distillation_type=DistillationType.RESPONSE,
                student_architecture=StudentArchitecture.SMALLER_SAME,
                alpha=0.6,
                beta=0.5
            )


class TestDistillationLoss:
    """Test DistillationLoss class."""
    
    def test_distillation_loss_forward(self):
        """Test forward pass of distillation loss."""
        loss_fn = DistillationLoss(temperature=4.0, alpha=0.7)
        
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        targets = torch.randint(0, 10, (4,))
        
        loss = loss_fn(student_logits, teacher_logits, targets)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0
    
    def test_distillation_loss_without_targets(self):
        """Test distillation loss without hard targets."""
        loss_fn = DistillationLoss(temperature=4.0, alpha=0.7)
        
        student_logits = torch.randn(4, 10)
        teacher_logits = torch.randn(4, 10)
        
        loss = loss_fn(student_logits, teacher_logits)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0


class TestFeatureDistillationLoss:
    """Test FeatureDistillationLoss class."""
    
    def test_feature_loss_same_shape(self):
        """Test feature loss with same shape tensors."""
        loss_fn = FeatureDistillationLoss()
        
        student_features = torch.randn(4, 64)
        teacher_features = torch.randn(4, 64)
        
        loss = loss_fn(student_features, teacher_features)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_feature_loss_different_shape(self):
        """Test feature loss with different shape tensors."""
        loss_fn = FeatureDistillationLoss()
        
        student_features = torch.randn(4, 32)
        teacher_features = torch.randn(4, 64)
        
        # Should handle dimension mismatch
        loss = loss_fn(student_features, teacher_features)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0


class TestDistillationAgent:
    """Test DistillationAgent class."""
    
    @pytest.fixture
    def agent_config(self):
        """Configuration for agent."""
        return {
            'compression_ratio': 0.5,
            'preserve_modules': []
        }
    
    @pytest.fixture
    def agent(self, agent_config):
        """Create DistillationAgent instance."""
        return DistillationAgent(agent_config)
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return SimpleModel(input_size=784, hidden_size=1024, output_size=10)
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.initialize()
        assert agent.name == "DistillationAgent"
    
    def test_agent_cleanup(self, agent):
        """Test agent cleanup."""
        agent.cleanup()
        assert len(agent._student_cache) == 0
    
    def test_can_optimize_valid_model(self, agent, simple_model):
        """Test can_optimize with valid model."""
        assert agent.can_optimize(simple_model)
    
    def test_can_optimize_small_model(self, agent):
        """Test can_optimize with small model."""
        small_model = SimpleModel(input_size=10, hidden_size=5, output_size=2)
        assert not agent.can_optimize(small_model)
    
    def test_can_optimize_no_classifier(self, agent):
        """Test can_optimize with model without classifier."""
        model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU())
        assert not agent.can_optimize(model)
    
    def test_estimate_impact(self, agent, simple_model):
        """Test impact estimation."""
        impact = agent.estimate_impact(simple_model)
        
        assert isinstance(impact, ImpactEstimate)
        assert 0 <= impact.performance_improvement <= 1
        assert 0 <= impact.size_reduction <= 1
        assert 0 <= impact.speed_improvement <= 1
        assert 0 <= impact.confidence <= 1
        assert impact.estimated_time_minutes > 0
    
    def test_optimize_response_distillation(self, agent, simple_model):
        """Test response-based distillation optimization."""
        config = {
            'distillation_type': 'response',
            'student_architecture': 'smaller_same',
            'num_epochs': 2,  # Small number for testing
            'batch_size': 4
        }
        
        result = agent.optimize(simple_model, config)
        
        assert isinstance(result, OptimizedModel)
        assert result.model is not None
        assert result.technique_used == "distillation_response"
        assert result.optimization_time > 0
        assert 'compression_ratio' in result.performance_metrics
    
    def test_optimize_feature_distillation(self, agent, simple_model):
        """Test feature-based distillation optimization."""
        config = {
            'distillation_type': 'feature',
            'student_architecture': 'smaller_same',
            'num_epochs': 2,
            'batch_size': 4
        }
        
        result = agent.optimize(simple_model, config)
        
        assert isinstance(result, OptimizedModel)
        assert result.technique_used == "distillation_feature"
    
    def test_optimize_with_progress_tracking(self, agent, simple_model):
        """Test optimization with progress tracking."""
        progress_updates = []
        
        def progress_callback(update):
            progress_updates.append(update)
        
        agent.add_progress_callback(progress_callback)
        
        config = {
            'distillation_type': 'response',
            'num_epochs': 2,
            'batch_size': 4
        }
        
        result = agent.optimize_with_tracking(simple_model, config)
        
        assert result.success
        assert len(progress_updates) > 0
        assert any(update.status.value == "completed" for update in progress_updates)
    
    def test_validate_result_valid(self, agent, simple_model):
        """Test validation with valid result."""
        # Create a smaller student model
        student_model = SimpleModel(input_size=784, hidden_size=512, output_size=10)
        
        validation = agent.validate_result(simple_model, student_model)
        
        assert isinstance(validation, ValidationResult)
        assert validation.is_valid
        assert 'compression_ratio' in validation.performance_metrics
    
    def test_validate_result_shape_mismatch(self, agent, simple_model):
        """Test validation with shape mismatch."""
        # Create model with different output size
        wrong_model = SimpleModel(input_size=784, hidden_size=1024, output_size=5)
        
        validation = agent.validate_result(simple_model, wrong_model)
        
        assert not validation.is_valid
        assert any("shape mismatch" in issue.lower() for issue in validation.issues)
    
    def test_validate_result_nan_output(self, agent, simple_model):
        """Test validation with NaN output."""
        # Create model that produces NaN
        nan_model = SimpleModel()
        
        # Corrupt weights to produce NaN
        with torch.no_grad():
            for param in nan_model.parameters():
                param.fill_(float('nan'))
        
        validation = agent.validate_result(simple_model, nan_model)
        
        assert not validation.is_valid
        assert not validation.accuracy_preserved
        assert any("nan" in issue.lower() for issue in validation.issues)
    
    def test_get_supported_techniques(self, agent):
        """Test getting supported techniques."""
        techniques = agent.get_supported_techniques()
        
        assert isinstance(techniques, list)
        assert "response" in techniques
        assert "feature" in techniques
        assert "attention" in techniques
        assert "progressive" in techniques
    
    def test_parse_config_defaults(self, agent):
        """Test config parsing with defaults."""
        config = {}
        parsed = agent._parse_config(config)
        
        assert parsed.distillation_type == DistillationType.RESPONSE
        assert parsed.student_architecture == StudentArchitecture.SMALLER_SAME
        assert parsed.temperature == 4.0
    
    def test_parse_config_custom(self, agent):
        """Test config parsing with custom values."""
        config = {
            'distillation_type': 'feature',
            'student_architecture': 'efficient',
            'temperature': 6.0,
            'alpha': 0.8,
            'beta': 0.2,
            'num_epochs': 20
        }
        
        parsed = agent._parse_config(config)
        
        assert parsed.distillation_type == DistillationType.FEATURE
        assert parsed.student_architecture == StudentArchitecture.EFFICIENT
        assert parsed.temperature == 6.0
        assert parsed.alpha == 0.8
        assert parsed.beta == 0.2
        assert parsed.num_epochs == 20
    
    def test_parse_config_invalid_values(self, agent):
        """Test config parsing with invalid values."""
        config = {
            'distillation_type': 'invalid_type',
            'student_architecture': 'invalid_arch'
        }
        
        parsed = agent._parse_config(config)
        
        # Should fall back to defaults
        assert parsed.distillation_type == DistillationType.RESPONSE
        assert parsed.student_architecture == StudentArchitecture.SMALLER_SAME
    
    def test_create_smaller_same_architecture(self, agent, simple_model):
        """Test creating smaller same architecture."""
        config = DistillationConfig(
            distillation_type=DistillationType.RESPONSE,
            student_architecture=StudentArchitecture.SMALLER_SAME,
            compression_ratio=0.5
        )
        
        student = agent._create_student_model(simple_model, config)
        
        assert isinstance(student, nn.Module)
        
        # Check that student has fewer parameters
        teacher_params = sum(p.numel() for p in simple_model.parameters())
        student_params = sum(p.numel() for p in student.parameters())
        assert student_params < teacher_params
    
    def test_create_dummy_input(self, agent, simple_model):
        """Test creating dummy input."""
        dummy_input = agent._create_dummy_input(simple_model)
        
        assert isinstance(dummy_input, torch.Tensor)
        assert dummy_input.shape[0] == 1  # Batch size 1
        assert dummy_input.shape[1] == 784  # Input size
    
    def test_optimization_cancellation(self, agent, simple_model):
        """Test optimization cancellation."""
        config = {
            'distillation_type': 'response',
            'num_epochs': 10,
            'batch_size': 4
        }
        
        # Cancel immediately
        agent.cancel_optimization()
        
        result = agent.optimize_with_tracking(simple_model, config)
        
        assert not result.success
        assert "cancelled" in result.error_message.lower()
    
    def test_optimization_with_snapshots(self, agent, simple_model):
        """Test optimization with snapshot creation."""
        config = {
            'distillation_type': 'response',
            'num_epochs': 2,
            'batch_size': 4
        }
        
        result = agent.optimize_with_tracking(simple_model, config)
        
        assert result.success
        assert len(result.snapshots) > 0
        assert any("original" in snapshot.checkpoint_name for snapshot in result.snapshots)


class TestDistillationIntegration:
    """Integration tests for distillation agent."""
    
    def test_end_to_end_distillation(self):
        """Test complete distillation workflow."""
        # Create teacher model
        teacher = SimpleModel(input_size=100, hidden_size=64, output_size=10)
        
        # Create agent
        agent = DistillationAgent({'compression_ratio': 0.6})
        agent.initialize()
        
        # Run distillation
        config = {
            'distillation_type': 'response',
            'student_architecture': 'smaller_same',
            'num_epochs': 3,
            'batch_size': 8,
            'temperature': 5.0,
            'alpha': 0.8,
            'beta': 0.2
        }
        
        result = agent.optimize_with_tracking(teacher, config)
        
        # Verify results
        assert result.success
        assert result.optimized_model is not None
        
        # Check compression
        teacher_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in result.optimized_model.parameters())
        compression_ratio = (teacher_params - student_params) / teacher_params
        assert compression_ratio > 0.1  # At least 10% compression
        
        # Validate result
        validation = agent.validate_result(teacher, result.optimized_model)
        assert validation.is_valid
        
        agent.cleanup()
    
    def test_distillation_with_different_architectures(self):
        """Test distillation with different student architectures."""
        teacher = SimpleModel(input_size=50, hidden_size=32, output_size=5)
        agent = DistillationAgent({})
        agent.initialize()
        
        architectures = ['smaller_same', 'efficient']
        
        for arch in architectures:
            config = {
                'distillation_type': 'response',
                'student_architecture': arch,
                'num_epochs': 2,
                'batch_size': 4
            }
            
            result = agent.optimize_with_tracking(teacher, config)
            assert result.success, f"Failed for architecture: {arch}"
        
        agent.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])