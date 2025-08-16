"""
Unit tests for the CompressionAgent.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import tempfile
import os
import math

from src.agents.optimization.compression import (
    CompressionAgent, CompressionType, DecompositionTarget, CompressionConfig,
    SVDLinear, LowRankLinear, TuckerConv2d
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


class ConvModel(nn.Module):
    """Simple convolutional model for testing."""
    
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(32, 10)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class TestCompressionConfig:
    """Test CompressionConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = CompressionConfig(
            compression_type=CompressionType.SVD
        )
        
        assert config.compression_type == CompressionType.SVD
        assert config.target_layers == DecompositionTarget.ALL
        assert config.compression_ratio == 0.5
        assert config.rank_ratio == 0.5
        assert config.preserve_modules == ['classifier', 'head', 'lm_head', 'embed_tokens']
        assert config.adaptive_rank == True
    
    def test_invalid_compression_ratio(self):
        """Test validation of compression ratio."""
        with pytest.raises(ValueError, match="Compression ratio must be between 0 and 1"):
            CompressionConfig(
                compression_type=CompressionType.SVD,
                compression_ratio=1.5
            )
        
        with pytest.raises(ValueError, match="Compression ratio must be between 0 and 1"):
            CompressionConfig(
                compression_type=CompressionType.SVD,
                compression_ratio=-0.1
            )
    
    def test_invalid_rank_ratio(self):
        """Test validation of rank ratio."""
        with pytest.raises(ValueError, match="Rank ratio must be between 0 and 1"):
            CompressionConfig(
                compression_type=CompressionType.SVD,
                rank_ratio=1.5
            )


class TestSVDLinear:
    """Test SVDLinear class."""
    
    def test_svd_linear_creation(self):
        """Test creating SVD linear layer."""
        original = nn.Linear(100, 50, bias=True)
        rank = 25
        
        svd_layer = SVDLinear(original, rank)
        
        assert svd_layer.rank == rank
        assert svd_layer.U.shape == (50, rank)
        assert svd_layer.S.shape == (rank,)
        assert svd_layer.Vt.shape == (rank, 100)
        assert svd_layer.bias is not None
    
    def test_svd_linear_forward(self):
        """Test forward pass of SVD linear layer."""
        original = nn.Linear(20, 10, bias=True)
        svd_layer = SVDLinear(original, rank=5)
        
        x = torch.randn(4, 20)
        output = svd_layer(x)
        
        assert output.shape == (4, 10)
    
    def test_svd_linear_no_bias(self):
        """Test SVD linear layer without bias."""
        original = nn.Linear(20, 10, bias=False)
        svd_layer = SVDLinear(original, rank=5)
        
        assert svd_layer.bias is None
        
        x = torch.randn(4, 20)
        output = svd_layer(x)
        assert output.shape == (4, 10)
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        original = nn.Linear(100, 50, bias=True)
        original_params = sum(p.numel() for p in original.parameters())
        
        svd_layer = SVDLinear(original, rank=25)
        compression_ratio = svd_layer.get_compression_ratio(original_params)
        
        assert 0 < compression_ratio < 1


class TestLowRankLinear:
    """Test LowRankLinear class."""
    
    def test_low_rank_linear_creation(self):
        """Test creating low-rank linear layer."""
        layer = LowRankLinear(100, 50, rank=25, bias=True)
        
        assert layer.rank == 25
        assert layer.A.shape == (50, 25)
        assert layer.B.shape == (25, 100)
        assert layer.bias is not None
    
    def test_low_rank_linear_forward(self):
        """Test forward pass of low-rank linear layer."""
        layer = LowRankLinear(20, 10, rank=5, bias=True)
        
        x = torch.randn(4, 20)
        output = layer(x)
        
        assert output.shape == (4, 10)
    
    def test_from_linear(self):
        """Test creating low-rank layer from existing linear layer."""
        original = nn.Linear(50, 30, bias=True)
        lr_layer = LowRankLinear.from_linear(original, rank=15)
        
        assert lr_layer.rank == 15
        assert lr_layer.A.shape == (30, 15)
        assert lr_layer.B.shape == (15, 50)
        assert lr_layer.bias is not None
        
        # Test forward pass
        x = torch.randn(2, 50)
        output = lr_layer(x)
        assert output.shape == (2, 30)


class TestTuckerConv2d:
    """Test TuckerConv2d class."""
    
    def test_tucker_conv2d_creation(self):
        """Test creating Tucker Conv2d layer."""
        original = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        ranks = (16, 8, 2, 2)  # (out_ch, in_ch, kh, kw)
        
        tucker_layer = TuckerConv2d(original, ranks)
        
        assert tucker_layer.core.shape == ranks
        assert tucker_layer.U1.shape == (32, 16)  # out_channels, r1
        assert tucker_layer.U2.shape == (16, 8)   # in_channels, r2
        assert tucker_layer.U3.shape == (3, 2)    # kh, r3
        assert tucker_layer.U4.shape == (3, 2)    # kw, r4
    
    def test_tucker_conv2d_forward(self):
        """Test forward pass of Tucker Conv2d layer."""
        original = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        ranks = (8, 4, 2, 2)
        
        tucker_layer = TuckerConv2d(original, ranks)
        
        x = torch.randn(2, 8, 32, 32)
        output = tucker_layer(x)
        
        assert output.shape == (2, 16, 32, 32)
    
    def test_tucker_conv2d_with_bias(self):
        """Test Tucker Conv2d layer with bias."""
        original = nn.Conv2d(4, 8, kernel_size=3, padding=1, bias=True)
        ranks = (4, 2, 2, 2)
        
        tucker_layer = TuckerConv2d(original, ranks)
        
        assert tucker_layer.bias is not None
        assert tucker_layer.bias.shape == (8,)


class TestCompressionAgent:
    """Test CompressionAgent class."""
    
    @pytest.fixture
    def agent_config(self):
        """Configuration for agent."""
        return {
            'compression_ratio': 0.6,
            'preserve_modules': []
        }
    
    @pytest.fixture
    def agent(self, agent_config):
        """Create CompressionAgent instance."""
        return CompressionAgent(agent_config)
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return SimpleModel(input_size=784, hidden_size=1024, output_size=10)
    
    @pytest.fixture
    def conv_model(self):
        """Create convolutional model for testing."""
        return ConvModel()
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.initialize()
        assert agent.name == "CompressionAgent"
    
    def test_agent_cleanup(self, agent):
        """Test agent cleanup."""
        agent.cleanup()
        # No specific state to check for compression agent
    
    def test_can_optimize_valid_model(self, agent, simple_model):
        """Test can_optimize with valid model."""
        assert agent.can_optimize(simple_model)
    
    def test_can_optimize_small_model(self, agent):
        """Test can_optimize with small model."""
        small_model = SimpleModel(input_size=10, hidden_size=5, output_size=2)
        assert not agent.can_optimize(small_model)
    
    def test_can_optimize_no_compressible_layers(self, agent):
        """Test can_optimize with model without compressible layers."""
        model = nn.Sequential(nn.ReLU(), nn.Dropout(0.5))
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
    
    def test_optimize_svd_compression(self, agent, simple_model):
        """Test SVD compression optimization."""
        config = {
            'compression_type': 'svd',
            'target_layers': 'linear',
            'compression_ratio': 0.5,
            'rank_ratio': 0.6
        }
        
        result = agent.optimize(simple_model, config)
        
        assert isinstance(result, OptimizedModel)
        assert result.model is not None
        assert result.technique_used == "compression_svd"
        assert result.optimization_time > 0
        assert 'compression_ratio' in result.performance_metrics
    
    def test_optimize_low_rank_compression(self, agent, simple_model):
        """Test low-rank compression optimization."""
        config = {
            'compression_type': 'low_rank',
            'target_layers': 'linear',
            'rank_ratio': 0.5
        }
        
        result = agent.optimize(simple_model, config)
        
        assert isinstance(result, OptimizedModel)
        assert result.technique_used == "compression_low_rank"
    
    def test_optimize_tucker_compression(self, agent, conv_model):
        """Test Tucker compression optimization."""
        config = {
            'compression_type': 'tucker',
            'target_layers': 'conv2d',
            'rank_ratio': 0.5
        }
        
        result = agent.optimize(conv_model, config)
        
        assert isinstance(result, OptimizedModel)
        assert result.technique_used == "compression_tucker"
    
    def test_optimize_mixed_compression(self, agent, conv_model):
        """Test mixed compression optimization."""
        config = {
            'compression_type': 'mixed',
            'target_layers': 'all',
            'rank_ratio': 0.6
        }
        
        result = agent.optimize(conv_model, config)
        
        assert isinstance(result, OptimizedModel)
        assert result.technique_used == "compression_mixed"
    
    def test_optimize_with_progress_tracking(self, agent, simple_model):
        """Test optimization with progress tracking."""
        progress_updates = []
        
        def progress_callback(update):
            progress_updates.append(update)
        
        agent.add_progress_callback(progress_callback)
        
        config = {
            'compression_type': 'svd',
            'rank_ratio': 0.5
        }
        
        result = agent.optimize_with_tracking(simple_model, config)
        
        assert result.success
        assert len(progress_updates) > 0
        assert any(update.status.value == "completed" for update in progress_updates)
    
    def test_validate_result_valid(self, agent, simple_model):
        """Test validation with valid result."""
        # Create compressed model manually
        config = {
            'compression_type': 'svd',
            'rank_ratio': 0.5
        }
        result = agent.optimize(simple_model, config)
        
        validation = agent.validate_result(simple_model, result.model)
        
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
    
    def test_validate_result_no_compression(self, agent, simple_model):
        """Test validation with no compression achieved."""
        # Use the same model (no compression)
        validation = agent.validate_result(simple_model, simple_model)
        
        assert not validation.is_valid
        assert any("insufficient compression" in issue.lower() for issue in validation.issues)
    
    def test_get_supported_techniques(self, agent):
        """Test getting supported techniques."""
        techniques = agent.get_supported_techniques()
        
        assert isinstance(techniques, list)
        assert "svd" in techniques
        assert "low_rank" in techniques
        assert "tucker" in techniques
        assert "mixed" in techniques
    
    def test_parse_config_defaults(self, agent):
        """Test config parsing with defaults."""
        config = {}
        parsed = agent._parse_config(config)
        
        assert parsed.compression_type == CompressionType.SVD
        assert parsed.target_layers == DecompositionTarget.ALL
        assert parsed.compression_ratio == 0.5
    
    def test_parse_config_custom(self, agent):
        """Test config parsing with custom values."""
        config = {
            'compression_type': 'tucker',
            'target_layers': 'conv2d',
            'compression_ratio': 0.7,
            'rank_ratio': 0.4,
            'svd_threshold': 1e-4,
            'adaptive_rank': False
        }
        
        parsed = agent._parse_config(config)
        
        assert parsed.compression_type == CompressionType.TUCKER
        assert parsed.target_layers == DecompositionTarget.CONV2D
        assert parsed.compression_ratio == 0.7
        assert parsed.rank_ratio == 0.4
        assert parsed.svd_threshold == 1e-4
        assert parsed.adaptive_rank == False
    
    def test_parse_config_invalid_values(self, agent):
        """Test config parsing with invalid values."""
        config = {
            'compression_type': 'invalid_type',
            'target_layers': 'invalid_target'
        }
        
        parsed = agent._parse_config(config)
        
        # Should fall back to defaults
        assert parsed.compression_type == CompressionType.SVD
        assert parsed.target_layers == DecompositionTarget.ALL
    
    def test_should_compress_layer(self, agent):
        """Test layer compression decision."""
        config = CompressionConfig(
            compression_type=CompressionType.SVD,
            target_layers=DecompositionTarget.LINEAR,
            preserve_modules=['classifier']
        )
        
        linear_layer = nn.Linear(10, 5)
        conv_layer = nn.Conv2d(3, 16, 3)
        
        # Should compress linear layer
        assert agent._should_compress_layer("layer1", linear_layer, config)
        
        # Should not compress conv layer when target is linear
        assert not agent._should_compress_layer("layer2", conv_layer, config)
        
        # Should not compress preserved module
        assert not agent._should_compress_layer("classifier", linear_layer, config)
    
    def test_determine_optimal_rank_svd(self, agent):
        """Test optimal rank determination for SVD."""
        layer = nn.Linear(100, 50)
        config = CompressionConfig(
            compression_type=CompressionType.SVD,
            compression_ratio=0.5,
            rank_ratio=0.8
        )
        
        rank = agent._determine_optimal_rank_svd(layer, config)
        
        assert isinstance(rank, int)
        assert 1 <= rank <= min(layer.weight.shape)
    
    def test_count_compressed_layers(self, agent, simple_model):
        """Test counting compressed layers."""
        # Initially no compressed layers
        count = agent._count_compressed_layers(simple_model)
        assert count == 0
        
        # Compress the model
        config = {
            'compression_type': 'svd',
            'rank_ratio': 0.5
        }
        result = agent.optimize(simple_model, config)
        
        # Should have compressed layers now
        count = agent._count_compressed_layers(result.model)
        assert count > 0
    
    def test_create_dummy_input(self, agent, simple_model):
        """Test creating dummy input."""
        dummy_input = agent._create_dummy_input(simple_model)
        
        assert isinstance(dummy_input, torch.Tensor)
        assert dummy_input.shape[0] == 1  # Batch size 1
        assert dummy_input.shape[1] == 784  # Input size
    
    def test_create_dummy_input_conv(self, agent, conv_model):
        """Test creating dummy input for conv model."""
        dummy_input = agent._create_dummy_input(conv_model)
        
        assert isinstance(dummy_input, torch.Tensor)
        assert dummy_input.shape[0] == 1  # Batch size 1
        assert dummy_input.shape[1] == 3   # Input channels
    
    def test_optimization_cancellation(self, agent, simple_model):
        """Test optimization cancellation."""
        config = {
            'compression_type': 'svd',
            'rank_ratio': 0.5
        }
        
        # Cancel immediately
        agent.cancel_optimization()
        
        result = agent.optimize_with_tracking(simple_model, config)
        
        assert not result.success
        assert "cancelled" in result.error_message.lower()
    
    def test_optimization_with_snapshots(self, agent, simple_model):
        """Test optimization with snapshot creation."""
        config = {
            'compression_type': 'svd',
            'rank_ratio': 0.5
        }
        
        result = agent.optimize_with_tracking(simple_model, config)
        
        assert result.success
        assert len(result.snapshots) > 0
        assert any("original" in snapshot.checkpoint_name for snapshot in result.snapshots)


class TestCompressionIntegration:
    """Integration tests for compression agent."""
    
    def test_end_to_end_svd_compression(self):
        """Test complete SVD compression workflow."""
        # Create model
        model = SimpleModel(input_size=80, hidden_size=40, output_size=8)
        
        # Create agent
        agent = CompressionAgent({'compression_ratio': 0.6})
        agent.initialize()
        
        # Run compression
        config = {
            'compression_type': 'svd',
            'target_layers': 'linear',
            'compression_ratio': 0.6,
            'rank_ratio': 0.5,
            'adaptive_rank': True
        }
        
        result = agent.optimize_with_tracking(model, config)
        
        # Verify results
        assert result.success
        assert result.optimized_model is not None
        
        # Check compression
        original_params = sum(p.numel() for p in model.parameters())
        compressed_params = sum(p.numel() for p in result.optimized_model.parameters())
        compression_ratio = (original_params - compressed_params) / original_params
        assert compression_ratio > 0.1  # At least 10% compression
        
        # Validate result
        validation = agent.validate_result(model, result.optimized_model)
        assert validation.is_valid
        
        agent.cleanup()
    
    def test_end_to_end_mixed_compression(self):
        """Test complete mixed compression workflow."""
        model = ConvModel()
        agent = CompressionAgent({})
        agent.initialize()
        
        config = {
            'compression_type': 'mixed',
            'target_layers': 'all',
            'compression_ratio': 0.5,
            'rank_ratio': 0.6
        }
        
        result = agent.optimize_with_tracking(model, config)
        
        assert result.success
        assert result.optimized_model is not None
        
        # Should have both SVD and Tucker compressed layers
        compressed_count = agent._count_compressed_layers(result.optimized_model)
        assert compressed_count > 0
        
        validation = agent.validate_result(model, result.optimized_model)
        assert validation.is_valid
        
        agent.cleanup()
    
    def test_compression_with_different_targets(self):
        """Test compression with different target layer types."""
        model = ConvModel()
        agent = CompressionAgent({})
        agent.initialize()
        
        targets = ['linear', 'conv2d', 'all']
        
        for target in targets:
            config = {
                'compression_type': 'svd' if target == 'linear' else 'tucker' if target == 'conv2d' else 'mixed',
                'target_layers': target,
                'rank_ratio': 0.5
            }
            
            result = agent.optimize_with_tracking(model, config)
            assert result.success, f"Failed for target: {target}"
        
        agent.cleanup()
    
    def test_compression_with_preservation(self):
        """Test compression with module preservation."""
        model = SimpleModel()
        agent = CompressionAgent({})
        agent.initialize()
        
        config = {
            'compression_type': 'svd',
            'target_layers': 'all',
            'preserve_modules': ['layers.4'],  # Preserve last layer
            'rank_ratio': 0.5
        }
        
        result = agent.optimize_with_tracking(model, config)
        
        assert result.success
        
        # Check that some layers were compressed but not all
        compressed_count = agent._count_compressed_layers(result.optimized_model)
        total_linear_layers = len([m for m in model.modules() if isinstance(m, nn.Linear)])
        
        assert 0 < compressed_count < total_linear_layers
        
        agent.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])