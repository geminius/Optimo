"""
Unit tests for QuantizationAgent.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
import sys
from typing import Dict, Any

# Import the agent
from src.agents.optimization.quantization import (
    QuantizationAgent, 
    QuantizationType, 
    QuantizationConfig,
    BITSANDBYTES_AVAILABLE
)
from src.agents.base import ImpactEstimate, ValidationResult, OptimizedModel


class SimpleTestModel(nn.Module):
    """Simple model for testing quantization."""
    
    def __init__(self, input_size: int = 768, hidden_size: int = 512, output_size: int = 256):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.layer3 = nn.Linear(output_size, 10)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x


class TinyTestModel(nn.Module):
    """Tiny model for testing size thresholds."""
    
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.layer(x)


class ConvTestModel(nn.Module):
    """Convolutional model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.fc = nn.Linear(32 * 220 * 220, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def quantization_config():
    """Default configuration for testing."""
    return {
        'preserve_modules': ['lm_head', 'embed_tokens'],
        'quantization_type': 'int8',
        'compute_dtype': torch.float16,
        'threshold': 6.0
    }


@pytest.fixture
def quantization_agent(quantization_config):
    """Create a QuantizationAgent instance for testing."""
    return QuantizationAgent(quantization_config)


@pytest.fixture
def simple_model():
    """Create a simple test model."""
    return SimpleTestModel()


@pytest.fixture
def tiny_model():
    """Create a tiny test model."""
    return TinyTestModel()


@pytest.fixture
def conv_model():
    """Create a convolutional test model."""
    return ConvTestModel()


class TestQuantizationAgent:
    """Test cases for QuantizationAgent."""
    
    def test_initialization(self, quantization_agent):
        """Test agent initialization."""
        assert quantization_agent.name == "QuantizationAgent"
        assert hasattr(quantization_agent, 'default_config')
        assert isinstance(quantization_agent.default_config, QuantizationConfig)
        
    def test_initialize_success(self, quantization_agent):
        """Test successful initialization."""
        result = quantization_agent.initialize()
        if BITSANDBYTES_AVAILABLE:
            assert result is True
        else:
            # Should still initialize even without bitsandbytes
            assert result is True
            
    def test_cleanup(self, quantization_agent):
        """Test cleanup method."""
        # Should not raise any exceptions
        quantization_agent.cleanup()
        
    def test_can_optimize_valid_model(self, quantization_agent, simple_model):
        """Test can_optimize with a valid model."""
        if BITSANDBYTES_AVAILABLE:
            assert quantization_agent.can_optimize(simple_model) is True
        else:
            assert quantization_agent.can_optimize(simple_model) is False
            
    def test_can_optimize_tiny_model(self, quantization_agent, tiny_model):
        """Test can_optimize with a model that's too small."""
        # Should return False for tiny models regardless of bitsandbytes availability
        assert quantization_agent.can_optimize(tiny_model) is False
        
    def test_can_optimize_no_quantizable_layers(self, quantization_agent):
        """Test can_optimize with model having no quantizable layers."""
        class NoQuantizableModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.activation = nn.ReLU()
                
            def forward(self, x):
                return self.activation(x)
        
        model = NoQuantizableModel()
        assert quantization_agent.can_optimize(model) is False
        
    def test_estimate_impact(self, quantization_agent, simple_model):
        """Test impact estimation."""
        impact = quantization_agent.estimate_impact(simple_model)
        
        assert isinstance(impact, ImpactEstimate)
        assert 0.0 <= impact.performance_improvement <= 1.0
        assert 0.0 <= impact.size_reduction <= 1.0
        assert 0.0 <= impact.speed_improvement <= 1.0
        assert 0.0 <= impact.confidence <= 1.0
        assert impact.estimated_time_minutes > 0
        
    def test_estimate_impact_conv_model(self, quantization_agent, conv_model):
        """Test impact estimation with convolutional model."""
        impact = quantization_agent.estimate_impact(conv_model)
        
        assert isinstance(impact, ImpactEstimate)
        # Should have some impact due to the linear layer
        assert impact.size_reduction > 0
        
    def test_get_supported_techniques(self, quantization_agent):
        """Test getting supported techniques."""
        techniques = quantization_agent.get_supported_techniques()
        
        assert isinstance(techniques, list)
        if BITSANDBYTES_AVAILABLE:
            assert "int4" in techniques
            assert "int8" in techniques
            assert "dynamic" in techniques
        else:
            assert len(techniques) == 0
            
    def test_parse_config_default(self, quantization_agent):
        """Test configuration parsing with defaults."""
        config = {}
        parsed = quantization_agent._parse_config(config)
        
        assert isinstance(parsed, QuantizationConfig)
        assert parsed.quantization_type == QuantizationType.INT8
        assert parsed.compute_dtype == torch.float16
        
    def test_parse_config_custom(self, quantization_agent):
        """Test configuration parsing with custom values."""
        config = {
            'quantization_type': 'int4',
            'compute_dtype': torch.float32,
            'quant_type': 'fp4',
            'use_double_quant': False,
            'threshold': 8.0,
            'preserve_modules': ['custom_module']
        }
        parsed = quantization_agent._parse_config(config)
        
        assert parsed.quantization_type == QuantizationType.INT4
        assert parsed.compute_dtype == torch.float32
        assert parsed.quant_type == 'fp4'
        assert parsed.use_double_quant is False
        assert parsed.threshold == 8.0
        assert 'custom_module' in parsed.preserve_modules
        
    def test_parse_config_invalid_type(self, quantization_agent):
        """Test configuration parsing with invalid quantization type."""
        config = {'quantization_type': 'invalid_type'}
        parsed = quantization_agent._parse_config(config)
        
        # Should default to INT8
        assert parsed.quantization_type == QuantizationType.INT8
        
    def test_should_preserve_module(self, quantization_agent):
        """Test module preservation logic."""
        preserve_modules = ['lm_head', 'embed_tokens', 'classifier']
        
        assert quantization_agent._should_preserve_module('model.lm_head', preserve_modules) is True
        assert quantization_agent._should_preserve_module('model.embed_tokens.weight', preserve_modules) is True
        assert quantization_agent._should_preserve_module('model.classifier.bias', preserve_modules) is True
        assert quantization_agent._should_preserve_module('model.layer1.weight', preserve_modules) is False
        
    def test_calculate_model_size(self, quantization_agent, simple_model):
        """Test model size calculation."""
        size = quantization_agent._calculate_model_size(simple_model)
        
        assert isinstance(size, int)
        assert size > 0
        
        # Verify calculation is reasonable
        expected_params = sum(p.numel() for p in simple_model.parameters())
        expected_size = expected_params * 4  # Assuming float32
        assert size == expected_size
        
    def test_count_quantized_layers_unquantized(self, quantization_agent, simple_model):
        """Test counting quantized layers in unquantized model."""
        count = quantization_agent._count_quantized_layers(simple_model)
        assert count == 0
        
    def test_create_dummy_input_linear(self, quantization_agent, simple_model):
        """Test dummy input creation for linear model."""
        dummy_input = quantization_agent._create_dummy_input(simple_model)
        
        assert isinstance(dummy_input, torch.Tensor)
        assert dummy_input.shape == (1, 768)  # Should match first layer input
        
    def test_create_dummy_input_conv(self, quantization_agent, conv_model):
        """Test dummy input creation for convolutional model."""
        dummy_input = quantization_agent._create_dummy_input(conv_model)
        
        assert isinstance(dummy_input, torch.Tensor)
        assert dummy_input.shape == (1, 3, 224, 224)  # Should match conv input
        
    @pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes not available")
    def test_optimize_int8(self, quantization_agent, simple_model):
        """Test 8-bit quantization optimization."""
        config = {'quantization_type': 'int8'}
        
        result = quantization_agent.optimize(simple_model, config)
        
        assert isinstance(result, OptimizedModel)
        assert result.model is not None
        assert result.technique_used == "quantization_int8"
        assert 'quantization_type' in result.optimization_metadata
        assert 'original_size_mb' in result.performance_metrics
        assert 'optimized_size_mb' in result.performance_metrics
        assert result.optimization_time > 0
        
    @pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes not available")
    def test_optimize_int4(self, quantization_agent, simple_model):
        """Test 4-bit quantization optimization."""
        config = {'quantization_type': 'int4'}
        
        result = quantization_agent.optimize(simple_model, config)
        
        assert isinstance(result, OptimizedModel)
        assert result.model is not None
        assert result.technique_used == "quantization_int4"
        
    def test_optimize_dynamic(self, quantization_agent, simple_model):
        """Test dynamic quantization optimization."""
        config = {'quantization_type': 'dynamic'}
        
        result = quantization_agent.optimize(simple_model, config)
        
        assert isinstance(result, OptimizedModel)
        assert result.model is not None
        assert result.technique_used == "quantization_dynamic"
        # Note: Dynamic quantization might not actually quantize if backend isn't configured
        
    def test_optimize_awq_fallback(self, quantization_agent, simple_model):
        """Test AWQ quantization (should fallback to 4-bit)."""
        config = {'quantization_type': 'awq'}
        
        if BITSANDBYTES_AVAILABLE:
            result = quantization_agent.optimize(simple_model, config)
            assert isinstance(result, OptimizedModel)
            assert result.model is not None
        
    def test_optimize_smoothquant_fallback(self, quantization_agent, simple_model):
        """Test SmoothQuant quantization (should fallback to 8-bit)."""
        config = {'quantization_type': 'smoothquant'}
        
        if BITSANDBYTES_AVAILABLE:
            result = quantization_agent.optimize(simple_model, config)
            assert isinstance(result, OptimizedModel)
            assert result.model is not None
        
    def test_optimize_invalid_type(self, quantization_agent, simple_model):
        """Test optimization with invalid quantization type."""
        config = {'quantization_type': 'invalid'}
        
        with pytest.raises(ValueError, match="quantization requires bitsandbytes|Unsupported quantization type"):
            quantization_agent.optimize(simple_model, config)
            
    @pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes not available")
    def test_validate_result_success(self, quantization_agent, simple_model):
        """Test successful validation of quantized model."""
        # First quantize the model
        config = {'quantization_type': 'int8'}
        result = quantization_agent.optimize(simple_model, config)
        
        # Then validate
        validation = quantization_agent.validate_result(simple_model, result.model)
        
        assert isinstance(validation, ValidationResult)
        assert validation.is_valid is True
        assert validation.accuracy_preserved is True
        assert 'quantized_layers' in validation.performance_metrics
        assert validation.performance_metrics['quantized_layers'] > 0
        
    def test_validate_result_with_issues(self, quantization_agent, simple_model):
        """Test validation with a problematic model."""
        # Create a model with missing modules
        class BrokenModel(nn.Module):
            def __init__(self):
                super().__init__()
                # Missing the layers that should be in simple_model
                
            def forward(self, x):
                return x
        
        broken_model = BrokenModel()
        validation = quantization_agent.validate_result(simple_model, broken_model)
        
        assert isinstance(validation, ValidationResult)
        assert validation.is_valid is False
        assert len(validation.issues) > 0
        
    def test_validate_result_inference_failure(self, quantization_agent):
        """Test validation when model inference fails."""
        class FailingModel(nn.Module):
            def forward(self, x):
                raise RuntimeError("Inference failed")
        
        original_model = SimpleTestModel()
        failing_model = FailingModel()
        
        validation = quantization_agent.validate_result(original_model, failing_model)
        
        assert isinstance(validation, ValidationResult)
        assert validation.is_valid is False
        assert any("inference test failed" in issue for issue in validation.issues)
        
    @pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes not available")
    def test_optimize_with_tracking(self, quantization_agent, simple_model):
        """Test optimization with progress tracking."""
        config = {'quantization_type': 'int8'}
        
        # Mock progress callback
        progress_updates = []
        def progress_callback(update):
            progress_updates.append(update)
        
        quantization_agent.add_progress_callback(progress_callback)
        
        result = quantization_agent.optimize_with_tracking(simple_model, config)
        
        assert result.success is True
        assert result.optimized_model is not None
        assert len(progress_updates) > 0
        
        # Check that we received progress updates
        statuses = [update.status for update in progress_updates]
        assert any("INITIALIZING" in str(status) for status in statuses)
        assert any("COMPLETED" in str(status) for status in statuses)
        
    def test_cancel_optimization(self, quantization_agent):
        """Test optimization cancellation."""
        quantization_agent.cancel_optimization()
        
        assert quantization_agent.is_cancelled() is True
        assert quantization_agent.get_current_status().value == "cancelled"
        
    def test_replace_module(self, quantization_agent, simple_model):
        """Test module replacement functionality."""
        # Create a new module to replace with
        new_module = nn.Linear(512, 256)
        
        # Replace layer2
        quantization_agent._replace_module(simple_model, 'layer2', new_module)
        
        # Verify replacement
        assert simple_model.layer2 is new_module
        assert simple_model.layer2.in_features == 512
        assert simple_model.layer2.out_features == 256


class TestQuantizationConfig:
    """Test cases for QuantizationConfig."""
    
    def test_default_initialization(self):
        """Test default QuantizationConfig initialization."""
        config = QuantizationConfig(quantization_type=QuantizationType.INT8)
        
        assert config.quantization_type == QuantizationType.INT8
        assert config.compute_dtype == torch.float16
        assert config.quant_type == "nf4"
        assert config.use_double_quant is True
        assert config.threshold == 6.0
        assert config.preserve_modules == []
        assert config.calibration_dataset_size == 512
        
    def test_custom_initialization(self):
        """Test QuantizationConfig with custom values."""
        preserve_modules = ['lm_head', 'classifier']
        config = QuantizationConfig(
            quantization_type=QuantizationType.INT4,
            compute_dtype=torch.float32,
            quant_type="fp4",
            use_double_quant=False,
            threshold=8.0,
            preserve_modules=preserve_modules,
            calibration_dataset_size=1024
        )
        
        assert config.quantization_type == QuantizationType.INT4
        assert config.compute_dtype == torch.float32
        assert config.quant_type == "fp4"
        assert config.use_double_quant is False
        assert config.threshold == 8.0
        assert config.preserve_modules == preserve_modules
        assert config.calibration_dataset_size == 1024


class TestQuantizationType:
    """Test cases for QuantizationType enum."""
    
    def test_enum_values(self):
        """Test QuantizationType enum values."""
        assert QuantizationType.INT4.value == "int4"
        assert QuantizationType.INT8.value == "int8"
        assert QuantizationType.DYNAMIC.value == "dynamic"
        assert QuantizationType.AWQ.value == "awq"
        assert QuantizationType.SMOOTHQUANT.value == "smoothquant"
        
    def test_enum_from_string(self):
        """Test creating QuantizationType from string."""
        assert QuantizationType("int4") == QuantizationType.INT4
        assert QuantizationType("int8") == QuantizationType.INT8
        assert QuantizationType("dynamic") == QuantizationType.DYNAMIC
        
        with pytest.raises(ValueError):
            QuantizationType("invalid_type")


# Integration tests
class TestQuantizationIntegration:
    """Integration tests for QuantizationAgent."""
    
    @pytest.mark.skipif(not BITSANDBYTES_AVAILABLE, reason="bitsandbytes not available")
    def test_full_quantization_workflow(self, quantization_agent, simple_model):
        """Test complete quantization workflow."""
        # Check if model can be optimized
        assert quantization_agent.can_optimize(simple_model) is True
        
        # Estimate impact
        impact = quantization_agent.estimate_impact(simple_model)
        assert impact.size_reduction > 0
        
        # Perform optimization
        config = {'quantization_type': 'int8'}
        result = quantization_agent.optimize(simple_model, config)
        assert result.model is not None
        
        # Validate result
        validation = quantization_agent.validate_result(simple_model, result.model)
        assert validation.is_valid is True
        
        # Check size reduction was achieved
        original_size = result.performance_metrics['original_size_mb']
        optimized_size = result.performance_metrics['optimized_size_mb']
        assert optimized_size < original_size
        
    def test_workflow_with_preservation(self, quantization_agent, simple_model):
        """Test quantization workflow with module preservation."""
        config = {
            'quantization_type': 'int8',
            'preserve_modules': ['layer1', 'layer3']  # Preserve first and last layers
        }
        
        if BITSANDBYTES_AVAILABLE:
            result = quantization_agent.optimize(simple_model, config)
            
            # Should still work but quantize fewer layers
            assert result.model is not None
            assert result.performance_metrics['quantized_layers'] < 3  # Should be less than total layers