"""
Quantization optimization agent for model compression using various quantization techniques.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Union
import logging
import time
import copy
from dataclasses import dataclass
from enum import Enum

try:
    import bitsandbytes as bnb
    from bitsandbytes.nn import Linear4bit, Linear8bitLt
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    logging.warning("bitsandbytes not available. Some quantization features will be disabled.")

try:
    from transformers import BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Some quantization features will be disabled.")

from ..base import BaseOptimizationAgent, ImpactEstimate, ValidationResult, OptimizedModel


class QuantizationType(Enum):
    """Supported quantization types."""
    INT4 = "int4"
    INT8 = "int8"
    DYNAMIC = "dynamic"
    AWQ = "awq"
    SMOOTHQUANT = "smoothquant"


@dataclass
class QuantizationConfig:
    """Configuration for quantization operations."""
    quantization_type: QuantizationType
    compute_dtype: torch.dtype = torch.float16
    quant_type: str = "nf4"  # For 4-bit quantization
    use_double_quant: bool = True  # For 4-bit quantization
    threshold: float = 6.0  # For 8-bit quantization
    preserve_modules: List[str] = None  # Modules to skip quantization
    calibration_dataset_size: int = 512  # For AWQ/SmoothQuant
    
    def __post_init__(self):
        if self.preserve_modules is None:
            self.preserve_modules = []


class QuantizationAgent(BaseOptimizationAgent):
    """
    Optimization agent that implements various quantization techniques including
    4-bit, 8-bit quantization using bitsandbytes, and advanced techniques like AWQ and SmoothQuant.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate dependencies
        if not BITSANDBYTES_AVAILABLE:
            self.logger.warning("bitsandbytes not available. Quantization capabilities will be limited.")
        
        # Default quantization settings
        self.default_config = QuantizationConfig(
            quantization_type=QuantizationType.INT8,
            preserve_modules=config.get('preserve_modules', ['lm_head', 'embed_tokens'])
        )
        
        # Supported layer types for quantization
        self.quantizable_layers = (nn.Linear, nn.Conv2d, nn.Conv1d)
        
    def initialize(self) -> bool:
        """Initialize the quantization agent."""
        try:
            if BITSANDBYTES_AVAILABLE:
                # Test bitsandbytes functionality
                test_tensor = torch.randn(10, 10)
                _ = bnb.functional.quantize_4bit(test_tensor)
                self.logger.info("bitsandbytes quantization functionality verified")
            
            self.logger.info("QuantizationAgent initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize QuantizationAgent: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources used by the agent."""
        # Clear any cached quantization data
        if hasattr(self, '_calibration_cache'):
            delattr(self, '_calibration_cache')
        self.logger.info("QuantizationAgent cleanup completed")
    
    def can_optimize(self, model: torch.nn.Module) -> bool:
        """Check if this agent can optimize the given model."""
        if not BITSANDBYTES_AVAILABLE:
            self.logger.warning("Cannot optimize: bitsandbytes not available")
            return False
        
        # Check if model has quantizable layers
        quantizable_layers_found = False
        for module in model.modules():
            if isinstance(module, self.quantizable_layers):
                quantizable_layers_found = True
                break
        
        if not quantizable_layers_found:
            self.logger.warning("No quantizable layers found in model")
            return False
        
        # Check model size (quantization is most beneficial for larger models)
        param_count = sum(p.numel() for p in model.parameters())
        if param_count < 1_000_000:  # Less than 1M parameters
            self.logger.warning(f"Model too small for quantization ({param_count:,} parameters)")
            return False
        
        return True
    
    def estimate_impact(self, model: torch.nn.Module) -> ImpactEstimate:
        """Estimate the impact of quantization on the model."""
        # Count quantizable parameters
        total_params = sum(p.numel() for p in model.parameters())
        quantizable_params = 0
        
        for module in model.modules():
            if isinstance(module, self.quantizable_layers):
                quantizable_params += sum(p.numel() for p in module.parameters())
        
        quantizable_ratio = quantizable_params / total_params if total_params > 0 else 0
        
        # Estimate size reduction based on quantization type
        # 4-bit: ~75% reduction, 8-bit: ~50% reduction
        size_reduction = quantizable_ratio * 0.5  # Conservative estimate for 8-bit
        
        # Performance improvement estimates (conservative)
        performance_improvement = min(0.3, quantizable_ratio * 0.4)  # Up to 30% improvement
        speed_improvement = min(0.2, quantizable_ratio * 0.3)  # Up to 20% speed improvement
        
        # Confidence based on model characteristics
        confidence = 0.8 if quantizable_ratio > 0.7 else 0.6
        
        # Estimated time based on model size
        estimated_time = max(5, int(total_params / 1_000_000 * 2))  # 2 minutes per million params
        
        return ImpactEstimate(
            performance_improvement=performance_improvement,
            size_reduction=size_reduction,
            speed_improvement=speed_improvement,
            confidence=confidence,
            estimated_time_minutes=estimated_time
        )
    
    def optimize(self, model: torch.nn.Module, config: Dict[str, Any]) -> OptimizedModel:
        """Execute quantization optimization on the model."""
        start_time = time.time()
        
        # Parse configuration
        quant_config = self._parse_config(config)
        
        # Create a copy of the model for optimization
        optimized_model = copy.deepcopy(model)
        
        # Track optimization metadata
        optimization_metadata = {
            "quantization_type": quant_config.quantization_type.value,
            "compute_dtype": str(quant_config.compute_dtype),
            "quant_type": quant_config.quant_type,
            "use_double_quant": quant_config.use_double_quant,
            "preserved_modules": quant_config.preserve_modules.copy()
        }
        
        try:
            # Execute quantization based on type
            if quant_config.quantization_type == QuantizationType.INT4:
                if not BITSANDBYTES_AVAILABLE:
                    raise ValueError("4-bit quantization requires bitsandbytes library")
                optimized_model = self._quantize_4bit(optimized_model, quant_config)
            elif quant_config.quantization_type == QuantizationType.INT8:
                if not BITSANDBYTES_AVAILABLE:
                    raise ValueError("8-bit quantization requires bitsandbytes library")
                optimized_model = self._quantize_8bit(optimized_model, quant_config)
            elif quant_config.quantization_type == QuantizationType.DYNAMIC:
                optimized_model = self._quantize_dynamic(optimized_model, quant_config)
            elif quant_config.quantization_type == QuantizationType.AWQ:
                optimized_model = self._quantize_awq(optimized_model, quant_config)
            elif quant_config.quantization_type == QuantizationType.SMOOTHQUANT:
                optimized_model = self._quantize_smoothquant(optimized_model, quant_config)
            else:
                raise ValueError(f"Unsupported quantization type: {quant_config.quantization_type}")
            
            # Calculate performance metrics
            original_size = self._calculate_model_size(model)
            optimized_size = self._calculate_model_size(optimized_model)
            size_reduction = (original_size - optimized_size) / original_size
            
            performance_metrics = {
                "original_size_mb": original_size / (1024 * 1024),
                "optimized_size_mb": optimized_size / (1024 * 1024),
                "size_reduction_ratio": size_reduction,
                "quantized_layers": self._count_quantized_layers(optimized_model)
            }
            
            optimization_time = time.time() - start_time
            
            return OptimizedModel(
                model=optimized_model,
                optimization_metadata=optimization_metadata,
                performance_metrics=performance_metrics,
                optimization_time=optimization_time,
                technique_used=f"quantization_{quant_config.quantization_type.value}"
            )
            
        except Exception as e:
            self.logger.error(f"Quantization failed: {e}")
            raise
    
    def validate_result(self, original: torch.nn.Module, optimized: torch.nn.Module) -> ValidationResult:
        """Validate the quantization result."""
        issues = []
        recommendations = []
        
        try:
            # Check if model structure is preserved
            original_modules = dict(original.named_modules())
            optimized_modules = dict(optimized.named_modules())
            
            # Verify critical modules exist
            for name, module in original_modules.items():
                if name not in optimized_modules:
                    issues.append(f"Module {name} missing in optimized model")
            
            # Check for quantized layers
            quantized_layers = self._count_quantized_layers(optimized)
            if quantized_layers == 0:
                issues.append("No layers were quantized")
            
            # Basic functionality test
            try:
                # Create dummy input for testing
                dummy_input = self._create_dummy_input(original)
                
                with torch.no_grad():
                    original_output = original(dummy_input)
                    optimized_output = optimized(dummy_input)
                
                # Check output shapes match
                if original_output.shape != optimized_output.shape:
                    issues.append(f"Output shape mismatch: {original_output.shape} vs {optimized_output.shape}")
                
                # Check for NaN or infinite values
                if torch.isnan(optimized_output).any():
                    issues.append("Optimized model produces NaN values")
                
                if torch.isinf(optimized_output).any():
                    issues.append("Optimized model produces infinite values")
                
                # Calculate output similarity (basic check)
                mse = torch.nn.functional.mse_loss(original_output, optimized_output).item()
                performance_metrics = {
                    "output_mse": mse,
                    "quantized_layers": quantized_layers
                }
                
                # Recommendations based on results
                if mse > 1.0:
                    recommendations.append("High output difference detected. Consider using higher precision quantization.")
                
                if quantized_layers < 5:
                    recommendations.append("Few layers quantized. Check quantization configuration.")
                
            except Exception as e:
                issues.append(f"Model inference test failed: {e}")
                performance_metrics = {"quantized_layers": quantized_layers}
            
            # Determine if validation passed
            is_valid = len(issues) == 0
            accuracy_preserved = len([issue for issue in issues if "NaN" in issue or "infinite" in issue]) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                accuracy_preserved=accuracy_preserved,
                performance_metrics=performance_metrics,
                issues=issues,
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            return ValidationResult(
                is_valid=False,
                accuracy_preserved=False,
                performance_metrics={},
                issues=[f"Validation error: {e}"],
                recommendations=["Check model compatibility and quantization configuration"]
            )
    
    def get_supported_techniques(self) -> List[str]:
        """Get list of quantization techniques supported by this agent."""
        techniques = []
        
        if BITSANDBYTES_AVAILABLE:
            techniques.extend(["int4", "int8", "dynamic"])
        
        # AWQ and SmoothQuant would require additional libraries
        # techniques.extend(["awq", "smoothquant"])
        
        return techniques
    
    def _parse_config(self, config: Dict[str, Any]) -> QuantizationConfig:
        """Parse and validate quantization configuration."""
        quant_type = config.get('quantization_type', 'int8')
        
        try:
            quantization_type = QuantizationType(quant_type)
        except ValueError:
            self.logger.warning(f"Unknown quantization type: {quant_type}, defaulting to int8")
            quantization_type = QuantizationType.INT8
        
        return QuantizationConfig(
            quantization_type=quantization_type,
            compute_dtype=config.get('compute_dtype', torch.float16),
            quant_type=config.get('quant_type', 'nf4'),
            use_double_quant=config.get('use_double_quant', True),
            threshold=config.get('threshold', 6.0),
            preserve_modules=config.get('preserve_modules', self.default_config.preserve_modules),
            calibration_dataset_size=config.get('calibration_dataset_size', 512)
        )
    
    def _quantize_4bit(self, model: torch.nn.Module, config: QuantizationConfig) -> torch.nn.Module:
        """Apply 4-bit quantization using bitsandbytes."""
        if not BITSANDBYTES_AVAILABLE:
            raise RuntimeError("bitsandbytes not available for 4-bit quantization")
        
        self._update_progress(self._current_status, 30.0, "Applying 4-bit quantization")
        
        quantized_layers = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not self._should_preserve_module(name, config.preserve_modules):
                # Replace Linear layer with 4-bit quantized version
                quantized_layer = Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=config.compute_dtype,
                    quant_type=config.quant_type,
                    use_double_quant=config.use_double_quant
                )
                
                # Copy weights and bias
                with torch.no_grad():
                    quantized_layer.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        quantized_layer.bias.data = module.bias.data.clone()
                
                # Replace the module
                self._replace_module(model, name, quantized_layer)
                quantized_layers += 1
        
        self.logger.info(f"Applied 4-bit quantization to {quantized_layers} layers")
        return model
    
    def _quantize_8bit(self, model: torch.nn.Module, config: QuantizationConfig) -> torch.nn.Module:
        """Apply 8-bit quantization using bitsandbytes."""
        if not BITSANDBYTES_AVAILABLE:
            raise RuntimeError("bitsandbytes not available for 8-bit quantization")
        
        self._update_progress(self._current_status, 30.0, "Applying 8-bit quantization")
        
        quantized_layers = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and not self._should_preserve_module(name, config.preserve_modules):
                # Replace Linear layer with 8-bit quantized version
                quantized_layer = Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    threshold=config.threshold
                )
                
                # Copy weights and bias
                with torch.no_grad():
                    quantized_layer.weight.data = module.weight.data.clone()
                    if module.bias is not None:
                        quantized_layer.bias.data = module.bias.data.clone()
                
                # Replace the module
                self._replace_module(model, name, quantized_layer)
                quantized_layers += 1
        
        self.logger.info(f"Applied 8-bit quantization to {quantized_layers} layers")
        return model
    
    def _quantize_dynamic(self, model: torch.nn.Module, config: QuantizationConfig) -> torch.nn.Module:
        """Apply dynamic quantization using PyTorch's built-in functionality."""
        self._update_progress(self._current_status, 30.0, "Applying dynamic quantization")
        
        try:
            # Use PyTorch's dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
            self.logger.info("Applied dynamic quantization")
            return quantized_model
        except Exception as e:
            self.logger.warning(f"Dynamic quantization failed: {e}, returning original model")
            # If dynamic quantization fails, return the original model
            # This can happen if the quantization backend isn't properly configured
            return model
    
    def _quantize_awq(self, model: torch.nn.Module, config: QuantizationConfig) -> torch.nn.Module:
        """Apply AWQ (Activation-aware Weight Quantization)."""
        # This would require the auto-gptq library and calibration data
        # For now, fall back to 4-bit quantization if available, otherwise dynamic
        self.logger.warning("AWQ not fully implemented, falling back to alternative quantization")
        if BITSANDBYTES_AVAILABLE:
            return self._quantize_4bit(model, config)
        else:
            return self._quantize_dynamic(model, config)
    
    def _quantize_smoothquant(self, model: torch.nn.Module, config: QuantizationConfig) -> torch.nn.Module:
        """Apply SmoothQuant technique."""
        # This would require additional implementation and calibration data
        # For now, fall back to 8-bit quantization if available, otherwise dynamic
        self.logger.warning("SmoothQuant not fully implemented, falling back to alternative quantization")
        if BITSANDBYTES_AVAILABLE:
            return self._quantize_8bit(model, config)
        else:
            return self._quantize_dynamic(model, config)
    
    def _should_preserve_module(self, module_name: str, preserve_modules: List[str]) -> bool:
        """Check if a module should be preserved from quantization."""
        for preserve_pattern in preserve_modules:
            if preserve_pattern in module_name:
                return True
        return False
    
    def _replace_module(self, model: torch.nn.Module, module_name: str, new_module: torch.nn.Module) -> None:
        """Replace a module in the model with a new module."""
        module_path = module_name.split('.')
        parent = model
        
        for part in module_path[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, module_path[-1], new_module)
    
    def _calculate_model_size(self, model: torch.nn.Module) -> int:
        """Calculate model size in bytes."""
        total_size = 0
        for param in model.parameters():
            total_size += param.numel() * param.element_size()
        return total_size
    
    def _count_quantized_layers(self, model: torch.nn.Module) -> int:
        """Count the number of quantized layers in the model."""
        count = 0
        for module in model.modules():
            if BITSANDBYTES_AVAILABLE and isinstance(module, (Linear4bit, Linear8bitLt)):
                count += 1
            elif hasattr(module, '_FLOAT_MODULE'):  # PyTorch quantized modules
                count += 1
        return count
    
    def _create_dummy_input(self, model: torch.nn.Module) -> torch.Tensor:
        """Create dummy input for model testing."""
        # This is a simple implementation - in practice, you'd want to
        # analyze the model to determine appropriate input shapes
        
        # Try to infer input shape from first layer
        first_layer = None
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                first_layer = module
                break
        
        if isinstance(first_layer, nn.Linear):
            return torch.randn(1, first_layer.in_features)
        elif isinstance(first_layer, nn.Conv2d):
            return torch.randn(1, first_layer.in_channels, 224, 224)  # Common image size
        elif isinstance(first_layer, nn.Conv1d):
            return torch.randn(1, first_layer.in_channels, 100)  # Common sequence length
        else:
            # Default fallback
            return torch.randn(1, 768)  # Common transformer hidden size