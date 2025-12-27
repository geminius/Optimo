"""
Model utility functions for the robotics optimization platform.

This module provides utility functions for model handling, input generation,
memory monitoring, and other model-related operations.
"""

import torch
import torch.nn as nn
import psutil
import logging
from typing import Tuple, Optional, Dict, Any, List
import numpy as np


logger = logging.getLogger(__name__)


def find_compatible_input(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """
    Find a compatible input tensor for the given model.
    
    This function attempts to determine the expected input shape for a model
    by analyzing its first layer or using common defaults for robotics models.
    
    Args:
        model: PyTorch model to analyze
        device: Device to create tensor on
        
    Returns:
        Compatible input tensor
    """
    try:
        # Try to find the first layer that expects input
        first_layer = None
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                first_layer = module
                break
        
        if first_layer is not None:
            if isinstance(first_layer, nn.Linear):
                # Linear layer - create 1D input
                batch_size = 1
                input_features = first_layer.in_features
                return torch.randn(batch_size, input_features).to(device)
            
            elif isinstance(first_layer, nn.Conv2d):
                # 2D Convolution - create image-like input
                batch_size = 1
                in_channels = first_layer.in_channels
                # Use common image sizes for robotics (224x224 is common for vision models)
                height, width = 224, 224
                return torch.randn(batch_size, in_channels, height, width).to(device)
            
            elif isinstance(first_layer, nn.Conv1d):
                # 1D Convolution - create sequence input
                batch_size = 1
                in_channels = first_layer.in_channels
                sequence_length = 100  # Default sequence length
                return torch.randn(batch_size, in_channels, sequence_length).to(device)
            
            elif isinstance(first_layer, nn.Conv3d):
                # 3D Convolution - create volume input
                batch_size = 1
                in_channels = first_layer.in_channels
                depth, height, width = 16, 64, 64  # Default 3D dimensions
                return torch.randn(batch_size, in_channels, depth, height, width).to(device)
        
        # Fallback: try common robotics model input shapes
        common_shapes = [
            (1, 3, 224, 224),    # RGB image (224x224)
            (1, 3, 256, 256),    # RGB image (256x256)
            (1, 1, 224, 224),    # Grayscale image
            (1, 512),            # Feature vector
            (1, 1024),           # Larger feature vector
            (1, 100, 512),       # Sequence input
            (1, 3, 480, 640),    # Camera resolution
        ]
        
        for shape in common_shapes:
            try:
                test_input = torch.randn(shape).to(device)
                # Try a forward pass to see if it works
                model.eval()
                with torch.no_grad():
                    _ = model(test_input)
                logger.info(f"Found compatible input shape: {shape}")
                return test_input
            except Exception:
                continue
        
        # Last resort: create a simple 1D input
        logger.warning("Could not determine input shape, using default 1D input")
        return torch.randn(1, 100).to(device)
        
    except Exception as e:
        logger.error(f"Error finding compatible input: {e}")
        # Emergency fallback
        return torch.randn(1, 100).to(device)


def get_memory_usage() -> float:
    """
    Get current memory usage in MB.
    
    Returns GPU memory usage if CUDA is available, otherwise system RAM usage.
    
    Returns:
        Memory usage in MB
    """
    try:
        if torch.cuda.is_available():
            # Get GPU memory usage
            memory_allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
            return memory_allocated
        else:
            # Get system RAM usage
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
            
    except Exception as e:
        logger.warning(f"Failed to get memory usage: {e}")
        return 0.0


def get_model_summary(model: torch.nn.Module) -> Dict[str, Any]:
    """
    Get a summary of model architecture and parameters.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary containing model summary information
    """
    try:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate model size in MB
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        # Count layers by type
        layer_counts = {}
        for name, module in model.named_modules():
            module_type = type(module).__name__
            layer_counts[module_type] = layer_counts.get(module_type, 0) + 1
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
            "model_size_mb": total_size_mb,
            "parameter_size_mb": param_size / (1024 * 1024),
            "buffer_size_mb": buffer_size / (1024 * 1024),
            "layer_counts": layer_counts
        }
        
    except Exception as e:
        logger.error(f"Failed to get model summary: {e}")
        return {
            "total_parameters": 0,
            "trainable_parameters": 0,
            "non_trainable_parameters": 0,
            "model_size_mb": 0.0,
            "parameter_size_mb": 0.0,
            "buffer_size_mb": 0.0,
            "layer_counts": {},
            "error": str(e)
        }


def estimate_model_flops(model: torch.nn.Module, input_tensor: torch.Tensor) -> int:
    """
    Estimate the number of FLOPs (Floating Point Operations) for a model.
    
    This is a simplified estimation based on layer types and sizes.
    
    Args:
        model: PyTorch model
        input_tensor: Sample input tensor
        
    Returns:
        Estimated FLOPs
    """
    try:
        total_flops = 0
        
        def flop_count_hook(module, input, output):
            nonlocal total_flops
            
            if isinstance(module, nn.Linear):
                # Linear layer: input_features * output_features
                flops = module.in_features * module.out_features
                if module.bias is not None:
                    flops += module.out_features
                total_flops += flops
            
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Convolution: output_elements * kernel_size * input_channels
                if hasattr(module, 'weight') and module.weight is not None:
                    kernel_flops = np.prod(module.kernel_size) * module.in_channels
                    
                    # Estimate output size
                    if isinstance(output, torch.Tensor):
                        output_elements = output.numel() // output.shape[0]  # Exclude batch dimension
                    else:
                        output_elements = 1000  # Fallback estimate
                    
                    flops = kernel_flops * output_elements
                    if module.bias is not None:
                        flops += output_elements
                    total_flops += flops
            
            elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                # BatchNorm: 2 operations per element (normalize + scale/shift)
                if isinstance(output, torch.Tensor):
                    total_flops += output.numel() * 2
            
            elif isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.GELU, nn.Sigmoid, nn.Tanh)):
                # Activation functions: 1 operation per element
                if isinstance(output, torch.Tensor):
                    total_flops += output.numel()
        
        # Register hooks
        hooks = []
        for module in model.modules():
            hook = module.register_forward_hook(flop_count_hook)
            hooks.append(hook)
        
        # Run forward pass
        model.eval()
        with torch.no_grad():
            _ = model(input_tensor)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return total_flops
        
    except Exception as e:
        logger.error(f"Failed to estimate FLOPs: {e}")
        return 0


def validate_model_compatibility(model: torch.nn.Module, device: torch.device) -> Tuple[bool, List[str]]:
    """
    Validate that a model is compatible with the optimization platform.
    
    Args:
        model: PyTorch model to validate
        device: Target device
        
    Returns:
        Tuple of (is_compatible, list_of_issues)
    """
    issues = []
    
    try:
        # Check if model can be moved to device
        try:
            model.to(device)
        except Exception as e:
            issues.append(f"Cannot move model to device {device}: {str(e)}")
        
        # Check if model can be set to eval mode
        try:
            model.eval()
        except Exception as e:
            issues.append(f"Cannot set model to eval mode: {str(e)}")
        
        # Check if we can find a compatible input
        try:
            input_tensor = find_compatible_input(model, device)
            
            # Try a forward pass
            with torch.no_grad():
                output = model(input_tensor)
            
            if output is None:
                issues.append("Model forward pass returns None")
            
        except Exception as e:
            issues.append(f"Model forward pass failed: {str(e)}")
        
        # Check for unsupported layer types
        unsupported_layers = []
        for name, module in model.named_modules():
            module_type = type(module).__name__
            # Add any layer types that are known to be problematic
            if module_type in ['CustomLayer', 'UnsupportedOp']:  # Example unsupported types
                unsupported_layers.append(f"{name}: {module_type}")
        
        if unsupported_layers:
            issues.append(f"Unsupported layer types found: {', '.join(unsupported_layers)}")
        
        # Check model size (warn if very large)
        try:
            summary = get_model_summary(model)
            if summary.get("model_size_mb", 0) > 5000:  # 5GB
                issues.append(f"Model is very large ({summary['model_size_mb']:.1f}MB), optimization may be slow")
        except Exception:
            pass
        
        return len(issues) == 0, issues
        
    except Exception as e:
        issues.append(f"Validation failed with error: {str(e)}")
        return False, issues


def prepare_model_for_optimization(model: torch.nn.Module, device: torch.device) -> torch.nn.Module:
    """
    Prepare a model for optimization by applying common preprocessing steps.
    
    Args:
        model: PyTorch model to prepare
        device: Target device
        
    Returns:
        Prepared model
    """
    try:
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        # Disable gradient computation for all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        # Apply any model-specific optimizations
        # (This could include things like fusing batch norm, etc.)
        
        logger.info("Model prepared for optimization")
        return model
        
    except Exception as e:
        logger.error(f"Failed to prepare model for optimization: {e}")
        raise


def compare_model_outputs(
    model1: torch.nn.Module, 
    model2: torch.nn.Module, 
    input_tensor: torch.Tensor,
    tolerance: float = 1e-5
) -> Dict[str, Any]:
    """
    Compare outputs of two models to check for consistency.
    
    Args:
        model1: First model
        model2: Second model  
        input_tensor: Input to test with
        tolerance: Numerical tolerance for comparison
        
    Returns:
        Dictionary with comparison results
    """
    try:
        model1.eval()
        model2.eval()
        
        with torch.no_grad():
            output1 = model1(input_tensor)
            output2 = model2(input_tensor)
        
        # Convert to numpy for easier comparison
        if isinstance(output1, torch.Tensor):
            out1_np = output1.cpu().numpy()
        else:
            out1_np = np.array(output1)
            
        if isinstance(output2, torch.Tensor):
            out2_np = output2.cpu().numpy()
        else:
            out2_np = np.array(output2)
        
        # Calculate differences
        abs_diff = np.abs(out1_np - out2_np)
        max_diff = np.max(abs_diff)
        mean_diff = np.mean(abs_diff)
        
        # Check if outputs are close
        are_close = np.allclose(out1_np, out2_np, atol=tolerance)
        
        return {
            "are_close": are_close,
            "max_absolute_difference": float(max_diff),
            "mean_absolute_difference": float(mean_diff),
            "tolerance_used": tolerance,
            "output1_shape": out1_np.shape,
            "output2_shape": out2_np.shape,
            "shapes_match": out1_np.shape == out2_np.shape
        }
        
    except Exception as e:
        logger.error(f"Failed to compare model outputs: {e}")
        return {
            "are_close": False,
            "error": str(e)
        }