"""
Shared utility functions for model operations.

This module provides common functionality used across multiple agents
to avoid code duplication and ensure consistency.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from pathlib import Path
import logging
import psutil

logger = logging.getLogger(__name__)


def find_compatible_input(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    """
    Find a compatible input shape for the model.
    
    This function attempts multiple strategies to determine the correct input shape:
    1. Infer from the first leaf layer (Linear or Conv)
    2. Try common input shapes
    3. Raise an error if nothing works
    
    Args:
        model: PyTorch model to find input for
        device: Device to create tensor on
        
    Returns:
        torch.Tensor: Compatible input tensor
        
    Raises:
        ValueError: If no compatible input shape can be found
    """
    # First, try to infer input size from first leaf layer
    try:
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                if hasattr(module, 'in_features'):
                    # Linear layer - try 1D input
                    input_size = module.in_features
                    dummy_input = torch.randn(1, input_size).to(device)
                    with torch.no_grad():
                        _ = model(dummy_input)
                    logger.info(f"Found compatible input shape from first layer: (1, {input_size})")
                    return dummy_input
                elif hasattr(module, 'in_channels'):
                    # Conv layer - try 2D input
                    in_channels = module.in_channels
                    dummy_input = torch.randn(1, in_channels, 224, 224).to(device)
                    with torch.no_grad():
                        _ = model(dummy_input)
                    logger.info(f"Found compatible input shape from first layer: (1, {in_channels}, 224, 224)")
                    return dummy_input
                break  # Only check first leaf layer
    except Exception as e:
        logger.debug(f"Could not infer from first layer: {e}")
    
    # Try common input shapes
    common_shapes = [
        (1, 3, 224, 224),  # Standard image
        (1, 3, 256, 256),  # Larger image
        (1, 1, 28, 28),    # MNIST-like
        (1, 512),          # 1D input
        (1, 1024),         # Larger 1D input
        (1, 1280),         # VLA models
        (1, 1000),         # Common large input
        (1, 2048),         # Very large input
        (1, 768),          # BERT-like
    ]
    
    for shape in common_shapes:
        try:
            dummy_input = torch.randn(shape).to(device)
            with torch.no_grad():
                _ = model(dummy_input)
            logger.info(f"Found compatible input shape: {shape}")
            return dummy_input
        except Exception:
            continue
    
    # If nothing works, raise an error instead of returning invalid input
    raise ValueError(
        "Could not find compatible input shape for model. "
        "Please ensure the model has a standard input format or provide input shape explicitly."
    )


def get_memory_usage() -> float:
    """
    Get current memory usage in MB.
    
    Returns GPU memory if available, otherwise system memory.
    
    Returns:
        float: Memory usage in MB
    """
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    else:
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count total number of parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_model_depth(model: torch.nn.Module) -> int:
    """
    Calculate approximate model depth (number of nested layers).
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Maximum depth of the model
    """
    max_depth = 0
    
    def calculate_depth(module, current_depth=0):
        nonlocal max_depth
        max_depth = max(max_depth, current_depth)
        
        for child in module.children():
            calculate_depth(child, current_depth + 1)
    
    calculate_depth(model)
    return max_depth


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Calculate model size in MB based on parameters.
    
    Assumes float32 (4 bytes per parameter).
    
    Args:
        model: PyTorch model
        
    Returns:
        float: Model size in MB
    """
    param_count = count_parameters(model)
    return (param_count * 4) / (1024 * 1024)


def load_model(
    model_path: str, 
    device: torch.device,
    weights_only: bool = False
) -> torch.nn.Module:
    """
    Load a PyTorch model from file.
    
    Args:
        model_path: Path to the model file
        device: Device to load model on
        weights_only: If True, only load weights (more secure)
        
    Returns:
        torch.nn.Module: Loaded model
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model format is unsupported
        ImportError: If custom model class is not importable
    """
    path = Path(model_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        # Try loading as PyTorch model
        if path.suffix in ['.pt', '.pth']:
            # Load model - custom classes must be importable via PYTHONPATH or sys.path
            # For test models, ensure test_models is in sys.path before calling this
            model = torch.load(model_path, map_location=device, weights_only=weights_only)
            
            if isinstance(model, dict) and 'model' in model:
                model = model['model']
            elif isinstance(model, dict) and 'state_dict' in model:
                # Need to reconstruct model architecture - this is a limitation
                raise ValueError("State dict found but no model architecture")
        else:
            raise ValueError(f"Unsupported model format: {path.suffix}")
        
        return model.to(device)
        
    except AttributeError as e:
        # This typically happens when a custom model class isn't importable
        if "Can't get attribute" in str(e):
            logger.error(
                f"Failed to load model: Custom model class not found. "
                f"Ensure the model's class is importable (e.g., add its directory to sys.path "
                f"or install the package). Error: {e}"
            )
            raise ImportError(
                f"Model class not found. For custom models, ensure the class is importable. "
                f"For test models, add 'test_models' to sys.path before loading. "
                f"Original error: {e}"
            ) from e
        raise
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def get_layer_types(model: torch.nn.Module) -> dict:
    """
    Get count of each layer type in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Dictionary mapping layer type names to counts
    """
    layer_types = {}
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            module_type = type(module).__name__
            layer_types[module_type] = layer_types.get(module_type, 0) + 1
    
    return layer_types


def create_dummy_input(
    model: torch.nn.Module, 
    device: torch.device,
    input_shape: Optional[Tuple[int, ...]] = None
) -> torch.Tensor:
    """
    Create dummy input for model testing.
    
    This is an alias for find_compatible_input for backward compatibility.
    
    Args:
        model: PyTorch model
        device: Device to create tensor on
        input_shape: Optional explicit input shape
        
    Returns:
        torch.Tensor: Dummy input tensor
    """
    if input_shape is not None:
        return torch.randn(input_shape).to(device)
    
    return find_compatible_input(model, device)
