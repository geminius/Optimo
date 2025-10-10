"""
Pruning optimization agent for model compression using structured and unstructured pruning techniques.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
import copy
import math
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..base import BaseOptimizationAgent, ImpactEstimate, ValidationResult, OptimizedModel


class PruningType(Enum):
    """Supported pruning types."""
    UNSTRUCTURED = "unstructured"
    STRUCTURED = "structured"
    MAGNITUDE = "magnitude"
    RANDOM = "random"
    GRADUAL = "gradual"


class SparsityPattern(Enum):
    """Supported sparsity patterns for structured pruning."""
    CHANNEL = "channel"
    FILTER = "filter"
    BLOCK = "block"
    N_M = "n_m"  # N:M sparsity (e.g., 2:4 sparsity)


@dataclass
class PruningConfig:
    """Configuration for pruning operations."""
    pruning_type: PruningType
    sparsity_ratio: float  # Target sparsity (0.0 to 1.0)
    sparsity_pattern: SparsityPattern = SparsityPattern.CHANNEL
    preserve_modules: List[str] = None  # Modules to skip pruning
    gradual_steps: int = 10  # For gradual pruning
    block_size: Tuple[int, int] = (4, 4)  # For block sparsity
    n_m_ratio: Tuple[int, int] = (2, 4)  # For N:M sparsity
    importance_metric: str = "magnitude"  # "magnitude", "gradient", "fisher"
    
    def __post_init__(self):
        if self.preserve_modules is None:
            self.preserve_modules = []
        
        # Validate sparsity ratio
        if not 0.0 <= self.sparsity_ratio <= 1.0:
            raise ValueError(f"Sparsity ratio must be between 0.0 and 1.0, got {self.sparsity_ratio}")


class PruningAgent(BaseOptimizationAgent):
    """
    Optimization agent that implements various pruning techniques including
    structured and unstructured pruning with different sparsity patterns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Default pruning settings
        self.default_config = PruningConfig(
            pruning_type=PruningType.MAGNITUDE,
            sparsity_ratio=0.5,
            preserve_modules=config.get('preserve_modules', ['classifier', 'head', 'lm_head', 'embed_tokens'])
        )
        
        # Supported layer types for pruning
        self.prunable_layers = (nn.Linear, nn.Conv2d, nn.Conv1d)
        
        # Cache for importance scores
        self._importance_cache = {}
        
    def initialize(self) -> bool:
        """Initialize the pruning agent."""
        try:
            # Test PyTorch pruning functionality
            test_layer = nn.Linear(10, 5)
            prune.random_unstructured(test_layer, name="weight", amount=0.1)
            prune.remove(test_layer, "weight")
            
            self.logger.info("PyTorch pruning functionality verified")
            self.logger.info("PruningAgent initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize PruningAgent: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources used by the agent."""
        # Clear importance score cache
        self._importance_cache.clear()
        self.logger.info("PruningAgent cleanup completed")
    
    def can_optimize(self, model: torch.nn.Module) -> bool:
        """Check if this agent can optimize the given model."""
        # Check if model has prunable layers
        prunable_layers_found = False
        for module in model.modules():
            if isinstance(module, self.prunable_layers):
                prunable_layers_found = True
                break
        
        if not prunable_layers_found:
            self.logger.warning("No prunable layers found in model")
            return False
        
        # Check model size (pruning is beneficial for models with redundancy)
        param_count = sum(p.numel() for p in model.parameters())
        if param_count < 100_000:  # Less than 100K parameters
            self.logger.warning(f"Model too small for effective pruning ({param_count:,} parameters)")
            return False
        
        return True
    
    def estimate_impact(self, model: torch.nn.Module) -> ImpactEstimate:
        """Estimate the impact of pruning on the model."""
        # Count prunable parameters
        total_params = sum(p.numel() for p in model.parameters())
        prunable_params = 0
        
        for module in model.modules():
            if isinstance(module, self.prunable_layers):
                prunable_params += sum(p.numel() for p in module.parameters())
        
        prunable_ratio = prunable_params / total_params if total_params > 0 else 0
        
        # Estimate size reduction based on default sparsity
        default_sparsity = 0.5  # 50% sparsity
        size_reduction = prunable_ratio * default_sparsity
        
        # Performance improvement estimates (conservative)
        # Pruning can improve inference speed but may reduce accuracy
        performance_improvement = min(0.2, prunable_ratio * 0.3)  # Up to 20% improvement
        speed_improvement = min(0.3, prunable_ratio * 0.4)  # Up to 30% speed improvement
        
        # Confidence based on model characteristics
        confidence = 0.7 if prunable_ratio > 0.6 else 0.5
        
        # Estimated time based on model size and pruning complexity
        estimated_time = max(3, int(total_params / 1_000_000 * 1.5))  # 1.5 minutes per million params
        
        return ImpactEstimate(
            performance_improvement=performance_improvement,
            size_reduction=size_reduction,
            speed_improvement=speed_improvement,
            confidence=confidence,
            estimated_time_minutes=estimated_time
        )
    
    def optimize(self, model: torch.nn.Module, config: Dict[str, Any]) -> OptimizedModel:
        """Execute pruning optimization on the model."""
        start_time = time.time()
        
        # Parse configuration
        prune_config = self._parse_config(config)
        
        # Create a copy of the model for optimization
        optimized_model = copy.deepcopy(model)
        
        # Track optimization metadata
        optimization_metadata = {
            "pruning_type": prune_config.pruning_type.value,
            "sparsity_ratio": prune_config.sparsity_ratio,
            "sparsity_pattern": prune_config.sparsity_pattern.value,
            "preserved_modules": prune_config.preserve_modules.copy(),
            "importance_metric": prune_config.importance_metric
        }
        
        try:
            # Execute pruning based on type
            if prune_config.pruning_type == PruningType.UNSTRUCTURED:
                optimized_model = self._prune_unstructured(optimized_model, prune_config)
            elif prune_config.pruning_type == PruningType.STRUCTURED:
                optimized_model = self._prune_structured(optimized_model, prune_config)
            elif prune_config.pruning_type == PruningType.MAGNITUDE:
                optimized_model = self._prune_magnitude_based(optimized_model, prune_config)
            elif prune_config.pruning_type == PruningType.RANDOM:
                optimized_model = self._prune_random(optimized_model, prune_config)
            elif prune_config.pruning_type == PruningType.GRADUAL:
                optimized_model = self._prune_gradual(optimized_model, prune_config)
            else:
                raise ValueError(f"Unsupported pruning type: {prune_config.pruning_type}")
            
            # Calculate performance metrics
            original_params = self._count_parameters(model)
            optimized_params = self._count_parameters(optimized_model)
            actual_sparsity = self._calculate_sparsity(optimized_model)
            
            performance_metrics = {
                "original_parameters": original_params,
                "optimized_parameters": optimized_params,
                "parameter_reduction_ratio": (original_params - optimized_params) / original_params,
                "actual_sparsity": actual_sparsity,
                "target_sparsity": prune_config.sparsity_ratio,
                "pruned_layers": self._count_pruned_layers(optimized_model)
            }
            
            optimization_time = time.time() - start_time
            
            return OptimizedModel(
                model=optimized_model,
                optimization_metadata=optimization_metadata,
                performance_metrics=performance_metrics,
                optimization_time=optimization_time,
                technique_used=f"pruning_{prune_config.pruning_type.value}"
            )
            
        except Exception as e:
            self.logger.error(f"Pruning failed: {e}")
            raise
    
    def validate_result(self, original: torch.nn.Module, optimized: torch.nn.Module) -> ValidationResult:
        """Validate the pruning result."""
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
            
            # Check for pruned layers
            pruned_layers = self._count_pruned_layers(optimized)
            if pruned_layers == 0:
                issues.append("No layers were pruned")
            
            # Calculate actual sparsity
            actual_sparsity = self._calculate_sparsity(optimized)
            
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
                
                # Calculate output similarity
                mse = torch.nn.functional.mse_loss(original_output, optimized_output).item()
                cosine_sim = torch.nn.functional.cosine_similarity(
                    original_output.flatten(), optimized_output.flatten(), dim=0
                ).item()
                
                performance_metrics = {
                    "output_mse": mse,
                    "cosine_similarity": cosine_sim,
                    "actual_sparsity": actual_sparsity,
                    "pruned_layers": pruned_layers
                }
                
                # Recommendations based on results
                if mse > 0.1:
                    recommendations.append("High output difference detected. Consider lower sparsity ratio.")
                
                if cosine_sim < 0.9:
                    recommendations.append("Low output similarity. Model behavior may have changed significantly.")
                
                if actual_sparsity < 0.1:
                    recommendations.append("Low sparsity achieved. Check pruning configuration.")
                
                if actual_sparsity > 0.9:
                    recommendations.append("Very high sparsity. Model may be over-pruned.")
                
            except Exception as e:
                issues.append(f"Model inference test failed: {e}")
                performance_metrics = {
                    "actual_sparsity": actual_sparsity,
                    "pruned_layers": pruned_layers
                }
            
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
                recommendations=["Check model compatibility and pruning configuration"]
            )
    
    def get_supported_techniques(self) -> List[str]:
        """Get list of pruning techniques supported by this agent."""
        return [
            "unstructured",
            "structured", 
            "magnitude",
            "random",
            "gradual"
        ]
    
    def _parse_config(self, config: Dict[str, Any]) -> PruningConfig:
        """Parse and validate pruning configuration."""
        prune_type = config.get('pruning_type', 'magnitude')
        
        try:
            pruning_type = PruningType(prune_type)
        except ValueError:
            self.logger.warning(f"Unknown pruning type: {prune_type}, defaulting to magnitude")
            pruning_type = PruningType.MAGNITUDE
        
        sparsity_pattern = SparsityPattern.CHANNEL
        if 'sparsity_pattern' in config:
            try:
                sparsity_pattern = SparsityPattern(config['sparsity_pattern'])
            except ValueError:
                self.logger.warning(f"Unknown sparsity pattern: {config['sparsity_pattern']}, defaulting to channel")
        
        return PruningConfig(
            pruning_type=pruning_type,
            sparsity_ratio=config.get('sparsity_ratio', 0.5),
            sparsity_pattern=sparsity_pattern,
            preserve_modules=config.get('preserve_modules', self.default_config.preserve_modules),
            gradual_steps=config.get('gradual_steps', 10),
            block_size=config.get('block_size', (4, 4)),
            n_m_ratio=config.get('n_m_ratio', (2, 4)),
            importance_metric=config.get('importance_metric', 'magnitude')
        )
    
    def _prune_unstructured(self, model: torch.nn.Module, config: PruningConfig) -> torch.nn.Module:
        """Apply unstructured pruning to the model."""
        self._update_progress(self._current_status, 30.0, "Applying unstructured pruning")
        
        pruned_layers = 0
        for name, module in model.named_modules():
            if isinstance(module, self.prunable_layers) and not self._should_preserve_module(name, config.preserve_modules):
                # Apply magnitude-based unstructured pruning to weights
                prune.l1_unstructured(module, name="weight", amount=config.sparsity_ratio)
                
                # Also prune bias if it exists and sparsity is high
                if hasattr(module, 'bias') and module.bias is not None and config.sparsity_ratio > 0.7:
                    prune.l1_unstructured(module, name="bias", amount=config.sparsity_ratio * 0.5)
                
                pruned_layers += 1
        
        # Make pruning permanent
        self._make_pruning_permanent(model)
        
        self.logger.info(f"Applied unstructured pruning to {pruned_layers} layers")
        return model
    
    def _prune_structured(self, model: torch.nn.Module, config: PruningConfig) -> torch.nn.Module:
        """Apply structured pruning to the model."""
        self._update_progress(self._current_status, 30.0, "Applying structured pruning")
        
        pruned_layers = 0
        for name, module in model.named_modules():
            if isinstance(module, self.prunable_layers) and not self._should_preserve_module(name, config.preserve_modules):
                
                if config.sparsity_pattern == SparsityPattern.CHANNEL:
                    self._prune_channels(module, config.sparsity_ratio)
                elif config.sparsity_pattern == SparsityPattern.FILTER:
                    self._prune_filters(module, config.sparsity_ratio)
                elif config.sparsity_pattern == SparsityPattern.BLOCK:
                    self._prune_blocks(module, config.sparsity_ratio, config.block_size)
                elif config.sparsity_pattern == SparsityPattern.N_M:
                    self._prune_n_m(module, config.n_m_ratio)
                
                pruned_layers += 1
        
        self.logger.info(f"Applied structured pruning to {pruned_layers} layers")
        return model
    
    def _prune_magnitude_based(self, model: torch.nn.Module, config: PruningConfig) -> torch.nn.Module:
        """Apply magnitude-based pruning to the model."""
        self._update_progress(self._current_status, 30.0, "Applying magnitude-based pruning")
        
        pruned_layers = 0
        for name, module in model.named_modules():
            if isinstance(module, self.prunable_layers) and not self._should_preserve_module(name, config.preserve_modules):
                # Use L1 norm for magnitude-based pruning
                prune.l1_unstructured(module, name="weight", amount=config.sparsity_ratio)
                pruned_layers += 1
        
        # Make pruning permanent
        self._make_pruning_permanent(model)
        
        self.logger.info(f"Applied magnitude-based pruning to {pruned_layers} layers")
        return model
    
    def _prune_random(self, model: torch.nn.Module, config: PruningConfig) -> torch.nn.Module:
        """Apply random pruning to the model."""
        self._update_progress(self._current_status, 30.0, "Applying random pruning")
        
        pruned_layers = 0
        for name, module in model.named_modules():
            if isinstance(module, self.prunable_layers) and not self._should_preserve_module(name, config.preserve_modules):
                # Apply random unstructured pruning
                prune.random_unstructured(module, name="weight", amount=config.sparsity_ratio)
                pruned_layers += 1
        
        # Make pruning permanent
        self._make_pruning_permanent(model)
        
        self.logger.info(f"Applied random pruning to {pruned_layers} layers")
        return model
    
    def _prune_gradual(self, model: torch.nn.Module, config: PruningConfig) -> torch.nn.Module:
        """Apply gradual pruning to the model."""
        self._update_progress(self._current_status, 30.0, "Applying gradual pruning")
        
        # Calculate sparsity schedule
        initial_sparsity = 0.0
        final_sparsity = config.sparsity_ratio
        steps = config.gradual_steps
        
        pruned_layers = 0
        for name, module in model.named_modules():
            if isinstance(module, self.prunable_layers) and not self._should_preserve_module(name, config.preserve_modules):
                
                # Apply gradual pruning in steps
                for step in range(steps):
                    current_sparsity = initial_sparsity + (final_sparsity - initial_sparsity) * (step + 1) / steps
                    
                    # Remove previous pruning mask if it exists
                    if hasattr(module, 'weight_mask'):
                        prune.remove(module, 'weight')
                    
                    # Apply new pruning level
                    prune.l1_unstructured(module, name="weight", amount=current_sparsity)
                    
                    # Update progress
                    progress = 30.0 + (step + 1) / steps * 40.0
                    self._update_progress(self._current_status, progress, f"Gradual pruning step {step + 1}/{steps}")
                
                pruned_layers += 1
        
        # Make pruning permanent
        self._make_pruning_permanent(model)
        
        self.logger.info(f"Applied gradual pruning to {pruned_layers} layers")
        return model
    
    def _prune_channels(self, module: torch.nn.Module, sparsity_ratio: float) -> None:
        """Apply channel-wise structured pruning."""
        if isinstance(module, nn.Linear):
            # For linear layers, prune input features (columns)
            weight = module.weight.data
            channel_importance = torch.norm(weight, dim=0)  # L2 norm across output dimension
            
            num_channels_to_prune = int(sparsity_ratio * weight.size(1))
            _, indices_to_prune = torch.topk(channel_importance, num_channels_to_prune, largest=False)
            
            # Zero out the selected channels
            weight[:, indices_to_prune] = 0
            
        elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
            # For conv layers, prune input channels
            weight = module.weight.data
            if isinstance(module, nn.Conv2d):
                # Reshape to compute norm properly: (out_channels, in_channels, h, w) -> norm over (out, h, w)
                channel_importance = torch.norm(weight.view(weight.size(0), weight.size(1), -1), dim=(0, 2))
            else:  # Conv1d
                # Reshape to compute norm properly: (out_channels, in_channels, length) -> norm over (out, length)
                channel_importance = torch.norm(weight.view(weight.size(0), weight.size(1), -1), dim=(0, 2))
            
            num_channels_to_prune = int(sparsity_ratio * weight.size(1))
            _, indices_to_prune = torch.topk(channel_importance, num_channels_to_prune, largest=False)
            
            # Zero out the selected channels
            weight[:, indices_to_prune] = 0
    
    def _prune_filters(self, module: torch.nn.Module, sparsity_ratio: float) -> None:
        """Apply filter-wise structured pruning."""
        if isinstance(module, nn.Linear):
            # For linear layers, prune output features (rows)
            weight = module.weight.data
            filter_importance = torch.norm(weight, dim=1)  # L2 norm across input dimension
            
            num_filters_to_prune = int(sparsity_ratio * weight.size(0))
            _, indices_to_prune = torch.topk(filter_importance, num_filters_to_prune, largest=False)
            
            # Zero out the selected filters
            weight[indices_to_prune, :] = 0
            
            # Also zero out corresponding bias if it exists
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data[indices_to_prune] = 0
            
        elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
            # For conv layers, prune output filters
            weight = module.weight.data
            if isinstance(module, nn.Conv2d):
                # Reshape to compute norm properly: (out_channels, in_channels, h, w) -> norm over (in, h, w)
                filter_importance = torch.norm(weight.view(weight.size(0), -1), dim=1)
            else:  # Conv1d
                # Reshape to compute norm properly: (out_channels, in_channels, length) -> norm over (in, length)
                filter_importance = torch.norm(weight.view(weight.size(0), -1), dim=1)
            
            num_filters_to_prune = int(sparsity_ratio * weight.size(0))
            _, indices_to_prune = torch.topk(filter_importance, num_filters_to_prune, largest=False)
            
            # Zero out the selected filters
            weight[indices_to_prune] = 0
            
            # Also zero out corresponding bias if it exists
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data[indices_to_prune] = 0
    
    def _prune_blocks(self, module: torch.nn.Module, sparsity_ratio: float, block_size: Tuple[int, int]) -> None:
        """Apply block-wise structured pruning."""
        if not isinstance(module, nn.Linear):
            self.logger.warning(f"Block pruning not implemented for {type(module)}")
            return
        
        weight = module.weight.data
        rows, cols = weight.shape
        block_h, block_w = block_size
        
        # Calculate number of blocks
        num_blocks_h = math.ceil(rows / block_h)
        num_blocks_w = math.ceil(cols / block_w)
        total_blocks = num_blocks_h * num_blocks_w
        
        # Calculate block importance scores
        block_scores = []
        block_positions = []
        
        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                start_row = i * block_h
                end_row = min((i + 1) * block_h, rows)
                start_col = j * block_w
                end_col = min((j + 1) * block_w, cols)
                
                block = weight[start_row:end_row, start_col:end_col]
                score = torch.norm(block).item()
                
                block_scores.append(score)
                block_positions.append((start_row, end_row, start_col, end_col))
        
        # Select blocks to prune
        num_blocks_to_prune = int(sparsity_ratio * total_blocks)
        block_indices = np.argsort(block_scores)[:num_blocks_to_prune]
        
        # Zero out selected blocks
        for idx in block_indices:
            start_row, end_row, start_col, end_col = block_positions[idx]
            weight[start_row:end_row, start_col:end_col] = 0
    
    def _prune_n_m(self, module: torch.nn.Module, n_m_ratio: Tuple[int, int]) -> None:
        """Apply N:M structured sparsity."""
        n, m = n_m_ratio
        if n >= m:
            self.logger.warning(f"Invalid N:M ratio {n}:{m}, N should be less than M")
            return
        
        weight = module.weight.data
        
        # Reshape weight to process in groups of M
        original_shape = weight.shape
        weight_flat = weight.view(-1)
        
        # Pad if necessary to make divisible by M
        remainder = weight_flat.numel() % m
        if remainder != 0:
            padding = m - remainder
            weight_flat = torch.cat([weight_flat, torch.zeros(padding, device=weight.device)])
        
        # Reshape to groups of M
        weight_groups = weight_flat.view(-1, m)
        
        # For each group of M, keep only the N largest magnitude values
        for i in range(weight_groups.size(0)):
            group = weight_groups[i]
            _, indices = torch.topk(torch.abs(group), n, largest=True)
            
            # Create mask to keep only top N values
            mask = torch.zeros_like(group)
            mask[indices] = 1
            
            # Apply mask
            weight_groups[i] = group * mask
        
        # Reshape back to original shape
        weight_pruned = weight_groups.view(-1)[:weight.numel()].view(original_shape)
        weight.copy_(weight_pruned)
    
    def _should_preserve_module(self, module_name: str, preserve_modules: List[str]) -> bool:
        """Check if a module should be preserved from pruning."""
        for preserve_pattern in preserve_modules:
            if preserve_pattern in module_name:
                return True
        return False
    
    def _make_pruning_permanent(self, model: torch.nn.Module) -> None:
        """Make pruning permanent by removing pruning masks."""
        for module in model.modules():
            if isinstance(module, self.prunable_layers):
                # Remove weight mask if it exists
                if hasattr(module, 'weight_mask'):
                    prune.remove(module, 'weight')
                
                # Remove bias mask if it exists
                if hasattr(module, 'bias_mask'):
                    prune.remove(module, 'bias')
    
    def _count_parameters(self, model: torch.nn.Module) -> int:
        """Count total number of parameters in the model."""
        return sum(p.numel() for p in model.parameters())
    
    def _calculate_sparsity(self, model: torch.nn.Module) -> float:
        """Calculate the overall sparsity of the model."""
        total_params = 0
        zero_params = 0
        
        for param in model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        return zero_params / total_params if total_params > 0 else 0.0
    
    def _count_pruned_layers(self, model: torch.nn.Module) -> int:
        """Count the number of layers that have been pruned."""
        pruned_count = 0
        
        for module in model.modules():
            if isinstance(module, self.prunable_layers):
                # Check if any parameters are zero (indicating pruning)
                for param in module.parameters():
                    if (param == 0).any():
                        pruned_count += 1
                        break
        
        return pruned_count
    
    def _create_dummy_input(self, model: torch.nn.Module) -> torch.Tensor:
        """Create dummy input for model testing."""
        from ...utils.model_utils import create_dummy_input
        device = next(model.parameters()).device if list(model.parameters()) else torch.device('cpu')
        return create_dummy_input(model, device)