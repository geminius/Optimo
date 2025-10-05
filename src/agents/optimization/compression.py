"""
Model compression optimization agent using tensor decomposition and other compression techniques.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import time
import copy
import math
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..base import BaseOptimizationAgent, ImpactEstimate, ValidationResult, OptimizedModel


class CompressionType(Enum):
    """Supported compression types."""
    SVD = "svd"  # Singular Value Decomposition
    CP = "cp"    # CANDECOMP/PARAFAC decomposition
    TUCKER = "tucker"  # Tucker decomposition
    LOW_RANK = "low_rank"  # Low-rank approximation
    HUFFMAN = "huffman"  # Huffman coding
    MIXED = "mixed"  # Mixed compression techniques


class DecompositionTarget(Enum):
    """Target layers for decomposition."""
    LINEAR = "linear"
    CONV2D = "conv2d"
    CONV1D = "conv1d"
    ALL = "all"


@dataclass
class CompressionConfig:
    """Configuration for model compression."""
    compression_type: CompressionType
    target_layers: DecompositionTarget = DecompositionTarget.ALL
    compression_ratio: float = 0.5  # Target compression ratio
    rank_ratio: float = 0.5  # Ratio of original rank to keep
    preserve_modules: List[str] = None  # Modules to skip compression
    svd_threshold: float = 1e-3  # Threshold for SVD truncation
    tucker_ranks: Optional[List[int]] = None  # Ranks for Tucker decomposition
    error_tolerance: float = 0.1  # Maximum acceptable approximation error
    adaptive_rank: bool = True  # Automatically determine ranks
    
    def __post_init__(self):
        if self.preserve_modules is None:
            self.preserve_modules = ['classifier', 'head', 'lm_head', 'embed_tokens']
        
        if not 0.0 < self.compression_ratio < 1.0:
            raise ValueError(f"Compression ratio must be between 0 and 1, got {self.compression_ratio}")
        
        if not 0.0 < self.rank_ratio <= 1.0:
            raise ValueError(f"Rank ratio must be between 0 and 1, got {self.rank_ratio}")


class SVDLinear(nn.Module):
    """Linear layer with SVD decomposition."""
    
    def __init__(self, original_layer: nn.Linear, rank: int):
        super().__init__()
        self.rank = rank
        
        # Perform SVD on the weight matrix
        U, S, Vt = torch.svd(original_layer.weight.data)
        
        # Keep only top-k singular values
        self.U = nn.Parameter(U[:, :rank])
        self.S = nn.Parameter(S[:rank])
        self.Vt = nn.Parameter(Vt[:, :rank])  # Vt is [in_features, rank] after truncation
        
        # Copy bias if it exists
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct weight matrix: W = U @ diag(S) @ Vt.T
        weight = self.U @ torch.diag(self.S) @ self.Vt.T
        return torch.nn.functional.linear(x, weight, self.bias)
    
    def get_compression_ratio(self, original_params: int) -> float:
        """Calculate compression ratio compared to original layer."""
        compressed_params = self.U.numel() + self.S.numel() + self.Vt.numel()
        if self.bias is not None:
            compressed_params += self.bias.numel()
        return compressed_params / original_params


class LowRankLinear(nn.Module):
    """Linear layer with low-rank factorization."""
    
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.rank = rank
        
        # Factorize W = AB where A is (out_features, rank) and B is (rank, in_features)
        self.A = nn.Parameter(torch.randn(out_features, rank) * math.sqrt(2.0 / rank))
        self.B = nn.Parameter(torch.randn(rank, in_features) * math.sqrt(2.0 / rank))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # W = AB
        weight = self.A @ self.B
        return torch.nn.functional.linear(x, weight, self.bias)
    
    @classmethod
    def from_linear(cls, linear_layer: nn.Linear, rank: int) -> 'LowRankLinear':
        """Create low-rank layer from existing linear layer."""
        layer = cls(linear_layer.in_features, linear_layer.out_features, rank, 
                   linear_layer.bias is not None)
        
        # Initialize using SVD of original weights
        U, S, Vt = torch.svd(linear_layer.weight.data)
        layer.A.data = U[:, :rank] @ torch.diag(torch.sqrt(S[:rank]))
        layer.B.data = torch.diag(torch.sqrt(S[:rank])) @ Vt[:, :rank].T
        
        if linear_layer.bias is not None:
            layer.bias.data = linear_layer.bias.data.clone()
        
        return layer


class TuckerConv2d(nn.Module):
    """Conv2d layer with Tucker decomposition."""
    
    def __init__(self, original_layer: nn.Conv2d, ranks: Tuple[int, int, int, int]):
        super().__init__()
        
        # Store original parameters
        self.stride = original_layer.stride
        self.padding = original_layer.padding
        self.dilation = original_layer.dilation
        self.groups = original_layer.groups
        
        # Tucker decomposition: W ≈ G ×₁ U₁ ×₂ U₂ ×₃ U₃ ×₄ U₄
        out_channels, in_channels, kh, kw = original_layer.weight.shape
        r1, r2, r3, r4 = ranks
        
        # Core tensor
        self.core = nn.Parameter(torch.randn(r1, r2, r3, r4))
        
        # Factor matrices
        self.U1 = nn.Parameter(torch.randn(out_channels, r1))
        self.U2 = nn.Parameter(torch.randn(in_channels, r2))
        self.U3 = nn.Parameter(torch.randn(kh, r3))
        self.U4 = nn.Parameter(torch.randn(kw, r4))
        
        # Initialize using truncated SVD approximation
        self._initialize_from_original(original_layer.weight.data)
        
        # Copy bias if it exists
        if original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.data.clone())
        else:
            self.register_parameter('bias', None)
    
    def _initialize_from_original(self, weight: torch.Tensor):
        """Initialize factors using SVD approximation."""
        # Simplified initialization - in practice would use proper Tucker decomposition
        out_channels, in_channels, kh, kw = weight.shape
        
        # Mode-1 unfolding (along output channels)
        W1 = weight.view(out_channels, -1)
        U, _, _ = torch.svd(W1)
        self.U1.data = U[:, :self.U1.shape[1]]
        
        # Mode-2 unfolding (along input channels)
        W2 = weight.permute(1, 0, 2, 3).contiguous().view(in_channels, -1)
        U, _, _ = torch.svd(W2)
        self.U2.data = U[:, :self.U2.shape[1]]
        
        # Initialize core tensor randomly (proper Tucker would compute this)
        nn.init.normal_(self.core, 0, 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Reconstruct weight tensor from Tucker factors
        # This is a simplified reconstruction
        weight = torch.einsum('ijkl,oi,pj,qk,rl->opqr', 
                             self.core, self.U1, self.U2, self.U3, self.U4)
        
        return torch.nn.functional.conv2d(x, weight, self.bias, 
                                        self.stride, self.padding, self.dilation, self.groups)


class CompressionAgent(BaseOptimizationAgent):
    """
    Optimization agent that implements various model compression techniques
    including tensor decomposition, low-rank approximation, and other methods.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Default compression settings
        self.default_config = CompressionConfig(
            compression_type=CompressionType.SVD,
            target_layers=DecompositionTarget.ALL,
            compression_ratio=config.get('compression_ratio', 0.5),
            preserve_modules=config.get('preserve_modules', [])
        )
        
        # Supported layer types for compression
        self.compressible_layers = (nn.Linear, nn.Conv2d, nn.Conv1d)
        
    def initialize(self) -> bool:
        """Initialize the compression agent."""
        try:
            # Test SVD functionality
            test_matrix = torch.randn(10, 20)
            U, S, Vt = torch.svd(test_matrix)
            
            # Test low-rank approximation
            rank = 5
            approx = U[:, :rank] @ torch.diag(S[:rank]) @ Vt[:rank, :]
            
            self.logger.info("Tensor decomposition functionality verified")
            self.logger.info("CompressionAgent initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize CompressionAgent: {e}")
            return False
    
    def cleanup(self) -> None:
        """Clean up resources used by the agent."""
        # No specific cleanup needed for compression
        self.logger.info("CompressionAgent cleanup completed")
    
    def can_optimize(self, model: torch.nn.Module) -> bool:
        """Check if this agent can optimize the given model."""
        # Check if model has compressible layers
        compressible_layers_found = False
        for module in model.modules():
            if isinstance(module, self.compressible_layers):
                compressible_layers_found = True
                break
        
        if not compressible_layers_found:
            self.logger.warning("No compressible layers found in model")
            return False
        
        # Check model size (compression is most beneficial for larger models)
        param_count = sum(p.numel() for p in model.parameters())
        if param_count < 10_000:  # Less than 10K parameters
            self.logger.warning(f"Model too small for effective compression ({param_count:,} parameters)")
            return False
        
        return True
    
    def estimate_impact(self, model: torch.nn.Module) -> ImpactEstimate:
        """Estimate the impact of compression on the model."""
        # Count compressible parameters
        total_params = sum(p.numel() for p in model.parameters())
        compressible_params = 0
        
        for module in model.modules():
            if isinstance(module, self.compressible_layers):
                compressible_params += sum(p.numel() for p in module.parameters())
        
        compressible_ratio = compressible_params / total_params if total_params > 0 else 0
        
        # Estimate compression based on default settings
        target_compression = self.default_config.compression_ratio
        size_reduction = compressible_ratio * target_compression
        
        # Performance improvement estimates (conservative)
        performance_improvement = min(0.1, size_reduction * 0.2)  # Slight improvement due to regularization
        speed_improvement = min(0.4, size_reduction * 0.6)  # Speed improvement from smaller model
        
        # Confidence based on model characteristics
        confidence = 0.75 if compressible_ratio > 0.7 else 0.6
        
        # Estimated time based on model size and decomposition complexity
        estimated_time = max(5, int(total_params / 1_000_000 * 3))  # 3 minutes per million params
        
        return ImpactEstimate(
            performance_improvement=performance_improvement,
            size_reduction=size_reduction,
            speed_improvement=speed_improvement,
            confidence=confidence,
            estimated_time_minutes=estimated_time
        )
    
    def optimize(self, model: torch.nn.Module, config: Dict[str, Any]) -> OptimizedModel:
        """Execute compression optimization on the model."""
        start_time = time.time()
        
        # Parse configuration
        compress_config = self._parse_config(config)
        
        # Create a copy of the model for optimization
        optimized_model = copy.deepcopy(model)
        
        # Track optimization metadata
        optimization_metadata = {
            "compression_type": compress_config.compression_type.value,
            "target_layers": compress_config.target_layers.value,
            "compression_ratio": compress_config.compression_ratio,
            "rank_ratio": compress_config.rank_ratio,
            "preserved_modules": compress_config.preserve_modules.copy(),
            "adaptive_rank": compress_config.adaptive_rank
        }
        
        try:
            # Execute compression based on type
            if compress_config.compression_type == CompressionType.SVD:
                optimized_model = self._compress_svd(optimized_model, compress_config)
            elif compress_config.compression_type == CompressionType.LOW_RANK:
                optimized_model = self._compress_low_rank(optimized_model, compress_config)
            elif compress_config.compression_type == CompressionType.TUCKER:
                optimized_model = self._compress_tucker(optimized_model, compress_config)
            elif compress_config.compression_type == CompressionType.CP:
                optimized_model = self._compress_cp(optimized_model, compress_config)
            elif compress_config.compression_type == CompressionType.HUFFMAN:
                optimized_model = self._compress_huffman(optimized_model, compress_config)
            elif compress_config.compression_type == CompressionType.MIXED:
                optimized_model = self._compress_mixed(optimized_model, compress_config)
            else:
                raise ValueError(f"Unsupported compression type: {compress_config.compression_type}")
            
            # Calculate performance metrics
            original_params = sum(p.numel() for p in model.parameters())
            optimized_params = sum(p.numel() for p in optimized_model.parameters())
            actual_compression = (original_params - optimized_params) / original_params
            
            performance_metrics = {
                "original_parameters": original_params,
                "optimized_parameters": optimized_params,
                "compression_ratio": actual_compression,
                "target_compression": compress_config.compression_ratio,
                "parameter_reduction": original_params - optimized_params,
                "compressed_layers": self._count_compressed_layers(optimized_model)
            }
            
            optimization_time = time.time() - start_time
            
            return OptimizedModel(
                model=optimized_model,
                optimization_metadata=optimization_metadata,
                performance_metrics=performance_metrics,
                optimization_time=optimization_time,
                technique_used=f"compression_{compress_config.compression_type.value}"
            )
            
        except Exception as e:
            self.logger.error(f"Model compression failed: {e}")
            raise
    
    def validate_result(self, original: torch.nn.Module, optimized: torch.nn.Module) -> ValidationResult:
        """Validate the compression result."""
        issues = []
        recommendations = []
        
        try:
            # Check parameter reduction
            original_params = sum(p.numel() for p in original.parameters())
            optimized_params = sum(p.numel() for p in optimized.parameters())
            compression_ratio = (original_params - optimized_params) / original_params
            
            if compression_ratio < 0.05:
                issues.append("Insufficient compression achieved")
            
            # Check for compressed layers
            compressed_layers = self._count_compressed_layers(optimized)
            if compressed_layers == 0:
                issues.append("No layers were compressed")
            
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
                
                # Calculate approximation error
                relative_error = mse / (torch.norm(original_output).item() ** 2 + 1e-8)
                
                performance_metrics = {
                    "output_mse": mse,
                    "cosine_similarity": cosine_sim,
                    "relative_error": relative_error,
                    "compression_ratio": compression_ratio,
                    "compressed_layers": compressed_layers
                }
                
                # Recommendations based on results
                if mse > 0.5:
                    recommendations.append("High approximation error. Consider lower compression ratio.")
                
                if cosine_sim < 0.85:
                    recommendations.append("Low output similarity. Check compression parameters.")
                
                if relative_error > 0.2:
                    recommendations.append("High relative error. Model behavior may be significantly affected.")
                
                if compressed_layers < 3:
                    recommendations.append("Few layers compressed. Check compression configuration.")
                
                if compression_ratio > 0.8:
                    recommendations.append("Very high compression achieved. Verify model performance.")
                
            except Exception as e:
                issues.append(f"Model inference test failed: {e}")
                performance_metrics = {
                    "compression_ratio": compression_ratio,
                    "compressed_layers": compressed_layers
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
                recommendations=["Check model compatibility and compression configuration"]
            )
    
    def get_supported_techniques(self) -> List[str]:
        """Get list of compression techniques supported by this agent."""
        return [
            "svd",
            "low_rank",
            "tucker",
            "cp",
            "huffman",
            "mixed"
        ]
    
    def _parse_config(self, config: Dict[str, Any]) -> CompressionConfig:
        """Parse and validate compression configuration."""
        compress_type = config.get('compression_type', 'svd')
        
        try:
            compression_type = CompressionType(compress_type)
        except ValueError:
            self.logger.warning(f"Unknown compression type: {compress_type}, defaulting to svd")
            compression_type = CompressionType.SVD
        
        target = config.get('target_layers', 'all')
        try:
            target_layers = DecompositionTarget(target)
        except ValueError:
            self.logger.warning(f"Unknown target layers: {target}, defaulting to all")
            target_layers = DecompositionTarget.ALL
        
        return CompressionConfig(
            compression_type=compression_type,
            target_layers=target_layers,
            compression_ratio=config.get('compression_ratio', 0.5),
            rank_ratio=config.get('rank_ratio', 0.5),
            preserve_modules=config.get('preserve_modules', self.default_config.preserve_modules),
            svd_threshold=config.get('svd_threshold', 1e-3),
            tucker_ranks=config.get('tucker_ranks'),
            error_tolerance=config.get('error_tolerance', 0.1),
            adaptive_rank=config.get('adaptive_rank', True)
        )
    
    def _compress_svd(self, model: torch.nn.Module, config: CompressionConfig) -> torch.nn.Module:
        """Apply SVD compression to the model."""
        self._update_progress(self._current_status, 30.0, "Applying SVD compression")
        
        compressed_layers = 0
        for name, module in model.named_modules():
            if (isinstance(module, nn.Linear) and 
                self._should_compress_layer(name, module, config)):
                
                # Determine rank
                if config.adaptive_rank:
                    rank = self._determine_optimal_rank_svd(module, config)
                else:
                    rank = max(1, int(min(module.weight.shape) * config.rank_ratio))
                
                # Replace with SVD layer
                svd_layer = SVDLinear(module, rank)
                self._replace_module(model, name, svd_layer)
                compressed_layers += 1
        
        self.logger.info(f"Applied SVD compression to {compressed_layers} layers")
        return model
    
    def _compress_low_rank(self, model: torch.nn.Module, config: CompressionConfig) -> torch.nn.Module:
        """Apply low-rank compression to the model."""
        self._update_progress(self._current_status, 30.0, "Applying low-rank compression")
        
        compressed_layers = 0
        for name, module in model.named_modules():
            if (isinstance(module, nn.Linear) and 
                self._should_compress_layer(name, module, config)):
                
                # Determine rank
                rank = max(1, int(min(module.weight.shape) * config.rank_ratio))
                
                # Replace with low-rank layer
                lr_layer = LowRankLinear.from_linear(module, rank)
                self._replace_module(model, name, lr_layer)
                compressed_layers += 1
        
        self.logger.info(f"Applied low-rank compression to {compressed_layers} layers")
        return model
    
    def _compress_tucker(self, model: torch.nn.Module, config: CompressionConfig) -> torch.nn.Module:
        """Apply Tucker decomposition compression to the model."""
        self._update_progress(self._current_status, 30.0, "Applying Tucker compression")
        
        compressed_layers = 0
        for name, module in model.named_modules():
            if (isinstance(module, nn.Conv2d) and 
                self._should_compress_layer(name, module, config)):
                
                # Determine ranks for Tucker decomposition
                if config.tucker_ranks:
                    ranks = tuple(config.tucker_ranks[:4])
                else:
                    # Use rank ratio for all modes
                    out_ch, in_ch, kh, kw = module.weight.shape
                    ranks = (
                        max(1, int(out_ch * config.rank_ratio)),
                        max(1, int(in_ch * config.rank_ratio)),
                        max(1, int(kh * config.rank_ratio)),
                        max(1, int(kw * config.rank_ratio))
                    )
                
                # Replace with Tucker layer
                tucker_layer = TuckerConv2d(module, ranks)
                self._replace_module(model, name, tucker_layer)
                compressed_layers += 1
        
        self.logger.info(f"Applied Tucker compression to {compressed_layers} layers")
        return model
    
    def _compress_cp(self, model: torch.nn.Module, config: CompressionConfig) -> torch.nn.Module:
        """Apply CP decomposition compression (simplified implementation)."""
        self._update_progress(self._current_status, 30.0, "Applying CP compression")
        
        # For simplicity, fall back to SVD for linear layers
        return self._compress_svd(model, config)
    
    def _compress_huffman(self, model: torch.nn.Module, config: CompressionConfig) -> torch.nn.Module:
        """Apply Huffman coding compression (simplified implementation)."""
        self._update_progress(self._current_status, 30.0, "Applying Huffman compression")
        
        # Huffman coding would require custom storage format
        # For now, fall back to SVD compression
        return self._compress_svd(model, config)
    
    def _compress_mixed(self, model: torch.nn.Module, config: CompressionConfig) -> torch.nn.Module:
        """Apply mixed compression techniques."""
        self._update_progress(self._current_status, 30.0, "Applying mixed compression")
        
        # Apply SVD to linear layers and Tucker to conv layers
        compressed_layers = 0
        
        for name, module in model.named_modules():
            if self._should_compress_layer(name, module, config):
                if isinstance(module, nn.Linear):
                    # Use SVD for linear layers
                    rank = max(1, int(min(module.weight.shape) * config.rank_ratio))
                    svd_layer = SVDLinear(module, rank)
                    self._replace_module(model, name, svd_layer)
                    compressed_layers += 1
                elif isinstance(module, nn.Conv2d):
                    # Use Tucker for conv layers
                    out_ch, in_ch, kh, kw = module.weight.shape
                    ranks = (
                        max(1, int(out_ch * config.rank_ratio)),
                        max(1, int(in_ch * config.rank_ratio)),
                        max(1, int(kh * config.rank_ratio)),
                        max(1, int(kw * config.rank_ratio))
                    )
                    tucker_layer = TuckerConv2d(module, ranks)
                    self._replace_module(model, name, tucker_layer)
                    compressed_layers += 1
        
        self.logger.info(f"Applied mixed compression to {compressed_layers} layers")
        return model
    
    def _should_compress_layer(self, name: str, module: torch.nn.Module, 
                              config: CompressionConfig) -> bool:
        """Check if a layer should be compressed."""
        # Check if module should be preserved
        for preserve_pattern in config.preserve_modules:
            if preserve_pattern in name:
                return False
        
        # Check target layer types
        if config.target_layers == DecompositionTarget.LINEAR:
            return isinstance(module, nn.Linear)
        elif config.target_layers == DecompositionTarget.CONV2D:
            return isinstance(module, nn.Conv2d)
        elif config.target_layers == DecompositionTarget.CONV1D:
            return isinstance(module, nn.Conv1d)
        elif config.target_layers == DecompositionTarget.ALL:
            return isinstance(module, self.compressible_layers)
        
        return False
    
    def _determine_optimal_rank_svd(self, module: nn.Linear, config: CompressionConfig) -> int:
        """Determine optimal rank for SVD based on singular values."""
        U, S, Vt = torch.svd(module.weight.data)
        
        # Find rank that preserves most energy while meeting compression target
        total_energy = torch.sum(S ** 2).item()
        cumulative_energy = torch.cumsum(S ** 2, dim=0)
        
        # Find rank that preserves desired energy
        energy_threshold = (1.0 - config.compression_ratio) * total_energy
        rank = torch.searchsorted(cumulative_energy, energy_threshold).item() + 1
        
        # Ensure rank is within bounds
        max_rank = min(module.weight.shape)
        rank = max(1, min(rank, int(max_rank * config.rank_ratio)))
        
        return rank
    
    def _replace_module(self, model: torch.nn.Module, module_name: str, new_module: torch.nn.Module) -> None:
        """Replace a module in the model with a new module."""
        module_path = module_name.split('.')
        parent = model
        
        for part in module_path[:-1]:
            parent = getattr(parent, part)
        
        setattr(parent, module_path[-1], new_module)
    
    def _count_compressed_layers(self, model: torch.nn.Module) -> int:
        """Count the number of compressed layers in the model."""
        count = 0
        for module in model.modules():
            if isinstance(module, (SVDLinear, LowRankLinear, TuckerConv2d)):
                count += 1
        return count
    
    def _create_dummy_input(self, model: torch.nn.Module) -> torch.Tensor:
        """Create dummy input for model testing."""
        # Try to infer input shape from first layer
        first_layer = None
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.Conv1d)):
                first_layer = module
                break
        
        if isinstance(first_layer, nn.Linear):
            return torch.randn(1, first_layer.in_features)
        elif isinstance(first_layer, nn.Conv2d):
            return torch.randn(1, first_layer.in_channels, 224, 224)
        elif isinstance(first_layer, nn.Conv1d):
            return torch.randn(1, first_layer.in_channels, 100)
        else:
            return torch.randn(1, 768)