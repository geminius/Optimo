"""
Demo script for the CompressionAgent - Model Compression using Tensor Decomposition.

This script demonstrates how to use the CompressionAgent to compress models
using various tensor decomposition and compression techniques.
"""

import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Any, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.optimization.compression import (
    CompressionAgent, CompressionType, DecompositionTarget, CompressionConfig,
    SVDLinear, LowRankLinear, TuckerConv2d
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinearModel(nn.Module):
    """Linear model for compression demonstration."""
    
    def __init__(self, input_size=784, hidden_sizes=[512, 256, 128], output_size=10):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class ConvModel(nn.Module):
    """Convolutional model for compression demonstration."""
    
    def __init__(self, num_classes=10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def create_linear_model() -> nn.Module:
    """Create a linear model for compression."""
    model = LinearModel(input_size=784, hidden_sizes=[512, 256, 128], output_size=10)
    
    # Initialize with reasonable weights
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    logger.info(f"Created linear model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def create_conv_model() -> nn.Module:
    """Create a convolutional model for compression."""
    model = ConvModel(num_classes=10)
    
    # Initialize with reasonable weights
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    logger.info(f"Created conv model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def demonstrate_svd_compression():
    """Demonstrate SVD-based model compression."""
    logger.info("=== SVD Compression Demo ===")
    
    # Create model
    model = create_linear_model()
    
    # Create compression agent
    agent_config = {
        'compression_ratio': 0.6,  # Target 60% compression
        'preserve_modules': ['layers.9']  # Preserve output layer
    }
    agent = CompressionAgent(agent_config)
    
    # Initialize agent
    if not agent.initialize():
        logger.error("Failed to initialize CompressionAgent")
        return
    
    # Estimate impact
    impact = agent.estimate_impact(model)
    logger.info(f"Estimated impact:")
    logger.info(f"  Performance improvement: {impact.performance_improvement:.2%}")
    logger.info(f"  Size reduction: {impact.size_reduction:.2%}")
    logger.info(f"  Speed improvement: {impact.speed_improvement:.2%}")
    logger.info(f"  Confidence: {impact.confidence:.2%}")
    logger.info(f"  Estimated time: {impact.estimated_time_minutes} minutes")
    
    # Configure SVD compression
    compression_config = {
        'compression_type': 'svd',
        'target_layers': 'linear',
        'compression_ratio': 0.6,
        'rank_ratio': 0.5,
        'adaptive_rank': True,
        'svd_threshold': 1e-3
    }
    
    # Add progress tracking
    progress_updates = []
    def track_progress(update):
        progress_updates.append(update)
        logger.info(f"Progress: {update.progress_percentage:.1f}% - {update.current_step}")
        if update.message:
            logger.info(f"  Message: {update.message}")
    
    agent.add_progress_callback(track_progress)
    
    # Perform compression
    logger.info("Starting SVD compression...")
    start_time = time.time()
    
    result = agent.optimize_with_tracking(model, compression_config)
    
    end_time = time.time()
    
    if result.success:
        logger.info(f"SVD compression completed successfully in {end_time - start_time:.2f} seconds")
        
        # Analyze results
        original_params = sum(p.numel() for p in model.parameters())
        compressed_params = result.performance_metrics['optimized_parameters']
        compression_ratio = result.performance_metrics['compression_ratio']
        
        logger.info(f"Results:")
        logger.info(f"  Original parameters: {original_params:,}")
        logger.info(f"  Compressed parameters: {compressed_params:,}")
        logger.info(f"  Compression ratio: {compression_ratio:.2%}")
        logger.info(f"  Compressed layers: {result.performance_metrics['compressed_layers']}")
        logger.info(f"  Optimization time: {result.optimization_time:.2f} seconds")
        
        # Validate result
        validation = agent.validate_result(model, result.optimized_model)
        logger.info(f"Validation result: {'PASSED' if validation.is_valid else 'FAILED'}")
        
        if validation.issues:
            logger.warning("Validation issues:")
            for issue in validation.issues:
                logger.warning(f"  - {issue}")
        
        if validation.recommendations:
            logger.info("Recommendations:")
            for rec in validation.recommendations:
                logger.info(f"  - {rec}")
        
        # Test inference
        test_input = torch.randn(1, 784)
        with torch.no_grad():
            original_output = model(test_input)
            compressed_output = result.optimized_model(test_input)
        
        mse = torch.nn.functional.mse_loss(original_output, compressed_output).item()
        cosine_sim = torch.nn.functional.cosine_similarity(
            original_output.flatten(), compressed_output.flatten(), dim=0
        ).item()
        
        logger.info(f"Output comparison:")
        logger.info(f"  MSE: {mse:.6f}")
        logger.info(f"  Cosine similarity: {cosine_sim:.4f}")
        
    else:
        logger.error(f"SVD compression failed: {result.error_message}")
    
    # Cleanup
    agent.cleanup()


def demonstrate_low_rank_compression():
    """Demonstrate low-rank compression."""
    logger.info("\n=== Low-Rank Compression Demo ===")
    
    model = create_linear_model()
    agent = CompressionAgent({'compression_ratio': 0.5})
    
    if not agent.initialize():
        logger.error("Failed to initialize CompressionAgent")
        return
    
    # Configure low-rank compression
    config = {
        'compression_type': 'low_rank',
        'target_layers': 'linear',
        'rank_ratio': 0.4,
        'preserve_modules': ['layers.9']
    }
    
    logger.info("Starting low-rank compression...")
    result = agent.optimize_with_tracking(model, config)
    
    if result.success:
        logger.info("Low-rank compression completed successfully")
        
        # Show compression metrics
        metrics = result.performance_metrics
        logger.info(f"Compression achieved: {metrics.get('compression_ratio', 0):.2%}")
        logger.info(f"Compressed layers: {metrics.get('compressed_layers', 0)}")
        
    else:
        logger.error(f"Low-rank compression failed: {result.error_message}")
    
    agent.cleanup()


def demonstrate_tucker_compression():
    """Demonstrate Tucker decomposition compression."""
    logger.info("\n=== Tucker Decomposition Demo ===")
    
    model = create_conv_model()
    agent = CompressionAgent({'compression_ratio': 0.7})
    
    if not agent.initialize():
        logger.error("Failed to initialize CompressionAgent")
        return
    
    # Configure Tucker compression
    config = {
        'compression_type': 'tucker',
        'target_layers': 'conv2d',
        'rank_ratio': 0.6,
        'tucker_ranks': [32, 16, 2, 2],  # Custom ranks for first layer
        'preserve_modules': ['classifier']
    }
    
    logger.info("Starting Tucker decomposition...")
    result = agent.optimize_with_tracking(model, config)
    
    if result.success:
        logger.info("Tucker compression completed successfully")
        
        # Analyze conv-specific results
        original_params = sum(p.numel() for p in model.parameters())
        compressed_params = result.performance_metrics['optimized_parameters']
        
        logger.info(f"Conv model compression:")
        logger.info(f"  Original: {original_params:,} parameters")
        logger.info(f"  Compressed: {compressed_params:,} parameters")
        logger.info(f"  Reduction: {(original_params - compressed_params) / original_params:.2%}")
        
    else:
        logger.error(f"Tucker compression failed: {result.error_message}")
    
    agent.cleanup()


def demonstrate_mixed_compression():
    """Demonstrate mixed compression techniques."""
    logger.info("\n=== Mixed Compression Demo ===")
    
    model = create_conv_model()
    agent = CompressionAgent({'compression_ratio': 0.6})
    
    if not agent.initialize():
        logger.error("Failed to initialize CompressionAgent")
        return
    
    # Configure mixed compression
    config = {
        'compression_type': 'mixed',
        'target_layers': 'all',
        'rank_ratio': 0.5,
        'preserve_modules': []
    }
    
    logger.info("Starting mixed compression (SVD + Tucker)...")
    result = agent.optimize_with_tracking(model, config)
    
    if result.success:
        logger.info("Mixed compression completed successfully")
        
        # Show detailed results
        metrics = result.performance_metrics
        logger.info(f"Mixed compression results:")
        logger.info(f"  Compression ratio: {metrics.get('compression_ratio', 0):.2%}")
        logger.info(f"  Compressed layers: {metrics.get('compressed_layers', 0)}")
        logger.info(f"  Technique used: {result.technique_used}")
        
    else:
        logger.error(f"Mixed compression failed: {result.error_message}")
    
    agent.cleanup()


def demonstrate_compression_comparison():
    """Compare different compression techniques."""
    logger.info("\n=== Compression Techniques Comparison ===")
    
    model = create_linear_model()
    
    techniques = [
        ('svd', 'SVD Decomposition'),
        ('low_rank', 'Low-Rank Factorization'),
        ('cp', 'CP Decomposition'),
        ('huffman', 'Huffman Coding')
    ]
    
    results = {}
    
    for technique_id, technique_name in techniques:
        logger.info(f"\nTesting {technique_name}...")
        
        agent = CompressionAgent({'compression_ratio': 0.6})
        agent.initialize()
        
        config = {
            'compression_type': technique_id,
            'target_layers': 'linear',
            'rank_ratio': 0.5
        }
        
        start_time = time.time()
        result = agent.optimize_with_tracking(model, config)
        end_time = time.time()
        
        if result.success:
            original_params = sum(p.numel() for p in model.parameters())
            compressed_params = result.performance_metrics['optimized_parameters']
            compression_ratio = (original_params - compressed_params) / original_params
            
            results[technique_name] = {
                'compression_ratio': compression_ratio,
                'compressed_params': compressed_params,
                'optimization_time': end_time - start_time,
                'success': True
            }
            
            logger.info(f"  ✓ Compression: {compression_ratio:.2%}")
            logger.info(f"  ✓ Parameters: {compressed_params:,}")
            logger.info(f"  ✓ Time: {end_time - start_time:.2f}s")
        else:
            results[technique_name] = {
                'success': False,
                'error': result.error_message
            }
            logger.error(f"  ✗ Failed: {result.error_message}")
        
        agent.cleanup()
    
    # Summary
    logger.info("\n=== Compression Comparison Summary ===")
    best_technique = None
    best_compression = 0
    
    for technique, result in results.items():
        if result['success']:
            compression = result['compression_ratio']
            logger.info(f"{technique}:")
            logger.info(f"  Compression: {compression:.2%}")
            logger.info(f"  Parameters: {result['compressed_params']:,}")
            logger.info(f"  Time: {result['optimization_time']:.2f}s")
            
            if compression > best_compression:
                best_compression = compression
                best_technique = technique
        else:
            logger.info(f"{technique}: FAILED - {result['error']}")
    
    if best_technique:
        logger.info(f"\nBest technique: {best_technique} ({best_compression:.2%} compression)")


def demonstrate_svd_linear_layer():
    """Demonstrate SVDLinear layer functionality."""
    logger.info("\n=== SVDLinear Layer Demo ===")
    
    # Create original linear layer
    original = nn.Linear(256, 128, bias=True)
    nn.init.xavier_uniform_(original.weight)
    nn.init.zeros_(original.bias)
    
    logger.info(f"Original layer: {original.in_features} -> {original.out_features}")
    logger.info(f"Original parameters: {sum(p.numel() for p in original.parameters()):,}")
    
    # Create SVD version with different ranks
    ranks = [32, 64, 96]
    
    for rank in ranks:
        svd_layer = SVDLinear(original, rank)
        
        svd_params = sum(p.numel() for p in svd_layer.parameters())
        original_params = sum(p.numel() for p in original.parameters())
        compression_ratio = svd_layer.get_compression_ratio(original_params)
        
        logger.info(f"\nSVD layer (rank {rank}):")
        logger.info(f"  Parameters: {svd_params:,}")
        logger.info(f"  Compression ratio: {compression_ratio:.2%}")
        logger.info(f"  U shape: {svd_layer.U.shape}")
        logger.info(f"  S shape: {svd_layer.S.shape}")
        logger.info(f"  Vt shape: {svd_layer.Vt.shape}")
        
        # Test forward pass
        test_input = torch.randn(4, 256)
        
        with torch.no_grad():
            original_output = original(test_input)
            svd_output = svd_layer(test_input)
        
        mse = torch.nn.functional.mse_loss(original_output, svd_output).item()
        logger.info(f"  Approximation MSE: {mse:.6f}")


def demonstrate_low_rank_linear_layer():
    """Demonstrate LowRankLinear layer functionality."""
    logger.info("\n=== LowRankLinear Layer Demo ===")
    
    # Create original linear layer
    original = nn.Linear(200, 100, bias=True)
    nn.init.xavier_uniform_(original.weight)
    nn.init.zeros_(original.bias)
    
    logger.info(f"Original layer: {original.in_features} -> {original.out_features}")
    logger.info(f"Original parameters: {sum(p.numel() for p in original.parameters()):,}")
    
    # Create low-rank versions
    ranks = [25, 50, 75]
    
    for rank in ranks:
        lr_layer = LowRankLinear.from_linear(original, rank)
        
        lr_params = sum(p.numel() for p in lr_layer.parameters())
        original_params = sum(p.numel() for p in original.parameters())
        compression_ratio = lr_params / original_params
        
        logger.info(f"\nLow-rank layer (rank {rank}):")
        logger.info(f"  Parameters: {lr_params:,}")
        logger.info(f"  Compression ratio: {compression_ratio:.2%}")
        logger.info(f"  A shape: {lr_layer.A.shape}")
        logger.info(f"  B shape: {lr_layer.B.shape}")
        
        # Test forward pass
        test_input = torch.randn(4, 200)
        
        with torch.no_grad():
            original_output = original(test_input)
            lr_output = lr_layer(test_input)
        
        mse = torch.nn.functional.mse_loss(original_output, lr_output).item()
        logger.info(f"  Approximation MSE: {mse:.6f}")


def demonstrate_tucker_conv_layer():
    """Demonstrate TuckerConv2d layer functionality."""
    logger.info("\n=== TuckerConv2d Layer Demo ===")
    
    # Create original conv layer
    original = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
    nn.init.xavier_uniform_(original.weight)
    nn.init.zeros_(original.bias)
    
    logger.info(f"Original conv: {original.in_channels} -> {original.out_channels}")
    logger.info(f"Kernel size: {original.kernel_size}")
    logger.info(f"Original parameters: {sum(p.numel() for p in original.parameters()):,}")
    
    # Create Tucker version with different rank configurations
    rank_configs = [
        (32, 32, 2, 2),  # Moderate compression
        (16, 16, 2, 2),  # High compression
        (48, 48, 3, 3)   # Low compression
    ]
    
    for ranks in rank_configs:
        tucker_layer = TuckerConv2d(original, ranks)
        
        tucker_params = sum(p.numel() for p in tucker_layer.parameters())
        original_params = sum(p.numel() for p in original.parameters())
        compression_ratio = tucker_params / original_params
        
        logger.info(f"\nTucker conv (ranks {ranks}):")
        logger.info(f"  Parameters: {tucker_params:,}")
        logger.info(f"  Compression ratio: {compression_ratio:.2%}")
        logger.info(f"  Core shape: {tucker_layer.core.shape}")
        logger.info(f"  U1 shape: {tucker_layer.U1.shape}")
        logger.info(f"  U2 shape: {tucker_layer.U2.shape}")
        
        # Test forward pass
        test_input = torch.randn(2, 64, 32, 32)
        
        with torch.no_grad():
            original_output = original(test_input)
            tucker_output = tucker_layer(test_input)
        
        mse = torch.nn.functional.mse_loss(original_output, tucker_output).item()
        logger.info(f"  Approximation MSE: {mse:.6f}")


def demonstrate_adaptive_rank_selection():
    """Demonstrate adaptive rank selection for SVD."""
    logger.info("\n=== Adaptive Rank Selection Demo ===")
    
    model = create_linear_model()
    agent = CompressionAgent({})
    agent.initialize()
    
    # Test adaptive vs fixed rank selection
    configs = [
        {'adaptive_rank': True, 'compression_ratio': 0.5, 'name': 'Adaptive'},
        {'adaptive_rank': False, 'rank_ratio': 0.5, 'name': 'Fixed Ratio'}
    ]
    
    for config in configs:
        logger.info(f"\nTesting {config['name']} rank selection...")
        
        compression_config = {
            'compression_type': 'svd',
            'target_layers': 'linear',
            'adaptive_rank': config['adaptive_rank']
        }
        
        if 'compression_ratio' in config:
            compression_config['compression_ratio'] = config['compression_ratio']
        if 'rank_ratio' in config:
            compression_config['rank_ratio'] = config['rank_ratio']
        
        result = agent.optimize_with_tracking(model, compression_config)
        
        if result.success:
            metrics = result.performance_metrics
            logger.info(f"  ✓ Compression: {metrics.get('compression_ratio', 0):.2%}")
            logger.info(f"  ✓ Compressed layers: {metrics.get('compressed_layers', 0)}")
        else:
            logger.error(f"  ✗ Failed: {result.error_message}")
    
    agent.cleanup()


def main():
    """Run all compression demonstrations."""
    logger.info("Model Compression Agent Demonstration")
    logger.info("=" * 50)
    
    try:
        # Basic compression techniques
        demonstrate_svd_compression()
        demonstrate_low_rank_compression()
        demonstrate_tucker_compression()
        demonstrate_mixed_compression()
        
        # Comparison and analysis
        demonstrate_compression_comparison()
        
        # Layer-level demonstrations
        demonstrate_svd_linear_layer()
        demonstrate_low_rank_linear_layer()
        demonstrate_tucker_conv_layer()
        
        # Advanced features
        demonstrate_adaptive_rank_selection()
        
        logger.info("\n" + "=" * 50)
        logger.info("All compression demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()