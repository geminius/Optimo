"""
Demo script for the ArchitectureSearchAgent - Neural Architecture Search Optimization.

This script demonstrates how to use the ArchitectureSearchAgent to find optimal
model architectures using various search strategies.
"""

import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Any, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.optimization.architecture_search import (
    ArchitectureSearchAgent, SearchStrategy, ArchitectureSpace, SearchConfig,
    ArchitectureCandidate, ArchitectureGenerator, ArchitectureEvaluator
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModel(nn.Module):
    """Baseline model for architecture search comparison."""
    
    def __init__(self, input_size=784, hidden_size=256, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)


def create_baseline_model() -> nn.Module:
    """Create a baseline model for architecture search."""
    model = BaselineModel(input_size=784, hidden_size=256, output_size=10)
    
    # Initialize with reasonable weights
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    logger.info(f"Created baseline model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def demonstrate_random_search():
    """Demonstrate random architecture search."""
    logger.info("=== Random Architecture Search Demo ===")
    
    # Create baseline model
    baseline_model = create_baseline_model()
    
    # Create architecture search agent
    agent_config = {
        'max_iterations': 20,
        'population_size': 10
    }
    agent = ArchitectureSearchAgent(agent_config)
    
    # Initialize agent
    if not agent.initialize():
        logger.error("Failed to initialize ArchitectureSearchAgent")
        return
    
    # Estimate impact
    impact = agent.estimate_impact(baseline_model)
    logger.info(f"Estimated impact:")
    logger.info(f"  Performance improvement: {impact.performance_improvement:.2%}")
    logger.info(f"  Size reduction: {impact.size_reduction:.2%}")
    logger.info(f"  Speed improvement: {impact.speed_improvement:.2%}")
    logger.info(f"  Confidence: {impact.confidence:.2%}")
    logger.info(f"  Estimated time: {impact.estimated_time_minutes} minutes")
    
    # Configure random search
    search_config = {
        'search_strategy': 'random',
        'search_space': 'full',
        'max_iterations': 15,
        'min_layers': 2,
        'max_layers': 6,
        'min_width': 64,
        'max_width': 512,
        'available_operations': ['linear', 'relu', 'gelu'],
        'performance_metric': 'accuracy'
    }
    
    # Add progress tracking
    progress_updates = []
    def track_progress(update):
        progress_updates.append(update)
        logger.info(f"Progress: {update.progress_percentage:.1f}% - {update.current_step}")
        if update.message:
            logger.info(f"  Message: {update.message}")
    
    agent.add_progress_callback(track_progress)
    
    # Perform architecture search
    logger.info("Starting random architecture search...")
    start_time = time.time()
    
    result = agent.optimize_with_tracking(baseline_model, search_config)
    
    end_time = time.time()
    
    if result.success:
        logger.info(f"Architecture search completed successfully in {end_time - start_time:.2f} seconds")
        
        # Analyze results
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        optimized_params = result.performance_metrics['optimized_parameters']
        
        logger.info(f"Results:")
        logger.info(f"  Baseline parameters: {baseline_params:,}")
        logger.info(f"  Optimized parameters: {optimized_params:,}")
        logger.info(f"  Parameter ratio: {optimized_params / baseline_params:.2f}")
        logger.info(f"  Architecture score: {result.performance_metrics['architecture_score']:.4f}")
        logger.info(f"  Number of layers: {result.performance_metrics['num_layers']}")
        logger.info(f"  Skip connections: {result.performance_metrics['skip_connections']}")
        logger.info(f"  Optimization time: {result.optimization_time:.2f} seconds")
        
        # Validate result
        validation = agent.validate_result(baseline_model, result.optimized_model)
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
            baseline_output = baseline_model(test_input)
            optimized_output = result.optimized_model(test_input)
        
        output_similarity = torch.nn.functional.cosine_similarity(
            baseline_output.flatten(), optimized_output.flatten(), dim=0
        ).item()
        logger.info(f"Output similarity: {output_similarity:.4f}")
        
    else:
        logger.error(f"Architecture search failed: {result.error_message}")
    
    # Cleanup
    agent.cleanup()


def demonstrate_evolutionary_search():
    """Demonstrate evolutionary architecture search."""
    logger.info("\n=== Evolutionary Architecture Search Demo ===")
    
    baseline_model = create_baseline_model()
    agent = ArchitectureSearchAgent({'max_iterations': 10, 'population_size': 8})
    
    if not agent.initialize():
        logger.error("Failed to initialize ArchitectureSearchAgent")
        return
    
    # Configure evolutionary search
    config = {
        'search_strategy': 'evolutionary',
        'search_space': 'width',
        'max_iterations': 8,
        'population_size': 6,
        'mutation_rate': 0.2,
        'crossover_rate': 0.8,
        'early_stopping_patience': 3,
        'min_layers': 3,
        'max_layers': 5,
        'min_width': 32,
        'max_width': 256
    }
    
    logger.info("Starting evolutionary architecture search...")
    result = agent.optimize_with_tracking(baseline_model, config)
    
    if result.success:
        logger.info("Evolutionary search completed successfully")
        
        # Show evolution metrics
        metrics = result.performance_metrics
        logger.info(f"Best architecture score: {metrics.get('architecture_score', 0):.4f}")
        logger.info(f"Final parameter count: {metrics.get('optimized_parameters', 0):,}")
        
    else:
        logger.error(f"Evolutionary search failed: {result.error_message}")
    
    agent.cleanup()


def demonstrate_search_spaces():
    """Demonstrate different architecture search spaces."""
    logger.info("\n=== Architecture Search Spaces Demo ===")
    
    baseline_model = create_baseline_model()
    
    search_spaces = [
        ('depth', 'Depth Search'),
        ('width', 'Width Search'),
        ('operations', 'Operations Search'),
        ('connections', 'Connections Search'),
        ('full', 'Full Search')
    ]
    
    results = {}
    
    for space_id, space_name in search_spaces:
        logger.info(f"\nTesting {space_name}...")
        
        agent = ArchitectureSearchAgent({'max_iterations': 5, 'population_size': 4})
        agent.initialize()
        
        config = {
            'search_strategy': 'random',
            'search_space': space_id,
            'max_iterations': 5,
            'min_layers': 2,
            'max_layers': 4,
            'min_width': 64,
            'max_width': 256
        }
        
        start_time = time.time()
        result = agent.optimize_with_tracking(baseline_model, config)
        end_time = time.time()
        
        if result.success:
            baseline_params = sum(p.numel() for p in baseline_model.parameters())
            optimized_params = result.performance_metrics['optimized_parameters']
            
            results[space_name] = {
                'parameter_ratio': optimized_params / baseline_params,
                'architecture_score': result.performance_metrics['architecture_score'],
                'optimization_time': end_time - start_time,
                'success': True
            }
            
            logger.info(f"  ✓ Parameter ratio: {optimized_params / baseline_params:.2f}")
            logger.info(f"  ✓ Score: {result.performance_metrics['architecture_score']:.4f}")
            logger.info(f"  ✓ Time: {end_time - start_time:.2f}s")
        else:
            results[space_name] = {
                'success': False,
                'error': result.error_message
            }
            logger.error(f"  ✗ Failed: {result.error_message}")
        
        agent.cleanup()
    
    # Summary
    logger.info("\n=== Search Spaces Summary ===")
    for space, result in results.items():
        if result['success']:
            logger.info(f"{space}:")
            logger.info(f"  Parameter ratio: {result['parameter_ratio']:.2f}")
            logger.info(f"  Architecture score: {result['architecture_score']:.4f}")
            logger.info(f"  Time: {result['optimization_time']:.2f}s")
        else:
            logger.info(f"{space}: FAILED - {result['error']}")


def demonstrate_search_strategies():
    """Compare different search strategies."""
    logger.info("\n=== Search Strategies Comparison ===")
    
    baseline_model = create_baseline_model()
    
    strategies = [
        ('random', 'Random Search'),
        ('evolutionary', 'Evolutionary Search'),
        ('gradient_based', 'Gradient-based Search'),
        ('progressive', 'Progressive Search')
    ]
    
    results = {}
    
    for strategy_id, strategy_name in strategies:
        logger.info(f"\nTesting {strategy_name}...")
        
        agent = ArchitectureSearchAgent({'max_iterations': 6, 'population_size': 4})
        agent.initialize()
        
        config = {
            'search_strategy': strategy_id,
            'search_space': 'full',
            'max_iterations': 6,
            'population_size': 4,
            'min_layers': 2,
            'max_layers': 4
        }
        
        start_time = time.time()
        result = agent.optimize_with_tracking(baseline_model, config)
        end_time = time.time()
        
        if result.success:
            results[strategy_name] = {
                'architecture_score': result.performance_metrics['architecture_score'],
                'parameter_count': result.performance_metrics['optimized_parameters'],
                'optimization_time': end_time - start_time,
                'success': True
            }
            
            logger.info(f"  ✓ Score: {result.performance_metrics['architecture_score']:.4f}")
            logger.info(f"  ✓ Parameters: {result.performance_metrics['optimized_parameters']:,}")
            logger.info(f"  ✓ Time: {end_time - start_time:.2f}s")
        else:
            results[strategy_name] = {
                'success': False,
                'error': result.error_message
            }
            logger.error(f"  ✗ Failed: {result.error_message}")
        
        agent.cleanup()
    
    # Find best strategy
    logger.info("\n=== Strategy Comparison Summary ===")
    best_strategy = None
    best_score = -1
    
    for strategy, result in results.items():
        if result['success']:
            score = result['architecture_score']
            logger.info(f"{strategy}:")
            logger.info(f"  Score: {score:.4f}")
            logger.info(f"  Parameters: {result['parameter_count']:,}")
            logger.info(f"  Time: {result['optimization_time']:.2f}s")
            
            if score > best_score:
                best_score = score
                best_strategy = strategy
        else:
            logger.info(f"{strategy}: FAILED - {result['error']}")
    
    if best_strategy:
        logger.info(f"\nBest strategy: {best_strategy} (score: {best_score:.4f})")


def demonstrate_architecture_generator():
    """Demonstrate architecture generation capabilities."""
    logger.info("\n=== Architecture Generator Demo ===")
    
    # Create generator with different configurations
    config = SearchConfig(
        search_strategy=SearchStrategy.RANDOM,
        search_space=ArchitectureSpace.FULL,
        min_layers=2,
        max_layers=5,
        min_width=32,
        max_width=256,
        available_operations=['linear', 'conv', 'relu', 'gelu']
    )
    
    generator = ArchitectureGenerator(config)
    
    logger.info("Generating sample architectures...")
    
    # Generate several architectures
    architectures = []
    for i in range(5):
        arch = generator.generate_random_architecture(input_size=784, output_size=10)
        architectures.append(arch)
        
        logger.info(f"Architecture {i+1}:")
        logger.info(f"  Layers: {len(arch.layers)}")
        logger.info(f"  Connections: {len(arch.connections)}")
        
        for j, layer in enumerate(arch.layers):
            logger.info(f"    Layer {j}: {layer['type']} "
                       f"({layer['input_size']} -> {layer['output_size']}) "
                       f"activation: {layer.get('activation', 'none')}")
    
    # Demonstrate mutation
    logger.info("\nDemonstrating architecture mutation...")
    original = architectures[0]
    mutated = generator.mutate_architecture(original)
    
    logger.info(f"Original layers: {len(original.layers)}")
    logger.info(f"Mutated layers: {len(mutated.layers)}")
    logger.info(f"Original connections: {len(original.connections)}")
    logger.info(f"Mutated connections: {len(mutated.connections)}")
    
    # Demonstrate crossover
    logger.info("\nDemonstrating architecture crossover...")
    parent1 = architectures[0]
    parent2 = architectures[1]
    offspring = generator.crossover_architectures(parent1, parent2)
    
    logger.info(f"Parent 1 layers: {len(parent1.layers)}")
    logger.info(f"Parent 2 layers: {len(parent2.layers)}")
    logger.info(f"Offspring layers: {len(offspring.layers)}")


def demonstrate_architecture_evaluator():
    """Demonstrate architecture evaluation capabilities."""
    logger.info("\n=== Architecture Evaluator Demo ===")
    
    baseline_model = create_baseline_model()
    
    config = SearchConfig(
        search_strategy=SearchStrategy.RANDOM,
        search_space=ArchitectureSpace.FULL
    )
    
    evaluator = ArchitectureEvaluator(config)
    
    # Create sample architecture
    layers = [
        {"type": "linear", "input_size": 784, "output_size": 128, "activation": "relu"},
        {"type": "linear", "input_size": 128, "output_size": 64, "activation": "gelu"},
        {"type": "linear", "input_size": 64, "output_size": 10, "activation": "relu"}
    ]
    
    candidate = ArchitectureCandidate(layers=layers, connections=[(0, 2)])
    
    logger.info("Evaluating sample architecture...")
    logger.info(f"Architecture layers: {len(candidate.layers)}")
    logger.info(f"Skip connections: {len(candidate.connections)}")
    
    # Evaluate the architecture
    evaluated = evaluator.evaluate_architecture(candidate, baseline_model)
    
    logger.info(f"Evaluation results:")
    logger.info(f"  Parameter count: {evaluated.parameter_count:,}")
    logger.info(f"  Inference time: {evaluated.inference_time:.6f}s")
    logger.info(f"  Validation accuracy: {evaluated.validation_accuracy:.4f}")
    logger.info(f"  Performance score: {evaluated.performance_score:.4f}")
    
    # Build and test the model
    model = evaluator._build_model_from_candidate(candidate)
    logger.info(f"Built model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test inference
    test_input = torch.randn(1, 784)
    with torch.no_grad():
        output = model(test_input)
    
    logger.info(f"Model output shape: {output.shape}")


def demonstrate_constrained_search():
    """Demonstrate architecture search with constraints."""
    logger.info("\n=== Constrained Architecture Search Demo ===")
    
    baseline_model = create_baseline_model()
    agent = ArchitectureSearchAgent({'max_iterations': 8})
    agent.initialize()
    
    # Search with parameter constraints
    config = {
        'search_strategy': 'evolutionary',
        'search_space': 'full',
        'max_iterations': 8,
        'population_size': 6,
        'constraint_budget': 50000,  # Max 50K parameters
        'min_layers': 2,
        'max_layers': 4,
        'min_width': 32,
        'max_width': 128,  # Smaller max width due to constraint
        'performance_metric': 'accuracy'
    }
    
    logger.info("Starting constrained architecture search...")
    logger.info(f"Parameter budget: {config['constraint_budget']:,}")
    
    result = agent.optimize_with_tracking(baseline_model, config)
    
    if result.success:
        optimized_params = result.performance_metrics['optimized_parameters']
        baseline_params = sum(p.numel() for p in baseline_model.parameters())
        
        logger.info(f"Constrained search completed:")
        logger.info(f"  Baseline parameters: {baseline_params:,}")
        logger.info(f"  Optimized parameters: {optimized_params:,}")
        logger.info(f"  Within budget: {'✓' if optimized_params <= config['constraint_budget'] else '✗'}")
        logger.info(f"  Architecture score: {result.performance_metrics['architecture_score']:.4f}")
    else:
        logger.error(f"Constrained search failed: {result.error_message}")
    
    agent.cleanup()


def main():
    """Run all architecture search demonstrations."""
    logger.info("Neural Architecture Search Agent Demonstration")
    logger.info("=" * 60)
    
    try:
        # Basic demonstrations
        demonstrate_random_search()
        demonstrate_evolutionary_search()
        
        # Search space and strategy comparisons
        demonstrate_search_spaces()
        demonstrate_search_strategies()
        
        # Component demonstrations
        demonstrate_architecture_generator()
        demonstrate_architecture_evaluator()
        
        # Advanced demonstrations
        demonstrate_constrained_search()
        
        logger.info("\n" + "=" * 60)
        logger.info("All architecture search demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()