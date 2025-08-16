#!/usr/bin/env python3
"""
Demo script for the PruningAgent showing various pruning techniques.
"""

import torch
import torch.nn as nn
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.optimization.pruning import PruningAgent, PruningType, SparsityPattern


def create_sample_model():
    """Create a sample neural network for demonstration."""
    return nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


def create_sample_cnn():
    """Create a sample CNN for demonstration."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )


def print_model_stats(model, name="Model"):
    """Print model statistics."""
    total_params = sum(p.numel() for p in model.parameters())
    zero_params = sum((p == 0).sum().item() for p in model.parameters())
    sparsity = zero_params / total_params if total_params > 0 else 0
    
    print(f"\n{name} Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Zero parameters: {zero_params:,}")
    print(f"  Sparsity: {sparsity:.2%}")


def demo_magnitude_pruning():
    """Demonstrate magnitude-based pruning."""
    print("\n" + "="*50)
    print("MAGNITUDE-BASED PRUNING DEMO")
    print("="*50)
    
    # Create model and agent
    model = create_sample_model()
    agent = PruningAgent({})
    
    if not agent.initialize():
        print("Failed to initialize PruningAgent")
        return
    
    print_model_stats(model, "Original Model")
    
    # Check if model can be optimized
    if not agent.can_optimize(model):
        print("Model cannot be optimized by PruningAgent")
        return
    
    # Get impact estimate
    impact = agent.estimate_impact(model)
    print(f"\nImpact Estimate:")
    print(f"  Performance improvement: {impact.performance_improvement:.2%}")
    print(f"  Size reduction: {impact.size_reduction:.2%}")
    print(f"  Speed improvement: {impact.speed_improvement:.2%}")
    print(f"  Confidence: {impact.confidence:.2%}")
    print(f"  Estimated time: {impact.estimated_time_minutes} minutes")
    
    # Configure pruning
    config = {
        'pruning_type': 'magnitude',
        'sparsity_ratio': 0.5,
        'preserve_modules': ['3']  # Preserve the final layer
    }
    
    print(f"\nApplying magnitude-based pruning with {config['sparsity_ratio']:.0%} sparsity...")
    
    # Apply optimization
    try:
        result = agent.optimize_with_tracking(model, config)
        
        if result.success:
            print("✓ Pruning completed successfully!")
            print_model_stats(result.optimized_model, "Pruned Model")
            
            # Print performance metrics
            metrics = result.performance_metrics
            print(f"\nPerformance Metrics:")
            print(f"  Parameter reduction: {metrics['parameter_reduction_ratio']:.2%}")
            print(f"  Actual sparsity: {metrics['actual_sparsity']:.2%}")
            print(f"  Pruned layers: {metrics['pruned_layers']}")
            print(f"  Optimization time: {result.optimization_time:.2f} seconds")
            
            # Validate result
            validation = agent.validate_result(model, result.optimized_model)
            print(f"\nValidation Result:")
            print(f"  Valid: {validation.is_valid}")
            print(f"  Accuracy preserved: {validation.accuracy_preserved}")
            if validation.issues:
                print(f"  Issues: {validation.issues}")
            if validation.recommendations:
                print(f"  Recommendations: {validation.recommendations}")
        else:
            print(f"✗ Pruning failed: {result.error_message}")
    
    except Exception as e:
        print(f"✗ Error during pruning: {e}")
    
    finally:
        agent.cleanup()


def demo_structured_pruning():
    """Demonstrate structured pruning."""
    print("\n" + "="*50)
    print("STRUCTURED PRUNING DEMO")
    print("="*50)
    
    # Create CNN model and agent
    model = create_sample_cnn()
    agent = PruningAgent({})
    
    if not agent.initialize():
        print("Failed to initialize PruningAgent")
        return
    
    print_model_stats(model, "Original CNN Model")
    
    # Configure structured pruning
    config = {
        'pruning_type': 'structured',
        'sparsity_ratio': 0.3,
        'sparsity_pattern': 'channel',
        'preserve_modules': ['6', '7']  # Preserve final layers
    }
    
    print(f"\nApplying structured channel pruning with {config['sparsity_ratio']:.0%} sparsity...")
    
    # Apply optimization
    try:
        result = agent.optimize_with_tracking(model, config)
        
        if result.success:
            print("✓ Structured pruning completed successfully!")
            print_model_stats(result.optimized_model, "Pruned CNN Model")
            
            # Test model functionality
            dummy_input = torch.randn(1, 3, 32, 32)
            with torch.no_grad():
                original_output = model(dummy_input)
                pruned_output = result.optimized_model(dummy_input)
            
            print(f"\nFunctionality Test:")
            print(f"  Original output shape: {original_output.shape}")
            print(f"  Pruned output shape: {pruned_output.shape}")
            print(f"  Output MSE: {torch.nn.functional.mse_loss(original_output, pruned_output).item():.6f}")
        else:
            print(f"✗ Structured pruning failed: {result.error_message}")
    
    except Exception as e:
        print(f"✗ Error during structured pruning: {e}")
    
    finally:
        agent.cleanup()


def demo_gradual_pruning():
    """Demonstrate gradual pruning."""
    print("\n" + "="*50)
    print("GRADUAL PRUNING DEMO")
    print("="*50)
    
    # Create model and agent
    model = create_sample_model()
    agent = PruningAgent({})
    
    if not agent.initialize():
        print("Failed to initialize PruningAgent")
        return
    
    print_model_stats(model, "Original Model")
    
    # Configure gradual pruning
    config = {
        'pruning_type': 'gradual',
        'sparsity_ratio': 0.7,
        'gradual_steps': 5,
        'preserve_modules': ['6']  # Preserve final layer
    }
    
    print(f"\nApplying gradual pruning to {config['sparsity_ratio']:.0%} sparsity in {config['gradual_steps']} steps...")
    
    # Add progress callback
    def progress_callback(update):
        print(f"  Progress: {update.progress_percentage:.1f}% - {update.current_step}")
    
    agent.add_progress_callback(progress_callback)
    
    # Apply optimization
    try:
        result = agent.optimize_with_tracking(model, config)
        
        if result.success:
            print("✓ Gradual pruning completed successfully!")
            print_model_stats(result.optimized_model, "Gradually Pruned Model")
            
            # Show snapshots
            print(f"\nSnapshots created: {len(result.snapshots)}")
            for i, snapshot in enumerate(result.snapshots):
                print(f"  {i+1}. {snapshot.checkpoint_name} at {snapshot.timestamp.strftime('%H:%M:%S')}")
        else:
            print(f"✗ Gradual pruning failed: {result.error_message}")
    
    except Exception as e:
        print(f"✗ Error during gradual pruning: {e}")
    
    finally:
        agent.cleanup()


def demo_comparison():
    """Compare different pruning techniques."""
    print("\n" + "="*50)
    print("PRUNING TECHNIQUES COMPARISON")
    print("="*50)
    
    # Create base model
    base_model = create_sample_model()
    print_model_stats(base_model, "Base Model")
    
    techniques = [
        {'name': 'Magnitude', 'config': {'pruning_type': 'magnitude', 'sparsity_ratio': 0.5}},
        {'name': 'Random', 'config': {'pruning_type': 'random', 'sparsity_ratio': 0.5}},
        {'name': 'Unstructured', 'config': {'pruning_type': 'unstructured', 'sparsity_ratio': 0.5}},
    ]
    
    results = []
    
    for technique in techniques:
        print(f"\n--- {technique['name']} Pruning ---")
        
        # Create fresh model copy
        model = create_sample_model()
        model.load_state_dict(base_model.state_dict())
        
        agent = PruningAgent({})
        if not agent.initialize():
            continue
        
        try:
            result = agent.optimize_with_tracking(model, technique['config'])
            
            if result.success:
                sparsity = result.performance_metrics['actual_sparsity']
                param_reduction = result.performance_metrics['parameter_reduction_ratio']
                time_taken = result.optimization_time
                
                results.append({
                    'name': technique['name'],
                    'sparsity': sparsity,
                    'param_reduction': param_reduction,
                    'time': time_taken
                })
                
                print(f"  ✓ Sparsity: {sparsity:.2%}")
                print(f"  ✓ Parameter reduction: {param_reduction:.2%}")
                print(f"  ✓ Time: {time_taken:.2f}s")
            else:
                print(f"  ✗ Failed: {result.error_message}")
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        finally:
            agent.cleanup()
    
    # Summary comparison
    if results:
        print(f"\n--- Summary ---")
        print(f"{'Technique':<12} {'Sparsity':<10} {'Param Reduction':<15} {'Time (s)':<10}")
        print("-" * 50)
        for result in results:
            print(f"{result['name']:<12} {result['sparsity']:<10.2%} {result['param_reduction']:<15.2%} {result['time']:<10.2f}")


def main():
    """Run all pruning demos."""
    print("PruningAgent Demo")
    print("This demo shows various pruning techniques for neural network optimization.")
    
    try:
        demo_magnitude_pruning()
        demo_structured_pruning()
        demo_gradual_pruning()
        demo_comparison()
        
        print("\n" + "="*50)
        print("All demos completed!")
        print("="*50)
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")


if __name__ == "__main__":
    main()