"""
Demo script for QuantizationAgent showing various quantization techniques.
"""

import torch
import torch.nn as nn
import time
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.optimization.quantization import QuantizationAgent, QuantizationType
from src.agents.base import ProgressUpdate


class DemoModel(nn.Module):
    """Demo model for quantization testing."""
    
    def __init__(self, input_size: int = 768, hidden_size: int = 1024, num_layers: int = 4):
        super().__init__()
        
        layers = []
        current_size = input_size
        
        for i in range(num_layers):
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            current_size = hidden_size
            
        # Final classification layer
        layers.append(nn.Linear(hidden_size, 10))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


def progress_callback(update: ProgressUpdate):
    """Callback function to track optimization progress."""
    print(f"[{update.timestamp.strftime('%H:%M:%S')}] "
          f"{update.status.value}: {update.progress_percentage:.1f}% - {update.current_step}")
    if update.message:
        print(f"  Message: {update.message}")


def print_model_info(model: nn.Module, title: str):
    """Print model information."""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate size
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    size_mb = total_size / (1024 * 1024)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {size_mb:.2f} MB")
    
    # Count layer types
    linear_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    print(f"Linear layers: {linear_layers}")


def test_quantization_technique(agent: QuantizationAgent, model: nn.Module, 
                              config: Dict[str, Any], technique_name: str):
    """Test a specific quantization technique."""
    print(f"\n{'='*60}")
    print(f"Testing {technique_name} Quantization")
    print(f"{'='*60}")
    
    # Create a copy of the model for this test
    import copy
    test_model = copy.deepcopy(model)
    
    try:
        # Check if agent can optimize this model
        can_optimize = agent.can_optimize(test_model)
        print(f"Can optimize: {can_optimize}")
        
        if not can_optimize:
            print(f"Skipping {technique_name} - model cannot be optimized")
            return
        
        # Estimate impact
        print("\nEstimating impact...")
        impact = agent.estimate_impact(test_model)
        print(f"Expected performance improvement: {impact.performance_improvement:.1%}")
        print(f"Expected size reduction: {impact.size_reduction:.1%}")
        print(f"Expected speed improvement: {impact.speed_improvement:.1%}")
        print(f"Confidence: {impact.confidence:.1%}")
        print(f"Estimated time: {impact.estimated_time_minutes} minutes")
        
        # Perform optimization with tracking
        print(f"\nPerforming {technique_name} quantization...")
        start_time = time.time()
        
        result = agent.optimize_with_tracking(test_model, config)
        
        optimization_time = time.time() - start_time
        print(f"Optimization completed in {optimization_time:.2f} seconds")
        
        if result.success:
            print("✅ Optimization successful!")
            
            # Print performance metrics
            print("\nPerformance Metrics:")
            for key, value in result.performance_metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
            
            # Print optimization metadata
            print("\nOptimization Metadata:")
            for key, value in result.optimization_metadata.items():
                print(f"  {key}: {value}")
            
            # Validate the result
            print("\nValidating optimized model...")
            validation = agent.validate_result(model, result.optimized_model)
            
            if validation.is_valid:
                print("✅ Validation passed!")
                if validation.accuracy_preserved:
                    print("✅ Accuracy preserved!")
                else:
                    print("⚠️  Accuracy may be affected")
            else:
                print("❌ Validation failed!")
                print("Issues:")
                for issue in validation.issues:
                    print(f"  - {issue}")
            
            if validation.recommendations:
                print("Recommendations:")
                for rec in validation.recommendations:
                    print(f"  - {rec}")
            
            # Test inference
            print("\nTesting inference...")
            dummy_input = torch.randn(1, 768)
            
            with torch.no_grad():
                try:
                    original_output = model(dummy_input)
                    optimized_output = result.optimized_model(dummy_input)
                    
                    # Calculate output difference
                    mse = torch.nn.functional.mse_loss(original_output, optimized_output).item()
                    print(f"Output MSE: {mse:.6f}")
                    
                    # Measure inference time
                    num_runs = 100
                    
                    # Original model timing
                    start_time = time.time()
                    for _ in range(num_runs):
                        _ = model(dummy_input)
                    original_time = (time.time() - start_time) / num_runs
                    
                    # Optimized model timing
                    start_time = time.time()
                    for _ in range(num_runs):
                        _ = result.optimized_model(dummy_input)
                    optimized_time = (time.time() - start_time) / num_runs
                    
                    speedup = original_time / optimized_time if optimized_time > 0 else 0
                    print(f"Original inference time: {original_time*1000:.2f} ms")
                    print(f"Optimized inference time: {optimized_time*1000:.2f} ms")
                    print(f"Speedup: {speedup:.2f}x")
                    
                except Exception as e:
                    print(f"❌ Inference test failed: {e}")
            
        else:
            print("❌ Optimization failed!")
            print(f"Error: {result.error_message}")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main demo function."""
    print("QuantizationAgent Demo")
    print("=" * 50)
    
    # Create demo model
    print("Creating demo model...")
    model = DemoModel(input_size=768, hidden_size=1024, num_layers=6)
    print_model_info(model, "Original Model")
    
    # Initialize quantization agent
    agent_config = {
        'preserve_modules': ['model.9'],  # Preserve final layer
        'log_level': 'INFO'
    }
    
    agent = QuantizationAgent(agent_config)
    
    # Add progress callback
    agent.add_progress_callback(progress_callback)
    
    # Initialize agent
    if not agent.initialize():
        print("❌ Failed to initialize QuantizationAgent")
        return
    
    print(f"\nSupported techniques: {agent.get_supported_techniques()}")
    
    # Test different quantization techniques
    techniques_to_test = [
        {
            'name': '8-bit',
            'config': {
                'quantization_type': 'int8',
                'threshold': 6.0
            }
        },
        {
            'name': '4-bit',
            'config': {
                'quantization_type': 'int4',
                'compute_dtype': torch.float16,
                'quant_type': 'nf4',
                'use_double_quant': True
            }
        },
        {
            'name': 'Dynamic',
            'config': {
                'quantization_type': 'dynamic'
            }
        }
    ]
    
    for technique in techniques_to_test:
        test_quantization_technique(
            agent, 
            model, 
            technique['config'], 
            technique['name']
        )
    
    # Test cancellation functionality
    print(f"\n{'='*60}")
    print("Testing Cancellation")
    print(f"{'='*60}")
    
    # Create another model copy
    test_model = DemoModel()
    
    # Start optimization and immediately cancel
    agent.cancel_optimization()
    config = {'quantization_type': 'int8'}
    result = agent.optimize_with_tracking(test_model, config)
    
    print(f"Cancellation test - Success: {result.success}")
    print(f"Error message: {result.error_message}")
    
    # Cleanup
    agent.cleanup()
    print("\n✅ Demo completed!")


if __name__ == "__main__":
    main()