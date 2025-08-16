"""
Demo script for the DistillationAgent - Knowledge Distillation Optimization.

This script demonstrates how to use the DistillationAgent to compress models
using knowledge distillation techniques.
"""

import torch
import torch.nn as nn
import logging
import time
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.optimization.distillation import (
    DistillationAgent, DistillationType, StudentArchitecture, DistillationConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TeacherModel(nn.Module):
    """Large teacher model for demonstration."""
    
    def __init__(self, input_size=784, hidden_size=512, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)


def create_sample_model() -> nn.Module:
    """Create a sample teacher model for distillation."""
    model = TeacherModel(input_size=784, hidden_size=512, output_size=10)
    
    # Initialize with reasonable weights
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    logger.info(f"Created teacher model with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model


def demonstrate_response_distillation():
    """Demonstrate response-based knowledge distillation."""
    logger.info("=== Response-based Knowledge Distillation Demo ===")
    
    # Create teacher model
    teacher_model = create_sample_model()
    
    # Create distillation agent
    agent_config = {
        'compression_ratio': 0.6,  # Target 60% compression
        'preserve_modules': []
    }
    agent = DistillationAgent(agent_config)
    
    # Initialize agent
    if not agent.initialize():
        logger.error("Failed to initialize DistillationAgent")
        return
    
    # Estimate impact
    impact = agent.estimate_impact(teacher_model)
    logger.info(f"Estimated impact:")
    logger.info(f"  Performance improvement: {impact.performance_improvement:.2%}")
    logger.info(f"  Size reduction: {impact.size_reduction:.2%}")
    logger.info(f"  Speed improvement: {impact.speed_improvement:.2%}")
    logger.info(f"  Confidence: {impact.confidence:.2%}")
    logger.info(f"  Estimated time: {impact.estimated_time_minutes} minutes")
    
    # Configure distillation
    distillation_config = {
        'distillation_type': 'response',
        'student_architecture': 'smaller_same',
        'temperature': 4.0,
        'alpha': 0.7,  # Weight for distillation loss
        'beta': 0.3,   # Weight for student loss
        'num_epochs': 5,
        'batch_size': 16,
        'learning_rate': 1e-4
    }
    
    # Add progress tracking
    progress_updates = []
    def track_progress(update):
        progress_updates.append(update)
        logger.info(f"Progress: {update.progress_percentage:.1f}% - {update.current_step}")
        if update.message:
            logger.info(f"  Message: {update.message}")
    
    agent.add_progress_callback(track_progress)
    
    # Perform distillation
    logger.info("Starting response-based distillation...")
    start_time = time.time()
    
    result = agent.optimize_with_tracking(teacher_model, distillation_config)
    
    end_time = time.time()
    
    if result.success:
        logger.info(f"Distillation completed successfully in {end_time - start_time:.2f} seconds")
        
        # Analyze results
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in result.optimized_model.parameters())
        compression_ratio = (teacher_params - student_params) / teacher_params
        
        logger.info(f"Results:")
        logger.info(f"  Teacher parameters: {teacher_params:,}")
        logger.info(f"  Student parameters: {student_params:,}")
        logger.info(f"  Compression ratio: {compression_ratio:.2%}")
        logger.info(f"  Optimization time: {result.optimization_time:.2f} seconds")
        
        # Validate result
        validation = agent.validate_result(teacher_model, result.optimized_model)
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
            teacher_output = teacher_model(test_input)
            student_output = result.optimized_model(test_input)
        
        output_similarity = torch.nn.functional.cosine_similarity(
            teacher_output.flatten(), student_output.flatten(), dim=0
        ).item()
        logger.info(f"Output similarity: {output_similarity:.4f}")
        
    else:
        logger.error(f"Distillation failed: {result.error_message}")
    
    # Cleanup
    agent.cleanup()


def demonstrate_feature_distillation():
    """Demonstrate feature-based knowledge distillation."""
    logger.info("\n=== Feature-based Knowledge Distillation Demo ===")
    
    teacher_model = create_sample_model()
    agent = DistillationAgent({'compression_ratio': 0.5})
    
    if not agent.initialize():
        logger.error("Failed to initialize DistillationAgent")
        return
    
    # Configure feature distillation
    config = {
        'distillation_type': 'feature',
        'student_architecture': 'efficient',
        'temperature': 5.0,
        'alpha': 0.8,
        'beta': 0.2,
        'num_epochs': 3,
        'batch_size': 8,
        'feature_layers': ['layers.2', 'layers.4']  # Intermediate layers
    }
    
    logger.info("Starting feature-based distillation...")
    result = agent.optimize_with_tracking(teacher_model, config)
    
    if result.success:
        logger.info("Feature distillation completed successfully")
        
        # Show compression metrics
        metrics = result.performance_metrics
        logger.info(f"Compression achieved: {metrics.get('compression_ratio', 0):.2%}")
        
    else:
        logger.error(f"Feature distillation failed: {result.error_message}")
    
    agent.cleanup()


def demonstrate_progressive_distillation():
    """Demonstrate progressive knowledge distillation."""
    logger.info("\n=== Progressive Knowledge Distillation Demo ===")
    
    teacher_model = create_sample_model()
    agent = DistillationAgent({'compression_ratio': 0.7})
    
    if not agent.initialize():
        logger.error("Failed to initialize DistillationAgent")
        return
    
    # Configure progressive distillation
    config = {
        'distillation_type': 'progressive',
        'student_architecture': 'smaller_same',
        'temperature': 6.0,
        'alpha': 0.6,
        'beta': 0.4,
        'num_epochs': 4,
        'batch_size': 12,
        'preserve_accuracy': 0.90  # Maintain 90% of original accuracy
    }
    
    logger.info("Starting progressive distillation...")
    result = agent.optimize_with_tracking(teacher_model, config)
    
    if result.success:
        logger.info("Progressive distillation completed successfully")
        
        # Analyze progressive results
        logger.info(f"Final compression: {result.performance_metrics.get('compression_ratio', 0):.2%}")
        logger.info(f"Technique used: {result.technique_used}")
        
    else:
        logger.error(f"Progressive distillation failed: {result.error_message}")
    
    agent.cleanup()


def demonstrate_distillation_comparison():
    """Compare different distillation techniques."""
    logger.info("\n=== Distillation Techniques Comparison ===")
    
    teacher_model = create_sample_model()
    
    techniques = [
        ('response', 'Response-based'),
        ('feature', 'Feature-based'),
        ('attention', 'Attention-based'),
        ('progressive', 'Progressive')
    ]
    
    results = {}
    
    for technique_id, technique_name in techniques:
        logger.info(f"\nTesting {technique_name} distillation...")
        
        agent = DistillationAgent({'compression_ratio': 0.6})
        agent.initialize()
        
        config = {
            'distillation_type': technique_id,
            'student_architecture': 'smaller_same',
            'num_epochs': 2,  # Quick test
            'batch_size': 8
        }
        
        start_time = time.time()
        result = agent.optimize_with_tracking(teacher_model, config)
        end_time = time.time()
        
        if result.success:
            teacher_params = sum(p.numel() for p in teacher_model.parameters())
            student_params = sum(p.numel() for p in result.optimized_model.parameters())
            compression = (teacher_params - student_params) / teacher_params
            
            results[technique_name] = {
                'compression_ratio': compression,
                'optimization_time': end_time - start_time,
                'success': True
            }
            
            logger.info(f"  ✓ Compression: {compression:.2%}")
            logger.info(f"  ✓ Time: {end_time - start_time:.2f}s")
        else:
            results[technique_name] = {
                'success': False,
                'error': result.error_message
            }
            logger.error(f"  ✗ Failed: {result.error_message}")
        
        agent.cleanup()
    
    # Summary
    logger.info("\n=== Comparison Summary ===")
    for technique, result in results.items():
        if result['success']:
            logger.info(f"{technique}:")
            logger.info(f"  Compression: {result['compression_ratio']:.2%}")
            logger.info(f"  Time: {result['optimization_time']:.2f}s")
        else:
            logger.info(f"{technique}: FAILED - {result['error']}")


def demonstrate_custom_student_architecture():
    """Demonstrate distillation with custom student architecture."""
    logger.info("\n=== Custom Student Architecture Demo ===")
    
    teacher_model = create_sample_model()
    
    # Create custom student architecture
    class CustomStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    custom_student = CustomStudent()
    
    # Configure agent with custom student
    agent_config = {
        'compression_ratio': 0.8,
        'custom_student': custom_student
    }
    agent = DistillationAgent(agent_config)
    agent.initialize()
    
    config = {
        'distillation_type': 'response',
        'student_architecture': 'custom',
        'temperature': 4.0,
        'num_epochs': 3,
        'batch_size': 16
    }
    
    logger.info("Starting distillation with custom student architecture...")
    result = agent.optimize_with_tracking(teacher_model, config)
    
    if result.success:
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in result.optimized_model.parameters())
        
        logger.info(f"Custom student distillation completed:")
        logger.info(f"  Teacher params: {teacher_params:,}")
        logger.info(f"  Student params: {student_params:,}")
        logger.info(f"  Compression: {(teacher_params - student_params) / teacher_params:.2%}")
    else:
        logger.error(f"Custom distillation failed: {result.error_message}")
    
    agent.cleanup()


def main():
    """Run all distillation demonstrations."""
    logger.info("Knowledge Distillation Agent Demonstration")
    logger.info("=" * 50)
    
    try:
        # Basic demonstrations
        demonstrate_response_distillation()
        demonstrate_feature_distillation()
        demonstrate_progressive_distillation()
        
        # Advanced demonstrations
        demonstrate_distillation_comparison()
        demonstrate_custom_student_architecture()
        
        logger.info("\n" + "=" * 50)
        logger.info("All distillation demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()