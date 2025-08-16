#!/usr/bin/env python3
"""
Unit tests for the PruningAgent.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from agents.optimization.pruning import PruningAgent, PruningType, SparsityPattern, PruningConfig
from agents.base import OptimizationStatus


class TestPruningAgent(unittest.TestCase):
    """Test cases for PruningAgent."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.agent = PruningAgent({})
        self.sample_model = self._create_sample_model()
        self.sample_cnn = self._create_sample_cnn()
    
    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'agent'):
            self.agent.cleanup()
    
    def _create_sample_model(self):
        """Create a sample neural network for testing."""
        return nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def _create_sample_cnn(self):
        """Create a sample CNN for testing."""
        return nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )
    
    def _create_tiny_model(self):
        """Create a tiny model that should not be prunable."""
        return nn.Linear(10, 5)  # Less than 100K parameters
    
    def test_initialization(self):
        """Test agent initialization."""
        self.assertTrue(self.agent.initialize())
        self.assertEqual(self.agent.name, "PruningAgent")
        self.assertIsNotNone(self.agent.logger)
    
    def test_can_optimize_valid_model(self):
        """Test can_optimize with a valid model."""
        self.assertTrue(self.agent.can_optimize(self.sample_model))
        self.assertTrue(self.agent.can_optimize(self.sample_cnn))
    
    def test_can_optimize_tiny_model(self):
        """Test can_optimize with a model that's too small."""
        tiny_model = self._create_tiny_model()
        self.assertFalse(self.agent.can_optimize(tiny_model))
    
    def test_can_optimize_no_prunable_layers(self):
        """Test can_optimize with a model that has no prunable layers."""
        model = nn.Sequential(nn.ReLU(), nn.Dropout(0.5))
        self.assertFalse(self.agent.can_optimize(model))
    
    def test_estimate_impact(self):
        """Test impact estimation."""
        impact = self.agent.estimate_impact(self.sample_model)
        
        self.assertGreaterEqual(impact.performance_improvement, 0.0)
        self.assertLessEqual(impact.performance_improvement, 1.0)
        self.assertGreaterEqual(impact.size_reduction, 0.0)
        self.assertLessEqual(impact.size_reduction, 1.0)
        self.assertGreaterEqual(impact.speed_improvement, 0.0)
        self.assertLessEqual(impact.speed_improvement, 1.0)
        self.assertGreaterEqual(impact.confidence, 0.0)
        self.assertLessEqual(impact.confidence, 1.0)
        self.assertGreater(impact.estimated_time_minutes, 0)
    
    def test_parse_config_default(self):
        """Test configuration parsing with defaults."""
        config = {}
        parsed = self.agent._parse_config(config)
        
        self.assertEqual(parsed.pruning_type, PruningType.MAGNITUDE)
        self.assertEqual(parsed.sparsity_ratio, 0.5)
        self.assertEqual(parsed.sparsity_pattern, SparsityPattern.CHANNEL)
    
    def test_parse_config_custom(self):
        """Test configuration parsing with custom values."""
        config = {
            'pruning_type': 'unstructured',
            'sparsity_ratio': 0.7,
            'sparsity_pattern': 'filter',
            'preserve_modules': ['final_layer'],
            'gradual_steps': 5
        }
        parsed = self.agent._parse_config(config)
        
        self.assertEqual(parsed.pruning_type, PruningType.UNSTRUCTURED)
        self.assertEqual(parsed.sparsity_ratio, 0.7)
        self.assertEqual(parsed.sparsity_pattern, SparsityPattern.FILTER)
        self.assertEqual(parsed.preserve_modules, ['final_layer'])
        self.assertEqual(parsed.gradual_steps, 5)
    
    def test_parse_config_invalid_values(self):
        """Test configuration parsing with invalid values."""
        config = {
            'pruning_type': 'invalid_type',
            'sparsity_pattern': 'invalid_pattern'
        }
        parsed = self.agent._parse_config(config)
        
        # Should fall back to defaults
        self.assertEqual(parsed.pruning_type, PruningType.MAGNITUDE)
        self.assertEqual(parsed.sparsity_pattern, SparsityPattern.CHANNEL)
    
    def test_magnitude_pruning(self):
        """Test magnitude-based pruning."""
        config = {
            'pruning_type': 'magnitude',
            'sparsity_ratio': 0.5
        }
        
        original_params = sum(p.numel() for p in self.sample_model.parameters())
        
        result = self.agent.optimize_with_tracking(self.sample_model, config)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.optimized_model)
        self.assertEqual(result.technique_used, "pruning_magnitude")
        
        # Check that sparsity was applied
        sparsity = self.agent._calculate_sparsity(result.optimized_model)
        self.assertGreater(sparsity, 0.0)
        
        # Check performance metrics
        metrics = result.performance_metrics
        self.assertIn('actual_sparsity', metrics)
        self.assertIn('pruned_layers', metrics)
        self.assertGreater(metrics['pruned_layers'], 0)
    
    def test_unstructured_pruning(self):
        """Test unstructured pruning."""
        config = {
            'pruning_type': 'unstructured',
            'sparsity_ratio': 0.3
        }
        
        result = self.agent.optimize_with_tracking(self.sample_model, config)
        
        self.assertTrue(result.success)
        self.assertEqual(result.technique_used, "pruning_unstructured")
        
        # Check that sparsity was applied
        sparsity = self.agent._calculate_sparsity(result.optimized_model)
        self.assertGreater(sparsity, 0.0)
    
    def test_structured_pruning(self):
        """Test structured pruning."""
        config = {
            'pruning_type': 'structured',
            'sparsity_ratio': 0.4,
            'sparsity_pattern': 'channel'
        }
        
        result = self.agent.optimize_with_tracking(self.sample_cnn, config)
        
        self.assertTrue(result.success)
        self.assertEqual(result.technique_used, "pruning_structured")
        
        # Check that sparsity was applied
        sparsity = self.agent._calculate_sparsity(result.optimized_model)
        self.assertGreater(sparsity, 0.0)
    
    def test_random_pruning(self):
        """Test random pruning."""
        config = {
            'pruning_type': 'random',
            'sparsity_ratio': 0.6
        }
        
        result = self.agent.optimize_with_tracking(self.sample_model, config)
        
        self.assertTrue(result.success)
        self.assertEqual(result.technique_used, "pruning_random")
        
        # Check that sparsity was applied
        sparsity = self.agent._calculate_sparsity(result.optimized_model)
        self.assertGreater(sparsity, 0.0)
    
    def test_gradual_pruning(self):
        """Test gradual pruning."""
        config = {
            'pruning_type': 'gradual',
            'sparsity_ratio': 0.5,
            'gradual_steps': 3
        }
        
        result = self.agent.optimize_with_tracking(self.sample_model, config)
        
        self.assertTrue(result.success)
        self.assertEqual(result.technique_used, "pruning_gradual")
        
        # Check that sparsity was applied
        sparsity = self.agent._calculate_sparsity(result.optimized_model)
        self.assertGreater(sparsity, 0.0)
    
    def test_preserve_modules(self):
        """Test that specified modules are preserved from pruning."""
        config = {
            'pruning_type': 'magnitude',
            'sparsity_ratio': 0.8,
            'preserve_modules': ['4']  # Preserve the last layer
        }
        
        # Get original weights of the last layer
        original_last_layer_weight = self.sample_model[4].weight.data.clone()
        
        result = self.agent.optimize_with_tracking(self.sample_model, config)
        
        self.assertTrue(result.success)
        
        # Check that the last layer wasn't pruned
        final_layer_weight = result.optimized_model[4].weight.data
        self.assertTrue(torch.equal(original_last_layer_weight, final_layer_weight))
    
    def test_validate_result_success(self):
        """Test validation of a successful pruning result."""
        config = {'pruning_type': 'magnitude', 'sparsity_ratio': 0.3}
        
        original_model = self._create_sample_model()
        result = self.agent.optimize_with_tracking(self.sample_model, config)
        
        validation = self.agent.validate_result(original_model, result.optimized_model)
        
        self.assertTrue(validation.is_valid)
        self.assertTrue(validation.accuracy_preserved)
        self.assertIn('actual_sparsity', validation.performance_metrics)
        self.assertIn('pruned_layers', validation.performance_metrics)
    
    def test_validate_result_with_issues(self):
        """Test validation when there are issues."""
        # Create a model with NaN values to simulate issues
        problematic_model = self._create_sample_model()
        problematic_model[0].weight.data.fill_(float('nan'))
        
        original_model = self._create_sample_model()
        validation = self.agent.validate_result(original_model, problematic_model)
        
        self.assertFalse(validation.accuracy_preserved)
        self.assertTrue(len(validation.issues) > 0)
    
    def test_sparsity_calculation(self):
        """Test sparsity calculation."""
        model = self._create_sample_model()
        
        # Initially no sparsity
        sparsity = self.agent._calculate_sparsity(model)
        self.assertEqual(sparsity, 0.0)
        
        # Zero out half the parameters in first layer
        first_layer = model[0]
        weight = first_layer.weight.data
        weight[:, :weight.size(1)//2] = 0
        
        # Should have some sparsity now
        sparsity = self.agent._calculate_sparsity(model)
        self.assertGreater(sparsity, 0.0)
    
    def test_count_pruned_layers(self):
        """Test counting of pruned layers."""
        model = self._create_sample_model()
        
        # Initially no pruned layers
        count = self.agent._count_pruned_layers(model)
        self.assertEqual(count, 0)
        
        # Zero out some weights in first layer
        model[0].weight.data[0, 0] = 0
        
        # Should count as one pruned layer
        count = self.agent._count_pruned_layers(model)
        self.assertEqual(count, 1)
    
    def test_should_preserve_module(self):
        """Test module preservation logic."""
        preserve_modules = ['classifier', 'head', 'final']
        
        self.assertTrue(self.agent._should_preserve_module('model.classifier', preserve_modules))
        self.assertTrue(self.agent._should_preserve_module('backbone.head.linear', preserve_modules))
        self.assertTrue(self.agent._should_preserve_module('final_layer', preserve_modules))
        self.assertFalse(self.agent._should_preserve_module('backbone.conv1', preserve_modules))
    
    def test_create_dummy_input(self):
        """Test dummy input creation for different model types."""
        # Test with linear model
        linear_input = self.agent._create_dummy_input(self.sample_model)
        self.assertEqual(linear_input.shape, (1, 1000))  # First layer has 1000 input features
        
        # Test with CNN model
        cnn_input = self.agent._create_dummy_input(self.sample_cnn)
        self.assertEqual(cnn_input.shape, (1, 3, 224, 224))  # Default image size
    
    def test_get_supported_techniques(self):
        """Test getting supported techniques."""
        techniques = self.agent.get_supported_techniques()
        
        expected_techniques = ["unstructured", "structured", "magnitude", "random", "gradual"]
        for technique in expected_techniques:
            self.assertIn(technique, techniques)
    
    def test_progress_tracking(self):
        """Test progress tracking during optimization."""
        progress_updates = []
        
        def progress_callback(update):
            progress_updates.append(update)
        
        self.agent.add_progress_callback(progress_callback)
        
        config = {'pruning_type': 'magnitude', 'sparsity_ratio': 0.5}
        result = self.agent.optimize_with_tracking(self.sample_model, config)
        
        self.assertTrue(result.success)
        self.assertGreater(len(progress_updates), 0)
        
        # Check that we got completion status
        final_update = progress_updates[-1]
        self.assertEqual(final_update.status, OptimizationStatus.COMPLETED)
    
    def test_cancellation(self):
        """Test optimization cancellation."""
        config = {'pruning_type': 'gradual', 'sparsity_ratio': 0.8, 'gradual_steps': 10}
        
        # Cancel immediately
        self.agent.cancel_optimization()
        
        result = self.agent.optimize_with_tracking(self.sample_model, config)
        
        self.assertFalse(result.success)
        self.assertEqual(result.technique_used, "cancelled")
        self.assertIn("cancelled", result.error_message.lower())
    
    def test_invalid_pruning_type(self):
        """Test handling of invalid pruning type."""
        config = {'pruning_type': 'invalid_type', 'sparsity_ratio': 0.5}
        
        # Should fall back to magnitude pruning
        result = self.agent.optimize_with_tracking(self.sample_model, config)
        
        self.assertTrue(result.success)
        self.assertEqual(result.technique_used, "pruning_magnitude")
    
    def test_extreme_sparsity_ratios(self):
        """Test handling of extreme sparsity ratios."""
        # Test very low sparsity
        config = {'pruning_type': 'magnitude', 'sparsity_ratio': 0.01}
        result = self.agent.optimize_with_tracking(self.sample_model, config)
        self.assertTrue(result.success)
        
        # Test very high sparsity
        config = {'pruning_type': 'magnitude', 'sparsity_ratio': 0.99}
        result = self.agent.optimize_with_tracking(self.sample_model, config)
        self.assertTrue(result.success)
    
    def test_n_m_sparsity(self):
        """Test N:M sparsity pattern."""
        model = nn.Linear(16, 8)  # Small model for easier testing
        
        # Apply 2:4 sparsity
        self.agent._prune_n_m(model, (2, 4))
        
        # Check that weights follow 2:4 pattern
        weight = model.weight.data
        weight_flat = weight.view(-1)
        
        # Group into sets of 4 and check each group has at most 2 non-zero values
        for i in range(0, len(weight_flat), 4):
            group = weight_flat[i:i+4]
            non_zero_count = (group != 0).sum().item()
            self.assertLessEqual(non_zero_count, 2)
    
    def test_block_sparsity(self):
        """Test block sparsity pattern."""
        model = nn.Linear(16, 16)  # Square model for easier testing
        
        # Apply block sparsity
        self.agent._prune_blocks(model, 0.5, (4, 4))
        
        # Check that some blocks are completely zero
        weight = model.weight.data
        zero_blocks = 0
        
        for i in range(0, 16, 4):
            for j in range(0, 16, 4):
                block = weight[i:i+4, j:j+4]
                if torch.all(block == 0):
                    zero_blocks += 1
        
        self.assertGreater(zero_blocks, 0)


class TestPruningConfig(unittest.TestCase):
    """Test cases for PruningConfig."""
    
    def test_valid_config(self):
        """Test creating a valid configuration."""
        config = PruningConfig(
            pruning_type=PruningType.MAGNITUDE,
            sparsity_ratio=0.5
        )
        
        self.assertEqual(config.pruning_type, PruningType.MAGNITUDE)
        self.assertEqual(config.sparsity_ratio, 0.5)
        self.assertEqual(config.preserve_modules, [])
    
    def test_invalid_sparsity_ratio(self):
        """Test that invalid sparsity ratios raise errors."""
        with self.assertRaises(ValueError):
            PruningConfig(
                pruning_type=PruningType.MAGNITUDE,
                sparsity_ratio=1.5  # Invalid: > 1.0
            )
        
        with self.assertRaises(ValueError):
            PruningConfig(
                pruning_type=PruningType.MAGNITUDE,
                sparsity_ratio=-0.1  # Invalid: < 0.0
            )
    
    def test_default_values(self):
        """Test default configuration values."""
        config = PruningConfig(
            pruning_type=PruningType.MAGNITUDE,
            sparsity_ratio=0.5
        )
        
        self.assertEqual(config.sparsity_pattern, SparsityPattern.CHANNEL)
        self.assertEqual(config.preserve_modules, [])
        self.assertEqual(config.gradual_steps, 10)
        self.assertEqual(config.block_size, (4, 4))
        self.assertEqual(config.n_m_ratio, (2, 4))
        self.assertEqual(config.importance_metric, "magnitude")


if __name__ == '__main__':
    unittest.main()