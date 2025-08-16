"""
Unit tests for the ArchitectureSearchAgent.
"""

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch
import tempfile
import os

from src.agents.optimization.architecture_search import (
    ArchitectureSearchAgent, SearchStrategy, ArchitectureSpace, SearchConfig,
    ArchitectureCandidate, ArchitectureGenerator, ArchitectureEvaluator
)
from src.agents.base import ImpactEstimate, ValidationResult, OptimizedModel


class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, input_size=784, hidden_size=1024, output_size=10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size)
        )
    
    def forward(self, x):
        return self.layers(x)


class TestSearchConfig:
    """Test SearchConfig class."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = SearchConfig(
            search_strategy=SearchStrategy.RANDOM,
            search_space=ArchitectureSpace.DEPTH
        )
        
        assert config.search_strategy == SearchStrategy.RANDOM
        assert config.search_space == ArchitectureSpace.DEPTH
        assert config.max_iterations == 50
        assert config.population_size == 20
        assert config.min_layers < config.max_layers
        assert config.min_width < config.max_width
    
    def test_invalid_layer_bounds(self):
        """Test validation of layer bounds."""
        with pytest.raises(ValueError, match="min_layers must be less than max_layers"):
            SearchConfig(
                search_strategy=SearchStrategy.RANDOM,
                search_space=ArchitectureSpace.DEPTH,
                min_layers=10,
                max_layers=5
            )
    
    def test_invalid_width_bounds(self):
        """Test validation of width bounds."""
        with pytest.raises(ValueError, match="min_width must be less than max_width"):
            SearchConfig(
                search_strategy=SearchStrategy.RANDOM,
                search_space=ArchitectureSpace.WIDTH,
                min_width=1000,
                max_width=100
            )


class TestArchitectureCandidate:
    """Test ArchitectureCandidate class."""
    
    def test_candidate_creation(self):
        """Test creating architecture candidate."""
        layers = [
            {"type": "linear", "input_size": 100, "output_size": 50, "activation": "relu"},
            {"type": "linear", "input_size": 50, "output_size": 10, "activation": "relu"}
        ]
        connections = [(0, 1)]
        
        candidate = ArchitectureCandidate(layers=layers, connections=connections)
        
        assert len(candidate.layers) == 2
        assert len(candidate.connections) == 1
        assert candidate.performance_score == 0.0
        assert candidate.parameter_count == 0
    
    def test_candidate_hash(self):
        """Test candidate hashing."""
        layers1 = [{"type": "linear", "size": 50, "activation": "relu"}]
        layers2 = [{"type": "linear", "size": 50, "activation": "relu"}]
        
        candidate1 = ArchitectureCandidate(layers=layers1, connections=[])
        candidate2 = ArchitectureCandidate(layers=layers2, connections=[])
        
        # Same structure should have same hash
        assert hash(candidate1) == hash(candidate2)


class TestArchitectureGenerator:
    """Test ArchitectureGenerator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SearchConfig(
            search_strategy=SearchStrategy.RANDOM,
            search_space=ArchitectureSpace.FULL,
            min_layers=2,
            max_layers=5,
            min_width=10,
            max_width=50
        )
    
    @pytest.fixture
    def generator(self, config):
        """Create generator instance."""
        return ArchitectureGenerator(config)
    
    def test_generate_random_architecture(self, generator):
        """Test generating random architecture."""
        candidate = generator.generate_random_architecture(input_size=100, output_size=10)
        
        assert isinstance(candidate, ArchitectureCandidate)
        assert len(candidate.layers) >= 2
        assert len(candidate.layers) <= 5
        assert candidate.layers[0]["input_size"] == 100
        assert candidate.layers[-1]["output_size"] == 10
    
    def test_mutate_architecture(self, generator):
        """Test architecture mutation."""
        original = generator.generate_random_architecture(input_size=100, output_size=10)
        mutated = generator.mutate_architecture(original)
        
        assert isinstance(mutated, ArchitectureCandidate)
        assert len(mutated.layers) == len(original.layers)
        # Performance should be reset
        assert mutated.performance_score == 0.0
    
    def test_crossover_architectures(self, generator):
        """Test architecture crossover."""
        parent1 = generator.generate_random_architecture(input_size=100, output_size=10)
        parent2 = generator.generate_random_architecture(input_size=100, output_size=10)
        
        offspring = generator.crossover_architectures(parent1, parent2)
        
        assert isinstance(offspring, ArchitectureCandidate)
        assert offspring.layers[-1]["output_size"] == 10  # Output size preserved
    
    def test_generate_with_different_search_spaces(self):
        """Test generation with different search spaces."""
        spaces = [ArchitectureSpace.DEPTH, ArchitectureSpace.WIDTH, 
                 ArchitectureSpace.OPERATIONS, ArchitectureSpace.CONNECTIONS]
        
        for space in spaces:
            config = SearchConfig(
                search_strategy=SearchStrategy.RANDOM,
                search_space=space,
                min_layers=2,
                max_layers=4
            )
            generator = ArchitectureGenerator(config)
            candidate = generator.generate_random_architecture(50, 5)
            
            assert isinstance(candidate, ArchitectureCandidate)
            assert len(candidate.layers) >= 2


class TestArchitectureEvaluator:
    """Test ArchitectureEvaluator class."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return SearchConfig(
            search_strategy=SearchStrategy.RANDOM,
            search_space=ArchitectureSpace.FULL
        )
    
    @pytest.fixture
    def evaluator(self, config):
        """Create evaluator instance."""
        return ArchitectureEvaluator(config)
    
    @pytest.fixture
    def reference_model(self):
        """Create reference model."""
        return SimpleModel(input_size=100, hidden_size=50, output_size=10)
    
    def test_evaluate_architecture(self, evaluator, reference_model):
        """Test architecture evaluation."""
        layers = [
            {"type": "linear", "input_size": 100, "output_size": 50, "activation": "relu"},
            {"type": "linear", "input_size": 50, "output_size": 10, "activation": "relu"}
        ]
        candidate = ArchitectureCandidate(layers=layers, connections=[])
        
        evaluated = evaluator.evaluate_architecture(candidate, reference_model)
        
        assert evaluated.parameter_count > 0
        assert evaluated.inference_time >= 0
        assert 0 <= evaluated.validation_accuracy <= 1
        assert evaluated.performance_score >= 0
    
    def test_build_model_from_candidate(self, evaluator):
        """Test building model from candidate."""
        layers = [
            {"type": "linear", "input_size": 100, "output_size": 50, "activation": "relu"},
            {"type": "linear", "input_size": 50, "output_size": 10, "activation": "gelu"}
        ]
        candidate = ArchitectureCandidate(layers=layers, connections=[])
        
        model = evaluator._build_model_from_candidate(candidate)
        
        assert isinstance(model, nn.Module)
        
        # Test forward pass
        dummy_input = torch.randn(1, 100)
        output = model(dummy_input)
        assert output.shape == (1, 10)
    
    def test_estimate_inference_time(self, evaluator, reference_model):
        """Test inference time estimation."""
        layers = [{"type": "linear", "input_size": 100, "output_size": 10, "activation": "relu"}]
        candidate = ArchitectureCandidate(layers=layers, connections=[])
        
        model = evaluator._build_model_from_candidate(candidate)
        inference_time = evaluator._estimate_inference_time(model, reference_model)
        
        assert inference_time >= 0
    
    def test_estimate_accuracy(self, evaluator, reference_model):
        """Test accuracy estimation."""
        layers = [{"type": "linear", "input_size": 100, "output_size": 10, "activation": "relu"}]
        candidate = ArchitectureCandidate(layers=layers, connections=[])
        
        model = evaluator._build_model_from_candidate(candidate)
        accuracy = evaluator._estimate_accuracy(model, reference_model)
        
        assert 0 <= accuracy <= 1


class TestArchitectureSearchAgent:
    """Test ArchitectureSearchAgent class."""
    
    @pytest.fixture
    def agent_config(self):
        """Configuration for agent."""
        return {
            'max_iterations': 10,
            'population_size': 5
        }
    
    @pytest.fixture
    def agent(self, agent_config):
        """Create ArchitectureSearchAgent instance."""
        return ArchitectureSearchAgent(agent_config)
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return SimpleModel(input_size=784, hidden_size=1024, output_size=10)
    
    def test_agent_initialization(self, agent):
        """Test agent initialization."""
        assert agent.initialize()
        assert agent.name == "ArchitectureSearchAgent"
    
    def test_agent_cleanup(self, agent):
        """Test agent cleanup."""
        agent.cleanup()
        assert agent.generator is None
        assert agent.evaluator is None
    
    def test_can_optimize_valid_model(self, agent, simple_model):
        """Test can_optimize with valid model."""
        assert agent.can_optimize(simple_model)
    
    def test_can_optimize_small_model(self, agent):
        """Test can_optimize with small model."""
        small_model = SimpleModel(input_size=10, hidden_size=5, output_size=2)
        assert not agent.can_optimize(small_model)
    
    def test_can_optimize_no_linear_layers(self, agent):
        """Test can_optimize with model without linear layers."""
        model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU())
        # Should still return True but with warning
        result = agent.can_optimize(model)
        # The current implementation checks for linear layers but doesn't fail
        assert isinstance(result, bool)
    
    def test_estimate_impact(self, agent, simple_model):
        """Test impact estimation."""
        impact = agent.estimate_impact(simple_model)
        
        assert isinstance(impact, ImpactEstimate)
        assert 0 <= impact.performance_improvement <= 1
        assert 0 <= impact.size_reduction <= 1
        assert 0 <= impact.speed_improvement <= 1
        assert 0 <= impact.confidence <= 1
        assert impact.estimated_time_minutes > 0
    
    def test_optimize_random_search(self, agent, simple_model):
        """Test random search optimization."""
        config = {
            'search_strategy': 'random',
            'search_space': 'depth',
            'max_iterations': 5,
            'min_layers': 2,
            'max_layers': 4
        }
        
        result = agent.optimize(simple_model, config)
        
        assert isinstance(result, OptimizedModel)
        assert result.model is not None
        assert result.technique_used == "nas_random"
        assert result.optimization_time > 0
        assert 'architecture_score' in result.performance_metrics
    
    def test_optimize_evolutionary_search(self, agent, simple_model):
        """Test evolutionary search optimization."""
        config = {
            'search_strategy': 'evolutionary',
            'search_space': 'width',
            'max_iterations': 3,
            'population_size': 4,
            'min_layers': 2,
            'max_layers': 3
        }
        
        result = agent.optimize(simple_model, config)
        
        assert isinstance(result, OptimizedModel)
        assert result.technique_used == "nas_evolutionary"
    
    def test_optimize_with_progress_tracking(self, agent, simple_model):
        """Test optimization with progress tracking."""
        progress_updates = []
        
        def progress_callback(update):
            progress_updates.append(update)
        
        agent.add_progress_callback(progress_callback)
        
        config = {
            'search_strategy': 'random',
            'max_iterations': 3
        }
        
        result = agent.optimize_with_tracking(simple_model, config)
        
        assert result.success
        assert len(progress_updates) > 0
        assert any(update.status.value == "completed" for update in progress_updates)
    
    def test_validate_result_valid(self, agent, simple_model):
        """Test validation with valid result."""
        # Create a different architecture
        optimized_model = SimpleModel(input_size=784, hidden_size=512, output_size=10)
        
        validation = agent.validate_result(simple_model, optimized_model)
        
        assert isinstance(validation, ValidationResult)
        assert validation.is_valid
        assert 'parameter_ratio' in validation.performance_metrics
    
    def test_validate_result_shape_mismatch(self, agent, simple_model):
        """Test validation with shape mismatch."""
        # Create model with different output size
        wrong_model = SimpleModel(input_size=784, hidden_size=1024, output_size=5)
        
        validation = agent.validate_result(simple_model, wrong_model)
        
        assert not validation.is_valid
        assert any("shape mismatch" in issue.lower() for issue in validation.issues)
    
    def test_validate_result_nan_output(self, agent, simple_model):
        """Test validation with NaN output."""
        # Create model that produces NaN
        nan_model = SimpleModel()
        
        # Corrupt weights to produce NaN
        with torch.no_grad():
            for param in nan_model.parameters():
                param.fill_(float('nan'))
        
        validation = agent.validate_result(simple_model, nan_model)
        
        assert not validation.is_valid
        assert not validation.accuracy_preserved
        assert any("nan" in issue.lower() for issue in validation.issues)
    
    def test_get_supported_techniques(self, agent):
        """Test getting supported techniques."""
        techniques = agent.get_supported_techniques()
        
        assert isinstance(techniques, list)
        assert "random" in techniques
        assert "evolutionary" in techniques
        assert "gradient_based" in techniques
        assert "progressive" in techniques
    
    def test_parse_config_defaults(self, agent):
        """Test config parsing with defaults."""
        config = {}
        parsed = agent._parse_config(config)
        
        assert parsed.search_strategy == SearchStrategy.EVOLUTIONARY
        assert parsed.search_space == ArchitectureSpace.FULL
        assert parsed.max_iterations == 50
    
    def test_parse_config_custom(self, agent):
        """Test config parsing with custom values."""
        config = {
            'search_strategy': 'random',
            'search_space': 'depth',
            'max_iterations': 20,
            'population_size': 10,
            'mutation_rate': 0.2,
            'min_layers': 3,
            'max_layers': 8
        }
        
        parsed = agent._parse_config(config)
        
        assert parsed.search_strategy == SearchStrategy.RANDOM
        assert parsed.search_space == ArchitectureSpace.DEPTH
        assert parsed.max_iterations == 20
        assert parsed.population_size == 10
        assert parsed.mutation_rate == 0.2
        assert parsed.min_layers == 3
        assert parsed.max_layers == 8
    
    def test_parse_config_invalid_values(self, agent):
        """Test config parsing with invalid values."""
        config = {
            'search_strategy': 'invalid_strategy',
            'search_space': 'invalid_space'
        }
        
        parsed = agent._parse_config(config)
        
        # Should fall back to defaults
        assert parsed.search_strategy == SearchStrategy.EVOLUTIONARY
        assert parsed.search_space == ArchitectureSpace.FULL
    
    def test_infer_model_sizes(self, agent, simple_model):
        """Test inferring model input/output sizes."""
        input_size, output_size = agent._infer_model_sizes(simple_model)
        
        assert input_size == 784
        assert output_size == 10
    
    def test_infer_model_sizes_no_linear(self, agent):
        """Test inferring sizes from model without linear layers."""
        model = nn.Sequential(nn.Conv2d(3, 16, 3), nn.ReLU())
        input_size, output_size = agent._infer_model_sizes(model)
        
        # Should return defaults
        assert input_size == 768
        assert output_size == 10
    
    def test_tournament_selection(self, agent):
        """Test tournament selection."""
        # Create population with known scores
        population = []
        for i in range(5):
            candidate = ArchitectureCandidate(layers=[], connections=[])
            candidate.performance_score = i * 0.2  # 0.0, 0.2, 0.4, 0.6, 0.8
            population.append(candidate)
        
        # Tournament should select higher scoring individuals more often
        selected = agent._tournament_selection(population, 3)
        assert selected in population
        assert selected.performance_score >= 0.0
    
    def test_create_dummy_input(self, agent, simple_model):
        """Test creating dummy input."""
        dummy_input = agent._create_dummy_input(simple_model)
        
        assert isinstance(dummy_input, torch.Tensor)
        assert dummy_input.shape[0] == 1  # Batch size 1
        assert dummy_input.shape[1] == 784  # Input size
    
    def test_optimization_cancellation(self, agent, simple_model):
        """Test optimization cancellation."""
        config = {
            'search_strategy': 'random',
            'max_iterations': 20
        }
        
        # Cancel immediately
        agent.cancel_optimization()
        
        result = agent.optimize_with_tracking(simple_model, config)
        
        assert not result.success
        assert "cancelled" in result.error_message.lower()
    
    def test_optimization_with_snapshots(self, agent, simple_model):
        """Test optimization with snapshot creation."""
        config = {
            'search_strategy': 'random',
            'max_iterations': 3
        }
        
        result = agent.optimize_with_tracking(simple_model, config)
        
        assert result.success
        assert len(result.snapshots) > 0
        assert any("original" in snapshot.checkpoint_name for snapshot in result.snapshots)


class TestArchitectureSearchIntegration:
    """Integration tests for architecture search agent."""
    
    def test_end_to_end_random_search(self):
        """Test complete random search workflow."""
        # Create model
        model = SimpleModel(input_size=50, hidden_size=32, output_size=5)
        
        # Create agent
        agent = ArchitectureSearchAgent({'max_iterations': 5, 'population_size': 3})
        agent.initialize()
        
        # Run search
        config = {
            'search_strategy': 'random',
            'search_space': 'full',
            'max_iterations': 5,
            'min_layers': 2,
            'max_layers': 4,
            'min_width': 16,
            'max_width': 64
        }
        
        result = agent.optimize_with_tracking(model, config)
        
        # Verify results
        assert result.success
        assert result.optimized_model is not None
        
        # Check that we found an architecture
        assert 'num_layers' in result.performance_metrics
        assert 'architecture_score' in result.performance_metrics
        
        # Validate result
        validation = agent.validate_result(model, result.optimized_model)
        assert validation.is_valid
        
        agent.cleanup()
    
    def test_end_to_end_evolutionary_search(self):
        """Test complete evolutionary search workflow."""
        model = SimpleModel(input_size=30, hidden_size=20, output_size=3)
        agent = ArchitectureSearchAgent({'max_iterations': 3, 'population_size': 4})
        agent.initialize()
        
        config = {
            'search_strategy': 'evolutionary',
            'search_space': 'width',
            'max_iterations': 3,
            'population_size': 4,
            'mutation_rate': 0.3,
            'crossover_rate': 0.7,
            'early_stopping_patience': 2
        }
        
        result = agent.optimize_with_tracking(model, config)
        
        assert result.success
        assert result.optimized_model is not None
        
        validation = agent.validate_result(model, result.optimized_model)
        assert validation.is_valid
        
        agent.cleanup()
    
    def test_search_with_different_spaces(self):
        """Test search with different architecture spaces."""
        model = SimpleModel(input_size=40, hidden_size=25, output_size=4)
        agent = ArchitectureSearchAgent({'max_iterations': 3})
        agent.initialize()
        
        spaces = ['depth', 'width', 'operations', 'full']
        
        for space in spaces:
            config = {
                'search_strategy': 'random',
                'search_space': space,
                'max_iterations': 3
            }
            
            result = agent.optimize_with_tracking(model, config)
            assert result.success, f"Failed for search space: {space}"
        
        agent.cleanup()


if __name__ == "__main__":
    pytest.main([__file__])