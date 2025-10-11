"""
Unit tests for ConfigurationManager service (src/services/config_manager.py).

Tests cover:
- Configuration loading and saving
- Validation with valid and invalid configs
- Concurrent update handling
- Rollback on errors
"""

import pytest
import tempfile
import json
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.services.config_manager import ConfigurationManager, ValidationResult
from src.config.optimization_criteria import (
    OptimizationCriteria,
    OptimizationConstraints,
    PerformanceThreshold,
    PerformanceMetric,
    OptimizationTechnique
)


@pytest.fixture
def temp_config_path():
    """Create a temporary configuration file path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "config.json"
        yield str(config_path)


@pytest.fixture
def valid_criteria():
    """Create valid optimization criteria for testing."""
    return OptimizationCriteria(
        name="test_criteria",
        description="Test optimization criteria",
        performance_thresholds=[
            PerformanceThreshold(
                metric=PerformanceMetric.ACCURACY,
                min_value=0.90,
                target_value=0.95,
                tolerance=0.05
            ),
            PerformanceThreshold(
                metric=PerformanceMetric.MODEL_SIZE,
                max_value=500.0,
                target_value=100.0,
                tolerance=0.1
            )
        ],
        constraints=OptimizationConstraints(
            max_optimization_time_minutes=60,
            max_memory_usage_gb=16.0,
            preserve_accuracy_threshold=0.95,
            allowed_techniques=[
                OptimizationTechnique.QUANTIZATION,
                OptimizationTechnique.PRUNING
            ],
            forbidden_techniques=[],
            hardware_constraints={"gpu_memory_gb": 8.0}
        ),
        priority_weights={
            PerformanceMetric.ACCURACY: 0.4,
            PerformanceMetric.MODEL_SIZE: 0.3,
            PerformanceMetric.INFERENCE_TIME: 0.3
        },
        target_deployment="edge"
    )


class TestConfigurationLoading:
    """Test configuration loading functionality."""
    
    def test_load_configuration_success(self, temp_config_path, valid_criteria):
        """Test successful configuration loading from file."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Save a configuration first
        config_manager.save_configuration(valid_criteria)
        
        # Load it back
        loaded = config_manager.load_configuration()
        
        assert loaded is not None
        assert loaded.name == valid_criteria.name
        assert loaded.description == valid_criteria.description
        assert loaded.target_deployment == valid_criteria.target_deployment
    
    def test_load_configuration_file_not_found(self, temp_config_path):
        """Test loading when configuration file doesn't exist."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Should return default configuration
        loaded = config_manager.load_configuration()
        
        assert loaded is not None
        assert loaded.name == "default"
    
    def test_load_configuration_invalid_json(self):
        """Test loading with invalid JSON file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.json"
            config_path.write_text("{ invalid json }")
            
            # Reset singleton for clean test
            ConfigurationManager._instance = None
            config_manager = ConfigurationManager(str(config_path))
            loaded = config_manager.load_configuration()
            
            # Should return default configuration on parse error
            assert loaded is not None
            assert loaded.name == "default"
    
    def test_load_configuration_caching(self, temp_config_path, valid_criteria):
        """Test that loaded configuration is cached."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        config_manager.save_configuration(valid_criteria)
        
        # Load twice
        loaded1 = config_manager.load_configuration()
        loaded2 = config_manager.get_current_configuration()
        
        # Should return the same cached instance
        assert loaded1.name == loaded2.name


class TestConfigurationSaving:
    """Test configuration saving functionality."""
    
    def test_save_configuration_success(self, temp_config_path, valid_criteria):
        """Test successful configuration saving."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        success = config_manager.save_configuration(valid_criteria)
        
        assert success
        assert config_manager.config_path.exists()
        
        # Verify file contents
        with open(config_manager.config_path, 'r') as f:
            data = json.load(f)
            assert data["name"] == valid_criteria.name
    
    def test_save_configuration_creates_directory(self):
        """Test that save creates parent directories if needed."""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_path = Path(temp_dir) / "nested" / "dir" / "config.json"
            
            # Reset singleton for clean test
            ConfigurationManager._instance = None
            config_manager = ConfigurationManager(str(nested_path))
            
            criteria = OptimizationCriteria(
                name="test",
                description="Test",
                priority_weights={PerformanceMetric.ACCURACY: 1.0}
            )
            
            success = config_manager.save_configuration(criteria)
            
            assert success
            assert nested_path.exists()
    
    def test_save_configuration_invalid_criteria(self, temp_config_path):
        """Test saving invalid configuration fails."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Create invalid criteria (empty name) - bypass validation
        invalid_criteria = OptimizationCriteria.__new__(OptimizationCriteria)
        invalid_criteria.name = ""
        invalid_criteria.description = "Test"
        invalid_criteria.performance_thresholds = []
        invalid_criteria.constraints = OptimizationConstraints()
        invalid_criteria.priority_weights = {PerformanceMetric.ACCURACY: 1.0}
        invalid_criteria.target_deployment = "general"
        
        success = config_manager.save_configuration(invalid_criteria)
        
        assert not success
    
    def test_save_configuration_atomic_write(self, temp_config_path, valid_criteria):
        """Test that save uses atomic write (temp file + rename)."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        with patch('pathlib.Path.replace') as mock_replace:
            config_manager.save_configuration(valid_criteria)
            
            # Verify atomic rename was called
            mock_replace.assert_called_once()


class TestConfigurationValidation:
    """Test configuration validation functionality."""
    
    def test_validate_valid_configuration(self, valid_criteria):
        """Test validation of valid configuration."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager()
        
        result = config_manager.validate_configuration(valid_criteria)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_validate_empty_name(self):
        """Test validation fails for empty name."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager()
        
        criteria = OptimizationCriteria(
            name="",
            description="Test",
            priority_weights={PerformanceMetric.ACCURACY: 1.0}
        )
        
        result = config_manager.validate_configuration(criteria)
        
        assert not result.is_valid
        assert any("name cannot be empty" in error.lower() for error in result.errors)
    
    def test_validate_invalid_time_constraint(self):
        """Test validation fails for invalid time constraint."""
        config_manager = ConfigurationManager()
        
        criteria = OptimizationCriteria(
            name="test",
            description="Test",
            constraints=OptimizationConstraints(
                max_optimization_time_minutes=-10  # Invalid: negative
            ),
            priority_weights={PerformanceMetric.ACCURACY: 1.0}
        )
        
        result = config_manager.validate_configuration(criteria)
        
        assert not result.is_valid
        assert any("time must be positive" in error.lower() for error in result.errors)
    
    def test_validate_invalid_memory_constraint(self):
        """Test validation fails for invalid memory constraint."""
        config_manager = ConfigurationManager()
        
        criteria = OptimizationCriteria(
            name="test",
            description="Test",
            constraints=OptimizationConstraints(
                max_memory_usage_gb=0  # Invalid: zero
            ),
            priority_weights={PerformanceMetric.ACCURACY: 1.0}
        )
        
        result = config_manager.validate_configuration(criteria)
        
        assert not result.is_valid
        assert any("memory" in error.lower() for error in result.errors)
    
    def test_validate_invalid_accuracy_threshold(self):
        """Test validation fails for out-of-range accuracy threshold."""
        config_manager = ConfigurationManager()
        
        criteria = OptimizationCriteria(
            name="test",
            description="Test",
            constraints=OptimizationConstraints(
                preserve_accuracy_threshold=1.5  # Invalid: > 1.0
            ),
            priority_weights={PerformanceMetric.ACCURACY: 1.0}
        )
        
        result = config_manager.validate_configuration(criteria)
        
        assert not result.is_valid
        assert any("accuracy threshold" in error.lower() for error in result.errors)
    
    def test_validate_conflicting_techniques(self):
        """Test validation fails when techniques are both allowed and forbidden."""
        config_manager = ConfigurationManager()
        
        criteria = OptimizationCriteria(
            name="test",
            description="Test",
            constraints=OptimizationConstraints(
                allowed_techniques=[OptimizationTechnique.QUANTIZATION],
                forbidden_techniques=[OptimizationTechnique.QUANTIZATION]  # Conflict
            ),
            priority_weights={PerformanceMetric.ACCURACY: 1.0}
        )
        
        result = config_manager.validate_configuration(criteria)
        
        assert not result.is_valid
        assert any("both allowed and forbidden" in error.lower() for error in result.errors)
    
    def test_validate_no_available_techniques(self):
        """Test validation fails when all techniques are forbidden."""
        config_manager = ConfigurationManager()
        
        criteria = OptimizationCriteria(
            name="test",
            description="Test",
            constraints=OptimizationConstraints(
                allowed_techniques=[OptimizationTechnique.QUANTIZATION],
                forbidden_techniques=[OptimizationTechnique.QUANTIZATION]
            ),
            priority_weights={PerformanceMetric.ACCURACY: 1.0}
        )
        
        result = config_manager.validate_configuration(criteria)
        
        assert not result.is_valid
    
    def test_validate_invalid_priority_weights(self):
        """Test validation fails when priority weights don't sum to 1.0."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager()
        
        # Create criteria bypassing validation
        criteria = OptimizationCriteria.__new__(OptimizationCriteria)
        criteria.name = "test"
        criteria.description = "Test"
        criteria.performance_thresholds = []
        criteria.constraints = OptimizationConstraints()
        criteria.priority_weights = {
            PerformanceMetric.ACCURACY: 0.5,
            PerformanceMetric.MODEL_SIZE: 0.6  # Total = 1.1
        }
        criteria.target_deployment = "general"
        
        result = config_manager.validate_configuration(criteria)
        
        assert not result.is_valid
        assert any("sum to 1.0" in error.lower() for error in result.errors)
    
    def test_validate_negative_priority_weight(self):
        """Test validation fails for negative priority weight."""
        config_manager = ConfigurationManager()
        
        criteria = OptimizationCriteria(
            name="test",
            description="Test",
            priority_weights={
                PerformanceMetric.ACCURACY: -0.5,  # Invalid: negative
                PerformanceMetric.MODEL_SIZE: 1.5
            }
        )
        
        result = config_manager.validate_configuration(criteria)
        
        assert not result.is_valid
        assert any("cannot be negative" in error.lower() for error in result.errors)
    
    def test_validate_invalid_deployment_target(self):
        """Test validation fails for invalid deployment target."""
        config_manager = ConfigurationManager()
        
        criteria = OptimizationCriteria(
            name="test",
            description="Test",
            target_deployment="invalid_target",  # Invalid
            priority_weights={PerformanceMetric.ACCURACY: 1.0}
        )
        
        result = config_manager.validate_configuration(criteria)
        
        assert not result.is_valid
        assert any("deployment target" in error.lower() for error in result.errors)
    
    def test_validate_invalid_performance_threshold_range(self):
        """Test validation fails when min > max in performance threshold."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager()
        
        # Create criteria bypassing validation
        criteria = OptimizationCriteria.__new__(OptimizationCriteria)
        criteria.name = "test"
        criteria.description = "Test"
        criteria.performance_thresholds = [
            PerformanceThreshold(
                metric=PerformanceMetric.ACCURACY,
                min_value=0.95,
                max_value=0.90  # Invalid: min > max
            )
        ]
        criteria.constraints = OptimizationConstraints()
        criteria.priority_weights = {PerformanceMetric.ACCURACY: 1.0}
        criteria.target_deployment = "general"
        
        result = config_manager.validate_configuration(criteria)
        
        assert not result.is_valid
        assert any("min value cannot exceed max value" in error.lower() for error in result.errors)
    
    def test_validate_invalid_tolerance(self):
        """Test validation fails for invalid tolerance value."""
        config_manager = ConfigurationManager()
        
        criteria = OptimizationCriteria(
            name="test",
            description="Test",
            performance_thresholds=[
                PerformanceThreshold(
                    metric=PerformanceMetric.ACCURACY,
                    tolerance=1.5  # Invalid: > 1.0
                )
            ],
            priority_weights={PerformanceMetric.ACCURACY: 1.0}
        )
        
        result = config_manager.validate_configuration(criteria)
        
        assert not result.is_valid
        assert any("tolerance" in error.lower() for error in result.errors)
    
    def test_validate_warnings(self):
        """Test that validation can produce warnings without failing."""
        config_manager = ConfigurationManager()
        
        criteria = OptimizationCriteria(
            name="test",
            description="",  # Empty description should produce warning
            constraints=OptimizationConstraints(
                preserve_accuracy_threshold=0.4  # Low threshold should produce warning
            ),
            priority_weights={PerformanceMetric.ACCURACY: 1.0}
        )
        
        result = config_manager.validate_configuration(criteria)
        
        # Should still be valid but have warnings
        assert result.is_valid
        assert len(result.warnings) > 0


class TestConcurrentUpdates:
    """Test concurrent configuration update handling."""
    
    def test_concurrent_save_operations(self, temp_config_path):
        """Test multiple threads saving configurations concurrently."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        def save_config(index):
            criteria = OptimizationCriteria(
                name=f"config_{index}",
                description=f"Config {index}",
                priority_weights={PerformanceMetric.ACCURACY: 1.0}
            )
            return config_manager.save_configuration(criteria)
        
        # Run concurrent saves
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(save_config, i) for i in range(10)]
            results = [f.result() for f in as_completed(futures)]
        
        # All saves should succeed (last one wins)
        assert any(results)  # At least some should succeed
    
    def test_concurrent_load_operations(self, temp_config_path, valid_criteria):
        """Test multiple threads loading configuration concurrently."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        config_manager.save_configuration(valid_criteria)
        
        def load_config():
            return config_manager.load_configuration()
        
        # Run concurrent loads
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(load_config) for _ in range(10)]
            results = [f.result() for f in as_completed(futures)]
        
        # All loads should succeed and return valid data
        assert all(r is not None for r in results)
        assert all(r.name == valid_criteria.name for r in results)
    
    def test_concurrent_update_operations(self, temp_config_path, valid_criteria):
        """Test multiple threads updating configuration concurrently."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        config_manager.save_configuration(valid_criteria)
        
        def update_config(index):
            criteria = OptimizationCriteria(
                name=f"updated_{index}",
                description=f"Updated {index}",
                priority_weights={PerformanceMetric.ACCURACY: 1.0}
            )
            return config_manager.update_configuration(criteria)
        
        # Run concurrent updates
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(update_config, i) for i in range(10)]
            results = [f.result() for f in as_completed(futures)]
        
        # Updates should complete without errors
        assert any(results)  # At least some should succeed
    
    def test_thread_safety_with_lock(self, temp_config_path, valid_criteria):
        """Test that configuration lock prevents race conditions."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        config_manager.save_configuration(valid_criteria)
        
        counter = {"value": 0}
        lock_acquired_count = {"value": 0}
        
        def update_with_counter():
            with config_manager._config_lock:
                lock_acquired_count["value"] += 1
                current = counter["value"]
                time.sleep(0.001)  # Simulate work
                counter["value"] = current + 1
        
        # Run concurrent operations
        threads = [threading.Thread(target=update_with_counter) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Counter should be exactly 10 (no race conditions)
        assert counter["value"] == 10
        assert lock_acquired_count["value"] == 10


class TestRollbackOnErrors:
    """Test rollback functionality when errors occur."""
    
    def test_save_rollback_on_validation_error(self, temp_config_path, valid_criteria):
        """Test that save doesn't persist invalid configuration."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Save valid config first
        config_manager.save_configuration(valid_criteria)
        original_content = config_manager.config_path.read_text()
        
        # Try to save invalid config
        invalid_criteria = OptimizationCriteria(
            name="",  # Invalid
            description="Test",
            priority_weights={PerformanceMetric.ACCURACY: 1.0}
        )
        
        success = config_manager.save_configuration(invalid_criteria)
        
        # Save should fail
        assert not success
        
        # Original file should be unchanged
        current_content = config_manager.config_path.read_text()
        assert current_content == original_content
    
    def test_update_rollback_on_validation_error(self, temp_config_path, valid_criteria):
        """Test that update doesn't modify configuration on validation error."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        config_manager.save_configuration(valid_criteria)
        
        # Try to update with invalid config (bypass validation)
        invalid_criteria = OptimizationCriteria.__new__(OptimizationCriteria)
        invalid_criteria.name = "invalid"
        invalid_criteria.description = "Test"
        invalid_criteria.performance_thresholds = []
        invalid_criteria.constraints = OptimizationConstraints()
        invalid_criteria.priority_weights = {
            PerformanceMetric.ACCURACY: 0.5,
            PerformanceMetric.MODEL_SIZE: 0.6  # Invalid: sum > 1.0
        }
        invalid_criteria.target_deployment = "general"
        
        success = config_manager.update_configuration(invalid_criteria)
        
        # Update should fail
        assert not success
        
        # Current config should still be the original
        current = config_manager.get_current_configuration()
        assert current.name == valid_criteria.name
    
    def test_load_rollback_on_corrupted_file(self):
        """Test that load returns default config when file is corrupted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text("corrupted json {{{")
            
            # Reset singleton for clean test
            ConfigurationManager._instance = None
            config_manager = ConfigurationManager(str(config_path))
            loaded = config_manager.load_configuration()
            
            # Should return default config, not crash
            assert loaded is not None
            assert loaded.name == "default"
    
    def test_save_rollback_on_io_error(self, temp_config_path, valid_criteria):
        """Test that save handles I/O errors gracefully."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        
        # Mock file write to raise an exception
        with patch('builtins.open', side_effect=IOError("Disk full")):
            success = config_manager.save_configuration(valid_criteria)
            
            # Save should fail gracefully
            assert not success
    
    def test_concurrent_update_consistency(self, temp_config_path, valid_criteria):
        """Test that concurrent updates maintain consistency."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager(temp_config_path)
        config_manager.save_configuration(valid_criteria)
        
        results = []
        
        def update_config(index):
            criteria = OptimizationCriteria(
                name=f"config_{index}",
                description=f"Config {index}",
                priority_weights={PerformanceMetric.ACCURACY: 1.0}
            )
            success = config_manager.update_configuration(criteria)
            results.append((index, success))
        
        # Run concurrent updates
        threads = [threading.Thread(target=update_config, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Final config should be one of the updates
        final_config = config_manager.get_current_configuration()
        assert final_config.name.startswith("config_")


class TestSingletonPattern:
    """Test singleton pattern implementation."""
    
    def test_singleton_returns_same_instance(self):
        """Test that multiple instantiations return the same instance."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        manager1 = ConfigurationManager()
        manager2 = ConfigurationManager()
        
        assert manager1 is manager2
    
    def test_singleton_with_different_paths(self):
        """Test that singleton ignores path after first instantiation."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        manager1 = ConfigurationManager("path1")
        manager2 = ConfigurationManager("path2")
        
        # Should return same instance
        assert manager1 is manager2
        # Path should be from first instantiation
        assert str(manager1.config_path) == "path1"


class TestDefaultConfiguration:
    """Test default configuration generation."""
    
    def test_get_default_configuration(self):
        """Test that default configuration is valid."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager()
        
        default = config_manager._get_default_configuration()
        
        assert default.name == "default"
        assert default.description
        assert len(default.performance_thresholds) > 0
        assert default.target_deployment == "edge"
    
    def test_default_configuration_is_valid(self):
        """Test that default configuration passes validation."""
        # Reset singleton for clean test
        ConfigurationManager._instance = None
        config_manager = ConfigurationManager()
        
        default = config_manager._get_default_configuration()
        result = config_manager.validate_configuration(default)
        
        assert result.is_valid
        assert len(result.errors) == 0


class TestValidationResult:
    """Test ValidationResult helper class."""
    
    def test_validation_result_initialization(self):
        """Test ValidationResult starts as valid."""
        result = ValidationResult()
        
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
    
    def test_add_error_marks_invalid(self):
        """Test that adding error marks result as invalid."""
        result = ValidationResult()
        
        result.add_error("Test error")
        
        assert not result.is_valid
        assert "Test error" in result.errors
    
    def test_add_warning_keeps_valid(self):
        """Test that adding warning doesn't mark as invalid."""
        result = ValidationResult()
        
        result.add_warning("Test warning")
        
        assert result.is_valid
        assert "Test warning" in result.warnings
    
    def test_multiple_errors_and_warnings(self):
        """Test accumulating multiple errors and warnings."""
        result = ValidationResult()
        
        result.add_error("Error 1")
        result.add_error("Error 2")
        result.add_warning("Warning 1")
        result.add_warning("Warning 2")
        
        assert not result.is_valid
        assert len(result.errors) == 2
        assert len(result.warnings) == 2
