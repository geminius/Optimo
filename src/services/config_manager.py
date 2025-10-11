"""
Configuration Manager for API-level optimization criteria management.

This module provides the ConfigurationManager service that handles
loading, persisting, and validating optimization criteria configurations
for the API endpoints.
"""

import logging
import json
import threading
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

from ..config.optimization_criteria import (
    OptimizationCriteria,
    OptimizationConstraints,
    OptimizationTechnique,
    PerformanceMetric,
    PerformanceThreshold
)


logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
    
    def add_error(self, error: str) -> None:
        """Add a validation error."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str) -> None:
        """Add a validation warning."""
        self.warnings.append(warning)


class ConfigurationManager:
    """
    Manages optimization criteria configuration for API endpoints.
    
    This service provides thread-safe configuration loading, persistence,
    and validation for the REST API. It's designed to work with the existing
    ConfigurationManager in src/config/optimization_criteria.py but provides
    a simpler interface focused on API needs.
    
    Responsibilities:
    - Load and persist optimization criteria configuration
    - Validate configuration values and constraints
    - Provide thread-safe configuration updates
    - Handle configuration file I/O
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, config_path: Optional[str] = None):
        """Singleton pattern to ensure single instance."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to configuration file (default: config/optimization_criteria.json)
        """
        # Only initialize once
        if self._initialized:
            return
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration file path
        if config_path is None:
            config_path = "config/optimization_criteria.json"
        self.config_path = Path(config_path)
        
        # Current configuration
        self._current_config: Optional[OptimizationCriteria] = None
        
        # Thread safety for configuration updates
        self._config_lock = threading.RLock()
        
        # Available optimization techniques (from agents)
        self._available_techniques = [
            OptimizationTechnique.QUANTIZATION,
            OptimizationTechnique.PRUNING,
            OptimizationTechnique.DISTILLATION,
            OptimizationTechnique.ARCHITECTURE_SEARCH,
            OptimizationTechnique.COMPRESSION
        ]
        
        self._initialized = True
        self.logger.info(f"ConfigurationManager initialized with config path: {self.config_path}")
    
    def load_configuration(self) -> Optional[OptimizationCriteria]:
        """
        Load configuration from file.
        
        Returns:
            OptimizationCriteria if loaded successfully, None otherwise
        """
        with self._config_lock:
            try:
                if not self.config_path.exists():
                    self.logger.warning(f"Configuration file not found: {self.config_path}")
                    return self._get_default_configuration()
                
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Convert dictionary to OptimizationCriteria
                criteria = self._dict_to_criteria(config_data)
                
                # Validate loaded configuration
                validation_result = self.validate_configuration(criteria)
                if not validation_result.is_valid:
                    self.logger.error(f"Loaded configuration is invalid: {validation_result.errors}")
                    return self._get_default_configuration()
                
                if validation_result.warnings:
                    for warning in validation_result.warnings:
                        self.logger.warning(f"Configuration warning: {warning}")
                
                self._current_config = criteria
                self.logger.info(f"Loaded configuration: {criteria.name}")
                return criteria
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse configuration file: {e}")
                return self._get_default_configuration()
            except Exception as e:
                self.logger.error(f"Failed to load configuration: {e}")
                return self._get_default_configuration()
    
    def save_configuration(self, criteria: OptimizationCriteria) -> bool:
        """
        Save configuration to file.
        
        Args:
            criteria: OptimizationCriteria to save
            
        Returns:
            True if saved successfully, False otherwise
        """
        with self._config_lock:
            try:
                # Validate before saving
                validation_result = self.validate_configuration(criteria)
                if not validation_result.is_valid:
                    self.logger.error(f"Cannot save invalid configuration: {validation_result.errors}")
                    return False
                
                # Ensure directory exists
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Convert criteria to dictionary
                config_data = self._criteria_to_dict(criteria)
                
                # Write to file with atomic operation (write to temp, then rename)
                temp_path = self.config_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(config_data, f, indent=2)
                
                # Atomic rename
                temp_path.replace(self.config_path)
                
                self._current_config = criteria
                self.logger.info(f"Saved configuration: {criteria.name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to save configuration: {e}")
                return False
    
    def get_current_configuration(self) -> OptimizationCriteria:
        """
        Get current active configuration.
        
        Returns:
            Current OptimizationCriteria (loads from file if not cached)
        """
        with self._config_lock:
            if self._current_config is None:
                self._current_config = self.load_configuration()
            
            if self._current_config is None:
                self._current_config = self._get_default_configuration()
            
            return self._current_config
    
    def update_configuration(self, criteria: OptimizationCriteria) -> bool:
        """
        Update configuration with validation and persistence.
        
        Args:
            criteria: New OptimizationCriteria
            
        Returns:
            True if updated successfully, False otherwise
        """
        with self._config_lock:
            try:
                # Validate new configuration
                validation_result = self.validate_configuration(criteria)
                if not validation_result.is_valid:
                    self.logger.error(f"Cannot update to invalid configuration: {validation_result.errors}")
                    return False
                
                # Save to file
                if not self.save_configuration(criteria):
                    return False
                
                # Update in-memory configuration
                self._current_config = criteria
                
                self.logger.info(f"Updated configuration: {criteria.name}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to update configuration: {e}")
                return False
    
    def validate_configuration(self, criteria: OptimizationCriteria) -> ValidationResult:
        """
        Validate optimization criteria configuration.
        
        Args:
            criteria: OptimizationCriteria to validate
            
        Returns:
            ValidationResult with validation status and messages
        """
        result = ValidationResult()
        
        try:
            # Basic validation
            if not criteria.name or not criteria.name.strip():
                result.add_error("Configuration name cannot be empty")
            
            if not criteria.description or not criteria.description.strip():
                result.add_warning("Configuration description is empty")
            
            # Validate constraints
            self._validate_constraints(criteria.constraints, result)
            
            # Validate performance thresholds
            self._validate_performance_thresholds(criteria.performance_thresholds, result)
            
            # Validate priority weights
            self._validate_priority_weights(criteria.priority_weights, result)
            
            # Validate deployment target
            self._validate_deployment_target(criteria.target_deployment, result)
            
            # Validate technique compatibility
            self._validate_technique_compatibility(criteria.constraints, result)
            
        except Exception as e:
            result.add_error(f"Validation error: {str(e)}")
        
        return result
    
    def _validate_constraints(self, constraints: OptimizationConstraints, result: ValidationResult) -> None:
        """Validate optimization constraints."""
        # Validate time constraint
        if constraints.max_optimization_time_minutes <= 0:
            result.add_error("Max optimization time must be positive")
        elif constraints.max_optimization_time_minutes > 1440:  # 24 hours
            result.add_warning("Max optimization time exceeds 24 hours")
        
        # Validate memory constraint
        if constraints.max_memory_usage_gb <= 0:
            result.add_error("Max memory usage must be positive")
        elif constraints.max_memory_usage_gb > 128:
            result.add_warning("Max memory usage exceeds 128 GB")
        
        # Validate accuracy threshold
        if constraints.preserve_accuracy_threshold < 0 or constraints.preserve_accuracy_threshold > 1:
            result.add_error("Accuracy threshold must be between 0 and 1")
        elif constraints.preserve_accuracy_threshold < 0.5:
            result.add_warning("Accuracy threshold below 50% may result in poor model quality")
        
        # Validate technique lists
        allowed_set = set(constraints.allowed_techniques)
        forbidden_set = set(constraints.forbidden_techniques)
        
        if allowed_set.intersection(forbidden_set):
            result.add_error("Techniques cannot be both allowed and forbidden")
        
        # Check if any techniques are available
        available_techniques = allowed_set - forbidden_set
        if not available_techniques:
            result.add_error("No optimization techniques are available (all are forbidden)")
        
        # Validate hardware constraints
        if constraints.hardware_constraints:
            if "gpu_memory_gb" in constraints.hardware_constraints:
                gpu_mem = constraints.hardware_constraints["gpu_memory_gb"]
                if gpu_mem <= 0:
                    result.add_error("GPU memory constraint must be positive")
            
            if "cpu_cores" in constraints.hardware_constraints:
                cpu_cores = constraints.hardware_constraints["cpu_cores"]
                if cpu_cores <= 0:
                    result.add_error("CPU cores constraint must be positive")
    
    def _validate_performance_thresholds(
        self, 
        thresholds: List[PerformanceThreshold], 
        result: ValidationResult
    ) -> None:
        """Validate performance thresholds."""
        for threshold in thresholds:
            # Validate tolerance
            if threshold.tolerance < 0 or threshold.tolerance > 1:
                result.add_error(f"Tolerance for {threshold.metric.value} must be between 0 and 1")
            
            # Validate min/max relationship
            if threshold.min_value is not None and threshold.max_value is not None:
                if threshold.min_value > threshold.max_value:
                    result.add_error(
                        f"Min value cannot exceed max value for {threshold.metric.value}"
                    )
            
            # Validate target value is within range
            if threshold.target_value is not None:
                if threshold.min_value is not None and threshold.target_value < threshold.min_value:
                    result.add_warning(
                        f"Target value for {threshold.metric.value} is below minimum"
                    )
                if threshold.max_value is not None and threshold.target_value > threshold.max_value:
                    result.add_warning(
                        f"Target value for {threshold.metric.value} exceeds maximum"
                    )
    
    def _validate_priority_weights(
        self, 
        priority_weights: Dict[PerformanceMetric, float], 
        result: ValidationResult
    ) -> None:
        """Validate priority weights."""
        if not priority_weights:
            result.add_warning("No priority weights specified")
            return
        
        # Check individual weights
        for metric, weight in priority_weights.items():
            if weight < 0:
                result.add_error(f"Priority weight for {metric.value} cannot be negative")
            elif weight > 1:
                result.add_error(f"Priority weight for {metric.value} cannot exceed 1.0")
        
        # Check total weight
        total_weight = sum(priority_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            result.add_error(f"Priority weights must sum to 1.0, got {total_weight:.3f}")
    
    def _validate_deployment_target(self, target_deployment: str, result: ValidationResult) -> None:
        """Validate deployment target."""
        valid_targets = {"edge", "cloud", "mobile", "general"}
        if target_deployment not in valid_targets:
            result.add_error(
                f"Invalid deployment target '{target_deployment}'. "
                f"Must be one of: {', '.join(valid_targets)}"
            )
    
    def _validate_technique_compatibility(
        self, 
        constraints: OptimizationConstraints, 
        result: ValidationResult
    ) -> None:
        """Validate that enabled techniques are available."""
        # Get enabled techniques (allowed - forbidden)
        allowed_set = set(constraints.allowed_techniques)
        forbidden_set = set(constraints.forbidden_techniques)
        enabled_techniques = allowed_set - forbidden_set
        
        # Check against available techniques
        available_set = set(self._available_techniques)
        unavailable = enabled_techniques - available_set
        
        if unavailable:
            unavailable_names = [t.value for t in unavailable]
            result.add_warning(
                f"Some enabled techniques are not available: {', '.join(unavailable_names)}"
            )
    
    def _dict_to_criteria(self, config_data: Dict[str, Any]) -> OptimizationCriteria:
        """Convert dictionary to OptimizationCriteria object."""
        # Convert performance thresholds
        thresholds = []
        for threshold_data in config_data.get("performance_thresholds", []):
            threshold = PerformanceThreshold(
                metric=PerformanceMetric(threshold_data["metric"]),
                min_value=threshold_data.get("min_value"),
                max_value=threshold_data.get("max_value"),
                target_value=threshold_data.get("target_value"),
                tolerance=threshold_data.get("tolerance", 0.05)
            )
            thresholds.append(threshold)
        
        # Convert constraints
        constraints_data = config_data.get("constraints", {})
        constraints = OptimizationConstraints(
            max_optimization_time_minutes=constraints_data.get("max_optimization_time_minutes", 60),
            max_memory_usage_gb=constraints_data.get("max_memory_usage_gb", 16.0),
            preserve_accuracy_threshold=constraints_data.get("preserve_accuracy_threshold", 0.95),
            allowed_techniques=[
                OptimizationTechnique(t) 
                for t in constraints_data.get("allowed_techniques", [t.value for t in OptimizationTechnique])
            ],
            forbidden_techniques=[
                OptimizationTechnique(t) 
                for t in constraints_data.get("forbidden_techniques", [])
            ],
            hardware_constraints=constraints_data.get("hardware_constraints", {})
        )
        
        # Convert priority weights
        priority_weights = {}
        for metric, weight in config_data.get("priority_weights", {}).items():
            priority_weights[PerformanceMetric(metric)] = weight
        
        return OptimizationCriteria(
            name=config_data["name"],
            description=config_data["description"],
            performance_thresholds=thresholds,
            constraints=constraints,
            priority_weights=priority_weights,
            target_deployment=config_data.get("target_deployment", "general")
        )
    
    def _criteria_to_dict(self, criteria: OptimizationCriteria) -> Dict[str, Any]:
        """Convert OptimizationCriteria to dictionary."""
        return {
            "name": criteria.name,
            "description": criteria.description,
            "performance_thresholds": [
                {
                    "metric": threshold.metric.value,
                    "min_value": threshold.min_value,
                    "max_value": threshold.max_value,
                    "target_value": threshold.target_value,
                    "tolerance": threshold.tolerance
                }
                for threshold in criteria.performance_thresholds
            ],
            "constraints": {
                "max_optimization_time_minutes": criteria.constraints.max_optimization_time_minutes,
                "max_memory_usage_gb": criteria.constraints.max_memory_usage_gb,
                "preserve_accuracy_threshold": criteria.constraints.preserve_accuracy_threshold,
                "allowed_techniques": [t.value for t in criteria.constraints.allowed_techniques],
                "forbidden_techniques": [t.value for t in criteria.constraints.forbidden_techniques],
                "hardware_constraints": criteria.constraints.hardware_constraints
            },
            "priority_weights": {
                metric.value: weight 
                for metric, weight in criteria.priority_weights.items()
            },
            "target_deployment": criteria.target_deployment
        }
    
    def _get_default_configuration(self) -> OptimizationCriteria:
        """Get default configuration when no configuration file exists."""
        self.logger.info("Using default configuration")
        
        return OptimizationCriteria(
            name="default",
            description="Default optimization criteria for robotics models",
            performance_thresholds=[
                PerformanceThreshold(
                    metric=PerformanceMetric.ACCURACY,
                    min_value=0.90,
                    target_value=0.95,
                    tolerance=0.05
                ),
                PerformanceThreshold(
                    metric=PerformanceMetric.MODEL_SIZE,
                    max_value=500.0,  # 500 MB
                    target_value=100.0,  # 100 MB
                    tolerance=0.1
                ),
                PerformanceThreshold(
                    metric=PerformanceMetric.INFERENCE_TIME,
                    max_value=100.0,  # 100 ms
                    target_value=50.0,  # 50 ms
                    tolerance=0.1
                )
            ],
            constraints=OptimizationConstraints(
                max_optimization_time_minutes=60,
                max_memory_usage_gb=16.0,
                preserve_accuracy_threshold=0.95,
                allowed_techniques=list(OptimizationTechnique),
                forbidden_techniques=[],
                hardware_constraints={}
            ),
            priority_weights={
                PerformanceMetric.ACCURACY: 0.4,
                PerformanceMetric.MODEL_SIZE: 0.3,
                PerformanceMetric.INFERENCE_TIME: 0.3
            },
            target_deployment="edge"
        )
