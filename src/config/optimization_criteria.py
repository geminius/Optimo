"""
Configuration classes for optimization criteria and constraints.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Set
from enum import Enum
import json
import threading
import time
from pathlib import Path
from datetime import datetime
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class OptimizationTechnique(Enum):
    """Supported optimization techniques."""
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    ARCHITECTURE_SEARCH = "architecture_search"
    COMPRESSION = "compression"


class PerformanceMetric(Enum):
    """Performance metrics for evaluation."""
    ACCURACY = "accuracy"
    INFERENCE_TIME = "inference_time"
    MEMORY_USAGE = "memory_usage"
    MODEL_SIZE = "model_size"
    THROUGHPUT = "throughput"
    FLOPS = "flops"


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric: PerformanceMetric
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    target_value: Optional[float] = None
    tolerance: float = 0.05  # 5% tolerance by default


@dataclass
class OptimizationConstraints:
    """Constraints for optimization process."""
    max_optimization_time_minutes: int = 60
    max_memory_usage_gb: float = 16.0
    preserve_accuracy_threshold: float = 0.95  # Minimum 95% accuracy preservation
    allowed_techniques: List[OptimizationTechnique] = field(default_factory=lambda: list(OptimizationTechnique))
    forbidden_techniques: List[OptimizationTechnique] = field(default_factory=list)
    hardware_constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationCriteria:
    """Complete optimization criteria configuration."""
    name: str
    description: str
    performance_thresholds: List[PerformanceThreshold] = field(default_factory=list)
    constraints: OptimizationConstraints = field(default_factory=OptimizationConstraints)
    priority_weights: Dict[PerformanceMetric, float] = field(default_factory=dict)
    target_deployment: str = "general"  # "edge", "cloud", "mobile", "general"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate the optimization criteria configuration."""
        # Validate priority weights sum to 1.0
        if self.priority_weights:
            total_weight = sum(self.priority_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                raise ValueError(f"Priority weights must sum to 1.0, got {total_weight}")
        
        # Validate performance thresholds
        for threshold in self.performance_thresholds:
            if threshold.min_value is not None and threshold.max_value is not None:
                if threshold.min_value > threshold.max_value:
                    raise ValueError(f"Min value cannot be greater than max value for {threshold.metric}")


@dataclass
class ConfigurationConflict:
    """Represents a configuration conflict."""
    config_name: str
    conflict_type: str
    description: str
    severity: str  # "error", "warning", "info"
    suggested_resolution: Optional[str] = None


class ConfigurationFileHandler(FileSystemEventHandler):
    """Handles file system events for configuration files."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logging.getLogger(__name__)
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            self.logger.info(f"Configuration file modified: {event.src_path}")
            self.config_manager._reload_configuration(Path(event.src_path))
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            self.logger.info(f"Configuration file created: {event.src_path}")
            self.config_manager._reload_configuration(Path(event.src_path))
    
    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            self.logger.info(f"Configuration file deleted: {event.src_path}")
            config_name = Path(event.src_path).stem
            self.config_manager._remove_configuration(config_name)


class ConfigurationManager:
    """Enhanced configuration manager with dynamic updates and conflict detection."""
    
    def __init__(self, config_dir: str = "config", enable_auto_reload: bool = True):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._configurations: Dict[str, OptimizationCriteria] = {}
        self._configuration_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._observers: List[Callable[[str, OptimizationCriteria], None]] = []
        self._conflicts: List[ConfigurationConflict] = []
        self.logger = logging.getLogger(__name__)
        
        # File system monitoring
        self._file_observer = None
        self._file_handler = None
        if enable_auto_reload:
            self._setup_file_monitoring()
        
        self._load_configurations()
    
    def _setup_file_monitoring(self):
        """Set up file system monitoring for dynamic configuration updates."""
        try:
            self._file_handler = ConfigurationFileHandler(self)
            self._file_observer = Observer()
            self._file_observer.schedule(self._file_handler, str(self.config_dir), recursive=False)
            self._file_observer.start()
            self.logger.info("File system monitoring enabled for configuration directory")
        except Exception as e:
            self.logger.warning(f"Failed to set up file monitoring: {e}")
    
    def _load_configurations(self):
        """Load all configuration files from the config directory."""
        with self._lock:
            config_files = self.config_dir.glob("*.json")
            for config_file in config_files:
                self._load_single_configuration(config_file)
            
            # Detect conflicts after loading all configurations
            self._detect_conflicts()
    
    def _load_single_configuration(self, config_file: Path):
        """Load a single configuration file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                criteria = self._dict_to_criteria(config_data)
                
                # Store metadata
                self._configuration_metadata[criteria.name] = {
                    "file_path": str(config_file),
                    "last_modified": datetime.fromtimestamp(config_file.stat().st_mtime),
                    "loaded_at": datetime.now()
                }
                
                self._configurations[criteria.name] = criteria
                self.logger.info(f"Loaded configuration: {criteria.name}")
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration from {config_file}: {e}")
    
    def _reload_configuration(self, config_file: Path):
        """Reload a specific configuration file."""
        with self._lock:
            old_config_name = None
            
            # Find existing configuration for this file
            for name, metadata in self._configuration_metadata.items():
                if metadata["file_path"] == str(config_file):
                    old_config_name = name
                    break
            
            # Load the new configuration
            self._load_single_configuration(config_file)
            
            # Get the new configuration name
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                new_config_name = config_data["name"]
            
            # If name changed, remove old configuration
            if old_config_name and old_config_name != new_config_name:
                self._configurations.pop(old_config_name, None)
                self._configuration_metadata.pop(old_config_name, None)
            
            # Detect conflicts and notify observers
            self._detect_conflicts()
            if new_config_name in self._configurations:
                self._notify_observers(new_config_name, self._configurations[new_config_name])
    
    def _remove_configuration(self, config_name: str):
        """Remove a configuration from memory."""
        with self._lock:
            if config_name in self._configurations:
                del self._configurations[config_name]
                self._configuration_metadata.pop(config_name, None)
                self.logger.info(f"Removed configuration: {config_name}")
                self._detect_conflicts()
    
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
            allowed_techniques=[OptimizationTechnique(t) for t in constraints_data.get("allowed_techniques", [t.value for t in OptimizationTechnique])],
            forbidden_techniques=[OptimizationTechnique(t) for t in constraints_data.get("forbidden_techniques", [])],
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
    
    def get_criteria(self, name: str) -> Optional[OptimizationCriteria]:
        """Get optimization criteria by name."""
        with self._lock:
            return self._configurations.get(name)
    
    def list_criteria(self) -> List[str]:
        """List all available criteria names."""
        with self._lock:
            return list(self._configurations.keys())
    
    def add_criteria(self, criteria: OptimizationCriteria) -> bool:
        """Add new optimization criteria."""
        with self._lock:
            # Validate before adding
            issues = self.validate_criteria(criteria)
            if issues:
                self.logger.error(f"Validation failed for criteria '{criteria.name}': {issues}")
                return False
            
            if criteria.name in self._configurations:
                self.logger.warning(f"Configuration '{criteria.name}' already exists, use update_criteria instead")
                return False
            
            self._configurations[criteria.name] = criteria
            
            # Store metadata
            config_file = self.config_dir / f"{criteria.name}.json"
            self._configuration_metadata[criteria.name] = {
                "file_path": str(config_file),
                "last_modified": datetime.now(),
                "loaded_at": datetime.now()
            }
            
            self._save_criteria(criteria)
            self._detect_conflicts()
            self._notify_observers(criteria.name, criteria)
            return True
    
    def update_criteria(self, name: str, criteria: OptimizationCriteria) -> bool:
        """Update existing optimization criteria."""
        with self._lock:
            if name not in self._configurations:
                return False
            
            # Validate before updating
            issues = self.validate_criteria(criteria)
            if issues:
                self.logger.error(f"Validation failed for criteria '{criteria.name}': {issues}")
                return False
            
            # If name changed, handle the rename
            if name != criteria.name:
                if criteria.name in self._configurations:
                    self.logger.error(f"Cannot rename to '{criteria.name}': name already exists")
                    return False
                
                # Remove old configuration
                del self._configurations[name]
                old_config_file = self.config_dir / f"{name}.json"
                if old_config_file.exists():
                    old_config_file.unlink()
                
                # Update metadata
                if name in self._configuration_metadata:
                    metadata = self._configuration_metadata.pop(name)
                    metadata["file_path"] = str(self.config_dir / f"{criteria.name}.json")
                    self._configuration_metadata[criteria.name] = metadata
            
            self._configurations[criteria.name] = criteria
            self._save_criteria(criteria)
            self._detect_conflicts()
            self._notify_observers(criteria.name, criteria)
            return True
    
    def remove_criteria(self, name: str) -> bool:
        """Remove optimization criteria."""
        with self._lock:
            if name not in self._configurations:
                return False
            
            del self._configurations[name]
            self._configuration_metadata.pop(name, None)
            
            config_file = self.config_dir / f"{name}.json"
            if config_file.exists():
                config_file.unlink()
            
            self._detect_conflicts()
            self.logger.info(f"Removed configuration: {name}")
            return True
    
    def _save_criteria(self, criteria: OptimizationCriteria) -> None:
        """Save criteria to file."""
        config_file = self.config_dir / f"{criteria.name}.json"
        config_data = self._criteria_to_dict(criteria)
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
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
            "priority_weights": {metric.value: weight for metric, weight in criteria.priority_weights.items()},
            "target_deployment": criteria.target_deployment
        }
    
    def _detect_conflicts(self):
        """Detect conflicts between configurations."""
        self._conflicts.clear()
        
        # Check for duplicate names
        names = list(self._configurations.keys())
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            for name in set(duplicates):
                self._conflicts.append(ConfigurationConflict(
                    config_name=name,
                    conflict_type="duplicate_name",
                    description=f"Multiple configurations with name '{name}'",
                    severity="error",
                    suggested_resolution="Rename one of the configurations"
                ))
        
        # Check for conflicting constraints
        for name1, config1 in self._configurations.items():
            for name2, config2 in self._configurations.items():
                if name1 >= name2:  # Avoid duplicate comparisons
                    continue
                
                conflicts = self._check_configuration_conflicts(config1, config2)
                for conflict in conflicts:
                    conflict.config_name = f"{name1} vs {name2}"
                    self._conflicts.append(conflict)
        
        # Log conflicts
        if self._conflicts:
            self.logger.warning(f"Detected {len(self._conflicts)} configuration conflicts")
            for conflict in self._conflicts:
                self.logger.warning(f"Conflict: {conflict.description}")
    
    def _check_configuration_conflicts(self, config1: OptimizationCriteria, config2: OptimizationCriteria) -> List[ConfigurationConflict]:
        """Check for conflicts between two configurations."""
        conflicts = []
        
        # Check for conflicting deployment targets with overlapping constraints
        if (config1.target_deployment == config2.target_deployment and 
            config1.target_deployment != "general"):
            
            # Check for conflicting technique restrictions
            forbidden1 = set(config1.constraints.forbidden_techniques)
            allowed1 = set(config1.constraints.allowed_techniques)
            forbidden2 = set(config2.constraints.forbidden_techniques)
            allowed2 = set(config2.constraints.allowed_techniques)
            
            # If one config forbids what another allows
            conflict_techniques = forbidden1.intersection(allowed2) or forbidden2.intersection(allowed1)
            if conflict_techniques:
                conflicts.append(ConfigurationConflict(
                    config_name="",
                    conflict_type="technique_conflict",
                    description=f"Conflicting technique restrictions for {config1.target_deployment} deployment: {conflict_techniques}",
                    severity="warning",
                    suggested_resolution="Review technique restrictions for consistency"
                ))
            
            # Check for conflicting performance thresholds
            threshold_conflicts = self._check_threshold_conflicts(config1.performance_thresholds, config2.performance_thresholds)
            conflicts.extend(threshold_conflicts)
        
        return conflicts
    
    def _check_threshold_conflicts(self, thresholds1: List[PerformanceThreshold], thresholds2: List[PerformanceThreshold]) -> List[ConfigurationConflict]:
        """Check for conflicts between performance thresholds."""
        conflicts = []
        
        # Group thresholds by metric
        metrics1 = {t.metric: t for t in thresholds1}
        metrics2 = {t.metric: t for t in thresholds2}
        
        for metric in metrics1.keys() & metrics2.keys():
            t1, t2 = metrics1[metric], metrics2[metric]
            
            # Check for incompatible ranges
            if (t1.min_value is not None and t2.max_value is not None and 
                t1.min_value > t2.max_value):
                conflicts.append(ConfigurationConflict(
                    config_name="",
                    conflict_type="threshold_conflict",
                    description=f"Incompatible {metric.value} thresholds: min {t1.min_value} > max {t2.max_value}",
                    severity="error",
                    suggested_resolution="Adjust threshold ranges to be compatible"
                ))
            
            if (t2.min_value is not None and t1.max_value is not None and 
                t2.min_value > t1.max_value):
                conflicts.append(ConfigurationConflict(
                    config_name="",
                    conflict_type="threshold_conflict",
                    description=f"Incompatible {metric.value} thresholds: min {t2.min_value} > max {t1.max_value}",
                    severity="error",
                    suggested_resolution="Adjust threshold ranges to be compatible"
                ))
        
        return conflicts
    
    def add_observer(self, observer: Callable[[str, OptimizationCriteria], None]):
        """Add an observer for configuration changes."""
        with self._lock:
            self._observers.append(observer)
    
    def remove_observer(self, observer: Callable[[str, OptimizationCriteria], None]):
        """Remove a configuration change observer."""
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)
    
    def _notify_observers(self, config_name: str, criteria: OptimizationCriteria):
        """Notify all observers of configuration changes."""
        for observer in self._observers:
            try:
                observer(config_name, criteria)
            except Exception as e:
                self.logger.error(f"Error notifying observer: {e}")
    
    def get_conflicts(self) -> List[ConfigurationConflict]:
        """Get current configuration conflicts."""
        with self._lock:
            return self._conflicts.copy()
    
    def resolve_conflict(self, conflict: ConfigurationConflict, resolution_action: str, **kwargs) -> bool:
        """Attempt to resolve a configuration conflict."""
        with self._lock:
            try:
                if conflict.conflict_type == "duplicate_name":
                    if resolution_action == "rename":
                        old_name = kwargs.get("old_name")
                        new_name = kwargs.get("new_name")
                        if old_name in self._configurations and new_name not in self._configurations:
                            config = self._configurations.pop(old_name)
                            config.name = new_name
                            self._configurations[new_name] = config
                            self._save_criteria(config)
                            self._detect_conflicts()
                            return True
                
                elif conflict.conflict_type == "technique_conflict":
                    if resolution_action == "merge_restrictions":
                        # Implementation for merging technique restrictions
                        pass
                
                elif conflict.conflict_type == "threshold_conflict":
                    if resolution_action == "adjust_thresholds":
                        # Implementation for adjusting conflicting thresholds
                        pass
                
                return False
            except Exception as e:
                self.logger.error(f"Error resolving conflict: {e}")
                return False
    
    def validate_criteria(self, criteria: OptimizationCriteria) -> List[str]:
        """Enhanced validation for optimization criteria."""
        issues = []
        
        try:
            criteria._validate_configuration()
        except ValueError as e:
            issues.append(str(e))
        
        # Validate technique consistency
        forbidden_set = set(criteria.constraints.forbidden_techniques)
        allowed_set = set(criteria.constraints.allowed_techniques)
        
        if forbidden_set.intersection(allowed_set):
            issues.append("Techniques cannot be both allowed and forbidden")
        
        # Validate hardware constraints
        hw_constraints = criteria.constraints.hardware_constraints
        if "gpu_memory_gb" in hw_constraints and hw_constraints["gpu_memory_gb"] <= 0:
            issues.append("GPU memory constraint must be positive")
        
        if "cpu_cores" in hw_constraints and hw_constraints["cpu_cores"] <= 0:
            issues.append("CPU cores constraint must be positive")
        
        # Validate performance thresholds
        for threshold in criteria.performance_thresholds:
            if threshold.tolerance < 0 or threshold.tolerance > 1:
                issues.append(f"Tolerance for {threshold.metric.value} must be between 0 and 1")
        
        # Validate deployment target
        valid_targets = {"edge", "cloud", "mobile", "general"}
        if criteria.target_deployment not in valid_targets:
            issues.append(f"Invalid deployment target: {criteria.target_deployment}")
        
        return issues
    
    def get_configuration_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a configuration."""
        with self._lock:
            return self._configuration_metadata.get(name)
    
    def reload_all_configurations(self):
        """Manually reload all configurations from disk."""
        with self._lock:
            self._configurations.clear()
            self._configuration_metadata.clear()
            self._load_configurations()
            self.logger.info("Reloaded all configurations")
    
    def shutdown(self):
        """Shutdown the configuration manager and stop file monitoring."""
        if self._file_observer:
            self._file_observer.stop()
            self._file_observer.join()
            self.logger.info("Configuration manager shutdown complete")