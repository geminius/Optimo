#!/usr/bin/env python3
"""
Demo script for the enhanced configuration management system.

This script demonstrates:
1. Creating and managing optimization criteria configurations
2. Dynamic configuration updates without system restart
3. Configuration conflict detection and resolution
4. Validation of configuration parameters
5. Observer pattern for configuration changes
"""

import sys
import tempfile
import json
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config.optimization_criteria import (
    ConfigurationManager,
    OptimizationCriteria,
    OptimizationConstraints,
    PerformanceThreshold,
    PerformanceMetric,
    OptimizationTechnique
)


def configuration_change_observer(config_name: str, criteria: OptimizationCriteria):
    """Observer function for configuration changes."""
    print(f"🔄 Configuration '{config_name}' was updated!")
    print(f"   Description: {criteria.description}")
    print(f"   Target deployment: {criteria.target_deployment}")


def demo_basic_configuration_management():
    """Demonstrate basic configuration management features."""
    print("=" * 60)
    print("🚀 Basic Configuration Management Demo")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create configuration manager
        config_manager = ConfigurationManager(temp_dir, enable_auto_reload=False)
        
        # Add observer for configuration changes
        config_manager.add_observer(configuration_change_observer)
        
        print("\n1. Creating optimization criteria configurations...")
        
        # Create edge deployment configuration
        edge_config = OptimizationCriteria(
            name="edge_deployment",
            description="Optimized for edge devices with limited resources",
            target_deployment="edge",
            performance_thresholds=[
                PerformanceThreshold(
                    metric=PerformanceMetric.MODEL_SIZE,
                    max_value=100.0,  # Max 100MB
                    tolerance=0.1
                ),
                PerformanceThreshold(
                    metric=PerformanceMetric.INFERENCE_TIME,
                    max_value=50.0,  # Max 50ms
                    tolerance=0.15
                ),
                PerformanceThreshold(
                    metric=PerformanceMetric.ACCURACY,
                    min_value=0.85,  # Min 85% accuracy
                    tolerance=0.05
                )
            ],
            constraints=OptimizationConstraints(
                max_optimization_time_minutes=30,
                max_memory_usage_gb=4.0,
                preserve_accuracy_threshold=0.90,
                allowed_techniques=[
                    OptimizationTechnique.QUANTIZATION,
                    OptimizationTechnique.PRUNING
                ],
                hardware_constraints={
                    "gpu_memory_gb": 2.0,
                    "cpu_cores": 2
                }
            ),
            priority_weights={
                PerformanceMetric.MODEL_SIZE: 0.4,
                PerformanceMetric.INFERENCE_TIME: 0.4,
                PerformanceMetric.ACCURACY: 0.2
            }
        )
        
        # Cloud deployment configuration
        cloud_config = OptimizationCriteria(
            name="cloud_deployment",
            description="Optimized for cloud deployment with high throughput",
            target_deployment="cloud",
            performance_thresholds=[
                PerformanceThreshold(
                    metric=PerformanceMetric.THROUGHPUT,
                    min_value=1000.0,  # Min 1000 requests/sec
                    tolerance=0.1
                ),
                PerformanceThreshold(
                    metric=PerformanceMetric.ACCURACY,
                    min_value=0.95,  # Min 95% accuracy
                    tolerance=0.02
                )
            ],
            constraints=OptimizationConstraints(
                max_optimization_time_minutes=120,
                max_memory_usage_gb=32.0,
                preserve_accuracy_threshold=0.98,
                allowed_techniques=list(OptimizationTechnique),
                hardware_constraints={
                    "gpu_memory_gb": 16.0,
                    "cpu_cores": 16
                }
            ),
            priority_weights={
                PerformanceMetric.THROUGHPUT: 0.5,
                PerformanceMetric.ACCURACY: 0.5
            }
        )
        
        # Add configurations
        success1 = config_manager.add_criteria(edge_config)
        success2 = config_manager.add_criteria(cloud_config)
        
        print(f"   ✅ Edge config added: {success1}")
        print(f"   ✅ Cloud config added: {success2}")
        
        print(f"\n2. Available configurations: {config_manager.list_criteria()}")
        
        # Demonstrate configuration retrieval
        retrieved_config = config_manager.get_criteria("edge_deployment")
        if retrieved_config:
            print(f"\n3. Retrieved edge config:")
            print(f"   Name: {retrieved_config.name}")
            print(f"   Description: {retrieved_config.description}")
            print(f"   Max model size: {retrieved_config.performance_thresholds[0].max_value}MB")
        
        config_manager.shutdown()


def demo_configuration_validation():
    """Demonstrate configuration validation features."""
    print("\n" + "=" * 60)
    print("🔍 Configuration Validation Demo")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_manager = ConfigurationManager(temp_dir, enable_auto_reload=False)
        
        print("\n1. Testing invalid configuration...")
        
        # Create configuration with some validation errors (but valid enough to create)
        try:
            invalid_config = OptimizationCriteria(
                name="invalid_config",
                description="Configuration with validation errors",
                target_deployment="invalid_target",  # Invalid deployment target
                performance_thresholds=[
                    PerformanceThreshold(
                        metric=PerformanceMetric.ACCURACY,
                        min_value=0.9,
                        max_value=0.8,  # Min > Max (invalid)
                        tolerance=1.5   # Tolerance > 1.0 (invalid)
                    )
                ],
                constraints=OptimizationConstraints(
                    allowed_techniques=[OptimizationTechnique.QUANTIZATION],
                    forbidden_techniques=[OptimizationTechnique.QUANTIZATION],  # Conflict
                    hardware_constraints={
                        "gpu_memory_gb": -1,  # Invalid negative value
                        "cpu_cores": 0        # Invalid zero value
                    }
                )
                # No priority_weights to avoid the validation error during creation
            )
            
            # Validate configuration
            issues = config_manager.validate_criteria(invalid_config)
            print(f"   Found {len(issues)} validation issues:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
            
            # Try to add invalid configuration
            success = config_manager.add_criteria(invalid_config)
            print(f"\n   ❌ Adding invalid config: {success}")
            
        except ValueError as e:
            print(f"   ❌ Configuration creation failed (as expected): {e}")
            
            # Create a simpler invalid config that can be created but fails validation
            simple_invalid_config = OptimizationCriteria(
                name="simple_invalid",
                description="Simple invalid config",
                target_deployment="invalid_target"
            )
            
            issues = config_manager.validate_criteria(simple_invalid_config)
            print(f"\n   Found {len(issues)} validation issues in simple config:")
            for i, issue in enumerate(issues, 1):
                print(f"   {i}. {issue}")
            
            success = config_manager.add_criteria(simple_invalid_config)
            print(f"\n   ❌ Adding simple invalid config: {success}")
        
        config_manager.shutdown()


def demo_conflict_detection():
    """Demonstrate configuration conflict detection."""
    print("\n" + "=" * 60)
    print("⚠️  Configuration Conflict Detection Demo")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_manager = ConfigurationManager(temp_dir, enable_auto_reload=False)
        
        print("\n1. Creating conflicting configurations...")
        
        # Create configurations with conflicting requirements
        config1 = OptimizationCriteria(
            name="mobile_config1",
            description="Mobile config requiring quantization",
            target_deployment="mobile",
            constraints=OptimizationConstraints(
                allowed_techniques=[OptimizationTechnique.QUANTIZATION],
                forbidden_techniques=[OptimizationTechnique.PRUNING]
            ),
            performance_thresholds=[
                PerformanceThreshold(
                    metric=PerformanceMetric.ACCURACY,
                    min_value=0.9  # Min 90%
                )
            ]
        )
        
        config2 = OptimizationCriteria(
            name="mobile_config2",
            description="Mobile config forbidding quantization",
            target_deployment="mobile",
            constraints=OptimizationConstraints(
                allowed_techniques=[OptimizationTechnique.PRUNING],
                forbidden_techniques=[OptimizationTechnique.QUANTIZATION]
            ),
            performance_thresholds=[
                PerformanceThreshold(
                    metric=PerformanceMetric.ACCURACY,
                    max_value=0.8  # Max 80% (conflicts with config1)
                )
            ]
        )
        
        config_manager.add_criteria(config1)
        config_manager.add_criteria(config2)
        
        # Check for conflicts
        conflicts = config_manager.get_conflicts()
        print(f"\n2. Detected {len(conflicts)} conflicts:")
        
        for i, conflict in enumerate(conflicts, 1):
            print(f"\n   Conflict {i}:")
            print(f"   Type: {conflict.conflict_type}")
            print(f"   Severity: {conflict.severity}")
            print(f"   Description: {conflict.description}")
            if conflict.suggested_resolution:
                print(f"   Suggested resolution: {conflict.suggested_resolution}")
        
        config_manager.shutdown()


def demo_dynamic_updates():
    """Demonstrate dynamic configuration updates."""
    print("\n" + "=" * 60)
    print("🔄 Dynamic Configuration Updates Demo")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_manager = ConfigurationManager(temp_dir, enable_auto_reload=False)
        config_manager.add_observer(configuration_change_observer)
        
        print("\n1. Creating initial configuration...")
        
        initial_config = OptimizationCriteria(
            name="dynamic_test",
            description="Initial configuration",
            target_deployment="general"
        )
        
        config_manager.add_criteria(initial_config)
        
        print("\n2. Updating configuration...")
        
        # Update the configuration
        updated_config = OptimizationCriteria(
            name="dynamic_test",
            description="Updated configuration with new parameters",
            target_deployment="edge",
            performance_thresholds=[
                PerformanceThreshold(
                    metric=PerformanceMetric.MODEL_SIZE,
                    max_value=50.0
                )
            ]
        )
        
        success = config_manager.update_criteria("dynamic_test", updated_config)
        print(f"   ✅ Configuration updated: {success}")
        
        # Verify the update
        retrieved = config_manager.get_criteria("dynamic_test")
        if retrieved:
            print(f"   New description: {retrieved.description}")
            print(f"   New target: {retrieved.target_deployment}")
            print(f"   New thresholds: {len(retrieved.performance_thresholds)}")
        
        print("\n3. Testing configuration metadata...")
        metadata = config_manager.get_configuration_metadata("dynamic_test")
        if metadata:
            print(f"   File path: {metadata['file_path']}")
            print(f"   Loaded at: {metadata['loaded_at']}")
        
        config_manager.shutdown()


def demo_manual_file_updates():
    """Demonstrate manual configuration file updates."""
    print("\n" + "=" * 60)
    print("📁 Manual File Updates Demo")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        config_manager = ConfigurationManager(temp_dir, enable_auto_reload=False)
        
        print("\n1. Creating configuration file manually...")
        
        # Create configuration file manually
        config_data = {
            "name": "manual_config",
            "description": "Manually created configuration",
            "performance_thresholds": [
                {
                    "metric": "accuracy",
                    "min_value": 0.85,
                    "tolerance": 0.05
                }
            ],
            "constraints": {
                "max_optimization_time_minutes": 45,
                "max_memory_usage_gb": 8.0,
                "preserve_accuracy_threshold": 0.90,
                "allowed_techniques": ["quantization", "pruning"],
                "forbidden_techniques": [],
                "hardware_constraints": {
                    "gpu_memory_gb": 4.0
                }
            },
            "priority_weights": {
                "accuracy": 1.0
            },
            "target_deployment": "edge"
        }
        
        config_file = Path(temp_dir) / "manual_config.json"
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"   Created file: {config_file}")
        
        # Reload configurations to pick up the new file
        config_manager.reload_all_configurations()
        
        # Verify the configuration was loaded
        loaded_config = config_manager.get_criteria("manual_config")
        if loaded_config:
            print(f"   ✅ Configuration loaded successfully")
            print(f"   Name: {loaded_config.name}")
            print(f"   Description: {loaded_config.description}")
            print(f"   Thresholds: {len(loaded_config.performance_thresholds)}")
        else:
            print(f"   ❌ Configuration not loaded")
        
        config_manager.shutdown()


def main():
    """Run all configuration management demos."""
    print("🤖 Robotics Model Optimization Platform")
    print("Configuration Management System Demo")
    print("=" * 60)
    
    try:
        demo_basic_configuration_management()
        demo_configuration_validation()
        demo_conflict_detection()
        demo_dynamic_updates()
        demo_manual_file_updates()
        
        print("\n" + "=" * 60)
        print("✅ All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()