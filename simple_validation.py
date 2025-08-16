#!/usr/bin/env python3
"""
Simple validation script to test basic integration functionality.
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, '.')

from src.integration.platform_integration import PlatformIntegrator
from src.integration.workflow_orchestrator import WorkflowOrchestrator
from src.integration.logging_integration import LoggingIntegrator
from src.integration.monitoring_integration import MonitoringIntegrator


async def test_basic_integration():
    """Test basic integration without full validation."""
    print("🔧 Testing basic integration...")
    
    # Create simple config
    config = {
        "logging": {"level": "INFO", "log_dir": "test_logs", "console_output": False},
        "monitoring": {"monitoring_interval_seconds": 1},
        "model_store": {"storage_path": "test_models"},
        "memory_manager": {"max_sessions": 10},
        "notification_service": {"enable_email": False},
        "monitoring_service": {"enable_metrics": True},
        "optimization_manager": {"max_concurrent_sessions": 3},
        "analysis_agent": {},
        "planning_agent": {},
        "evaluation_agent": {},
        "quantization_agent": {},
        "pruning_agent": {},
        "distillation_agent": {},
        "compression_agent": {},
        "architecture_search_agent": {}
    }
    
    # Mock all components to return successful initialization
    mock_components = {}
    
    def create_mock_component(name):
        mock = MagicMock()
        mock.initialize.return_value = True
        mock.cleanup.return_value = None
        mock.__class__.__name__ = name
        return mock
    
    # Create mocks for all components
    mock_components['ModelStore'] = create_mock_component('ModelStore')
    mock_components['MemoryManager'] = create_mock_component('MemoryManager')
    mock_components['NotificationService'] = create_mock_component('NotificationService')
    mock_components['MonitoringService'] = create_mock_component('MonitoringService')
    mock_components['AnalysisAgent'] = create_mock_component('AnalysisAgent')
    mock_components['PlanningAgent'] = create_mock_component('PlanningAgent')
    mock_components['EvaluationAgent'] = create_mock_component('EvaluationAgent')
    mock_components['QuantizationAgent'] = create_mock_component('QuantizationAgent')
    mock_components['PruningAgent'] = create_mock_component('PruningAgent')
    mock_components['DistillationAgent'] = create_mock_component('DistillationAgent')
    mock_components['CompressionAgent'] = create_mock_component('CompressionAgent')
    mock_components['ArchitectureSearchAgent'] = create_mock_component('ArchitectureSearchAgent')
    
    # Create optimization manager mock
    optimization_manager_mock = MagicMock()
    optimization_manager_mock.initialize.return_value = True
    optimization_manager_mock.cleanup.return_value = None
    optimization_manager_mock.get_active_sessions.return_value = []
    optimization_manager_mock.analysis_agent = mock_components['AnalysisAgent']
    optimization_manager_mock.planning_agent = mock_components['PlanningAgent']
    optimization_manager_mock.evaluation_agent = mock_components['EvaluationAgent']
    optimization_manager_mock.optimization_agents = {
        'quantization': mock_components['QuantizationAgent'],
        'pruning': mock_components['PruningAgent'],
        'distillation': mock_components['DistillationAgent'],
        'compression': mock_components['CompressionAgent'],
        'architecture_search': mock_components['ArchitectureSearchAgent']
    }
    optimization_manager_mock.model_store = mock_components['ModelStore']
    optimization_manager_mock.__class__.__name__ = 'OptimizationManager'
    
    mock_components['OptimizationManager'] = optimization_manager_mock
    
    # Mock model store methods
    mock_components['ModelStore'].get_model_metadata.return_value = None
    
    with patch.multiple(
        'src.integration.platform_integration',
        **mock_components
    ):
        # Create platform integrator
        integrator = PlatformIntegrator(config)
        
        try:
            # Test initialization
            success = await integrator.initialize_platform()
            if not success:
                print("❌ Platform initialization failed")
                return False
            
            print("✅ Platform initialized successfully")
            
            # Test that components are available
            if not integrator.optimization_manager:
                print("❌ Optimization manager not available")
                return False
            
            if not integrator.model_store:
                print("❌ Model store not available")
                return False
            
            if len(integrator.optimization_agents) == 0:
                print("❌ No optimization agents available")
                return False
            
            print("✅ All core components available")
            
            # Test platform status
            status = integrator.get_platform_status()
            if not status["initialized"]:
                print("❌ Platform status shows not initialized")
                return False
            
            print("✅ Platform status is healthy")
            
            # Test shutdown
            await integrator.shutdown_platform()
            print("✅ Platform shutdown successful")
            
            return True
            
        except Exception as e:
            print(f"❌ Integration test failed: {e}")
            return False


async def test_logging_integration():
    """Test logging integration."""
    print("\n📝 Testing logging integration...")
    
    config = {
        "level": "INFO",
        "log_dir": "test_logs",
        "json_format": False,
        "console_output": False
    }
    
    try:
        logging_integrator = LoggingIntegrator(config)
        await logging_integrator.initialize()
        
        # Test component logging setup
        mock_components = [MagicMock() for _ in range(3)]
        for i, component in enumerate(mock_components):
            component.__class__.__name__ = f"TestComponent{i}"
        
        await logging_integrator.setup_component_logging(mock_components)
        
        # Test statistics
        stats = logging_integrator.get_log_statistics()
        if stats["total_handlers"] == 0:
            print("❌ No log handlers created")
            return False
        
        print("✅ Logging integration successful")
        
        await logging_integrator.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Logging integration failed: {e}")
        return False


async def test_monitoring_integration():
    """Test monitoring integration."""
    print("\n📊 Testing monitoring integration...")
    
    config = {
        "monitoring_interval_seconds": 1,
        "health_check_interval_seconds": 2,
        "metrics_retention_hours": 1
    }
    
    try:
        monitoring_integrator = MonitoringIntegrator(config)
        await monitoring_integrator.initialize()
        
        # Test system metrics
        metrics = monitoring_integrator.get_system_metrics()
        if metrics.cpu_percent < 0:
            print("❌ Invalid CPU metrics")
            return False
        
        # Test component monitoring
        mock_component = MagicMock()
        monitoring_integrator.add_monitored_component("test_component", mock_component)
        
        # Wait a bit for health check
        await asyncio.sleep(0.1)
        
        # Test health status
        health = monitoring_integrator.get_component_health()
        if "test_component" not in health:
            print("❌ Component not found in health status")
            return False
        
        print("✅ Monitoring integration successful")
        
        await monitoring_integrator.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Monitoring integration failed: {e}")
        return False


async def main():
    """Main validation function."""
    print("🚀 SIMPLE INTEGRATION VALIDATION")
    print("="*40)
    
    tests = [
        ("Basic Integration", test_basic_integration),
        ("Logging Integration", test_logging_integration),
        ("Monitoring Integration", test_monitoring_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            results[test_name] = success
            
            if success:
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*40)
    print("VALIDATION SUMMARY")
    print("="*40)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  - {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} VALIDATION TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))