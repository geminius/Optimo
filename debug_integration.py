#!/usr/bin/env python3
"""
Debug integration script to identify initialization issues.
"""

import asyncio
import sys
import logging
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.insert(0, '.')

from src.integration.platform_integration import PlatformIntegrator

# Set up logging to see what's happening
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')


async def debug_initialization():
    """Debug the initialization process."""
    print("🔍 Debugging platform initialization...")
    
    config = {
        "logging": {"level": "DEBUG", "log_dir": "test_logs", "console_output": True},
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
    
    # Create detailed mocks
    def create_detailed_mock(name):
        mock = MagicMock()
        mock.initialize.return_value = True
        mock.cleanup.return_value = None
        mock.__class__.__name__ = name
        print(f"Created mock for {name}")
        return mock
    
    # Mock all components
    mocks = {
        'ModelStore': create_detailed_mock('ModelStore'),
        'MemoryManager': create_detailed_mock('MemoryManager'),
        'NotificationService': create_detailed_mock('NotificationService'),
        'MonitoringService': create_detailed_mock('MonitoringService'),
        'AnalysisAgent': create_detailed_mock('AnalysisAgent'),
        'PlanningAgent': create_detailed_mock('PlanningAgent'),
        'EvaluationAgent': create_detailed_mock('EvaluationAgent'),
        'QuantizationAgent': create_detailed_mock('QuantizationAgent'),
        'PruningAgent': create_detailed_mock('PruningAgent'),
        'DistillationAgent': create_detailed_mock('DistillationAgent'),
        'CompressionAgent': create_detailed_mock('CompressionAgent'),
        'ArchitectureSearchAgent': create_detailed_mock('ArchitectureSearchAgent'),
    }
    
    # Create optimization manager mock with proper structure
    optimization_manager_mock = MagicMock()
    optimization_manager_mock.initialize.return_value = True
    optimization_manager_mock.cleanup.return_value = None
    optimization_manager_mock.get_active_sessions.return_value = []
    optimization_manager_mock.__class__.__name__ = 'OptimizationManager'
    
    # Ensure the mock actually returns what we expect
    print(f"Mock get_active_sessions returns: {optimization_manager_mock.get_active_sessions()}")
    print(f"Type: {type(optimization_manager_mock.get_active_sessions())}")
    
    # Set up the optimization manager attributes that will be checked
    optimization_manager_mock.model_store = mocks['ModelStore']
    optimization_manager_mock.analysis_agent = mocks['AnalysisAgent']
    optimization_manager_mock.planning_agent = mocks['PlanningAgent']
    optimization_manager_mock.evaluation_agent = mocks['EvaluationAgent']
    optimization_manager_mock.optimization_agents = {
        'quantization': mocks['QuantizationAgent'],
        'pruning': mocks['PruningAgent'],
        'distillation': mocks['DistillationAgent'],
        'compression': mocks['CompressionAgent'],
        'architecture_search': mocks['ArchitectureSearchAgent']
    }
    
    mocks['OptimizationManager'] = optimization_manager_mock
    
    # Set up model store methods
    mocks['ModelStore'].get_model_metadata.return_value = None
    
    print("All mocks created, starting initialization...")
    
    with patch.multiple('src.integration.platform_integration', **mocks):
        integrator = PlatformIntegrator(config)
        
        try:
            print("Calling initialize_platform...")
            success = await integrator.initialize_platform()
            print(f"Initialization result: {success}")
            
            if success:
                print("✅ Initialization successful!")
                
                # Check components
                print(f"Optimization manager: {integrator.optimization_manager}")
                print(f"Model store: {integrator.model_store}")
                print(f"Optimization agents: {len(integrator.optimization_agents)}")
                
                # Test platform status
                status = integrator.get_platform_status()
                print(f"Platform status: {status}")
                
                await integrator.shutdown_platform()
                return True
            else:
                print("❌ Initialization failed")
                return False
                
        except Exception as e:
            print(f"❌ Exception during initialization: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    asyncio.run(debug_initialization())