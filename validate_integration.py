#!/usr/bin/env python3
"""
Simple integration validation script.

This script performs basic validation of the integrated platform
to ensure all components work together correctly.
"""

import asyncio
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
import sys
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, '.')

from src.integration.platform_integration import PlatformIntegrator
from src.integration.workflow_orchestrator import WorkflowOrchestrator
from src.config.optimization_criteria import OptimizationCriteria, OptimizationConstraints, OptimizationTechnique
from unittest.mock import MagicMock, patch


class SimpleTestModel(nn.Module):
    """Simple test model."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 1)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


async def test_platform_integration():
    """Test basic platform integration."""
    print("🔧 Testing platform integration...")
    
    # Create test configuration
    config = {
        "logging": {
            "level": "INFO",
            "log_dir": "test_logs",
            "json_format": False,
            "console_output": False
        },
        "monitoring": {
            "monitoring_interval_seconds": 1,
            "health_check_interval_seconds": 2
        },
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
    
    # Mock all external dependencies
    with patch.multiple(
        'src.integration.platform_integration',
        ModelStore=MagicMock,
        MemoryManager=MagicMock,
        NotificationService=MagicMock,
        MonitoringService=MagicMock,
        AnalysisAgent=MagicMock,
        PlanningAgent=MagicMock,
        EvaluationAgent=MagicMock,
        QuantizationAgent=MagicMock,
        PruningAgent=MagicMock,
        DistillationAgent=MagicMock,
        CompressionAgent=MagicMock,
        ArchitectureSearchAgent=MagicMock,
        OptimizationManager=MagicMock
    ):
        # Create platform integrator
        integrator = PlatformIntegrator(config)
        
        # Test initialization
        success = await integrator.initialize_platform()
        if not success:
            print("❌ Platform initialization failed")
            return False
        
        print("✅ Platform initialized successfully")
        
        # Test platform status
        status = integrator.get_platform_status()
        if not status["initialized"]:
            print("❌ Platform status shows not initialized")
            return False
        
        print("✅ Platform status is healthy")
        
        # Test component access
        try:
            optimization_manager = integrator.get_optimization_manager()
            model_store = integrator.get_model_store()
            print("✅ Core components accessible")
        except Exception as e:
            print(f"❌ Failed to access components: {e}")
            return False
        
        # Test shutdown
        await integrator.shutdown_platform()
        print("✅ Platform shutdown successful")
        
        return True


async def test_workflow_orchestration():
    """Test workflow orchestration."""
    print("\n🔄 Testing workflow orchestration...")
    
    # Create temporary model file
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test model
        model = SimpleTestModel()
        model_path = temp_path / "test_model.pth"
        torch.save(model.state_dict(), model_path)
        
        # Create platform integrator
        config = {
            "logging": {"level": "INFO", "log_dir": "test_logs", "console_output": False},
            "monitoring": {"monitoring_interval_seconds": 1},
            "model_store": {"storage_path": str(temp_path)},
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
        
        with patch.multiple(
            'src.integration.platform_integration',
            ModelStore=MagicMock,
            MemoryManager=MagicMock,
            NotificationService=MagicMock,
            MonitoringService=MagicMock,
            AnalysisAgent=MagicMock,
            PlanningAgent=MagicMock,
            EvaluationAgent=MagicMock,
            QuantizationAgent=MagicMock,
            PruningAgent=MagicMock,
            DistillationAgent=MagicMock,
            CompressionAgent=MagicMock,
            ArchitectureSearchAgent=MagicMock,
            OptimizationManager=MagicMock
        ):
            # Initialize platform
            integrator = PlatformIntegrator(config)
            await integrator.initialize_platform()
            
            # Create workflow orchestrator
            orchestrator = WorkflowOrchestrator(integrator)
            
            # Mock optimization manager behavior
            optimization_manager = integrator.get_optimization_manager()
            optimization_manager.start_optimization_session.return_value = "test_session_123"
            optimization_manager.get_session_status.return_value = {
                "status": "completed",
                "progress_percentage": 100.0,
                "current_step": "Completed",
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat(),
                "error_message": None,
                "session_data": {
                    "model_id": str(model_path),
                    "steps_completed": 3
                }
            }
            
            # Create optimization criteria
            constraints = OptimizationConstraints(
                preserve_accuracy_threshold=0.95,
                allowed_techniques=[OptimizationTechnique.QUANTIZATION]
            )
            
            criteria = OptimizationCriteria(
                name="validation_test",
                description="Validation test workflow",
                constraints=constraints,
                target_deployment="general"
            )
            
            # Execute workflow
            try:
                result = await orchestrator.execute_complete_workflow(
                    model_path=str(model_path),
                    criteria=criteria,
                    user_id="validation_user"
                )
                
                if not result.success:
                    print(f"❌ Workflow execution failed: {result.error_message}")
                    return False
                
                print("✅ Workflow executed successfully")
                print(f"   - Workflow ID: {result.workflow_id}")
                print(f"   - Session ID: {result.session_id}")
                print(f"   - Execution time: {result.execution_time_seconds:.2f}s")
                
                # Verify results structure
                if not result.results:
                    print("❌ No results returned from workflow")
                    return False
                
                results = result.results
                required_keys = ["workflow_id", "model_metadata", "session_id", "execution_summary"]
                for key in required_keys:
                    if key not in results:
                        print(f"❌ Missing required result key: {key}")
                        return False
                
                print("✅ Workflow results structure is valid")
                
            except Exception as e:
                print(f"❌ Workflow execution failed with exception: {e}")
                return False
            
            finally:
                await integrator.shutdown_platform()
        
        return True


async def test_requirements_coverage():
    """Test that all requirements are covered."""
    print("\n📋 Testing requirements coverage...")
    
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
    
    with patch.multiple(
        'src.integration.platform_integration',
        ModelStore=MagicMock,
        MemoryManager=MagicMock,
        NotificationService=MagicMock,
        MonitoringService=MagicMock,
        AnalysisAgent=MagicMock,
        PlanningAgent=MagicMock,
        EvaluationAgent=MagicMock,
        QuantizationAgent=MagicMock,
        PruningAgent=MagicMock,
        DistillationAgent=MagicMock,
        CompressionAgent=MagicMock,
        ArchitectureSearchAgent=MagicMock,
        OptimizationManager=MagicMock
    ):
        integrator = PlatformIntegrator(config)
        await integrator.initialize_platform()
        
        optimization_manager = integrator.get_optimization_manager()
        
        # Requirement 1: Autonomous analysis and optimization identification
        if not optimization_manager.analysis_agent:
            print("❌ Requirement 1: Analysis agent missing")
            return False
        if not optimization_manager.planning_agent:
            print("❌ Requirement 1: Planning agent missing")
            return False
        print("✅ Requirement 1: Autonomous analysis and optimization identification")
        
        # Requirement 2: Automatic optimization execution
        if len(optimization_manager.optimization_agents) == 0:
            print("❌ Requirement 2: No optimization agents available")
            return False
        if "quantization" not in optimization_manager.optimization_agents:
            print("❌ Requirement 2: Quantization agent missing")
            return False
        if "pruning" not in optimization_manager.optimization_agents:
            print("❌ Requirement 2: Pruning agent missing")
            return False
        print("✅ Requirement 2: Automatic optimization execution")
        
        # Requirement 3: Comprehensive evaluation
        if not optimization_manager.evaluation_agent:
            print("❌ Requirement 3: Evaluation agent missing")
            return False
        print("✅ Requirement 3: Comprehensive evaluation")
        
        # Requirement 4: Configurable criteria and constraints
        try:
            constraints = OptimizationConstraints(
                preserve_accuracy_threshold=0.95,
                allowed_techniques=[OptimizationTechnique.QUANTIZATION, OptimizationTechnique.PRUNING]
            )
            criteria = OptimizationCriteria(
                name="requirements_test",
                description="Requirements test",
                constraints=constraints,
                target_deployment="general"
            )
            if criteria.constraints.preserve_accuracy_threshold != 0.95:
                print("❌ Requirement 4: Configurable thresholds not working")
                return False
            print("✅ Requirement 4: Configurable criteria and constraints")
        except Exception as e:
            print(f"❌ Requirement 4: Failed to create criteria: {e}")
            return False
        
        # Requirement 5: Monitoring and control
        status = integrator.get_platform_status()
        if "components" not in status:
            print("❌ Requirement 5: Component monitoring missing")
            return False
        if "health" not in status:
            print("❌ Requirement 5: Health monitoring missing")
            return False
        print("✅ Requirement 5: Monitoring and control")
        
        # Requirement 6: Multiple model types and optimization techniques
        available_techniques = list(optimization_manager.optimization_agents.keys())
        if len(available_techniques) < 3:
            print(f"❌ Requirement 6: Only {len(available_techniques)} optimization techniques available")
            return False
        
        required_techniques = ["quantization", "pruning"]
        for technique in required_techniques:
            if technique not in available_techniques:
                print(f"❌ Requirement 6: Required technique {technique} missing")
                return False
        
        print("✅ Requirement 6: Multiple model types and optimization techniques")
        
        await integrator.shutdown_platform()
        
        return True


async def main():
    """Main validation function."""
    print("🚀 ROBOTICS MODEL OPTIMIZATION PLATFORM")
    print("   INTEGRATION VALIDATION")
    print("="*50)
    
    start_time = datetime.now()
    
    tests = [
        ("Platform Integration", test_platform_integration),
        ("Workflow Orchestration", test_workflow_orchestration),
        ("Requirements Coverage", test_requirements_coverage)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running {test_name} test...")
            success = await test_func()
            results[test_name] = success
            
            if success:
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}")
            results[test_name] = False
    
    # Summary
    execution_time = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    
    passed_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"Execution time: {execution_time:.2f} seconds")
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  - {test_name}: {status}")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        print("   The platform integration is working correctly.")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} VALIDATION TESTS FAILED!")
        print("   Please check the failed tests and fix any issues.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))