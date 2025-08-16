"""
Final Integration Validation Tests - Comprehensive validation of the complete platform.

This module provides the final validation tests that ensure all components
work together correctly and all requirements are satisfied.
"""

import pytest
import asyncio
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json
import time
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.main import RoboticsOptimizationPlatform
from src.config.optimization_criteria import OptimizationCriteria, OptimizationConstraints, OptimizationTechnique


class TestModel(nn.Module):
    """Test model for validation."""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create necessary directories
        (workspace / "models").mkdir()
        (workspace / "logs").mkdir()
        (workspace / "uploads").mkdir()
        
        yield workspace


@pytest.fixture
def test_model_file(temp_workspace):
    """Create a test model file."""
    model = TestModel()
    model_path = temp_workspace / "models" / "test_model.pth"
    torch.save(model.state_dict(), model_path)
    return model_path


@pytest.fixture
def platform_config(temp_workspace):
    """Create platform configuration for testing."""
    return {
        "logging": {
            "level": "INFO",
            "log_dir": str(temp_workspace / "logs"),
            "json_format": False,
            "console_output": False
        },
        "monitoring": {
            "monitoring_interval_seconds": 1,
            "health_check_interval_seconds": 2,
            "metrics_retention_hours": 1
        },
        "model_store": {
            "storage_path": str(temp_workspace / "models"),
            "max_models": 100
        },
        "memory_manager": {
            "max_sessions": 10,
            "cleanup_interval_minutes": 1
        },
        "notification_service": {
            "enable_email": False,
            "enable_webhook": False
        },
        "monitoring_service": {
            "enable_metrics": True,
            "metrics_interval_seconds": 5
        },
        "optimization_manager": {
            "max_concurrent_sessions": 3,
            "auto_rollback_on_failure": True
        },
        "analysis_agent": {},
        "planning_agent": {},
        "evaluation_agent": {},
        "quantization_agent": {},
        "pruning_agent": {},
        "distillation_agent": {},
        "compression_agent": {},
        "architecture_search_agent": {}
    }


class TestFinalIntegrationValidation:
    """Final integration validation tests."""
    
    @pytest.mark.asyncio
    async def test_complete_platform_lifecycle(self, platform_config):
        """Test complete platform lifecycle from startup to shutdown."""
        # Create platform
        platform = RoboticsOptimizationPlatform()
        platform.config = platform_config
        
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
            # Test startup
            assert not platform.is_running
            success = await platform.start()
            assert success
            assert platform.is_running
            assert platform.platform_integrator is not None
            assert platform.workflow_orchestrator is not None
            
            # Test platform status
            status = platform.get_platform_status()
            assert status["running"]
            assert status["initialized"]
            
            # Test shutdown
            await platform.stop()
            assert not platform.is_running
    
    @pytest.mark.asyncio
    async def test_end_to_end_optimization_workflow(self, platform_config, test_model_file):
        """Test complete end-to-end optimization workflow."""
        platform = RoboticsOptimizationPlatform()
        platform.config = platform_config
        
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
            # Start platform
            await platform.start()
            
            # Mock optimization manager behavior
            optimization_manager = platform.platform_integrator.get_optimization_manager()
            optimization_manager.start_optimization_session.return_value = "test_session_123"
            optimization_manager.get_session_status.return_value = {
                "status": "completed",
                "progress_percentage": 100.0,
                "current_step": "Completed",
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat(),
                "error_message": None,
                "session_data": {
                    "model_id": str(test_model_file),
                    "steps_completed": 3
                }
            }
            
            # Create optimization criteria
            constraints = OptimizationConstraints(
                preserve_accuracy_threshold=0.95,
                allowed_techniques=[OptimizationTechnique.QUANTIZATION, OptimizationTechnique.PRUNING]
            )
            
            criteria = OptimizationCriteria(
                name="final_validation_test",
                description="Final validation optimization test",
                constraints=constraints,
                target_deployment="general"
            )
            
            # Execute optimization workflow
            result = await platform.optimize_model(
                model_path=str(test_model_file),
                criteria=criteria,
                user_id="validation_test_user"
            )
            
            # Validate results
            assert result["success"]
            assert result["workflow_id"] is not None
            assert result["session_id"] == "test_session_123"
            assert result["execution_time_seconds"] > 0
            assert result["error_message"] is None
            assert result["results"] is not None
            
            # Validate result structure
            results = result["results"]
            assert "workflow_id" in results
            assert "model_metadata" in results
            assert "session_id" in results
            assert "execution_summary" in results
            
            await platform.stop()
    
    @pytest.mark.asyncio
    async def test_all_optimization_techniques_available(self, platform_config):
        """Test that all required optimization techniques are available."""
        platform = RoboticsOptimizationPlatform()
        platform.config = platform_config
        
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
            await platform.start()
            
            # Get optimization manager
            optimization_manager = platform.platform_integrator.get_optimization_manager()
            
            # Verify all required optimization agents are available
            expected_agents = ["quantization", "pruning", "distillation", "compression", "architecture_search"]
            available_agents = list(optimization_manager.optimization_agents.keys())
            
            for agent_name in expected_agents:
                assert agent_name in available_agents, f"Optimization agent {agent_name} not available"
            
            await platform.stop()
    
    @pytest.mark.asyncio
    async def test_comprehensive_logging_and_monitoring(self, platform_config):
        """Test comprehensive logging and monitoring functionality."""
        platform = RoboticsOptimizationPlatform()
        platform.config = platform_config
        
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
            await platform.start()
            
            # Verify logging integration
            assert platform.platform_integrator.logging_integrator is not None
            
            # Verify monitoring integration
            assert platform.platform_integrator.monitoring_integrator is not None
            
            # Test platform status includes monitoring data
            status = platform.get_platform_status()
            assert "components" in status
            assert "health" in status
            
            # Verify log directory was created
            log_dir = Path(platform_config["logging"]["log_dir"])
            assert log_dir.exists()
            
            await platform.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, platform_config, test_model_file):
        """Test error handling and recovery mechanisms."""
        platform = RoboticsOptimizationPlatform()
        platform.config = platform_config
        
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
            await platform.start()
            
            # Mock optimization manager to simulate failure
            optimization_manager = platform.platform_integrator.get_optimization_manager()
            optimization_manager.start_optimization_session.side_effect = Exception("Simulated optimization failure")
            
            # Create criteria
            constraints = OptimizationConstraints(
                preserve_accuracy_threshold=0.95,
                allowed_techniques=[OptimizationTechnique.QUANTIZATION]
            )
            
            criteria = OptimizationCriteria(
                name="error_handling_test",
                description="Error handling test",
                constraints=constraints,
                target_deployment="general"
            )
            
            # Execute workflow (should handle error gracefully)
            result = await platform.optimize_model(
                model_path=str(test_model_file),
                criteria=criteria,
                user_id="error_test_user"
            )
            
            # Verify error was handled gracefully
            assert not result["success"]
            assert result["error_message"] is not None
            assert "Simulated optimization failure" in result["error_message"]
            
            # Platform should still be running
            assert platform.is_running
            
            await platform.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_optimization_sessions(self, platform_config, test_model_file):
        """Test multiple concurrent optimization sessions."""
        platform = RoboticsOptimizationPlatform()
        platform.config = platform_config
        
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
            await platform.start()
            
            # Mock optimization manager for concurrent sessions
            optimization_manager = platform.platform_integrator.get_optimization_manager()
            
            session_counter = 0
            def mock_start_session(*args, **kwargs):
                nonlocal session_counter
                session_counter += 1
                return f"concurrent_session_{session_counter}"
            
            optimization_manager.start_optimization_session.side_effect = mock_start_session
            optimization_manager.get_session_status.return_value = {
                "status": "completed",
                "progress_percentage": 100.0,
                "current_step": "Completed",
                "start_time": datetime.now().isoformat(),
                "last_update": datetime.now().isoformat(),
                "error_message": None,
                "session_data": {
                    "model_id": str(test_model_file),
                    "steps_completed": 3
                }
            }
            
            # Create criteria
            constraints = OptimizationConstraints(
                preserve_accuracy_threshold=0.95,
                allowed_techniques=[OptimizationTechnique.QUANTIZATION]
            )
            
            criteria = OptimizationCriteria(
                name="concurrent_test",
                description="Concurrent optimization test",
                constraints=constraints,
                target_deployment="general"
            )
            
            # Start multiple concurrent optimizations
            tasks = []
            for i in range(3):
                task = asyncio.create_task(
                    platform.optimize_model(
                        model_path=str(test_model_file),
                        criteria=criteria,
                        user_id=f"concurrent_user_{i}"
                    )
                )
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks)
            
            # Verify all succeeded
            assert len(results) == 3
            for result in results:
                assert result["success"]
                assert result["session_id"] is not None
            
            # Verify unique session IDs
            session_ids = [result["session_id"] for result in results]
            assert len(set(session_ids)) == 3
            
            await platform.stop()
    
    @pytest.mark.asyncio
    async def test_requirements_coverage_validation(self, platform_config, test_model_file):
        """Validate that all specified requirements are covered."""
        platform = RoboticsOptimizationPlatform()
        platform.config = platform_config
        
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
            await platform.start()
            
            optimization_manager = platform.platform_integrator.get_optimization_manager()
            
            # Requirement 1: Autonomous analysis and optimization identification
            assert optimization_manager.analysis_agent is not None, "Analysis agent required for Requirement 1"
            assert optimization_manager.planning_agent is not None, "Planning agent required for Requirement 1"
            
            # Requirement 2: Automatic optimization execution
            assert len(optimization_manager.optimization_agents) > 0, "Optimization agents required for Requirement 2"
            assert "quantization" in optimization_manager.optimization_agents, "Quantization agent required"
            assert "pruning" in optimization_manager.optimization_agents, "Pruning agent required"
            
            # Requirement 3: Comprehensive evaluation
            assert optimization_manager.evaluation_agent is not None, "Evaluation agent required for Requirement 3"
            
            # Requirement 4: Configurable criteria and constraints
            # Test that we can create and use optimization criteria
            constraints = OptimizationConstraints(
                preserve_accuracy_threshold=0.95,
                allowed_techniques=[OptimizationTechnique.QUANTIZATION, OptimizationTechnique.PRUNING]
            )
            
            criteria = OptimizationCriteria(
                name="requirements_validation",
                description="Requirements validation test",
                constraints=constraints,
                target_deployment="general"
            )
            
            assert criteria.constraints.preserve_accuracy_threshold == 0.95, "Configurable thresholds required"
            assert len(criteria.constraints.allowed_techniques) == 2, "Configurable techniques required"
            
            # Requirement 5: Monitoring and control
            status = platform.get_platform_status()
            assert "components" in status, "Component monitoring required for Requirement 5"
            assert "health" in status, "Health monitoring required for Requirement 5"
            assert platform.platform_integrator.monitoring_integrator is not None, "Monitoring integration required"
            
            # Requirement 6: Multiple model types and optimization techniques
            expected_techniques = ["quantization", "pruning", "distillation", "compression", "architecture_search"]
            available_techniques = list(optimization_manager.optimization_agents.keys())
            
            # Should support multiple techniques
            assert len(available_techniques) >= 3, "Multiple optimization techniques required for Requirement 6"
            
            # Should support common techniques
            common_techniques = ["quantization", "pruning"]
            for technique in common_techniques:
                assert technique in available_techniques, f"Common technique {technique} required"
            
            await platform.stop()
    
    @pytest.mark.asyncio
    async def test_platform_configuration_validation(self, platform_config):
        """Test platform configuration validation and customization."""
        # Test with custom configuration
        custom_config = platform_config.copy()
        custom_config["optimization_manager"]["max_concurrent_sessions"] = 10
        custom_config["logging"]["level"] = "DEBUG"
        
        platform = RoboticsOptimizationPlatform()
        platform.config = custom_config
        
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
            await platform.start()
            
            # Verify configuration was applied
            assert platform.config["optimization_manager"]["max_concurrent_sessions"] == 10
            assert platform.config["logging"]["level"] == "DEBUG"
            
            await platform.stop()
    
    @pytest.mark.asyncio
    async def test_integration_robustness(self, platform_config):
        """Test integration robustness under various conditions."""
        platform = RoboticsOptimizationPlatform()
        platform.config = platform_config
        
        # Test multiple start/stop cycles
        for i in range(3):
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
                # Start platform
                success = await platform.start()
                assert success, f"Platform start failed on cycle {i+1}"
                assert platform.is_running
                
                # Verify platform is functional
                status = platform.get_platform_status()
                assert status["running"]
                assert status["initialized"]
                
                # Stop platform
                await platform.stop()
                assert not platform.is_running
    
    def test_import_validation(self):
        """Test that all required modules can be imported successfully."""
        # Test core integration imports
        try:
            from src.integration.platform_integration import PlatformIntegrator
            from src.integration.workflow_orchestrator import WorkflowOrchestrator
            from src.integration.logging_integration import LoggingIntegrator
            from src.integration.monitoring_integration import MonitoringIntegrator
            from src.main import RoboticsOptimizationPlatform
            
            # If we get here, all imports worked
            assert True
        except ImportError as e:
            pytest.fail(f"Failed to import required modules: {e}")
    
    def test_configuration_structure_validation(self):
        """Test that configuration structure is valid."""
        platform = RoboticsOptimizationPlatform()
        config = platform.config
        
        # Verify required configuration sections exist
        required_sections = [
            "logging", "monitoring", "model_store", "memory_manager",
            "notification_service", "monitoring_service", "optimization_manager",
            "analysis_agent", "planning_agent", "evaluation_agent"
        ]
        
        for section in required_sections:
            assert section in config, f"Required configuration section {section} missing"
        
        # Verify optimization agent configurations
        optimization_agents = [
            "quantization_agent", "pruning_agent", "distillation_agent",
            "compression_agent", "architecture_search_agent"
        ]
        
        for agent in optimization_agents:
            assert agent in config, f"Optimization agent configuration {agent} missing"