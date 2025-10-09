"""
Complete Platform Integration Tests - End-to-end testing of the integrated platform.

This module provides comprehensive integration tests that validate the complete
platform functionality from initialization through optimization workflows.
"""

import pytest
import pytest_asyncio
import asyncio
import tempfile
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import json
import time
from datetime import datetime

from src.integration.platform_integration import PlatformIntegrator
from src.integration.workflow_orchestrator import WorkflowOrchestrator, WorkflowPhase
from src.integration.logging_integration import LoggingIntegrator
from src.integration.monitoring_integration import MonitoringIntegrator, HealthStatus
from src.config.optimization_criteria import OptimizationCriteria, OptimizationConstraints, OptimizationTechnique


class SimpleTestModel(nn.Module):
    """Simple model for integration testing."""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for test models."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_model(temp_model_dir):
    """Create and save a test model."""
    model = SimpleTestModel()
    model_path = temp_model_dir / "test_model.pth"
    torch.save(model.state_dict(), model_path)
    return model_path, model


@pytest.fixture
def platform_config():
    """Create platform configuration for testing."""
    return {
        "logging": {
            "level": "INFO",
            "log_dir": "test_logs",
            "json_format": False,
            "console_output": True
        },
        "monitoring": {
            "monitoring_interval_seconds": 5,
            "health_check_interval_seconds": 10,
            "metrics_retention_hours": 1
        },
        "model_store": {
            "storage_path": "test_models",
            "max_models": 100
        },
        "memory_manager": {
            "max_sessions": 10,
            "cleanup_interval_minutes": 5
        },
        "notification_service": {
            "enable_email": False,
            "enable_webhook": False
        },
        "monitoring_service": {
            "enable_metrics": True,
            "metrics_interval": 30
        },
        "optimization_manager": {
            "max_concurrent_sessions": 2,
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


@pytest_asyncio.fixture
async def platform_integrator(platform_config):
    """Create and initialize platform integrator."""
    integrator = PlatformIntegrator(platform_config)
    
    # Create mock classes with initialize methods
    def create_mock_with_initialize():
        mock = MagicMock()
        mock.return_value.initialize.return_value = True
        mock.return_value.cleanup.return_value = None
        return mock
    
    # Create special mock for OptimizationManager that returns list from get_active_sessions
    def create_optimization_manager_mock():
        mock = MagicMock()
        mock.return_value.initialize.return_value = True
        mock.return_value.cleanup.return_value = None
        mock.return_value.get_active_sessions.return_value = []  # Return empty list
        return mock
    
    # Mock the actual agent initialization to avoid dependencies
    with patch.multiple(
        'src.integration.platform_integration',
        ModelStore=create_mock_with_initialize(),
        MemoryManager=create_mock_with_initialize(),
        NotificationService=create_mock_with_initialize(),
        MonitoringService=create_mock_with_initialize(),
        AnalysisAgent=create_mock_with_initialize(),
        PlanningAgent=create_mock_with_initialize(),
        EvaluationAgent=create_mock_with_initialize(),
        QuantizationAgent=create_mock_with_initialize(),
        PruningAgent=create_mock_with_initialize(),
        DistillationAgent=create_mock_with_initialize(),
        CompressionAgent=create_mock_with_initialize(),
        ArchitectureSearchAgent=create_mock_with_initialize(),
        OptimizationManager=create_optimization_manager_mock()
    ):
        success = await integrator.initialize_platform()
        assert success, "Platform initialization should succeed"
        
        yield integrator
        
        await integrator.shutdown_platform()


class TestCompletePlatformIntegration:
    """Test complete platform integration functionality."""
    
    @pytest.mark.asyncio
    async def test_platform_initialization_and_shutdown(self, platform_config):
        """Test complete platform initialization and shutdown cycle."""
        integrator = PlatformIntegrator(platform_config)
        
        # Create mock OptimizationManager that returns proper types
        mock_opt_manager = MagicMock()
        mock_opt_manager.return_value.get_active_sessions.return_value = []
        mock_opt_manager.return_value.initialize.return_value = True
        
        # Mock dependencies
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
            OptimizationManager=mock_opt_manager
        ):
            # Test initialization
            assert not integrator.is_initialized
            success = await integrator.initialize_platform()
            assert success
            assert integrator.is_initialized
            
            # Test platform status
            status = integrator.get_platform_status()
            assert status["initialized"]
            assert status["health"] in ["healthy", "degraded"]
            assert "components" in status
            
            # Test shutdown
            await integrator.shutdown_platform()
            assert not integrator.is_initialized
    
    @pytest.mark.asyncio
    async def test_component_wiring(self, platform_integrator):
        """Test that all components are properly wired together."""
        # Verify optimization manager is available
        optimization_manager = platform_integrator.get_optimization_manager()
        assert optimization_manager is not None
        
        # Verify model store is available
        model_store = platform_integrator.get_model_store()
        assert model_store is not None
        
        # Verify platform status shows all components
        status = platform_integrator.get_platform_status()
        components = status["components"]
        
        expected_components = [
            "model_store", "memory_manager", "notification_service",
            "monitoring_service", "optimization_manager", "analysis_agent",
            "planning_agent", "evaluation_agent", "optimization_agents"
        ]
        
        for component in expected_components:
            assert component in components, f"Component {component} not found in status"
    
    @pytest.mark.asyncio
    async def test_logging_integration(self, platform_config):
        """Test logging integration functionality."""
        logging_config = platform_config["logging"]
        logging_integrator = LoggingIntegrator(logging_config)
        
        # Initialize logging
        await logging_integrator.initialize()
        
        # Test component logging setup
        mock_components = [MagicMock() for _ in range(3)]
        for i, component in enumerate(mock_components):
            component.__class__.__name__ = f"TestComponent{i}"
        
        await logging_integrator.setup_component_logging(mock_components)
        
        # Verify loggers were created
        stats = logging_integrator.get_log_statistics()
        assert stats["total_loggers"] >= len(mock_components)
        assert stats["total_handlers"] > 0
        
        # Cleanup
        await logging_integrator.cleanup()
    
    @pytest.mark.asyncio
    async def test_monitoring_integration(self, platform_config):
        """Test monitoring integration functionality."""
        monitoring_config = platform_config["monitoring"]
        monitoring_integrator = MonitoringIntegrator(monitoring_config)
        
        # Initialize monitoring
        await monitoring_integrator.initialize()
        
        # Add mock components
        mock_components = {
            "test_component_1": MagicMock(),
            "test_component_2": MagicMock()
        }
        
        for name, component in mock_components.items():
            monitoring_integrator.add_monitored_component(name, component)
        
        # Wait for health checks
        await asyncio.sleep(0.1)
        
        # Check system metrics
        metrics = monitoring_integrator.get_system_metrics()
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        assert metrics.disk_percent >= 0
        
        # Check component health
        health = monitoring_integrator.get_component_health()
        assert len(health) == len(mock_components)
        
        # Check platform health summary
        summary = monitoring_integrator.get_platform_health_summary()
        assert "overall_status" in summary
        assert "total_components" in summary
        
        # Cleanup
        await monitoring_integrator.cleanup()
    
    @pytest.mark.asyncio
    async def test_complete_workflow_orchestration(self, platform_integrator, test_model):
        """Test complete workflow orchestration from upload to evaluation."""
        model_path, model = test_model
        
        # Create workflow orchestrator
        orchestrator = WorkflowOrchestrator(platform_integrator)
        
        # Create optimization criteria
        constraints = OptimizationConstraints(
            preserve_accuracy_threshold=0.95,
            allowed_techniques=[OptimizationTechnique.QUANTIZATION]
        )
        
        criteria = OptimizationCriteria(
            name="test_integration_workflow",
            description="Integration test workflow",
            constraints=constraints,
            target_deployment="general"
        )
        
        # Mock the optimization manager to simulate successful workflow
        optimization_manager = platform_integrator.get_optimization_manager()
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
        
        # Execute complete workflow
        result = await orchestrator.execute_complete_workflow(
            model_path=str(model_path),
            criteria=criteria,
            user_id="test_user"
        )
        
        # Verify workflow result
        assert result.success
        assert result.session_id == "test_session_123"
        assert result.final_phase == WorkflowPhase.COMPLETION
        assert result.execution_time_seconds > 0
        assert result.results is not None
        
        # Verify workflow results structure
        results = result.results
        assert "workflow_id" in results
        assert "model_metadata" in results
        assert "session_id" in results
        assert "execution_summary" in results
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, platform_integrator, test_model):
        """Test workflow error handling and recovery."""
        model_path, model = test_model
        
        orchestrator = WorkflowOrchestrator(platform_integrator)
        
        # Create criteria
        constraints = OptimizationConstraints(
            preserve_accuracy_threshold=0.95,
            allowed_techniques=[OptimizationTechnique.QUANTIZATION]
        )
        
        criteria = OptimizationCriteria(
            name="test_error_workflow",
            description="Error handling test",
            constraints=constraints,
            target_deployment="general"
        )
        
        # Mock optimization manager to simulate failure
        optimization_manager = platform_integrator.get_optimization_manager()
        optimization_manager.start_optimization_session.side_effect = Exception("Simulated failure")
        
        # Execute workflow (should handle error gracefully)
        result = await orchestrator.execute_complete_workflow(
            model_path=str(model_path),
            criteria=criteria,
            user_id="test_user"
        )
        
        # Verify error handling
        assert not result.success
        assert result.error_message is not None
        assert "Simulated failure" in result.error_message
        assert result.final_phase != WorkflowPhase.COMPLETION
    
    @pytest.mark.asyncio
    async def test_concurrent_workflows(self, platform_integrator, test_model):
        """Test multiple concurrent workflows."""
        model_path, model = test_model
        
        orchestrator = WorkflowOrchestrator(platform_integrator)
        
        # Create criteria
        constraints = OptimizationConstraints(
            preserve_accuracy_threshold=0.95,
            allowed_techniques=[OptimizationTechnique.QUANTIZATION]
        )
        
        criteria = OptimizationCriteria(
            name="test_concurrent_workflow",
            description="Concurrent workflow test",
            constraints=constraints,
            target_deployment="general"
        )
        
        # Mock optimization manager for successful workflows
        optimization_manager = platform_integrator.get_optimization_manager()
        
        session_counter = 0
        def mock_start_session(*args, **kwargs):
            nonlocal session_counter
            session_counter += 1
            return f"test_session_{session_counter}"
        
        optimization_manager.start_optimization_session.side_effect = mock_start_session
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
        
        # Start multiple concurrent workflows
        workflow_tasks = []
        for i in range(3):
            task = asyncio.create_task(
                orchestrator.execute_complete_workflow(
                    model_path=str(model_path),
                    criteria=criteria,
                    user_id=f"test_user_{i}"
                )
            )
            workflow_tasks.append(task)
        
        # Wait for all workflows to complete
        results = await asyncio.gather(*workflow_tasks)
        
        # Verify all workflows succeeded
        assert len(results) == 3
        for result in results:
            assert result.success
            assert result.session_id is not None
            assert result.final_phase == WorkflowPhase.COMPLETION
        
        # Verify unique session IDs
        session_ids = [result.session_id for result in results]
        assert len(set(session_ids)) == 3  # All unique
    
    @pytest.mark.asyncio
    async def test_platform_health_monitoring(self, platform_integrator):
        """Test platform health monitoring and alerting."""
        # Get platform status
        status = platform_integrator.get_platform_status()
        assert status["initialized"]
        
        # Verify monitoring is active
        if hasattr(platform_integrator, 'monitoring_integrator') and platform_integrator.monitoring_integrator:
            monitoring_stats = platform_integrator.monitoring_integrator.get_monitoring_statistics()
            assert "monitored_components" in monitoring_stats
            assert "monitoring_active" in monitoring_stats
    
    @pytest.mark.asyncio
    async def test_platform_resource_cleanup(self, platform_config):
        """Test proper resource cleanup during shutdown."""
        integrator = PlatformIntegrator(platform_config)
        
        # Create mock OptimizationManager that returns proper types
        mock_opt_manager = MagicMock()
        mock_opt_manager.return_value.get_active_sessions.return_value = []
        mock_opt_manager.return_value.initialize.return_value = True
        
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
            OptimizationManager=mock_opt_manager
        ):
            # Initialize platform
            await integrator.initialize_platform()
            assert integrator.is_initialized
            
            # Verify components are initialized
            assert integrator.optimization_manager is not None
            assert integrator.model_store is not None
            assert len(integrator.optimization_agents) > 0
            
            # Shutdown platform
            await integrator.shutdown_platform()
            
            # Verify cleanup
            assert not integrator.is_initialized
            
            # Verify cleanup methods were called on mocked components
            if integrator.optimization_manager:
                integrator.optimization_manager.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_integration_validation(self, platform_integrator):
        """Test integration validation functionality."""
        # Platform should be properly initialized and validated
        assert platform_integrator.is_initialized
        
        # Test that we can get required components
        optimization_manager = platform_integrator.get_optimization_manager()
        assert optimization_manager is not None
        
        model_store = platform_integrator.get_model_store()
        assert model_store is not None
        
        # Test platform status
        status = platform_integrator.get_platform_status()
        assert status["initialized"]
        assert status["health"] in ["healthy", "degraded"]
        
        # Verify all expected components are present
        components = status["components"]
        required_components = [
            "model_store", "memory_manager", "notification_service",
            "monitoring_service", "optimization_manager", "analysis_agent",
            "planning_agent", "evaluation_agent"
        ]
        
        for component in required_components:
            assert components.get(component), f"Required component {component} not available"
    
    @pytest.mark.asyncio
    async def test_end_to_end_requirements_coverage(self, platform_integrator, test_model):
        """Test that the integration covers all specified requirements."""
        model_path, model = test_model
        
        # Requirement 1: Autonomous analysis and optimization identification
        optimization_manager = platform_integrator.get_optimization_manager()
        assert optimization_manager.analysis_agent is not None
        assert optimization_manager.planning_agent is not None
        
        # Requirement 2: Automatic optimization execution
        assert len(optimization_manager.optimization_agents) > 0
        assert "quantization" in optimization_manager.optimization_agents
        assert "pruning" in optimization_manager.optimization_agents
        
        # Requirement 3: Comprehensive evaluation
        assert optimization_manager.evaluation_agent is not None
        
        # Requirement 4: Configurable criteria and constraints
        constraints = OptimizationConstraints(
            preserve_accuracy_threshold=0.95,
            allowed_techniques=[OptimizationTechnique.QUANTIZATION]
        )
        
        criteria = OptimizationCriteria(
            name="requirements_test",
            description="Requirements coverage test",
            constraints=constraints,
            target_deployment="general"
        )
        
        # Should be able to create criteria without errors
        assert criteria.name == "requirements_test"
        assert criteria.constraints.preserve_accuracy_threshold == 0.95
        
        # Requirement 5: Monitoring and control
        status = platform_integrator.get_platform_status()
        assert "components" in status
        assert "health" in status
        
        # Requirement 6: Multiple model types and optimization techniques
        expected_agents = ["quantization", "pruning", "distillation", "compression", "architecture_search"]
        available_agents = list(optimization_manager.optimization_agents.keys())
        
        # Should have multiple optimization techniques available
        assert len(available_agents) >= 2
        
        # Test model store supports different formats (mocked)
        model_store = platform_integrator.get_model_store()
        assert model_store is not None