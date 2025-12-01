"""
Platform Integration - Comprehensive integration of all platform components.

This module provides the PlatformIntegrator class that wires together all
agents, services, and interfaces into a cohesive platform.
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
import time

from ..services.optimization_manager import OptimizationManager
from ..services.memory_manager import MemoryManager
from ..services.notification_service import NotificationService
from ..services.monitoring_service import MonitoringService
from ..models.store import ModelStore
from ..agents.analysis.agent import AnalysisAgent
from ..agents.planning.agent import PlanningAgent
from ..agents.evaluation.agent import EvaluationAgent
from ..agents.optimization.quantization import QuantizationAgent
from ..agents.optimization.pruning import PruningAgent
from ..agents.optimization.distillation import DistillationAgent
from ..agents.optimization.compression import CompressionAgent
from ..agents.optimization.architecture_search import ArchitectureSearchAgent
from ..config.optimization_criteria import OptimizationCriteria
from .logging_integration import LoggingIntegrator
from .monitoring_integration import MonitoringIntegrator


logger = logging.getLogger(__name__)


class PlatformIntegrator:
    """
    Comprehensive platform integrator that wires together all components.
    
    Responsibilities:
    - Initialize and configure all platform components
    - Establish proper dependencies and communication channels
    - Provide unified platform lifecycle management
    - Ensure proper error handling and recovery across components
    - Coordinate platform-wide monitoring and logging
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Core services
        self.model_store: Optional[ModelStore] = None
        self.memory_manager: Optional[MemoryManager] = None
        self.notification_service: Optional[NotificationService] = None
        self.monitoring_service: Optional[MonitoringService] = None
        self.optimization_manager: Optional[OptimizationManager] = None
        
        # Agents
        self.analysis_agent: Optional[AnalysisAgent] = None
        self.planning_agent: Optional[PlanningAgent] = None
        self.evaluation_agent: Optional[EvaluationAgent] = None
        self.optimization_agents: Dict[str, Any] = {}
        
        # Integration components
        self.logging_integrator: Optional[LoggingIntegrator] = None
        self.monitoring_integrator: Optional[MonitoringIntegrator] = None
        
        # Platform state
        self.is_initialized = False
        self.initialization_lock = threading.Lock()
        
        self.logger.info("PlatformIntegrator created")
    
    async def initialize_platform(self) -> bool:
        """
        Initialize the complete platform with all components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        with self.initialization_lock:
            if self.is_initialized:
                self.logger.warning("Platform already initialized")
                return True
            
            try:
                self.logger.info("Starting platform initialization")
                
                # Phase 1: Initialize logging and monitoring infrastructure
                await self._initialize_infrastructure()
                
                # Phase 2: Initialize core services
                await self._initialize_core_services()
                
                # Phase 3: Initialize agents
                await self._initialize_agents()
                
                # Phase 4: Wire components together
                await self._wire_components()
                
                # Phase 5: Validate integration
                await self._validate_integration()
                
                self.is_initialized = True
                self.logger.info("Platform initialization completed successfully")
                return True
                
            except Exception as e:
                self.logger.error(f"Platform initialization failed: {e}")
                await self._cleanup_partial_initialization()
                return False
    
    async def shutdown_platform(self) -> None:
        """Shutdown the platform gracefully."""
        if not self.is_initialized:
            self.logger.warning("Platform not initialized, nothing to shutdown")
            return
        
        try:
            self.logger.info("Starting platform shutdown")
            
            # Shutdown in reverse order of initialization
            await self._shutdown_agents()
            await self._shutdown_core_services()
            await self._shutdown_infrastructure()
            
            self.is_initialized = False
            self.logger.info("Platform shutdown completed")
            
        except Exception as e:
            self.logger.error(f"Error during platform shutdown: {e}")
    
    def get_optimization_manager(self) -> OptimizationManager:
        """Get the optimization manager instance."""
        if not self.is_initialized or not self.optimization_manager:
            raise RuntimeError("Platform not initialized or optimization manager not available")
        return self.optimization_manager
    
    def get_model_store(self) -> ModelStore:
        """Get the model store instance."""
        if not self.is_initialized or not self.model_store:
            raise RuntimeError("Platform not initialized or model store not available")
        return self.model_store
    
    def get_memory_manager(self) -> MemoryManager:
        """Get the memory manager instance."""
        if not self.is_initialized or not self.memory_manager:
            raise RuntimeError("Platform not initialized or memory manager not available")
        return self.memory_manager
    
    def get_notification_service(self) -> NotificationService:
        """Get the notification service instance."""
        if not self.is_initialized or not self.notification_service:
            raise RuntimeError("Platform not initialized or notification service not available")
        return self.notification_service
    
    def get_platform_status(self) -> Dict[str, Any]:
        """Get comprehensive platform status."""
        status = {
            "initialized": self.is_initialized,
            "components": {},
            "health": "unknown"
        }
        
        if self.is_initialized:
            # Check component health
            components = {
                "model_store": self.model_store is not None,
                "memory_manager": self.memory_manager is not None,
                "notification_service": self.notification_service is not None,
                "monitoring_service": self.monitoring_service is not None,
                "optimization_manager": self.optimization_manager is not None,
                "analysis_agent": self.analysis_agent is not None,
                "planning_agent": self.planning_agent is not None,
                "evaluation_agent": self.evaluation_agent is not None,
                "optimization_agents": len(self.optimization_agents) > 0
            }
            
            status["components"] = components
            
            # Determine overall health
            all_healthy = all(components.values())
            status["health"] = "healthy" if all_healthy else "degraded"
            
            # Add detailed agent status
            status["optimization_agents"] = list(self.optimization_agents.keys())
            
            # Add monitoring data if available
            if self.monitoring_service:
                try:
                    status["metrics"] = self.monitoring_service.get_current_metrics()
                except Exception as e:
                    self.logger.warning(f"Failed to get monitoring metrics: {e}")
        
        return status
    
    async def _initialize_infrastructure(self) -> None:
        """Initialize logging and monitoring infrastructure."""
        self.logger.info("Initializing infrastructure components")
        
        # Initialize logging integration
        logging_config = self.config.get("logging", {})
        self.logging_integrator = LoggingIntegrator(logging_config)
        await self.logging_integrator.initialize()
        
        # Initialize monitoring integration
        monitoring_config = self.config.get("monitoring", {})
        self.monitoring_integrator = MonitoringIntegrator(monitoring_config)
        await self.monitoring_integrator.initialize()
        
        self.logger.info("Infrastructure initialization completed")
    
    async def _initialize_core_services(self) -> None:
        """Initialize core platform services."""
        self.logger.info("Initializing core services")
        
        # Initialize model store
        model_store_config = self.config.get("model_store", {})
        storage_path = model_store_config.get("storage_path", "models")
        enable_versioning = model_store_config.get("enable_versioning", False)
        self.model_store = ModelStore(storage_path, enable_versioning)
        # ModelStore doesn't have initialize() method - it initializes in __init__
        
        # Initialize memory manager
        memory_config = self.config.get("memory_manager", {})
        self.memory_manager = MemoryManager(memory_config)
        if not self.memory_manager.initialize():
            raise RuntimeError("Failed to initialize MemoryManager")
        
        # Initialize notification service
        # NotificationService doesn't take config and doesn't have initialize() method
        self.notification_service = NotificationService()
        
        # Initialize monitoring service (pass notification_service, not config)
        self.monitoring_service = MonitoringService(self.notification_service)
        
        self.logger.info("Core services initialization completed")
    
    async def _initialize_agents(self) -> None:
        """Initialize all platform agents."""
        self.logger.info("Initializing platform agents")
        
        # Initialize analysis agent
        analysis_config = self.config.get("analysis_agent", {})
        self.analysis_agent = AnalysisAgent(analysis_config)
        if not self.analysis_agent.initialize():
            raise RuntimeError("Failed to initialize AnalysisAgent")
        
        # Initialize planning agent
        planning_config = self.config.get("planning_agent", {})
        self.planning_agent = PlanningAgent(planning_config)
        if not self.planning_agent.initialize():
            raise RuntimeError("Failed to initialize PlanningAgent")
        
        # Initialize evaluation agent
        evaluation_config = self.config.get("evaluation_agent", {})
        self.evaluation_agent = EvaluationAgent(evaluation_config)
        if not self.evaluation_agent.initialize():
            raise RuntimeError("Failed to initialize EvaluationAgent")
        
        # Initialize optimization agents
        await self._initialize_optimization_agents()
        
        self.logger.info("Agents initialization completed")
    
    async def _initialize_optimization_agents(self) -> None:
        """Initialize all optimization agents."""
        self.logger.info("Initializing optimization agents")
        
        # Quantization agent
        quantization_config = self.config.get("quantization_agent", {})
        quantization_agent = QuantizationAgent(quantization_config)
        if quantization_agent.initialize():
            self.optimization_agents["quantization"] = quantization_agent
            self.logger.info("QuantizationAgent initialized")
        else:
            self.logger.warning("Failed to initialize QuantizationAgent")
        
        # Pruning agent
        pruning_config = self.config.get("pruning_agent", {})
        pruning_agent = PruningAgent(pruning_config)
        if pruning_agent.initialize():
            self.optimization_agents["pruning"] = pruning_agent
            self.logger.info("PruningAgent initialized")
        else:
            self.logger.warning("Failed to initialize PruningAgent")
        
        # Distillation agent
        distillation_config = self.config.get("distillation_agent", {})
        distillation_agent = DistillationAgent(distillation_config)
        if distillation_agent.initialize():
            self.optimization_agents["distillation"] = distillation_agent
            self.logger.info("DistillationAgent initialized")
        else:
            self.logger.warning("Failed to initialize DistillationAgent")
        
        # Compression agent
        compression_config = self.config.get("compression_agent", {})
        compression_agent = CompressionAgent(compression_config)
        if compression_agent.initialize():
            self.optimization_agents["compression"] = compression_agent
            self.logger.info("CompressionAgent initialized")
        else:
            self.logger.warning("Failed to initialize CompressionAgent")
        
        # Architecture search agent
        arch_search_config = self.config.get("architecture_search_agent", {})
        arch_search_agent = ArchitectureSearchAgent(arch_search_config)
        if arch_search_agent.initialize():
            self.optimization_agents["architecture_search"] = arch_search_agent
            self.logger.info("ArchitectureSearchAgent initialized")
        else:
            self.logger.warning("Failed to initialize ArchitectureSearchAgent")
        
        self.logger.info(f"Initialized {len(self.optimization_agents)} optimization agents")
    
    async def _wire_components(self) -> None:
        """Wire all components together with proper dependencies."""
        self.logger.info("Wiring platform components")
        
        # Create optimization manager with all dependencies
        optimization_config = self.config.get("optimization_manager", {})
        self.optimization_manager = OptimizationManager(optimization_config)
        
        # Inject dependencies into optimization manager
        self.optimization_manager.model_store = self.model_store
        self.optimization_manager.memory_manager = self.memory_manager
        self.optimization_manager.notification_service = self.notification_service
        self.optimization_manager.monitoring_service = self.monitoring_service
        
        # Inject agents
        self.optimization_manager.analysis_agent = self.analysis_agent
        self.optimization_manager.planning_agent = self.planning_agent
        self.optimization_manager.evaluation_agent = self.evaluation_agent
        self.optimization_manager.optimization_agents = self.optimization_agents
        
        # Initialize optimization manager
        if not self.optimization_manager.initialize():
            raise RuntimeError("Failed to initialize OptimizationManager")
        
        # Connect progress updates from OptimizationManager to NotificationService
        def progress_callback(session_id: str, update):
            """Forward progress updates to notification service."""
            try:
                self.notification_service.update_progress(
                    session_id=session_id,
                    current_step=update.current_step,
                    step_name=update.step_name,
                    metadata=update.metadata
                )
            except Exception as e:
                self.logger.error(f"Error forwarding progress update: {e}")
        
        self.optimization_manager.add_progress_callback(progress_callback)
        
        # Start monitoring service
        monitoring_interval = self.config.get("monitoring", {}).get("interval", 30)
        self.monitoring_service.start_monitoring(interval=monitoring_interval)
        
        # Set up cross-component communication
        await self._setup_component_communication()
        
        self.logger.info("Component wiring completed")
    
    async def _setup_component_communication(self) -> None:
        """Set up communication channels between components."""
        self.logger.info("Setting up component communication")
        
        # Connect monitoring integrator to all components
        if self.monitoring_integrator:
            # Monitor optimization manager
            if self.optimization_manager:
                self.monitoring_integrator.add_monitored_component(
                    "optimization_manager", self.optimization_manager
                )
            
            # Monitor agents
            for agent_name, agent in self.optimization_agents.items():
                self.monitoring_integrator.add_monitored_component(
                    f"agent_{agent_name}", agent
                )
            
            # Monitor core services
            if self.memory_manager:
                self.monitoring_integrator.add_monitored_component(
                    "memory_manager", self.memory_manager
                )
            
            if self.model_store:
                self.monitoring_integrator.add_monitored_component(
                    "model_store", self.model_store
                )
        
        # Set up logging integration
        if self.logging_integrator:
            await self.logging_integrator.setup_component_logging([
                self.optimization_manager,
                self.model_store,
                self.memory_manager,
                self.notification_service,
                self.monitoring_service,
                self.analysis_agent,
                self.planning_agent,
                self.evaluation_agent
            ] + list(self.optimization_agents.values()))
        
        self.logger.info("Component communication setup completed")
    
    async def _validate_integration(self) -> None:
        """Validate that all components are properly integrated."""
        self.logger.info("Validating platform integration")
        
        # Test basic component connectivity
        validation_errors = []
        
        # Validate optimization manager
        if not self.optimization_manager:
            validation_errors.append("OptimizationManager not initialized")
        elif not hasattr(self.optimization_manager, 'model_store') or not self.optimization_manager.model_store:
            validation_errors.append("OptimizationManager missing ModelStore dependency")
        
        # Validate agents
        required_agents = ["analysis_agent", "planning_agent", "evaluation_agent"]
        for agent_name in required_agents:
            agent = getattr(self, agent_name, None)
            if not agent:
                validation_errors.append(f"{agent_name} not initialized")
        
        # Validate optimization agents
        if len(self.optimization_agents) == 0:
            validation_errors.append("No optimization agents initialized")
        
        # Validate services
        required_services = ["model_store", "memory_manager", "notification_service", "monitoring_service"]
        for service_name in required_services:
            service = getattr(self, service_name, None)
            if not service:
                validation_errors.append(f"{service_name} not initialized")
        
        if validation_errors:
            error_msg = "Integration validation failed: " + "; ".join(validation_errors)
            raise RuntimeError(error_msg)
        
        # Test basic functionality
        try:
            # Test model store (should handle gracefully)
            try:
                self.model_store.get_model_metadata("test")
            except Exception:
                # This is expected for non-existent models, ignore
                pass
            
            # Test optimization manager
            try:
                active_sessions = self.optimization_manager.get_active_sessions()
                if not isinstance(active_sessions, list):
                    raise RuntimeError(f"get_active_sessions() should return a list, got {type(active_sessions)}: {active_sessions}")
            except Exception as e:
                raise RuntimeError(f"get_active_sessions() failed: {type(e).__name__}: {e}")
            
            self.logger.info("Integration validation completed successfully")
            
        except Exception as e:
            raise RuntimeError(f"Integration validation failed during functionality test: {e}")
    
    async def _cleanup_partial_initialization(self) -> None:
        """Clean up partially initialized components."""
        self.logger.info("Cleaning up partial initialization")
        
        # Clean up in reverse order
        if self.optimization_manager:
            self.optimization_manager.cleanup()
        
        for agent in self.optimization_agents.values():
            if hasattr(agent, 'cleanup'):
                agent.cleanup()
        
        for agent in [self.evaluation_agent, self.planning_agent, self.analysis_agent]:
            if agent and hasattr(agent, 'cleanup'):
                agent.cleanup()
        
        for service in [self.monitoring_service, self.notification_service, self.memory_manager, self.model_store]:
            if service and hasattr(service, 'cleanup'):
                service.cleanup()
        
        if self.monitoring_integrator:
            await self.monitoring_integrator.cleanup()
        
        if self.logging_integrator:
            await self.logging_integrator.cleanup()
    
    async def _shutdown_agents(self) -> None:
        """Shutdown all agents."""
        self.logger.info("Shutting down agents")
        
        if self.optimization_manager:
            self.optimization_manager.cleanup()
        
        for agent in self.optimization_agents.values():
            if hasattr(agent, 'cleanup'):
                agent.cleanup()
        
        for agent in [self.evaluation_agent, self.planning_agent, self.analysis_agent]:
            if agent and hasattr(agent, 'cleanup'):
                agent.cleanup()
    
    async def _shutdown_core_services(self) -> None:
        """Shutdown core services."""
        self.logger.info("Shutting down core services")
        
        # Stop monitoring service first
        if self.monitoring_service:
            try:
                self.monitoring_service.stop_monitoring()
            except Exception as e:
                self.logger.error(f"Error stopping monitoring service: {e}")
        
        for service in [self.monitoring_service, self.notification_service, self.memory_manager, self.model_store]:
            if service and hasattr(service, 'cleanup'):
                service.cleanup()
    
    async def _shutdown_infrastructure(self) -> None:
        """Shutdown infrastructure components."""
        self.logger.info("Shutting down infrastructure")
        
        if self.monitoring_integrator:
            await self.monitoring_integrator.cleanup()
        
        if self.logging_integrator:
            await self.logging_integrator.cleanup()