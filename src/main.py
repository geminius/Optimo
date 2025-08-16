#!/usr/bin/env python3
"""
Main entry point for the Robotics Model Optimization Platform.

This module provides the complete integrated platform with all components
properly wired together for production use.
"""

import asyncio
import logging
import signal
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
import argparse

from .integration.platform_integration import PlatformIntegrator
from .integration.workflow_orchestrator import WorkflowOrchestrator
from .api.main import app
from .config.optimization_criteria import OptimizationCriteria, OptimizationConstraints, OptimizationTechnique


logger = logging.getLogger(__name__)


class RoboticsOptimizationPlatform:
    """
    Complete robotics model optimization platform.
    
    Integrates all components and provides a unified interface for
    model optimization workflows.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.platform_integrator: Optional[PlatformIntegrator] = None
        self.workflow_orchestrator: Optional[WorkflowOrchestrator] = None
        self.is_running = False
        
        # Set up logging
        self._setup_logging()
        
        logger.info("RoboticsOptimizationPlatform initialized")
    
    async def start(self) -> bool:
        """Start the complete platform."""
        try:
            logger.info("Starting Robotics Model Optimization Platform")
            
            # Initialize platform integrator
            self.platform_integrator = PlatformIntegrator(self.config)
            success = await self.platform_integrator.initialize_platform()
            
            if not success:
                logger.error("Failed to initialize platform")
                return False
            
            # Initialize workflow orchestrator
            self.workflow_orchestrator = WorkflowOrchestrator(self.platform_integrator)
            
            # Set up signal handlers for graceful shutdown
            self._setup_signal_handlers()
            
            # Inject platform integrator into FastAPI app
            app.state.platform_integrator = self.platform_integrator
            app.state.workflow_orchestrator = self.workflow_orchestrator
            
            # Update the optimization manager dependency in the app
            optimization_manager = self.platform_integrator.get_optimization_manager()
            app.state.optimization_manager = optimization_manager
            
            self.is_running = True
            logger.info("Platform started successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start platform: {e}")
            await self._cleanup()
            return False
    
    async def stop(self) -> None:
        """Stop the platform gracefully."""
        if not self.is_running:
            logger.warning("Platform not running")
            return
        
        try:
            logger.info("Stopping Robotics Model Optimization Platform")
            
            self.is_running = False
            
            # Cleanup components
            await self._cleanup()
            
            logger.info("Platform stopped successfully")
            
        except Exception as e:
            logger.error(f"Error during platform shutdown: {e}")
    
    async def optimize_model(
        self,
        model_path: str,
        criteria: OptimizationCriteria,
        user_id: str = "system"
    ) -> Dict[str, Any]:
        """
        Optimize a model using the complete workflow.
        
        Args:
            model_path: Path to the model file
            criteria: Optimization criteria
            user_id: User ID for tracking
            
        Returns:
            Optimization results
        """
        if not self.is_running or not self.workflow_orchestrator:
            raise RuntimeError("Platform not running")
        
        logger.info(f"Starting model optimization: {model_path}")
        
        result = await self.workflow_orchestrator.execute_complete_workflow(
            model_path=model_path,
            criteria=criteria,
            user_id=user_id
        )
        
        if result.success:
            logger.info(f"Model optimization completed successfully: {result.workflow_id}")
        else:
            logger.error(f"Model optimization failed: {result.error_message}")
        
        return {
            "success": result.success,
            "workflow_id": result.workflow_id,
            "session_id": result.session_id,
            "execution_time_seconds": result.execution_time_seconds,
            "error_message": result.error_message,
            "results": result.results
        }
    
    def get_platform_status(self) -> Dict[str, Any]:
        """Get comprehensive platform status."""
        if not self.platform_integrator:
            return {
                "running": False,
                "error": "Platform not initialized"
            }
        
        status = self.platform_integrator.get_platform_status()
        status["running"] = self.is_running
        
        return status
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load platform configuration."""
        default_config = {
            "logging": {
                "level": "INFO",
                "log_dir": "logs",
                "json_format": True,
                "console_output": True,
                "max_file_size_mb": 100,
                "backup_count": 5
            },
            "monitoring": {
                "monitoring_interval_seconds": 30,
                "health_check_interval_seconds": 60,
                "metrics_retention_hours": 24,
                "alert_thresholds": {
                    "cpu_percent": 80.0,
                    "memory_percent": 85.0,
                    "disk_percent": 90.0
                }
            },
            "model_store": {
                "storage_path": "models",
                "max_models": 1000,
                "cleanup_interval_hours": 24
            },
            "memory_manager": {
                "max_sessions": 50,
                "cleanup_interval_minutes": 30,
                "session_timeout_hours": 4
            },
            "notification_service": {
                "enable_email": False,
                "enable_webhook": True,
                "webhook_url": None
            },
            "monitoring_service": {
                "enable_metrics": True,
                "metrics_interval_seconds": 60,
                "enable_alerts": True
            },
            "optimization_manager": {
                "max_concurrent_sessions": 5,
                "auto_rollback_on_failure": True,
                "snapshot_frequency": 1,
                "session_timeout_minutes": 240
            },
            "analysis_agent": {
                "enable_profiling": True,
                "profiling_samples": 100
            },
            "planning_agent": {
                "enable_cost_analysis": True,
                "default_strategy": "conservative"
            },
            "evaluation_agent": {
                "enable_benchmarks": True,
                "benchmark_timeout_minutes": 30
            },
            "quantization_agent": {
                "default_bits": 8,
                "enable_dynamic": True
            },
            "pruning_agent": {
                "default_sparsity": 0.5,
                "enable_structured": True
            },
            "distillation_agent": {
                "temperature": 4.0,
                "alpha": 0.7
            },
            "compression_agent": {
                "compression_ratio": 0.5,
                "enable_svd": True
            },
            "architecture_search_agent": {
                "search_space": "efficient",
                "max_iterations": 100
            }
        }
        
        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        user_config = json.load(f)
                    
                    # Merge configurations
                    self._merge_config(default_config, user_config)
                    logger.info(f"Loaded configuration from: {config_path}")
                    
                except Exception as e:
                    logger.warning(f"Failed to load config file {config_path}: {e}")
                    logger.info("Using default configuration")
            else:
                logger.warning(f"Config file not found: {config_path}")
                logger.info("Using default configuration")
        
        return default_config
    
    def _merge_config(self, base_config: Dict[str, Any], user_config: Dict[str, Any]) -> None:
        """Merge user configuration into base configuration."""
        for key, value in user_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def _setup_logging(self) -> None:
        """Set up basic logging configuration."""
        log_config = self.config.get("logging", {})
        log_level = log_config.get("level", "INFO")
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.stop())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def _cleanup(self) -> None:
        """Clean up platform resources."""
        if self.platform_integrator:
            await self.platform_integrator.shutdown_platform()
            self.platform_integrator = None
        
        self.workflow_orchestrator = None


async def main():
    """Main entry point for the platform."""
    parser = argparse.ArgumentParser(
        description="Robotics Model Optimization Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--api-only",
        action="store_true",
        help="Run only the API server (for development)"
    )
    
    parser.add_argument(
        "--test-workflow",
        type=str,
        help="Test workflow with specified model path"
    )
    
    args = parser.parse_args()
    
    # Create platform
    platform = RoboticsOptimizationPlatform(args.config)
    
    try:
        # Start platform
        success = await platform.start()
        if not success:
            logger.error("Failed to start platform")
            return 1
        
        # Handle different modes
        if args.test_workflow:
            # Test workflow mode
            await test_workflow(platform, args.test_workflow)
        elif args.api_only:
            # API-only mode
            logger.info("Running in API-only mode")
            logger.info("Platform is ready. Use Ctrl+C to stop.")
            
            # Keep running until interrupted
            try:
                while platform.is_running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
        else:
            # Full platform mode
            logger.info("Platform is running. Use Ctrl+C to stop.")
            
            # Keep running until interrupted
            try:
                while platform.is_running:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
        
        return 0
        
    except Exception as e:
        logger.error(f"Platform error: {e}")
        return 1
    
    finally:
        await platform.stop()


async def test_workflow(platform: RoboticsOptimizationPlatform, model_path: str):
    """Test the complete workflow with a model."""
    logger.info(f"Testing workflow with model: {model_path}")
    
    # Create test criteria
    constraints = OptimizationConstraints(
        preserve_accuracy_threshold=0.95,
        allowed_techniques=[OptimizationTechnique.QUANTIZATION, OptimizationTechnique.PRUNING]
    )
    
    criteria = OptimizationCriteria(
        name="test_workflow",
        description="Test workflow execution",
        constraints=constraints,
        target_deployment="general"
    )
    
    # Execute workflow
    result = await platform.optimize_model(
        model_path=model_path,
        criteria=criteria,
        user_id="test_user"
    )
    
    # Print results
    print("\n" + "="*60)
    print("WORKFLOW TEST RESULTS")
    print("="*60)
    print(f"Success: {result['success']}")
    print(f"Workflow ID: {result['workflow_id']}")
    print(f"Session ID: {result['session_id']}")
    print(f"Execution Time: {result['execution_time_seconds']:.2f}s")
    
    if result['error_message']:
        print(f"Error: {result['error_message']}")
    
    if result['results']:
        print("\nDetailed Results:")
        print(json.dumps(result['results'], indent=2, default=str))
    
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())