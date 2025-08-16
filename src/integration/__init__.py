"""
Integration module for the robotics model optimization platform.

This module provides comprehensive integration of all platform components,
ensuring proper wiring, initialization, and coordination between services.
"""

from .platform_integration import PlatformIntegrator
from .workflow_orchestrator import WorkflowOrchestrator
from .monitoring_integration import MonitoringIntegrator
from .logging_integration import LoggingIntegrator

__all__ = [
    'PlatformIntegrator',
    'WorkflowOrchestrator', 
    'MonitoringIntegrator',
    'LoggingIntegrator'
]