"""
Monitoring Service for comprehensive system monitoring and health checks.

This module provides monitoring capabilities for the robotics model optimization platform,
including performance monitoring, resource usage tracking, and health checks.
"""

import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import json

from .notification_service import NotificationService, NotificationType, AlertSeverity

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    timestamp: datetime
    cpu_usage_percent: float
    memory_usage_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    active_sessions: int
    optimization_queue_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_usage_percent': self.cpu_usage_percent,
            'memory_usage_percent': self.memory_usage_percent,
            'memory_used_mb': self.memory_used_mb,
            'memory_available_mb': self.memory_available_mb,
            'disk_usage_percent': self.disk_usage_percent,
            'disk_free_gb': self.disk_free_gb,
            'active_sessions': self.active_sessions,
            'optimization_queue_size': self.optimization_queue_size
        }


@dataclass
class HealthCheck:
    """Health check result."""
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'component': self.component,
            'status': self.status.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }


class MonitoringService:
    """
    Service for monitoring system performance and health.
    
    Provides comprehensive monitoring capabilities including resource usage tracking,
    performance metrics collection, and automated health checks.
    """
    
    def __init__(self, notification_service: NotificationService):
        self.notification_service = notification_service
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._monitoring_interval = 30  # seconds
        self._metrics_history: List[PerformanceMetrics] = []
        self._health_checks: Dict[str, HealthCheck] = {}
        self._health_check_callbacks: Dict[str, Callable[[], HealthCheck]] = {}
        self._lock = threading.Lock()
        
        # Performance thresholds
        self._cpu_warning_threshold = 80.0
        self._cpu_critical_threshold = 95.0
        self._memory_warning_threshold = 80.0
        self._memory_critical_threshold = 95.0
        self._disk_warning_threshold = 85.0
        self._disk_critical_threshold = 95.0
        
        # Register default health checks
        self._register_default_health_checks()
        
        logger.info("Monitoring service initialized")
    
    def _register_default_health_checks(self):
        """Register default system health checks."""
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("disk_space", self._check_disk_space)
        self.register_health_check("memory_usage", self._check_memory_usage)
    
    def start_monitoring(self, interval: int = 30):
        """
        Start continuous monitoring.
        
        Args:
            interval: Monitoring interval in seconds
        """
        if self._monitoring_active:
            logger.warning("Monitoring is already active")
            return
        
        self._monitoring_interval = interval
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        logger.info(f"Started monitoring with {interval}s interval")
        self.notification_service.send_notification(
            NotificationType.INFO,
            "Monitoring Started",
            f"System monitoring started with {interval}s interval"
        )
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self._monitoring_active:
            logger.warning("Monitoring is not active")
            return
        
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        
        logger.info("Stopped monitoring")
        self.notification_service.send_notification(
            NotificationType.INFO,
            "Monitoring Stopped",
            "System monitoring has been stopped"
        )
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        logger.info("Monitoring loop started")
        
        while self._monitoring_active:
            try:
                # Collect performance metrics
                metrics = self._collect_performance_metrics()
                
                with self._lock:
                    self._metrics_history.append(metrics)
                    # Keep only last 1000 metrics (about 8 hours at 30s interval)
                    if len(self._metrics_history) > 1000:
                        self._metrics_history = self._metrics_history[-1000:]
                
                # Check for performance issues
                self._check_performance_thresholds(metrics)
                
                # Run health checks
                self._run_health_checks()
                
                # Sleep until next interval
                time.sleep(self._monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self.notification_service.create_alert(
                    AlertSeverity.HIGH,
                    "Monitoring Error",
                    f"Error in monitoring loop: {str(e)}"
                )
                time.sleep(self._monitoring_interval)
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / (1024 * 1024)
        memory_available_mb = memory.available / (1024 * 1024)
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024 * 1024 * 1024)
        
        # Active sessions (placeholder - would be integrated with actual session tracking)
        active_sessions = len(self.notification_service._progress_sessions)
        
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage_percent=cpu_percent,
            memory_usage_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            memory_available_mb=memory_available_mb,
            disk_usage_percent=disk_percent,
            disk_free_gb=disk_free_gb,
            active_sessions=active_sessions
        )
        
        logger.debug(f"Collected metrics: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%")
        return metrics
    
    def _check_performance_thresholds(self, metrics: PerformanceMetrics):
        """Check if performance metrics exceed thresholds."""
        # Check CPU usage
        if metrics.cpu_usage_percent >= self._cpu_critical_threshold:
            self.notification_service.create_alert(
                AlertSeverity.CRITICAL,
                "Critical CPU Usage",
                f"CPU usage is {metrics.cpu_usage_percent:.1f}% (threshold: {self._cpu_critical_threshold}%)"
            )
        elif metrics.cpu_usage_percent >= self._cpu_warning_threshold:
            self.notification_service.create_alert(
                AlertSeverity.MEDIUM,
                "High CPU Usage",
                f"CPU usage is {metrics.cpu_usage_percent:.1f}% (threshold: {self._cpu_warning_threshold}%)"
            )
        
        # Check memory usage
        if metrics.memory_usage_percent >= self._memory_critical_threshold:
            self.notification_service.create_alert(
                AlertSeverity.CRITICAL,
                "Critical Memory Usage",
                f"Memory usage is {metrics.memory_usage_percent:.1f}% (threshold: {self._memory_critical_threshold}%)"
            )
        elif metrics.memory_usage_percent >= self._memory_warning_threshold:
            self.notification_service.create_alert(
                AlertSeverity.MEDIUM,
                "High Memory Usage",
                f"Memory usage is {metrics.memory_usage_percent:.1f}% (threshold: {self._memory_warning_threshold}%)"
            )
        
        # Check disk usage
        if metrics.disk_usage_percent >= self._disk_critical_threshold:
            self.notification_service.create_alert(
                AlertSeverity.CRITICAL,
                "Critical Disk Usage",
                f"Disk usage is {metrics.disk_usage_percent:.1f}% (threshold: {self._disk_critical_threshold}%)"
            )
        elif metrics.disk_usage_percent >= self._disk_warning_threshold:
            self.notification_service.create_alert(
                AlertSeverity.MEDIUM,
                "High Disk Usage",
                f"Disk usage is {metrics.disk_usage_percent:.1f}% (threshold: {self._disk_warning_threshold}%)"
            )
    
    def register_health_check(self, component: str, callback: Callable[[], HealthCheck]):
        """
        Register a health check callback.
        
        Args:
            component: Component name
            callback: Function that returns a HealthCheck
        """
        self._health_check_callbacks[component] = callback
        logger.info(f"Registered health check for component: {component}")
    
    def _run_health_checks(self):
        """Run all registered health checks."""
        for component, callback in self._health_check_callbacks.items():
            try:
                health_check = callback()
                
                with self._lock:
                    previous_check = self._health_checks.get(component)
                    self._health_checks[component] = health_check
                
                # Send alert if status changed to warning or critical
                if (previous_check is None or 
                    previous_check.status != health_check.status):
                    
                    if health_check.status == HealthStatus.CRITICAL:
                        self.notification_service.create_alert(
                            AlertSeverity.CRITICAL,
                            f"Health Check Failed: {component}",
                            health_check.message
                        )
                    elif health_check.status == HealthStatus.WARNING:
                        self.notification_service.create_alert(
                            AlertSeverity.MEDIUM,
                            f"Health Check Warning: {component}",
                            health_check.message
                        )
                
            except Exception as e:
                logger.error(f"Error running health check for {component}: {e}")
                error_check = HealthCheck(
                    component=component,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check failed: {str(e)}",
                    timestamp=datetime.now()
                )
                
                with self._lock:
                    self._health_checks[component] = error_check
    
    def _check_system_resources(self) -> HealthCheck:
        """Check overall system resource health."""
        try:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > 90 or memory_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"System resources critical: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            elif cpu_percent > 75 or memory_percent > 75:
                status = HealthStatus.WARNING
                message = f"System resources high: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"System resources normal: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%"
            
            return HealthCheck(
                component="system_resources",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component="system_resources",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check system resources: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_disk_space(self) -> HealthCheck:
        """Check disk space health."""
        try:
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024 * 1024 * 1024)
            
            if disk_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Disk space critical: {disk_percent:.1f}% used, {disk_free_gb:.1f}GB free"
            elif disk_percent > 85:
                status = HealthStatus.WARNING
                message = f"Disk space low: {disk_percent:.1f}% used, {disk_free_gb:.1f}GB free"
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space normal: {disk_percent:.1f}% used, {disk_free_gb:.1f}GB free"
            
            return HealthCheck(
                component="disk_space",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={
                    'disk_percent': disk_percent,
                    'disk_free_gb': disk_free_gb
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check disk space: {str(e)}",
                timestamp=datetime.now()
            )
    
    def _check_memory_usage(self) -> HealthCheck:
        """Check memory usage health."""
        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024 * 1024 * 1024)
            
            if memory_percent > 95:
                status = HealthStatus.CRITICAL
                message = f"Memory usage critical: {memory_percent:.1f}% used, {memory_available_gb:.1f}GB available"
            elif memory_percent > 80:
                status = HealthStatus.WARNING
                message = f"Memory usage high: {memory_percent:.1f}% used, {memory_available_gb:.1f}GB available"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {memory_percent:.1f}% used, {memory_available_gb:.1f}GB available"
            
            return HealthCheck(
                component="memory_usage",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={
                    'memory_percent': memory_percent,
                    'memory_available_gb': memory_available_gb
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component="memory_usage",
                status=HealthStatus.UNKNOWN,
                message=f"Failed to check memory usage: {str(e)}",
                timestamp=datetime.now()
            )
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics."""
        with self._lock:
            return self._metrics_history[-1] if self._metrics_history else None
    
    def get_metrics_history(
        self,
        hours: int = 1,
        limit: Optional[int] = None
    ) -> List[PerformanceMetrics]:
        """
        Get performance metrics history.
        
        Args:
            hours: Number of hours of history to return
            limit: Maximum number of metrics to return
            
        Returns:
            List of performance metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            filtered_metrics = [
                m for m in self._metrics_history
                if m.timestamp >= cutoff_time
            ]
        
        if limit:
            filtered_metrics = filtered_metrics[-limit:]
        
        return filtered_metrics
    
    def get_health_status(self) -> Dict[str, HealthCheck]:
        """Get current health status for all components."""
        with self._lock:
            return self._health_checks.copy()
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        with self._lock:
            health_checks = list(self._health_checks.values())
        
        if not health_checks:
            return HealthStatus.UNKNOWN
        
        # Overall health is the worst status among all components
        if any(hc.status == HealthStatus.CRITICAL for hc in health_checks):
            return HealthStatus.CRITICAL
        elif any(hc.status == HealthStatus.WARNING for hc in health_checks):
            return HealthStatus.WARNING
        elif any(hc.status == HealthStatus.UNKNOWN for hc in health_checks):
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY
    
    def set_thresholds(
        self,
        cpu_warning: Optional[float] = None,
        cpu_critical: Optional[float] = None,
        memory_warning: Optional[float] = None,
        memory_critical: Optional[float] = None,
        disk_warning: Optional[float] = None,
        disk_critical: Optional[float] = None
    ):
        """
        Update performance thresholds.
        
        Args:
            cpu_warning: CPU warning threshold percentage
            cpu_critical: CPU critical threshold percentage
            memory_warning: Memory warning threshold percentage
            memory_critical: Memory critical threshold percentage
            disk_warning: Disk warning threshold percentage
            disk_critical: Disk critical threshold percentage
        """
        if cpu_warning is not None:
            self._cpu_warning_threshold = cpu_warning
        if cpu_critical is not None:
            self._cpu_critical_threshold = cpu_critical
        if memory_warning is not None:
            self._memory_warning_threshold = memory_warning
        if memory_critical is not None:
            self._memory_critical_threshold = memory_critical
        if disk_warning is not None:
            self._disk_warning_threshold = disk_warning
        if disk_critical is not None:
            self._disk_critical_threshold = disk_critical
        
        logger.info("Updated performance thresholds")
        self.notification_service.send_notification(
            NotificationType.INFO,
            "Thresholds Updated",
            "Performance monitoring thresholds have been updated"
        )