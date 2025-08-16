"""
Monitoring Integration - Comprehensive monitoring across all platform components.

This module provides centralized monitoring, metrics collection, and health
checking for all platform components.
"""

import logging
import asyncio
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import psutil
import json


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a component."""
    component_name: str
    status: HealthStatus
    last_check: datetime
    message: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-level metrics."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io: Dict[str, int]
    process_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "disk_percent": self.disk_percent,
            "network_io": self.network_io,
            "process_count": self.process_count
        }


class MonitoringIntegrator:
    """
    Comprehensive monitoring integration for the platform.
    
    Provides system monitoring, component health checking, metrics collection,
    and alerting capabilities across all platform components.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Configuration
        self.monitoring_interval = config.get("monitoring_interval_seconds", 30)
        self.health_check_interval = config.get("health_check_interval_seconds", 60)
        self.metrics_retention_hours = config.get("metrics_retention_hours", 24)
        self.alert_thresholds = config.get("alert_thresholds", {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0
        })
        
        # Monitoring state
        self.monitored_components: Dict[str, Any] = {}
        self.component_health: Dict[str, ComponentHealth] = {}
        self.system_metrics_history: List[SystemMetrics] = []
        self.alert_callbacks: List[Callable[[str, HealthStatus, str], None]] = []
        
        # Monitoring control
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
        self.health_check_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        self.logger.info("MonitoringIntegrator initialized")
    
    async def initialize(self) -> None:
        """Initialize monitoring system."""
        try:
            self.logger.info("Initializing monitoring integration")
            
            # Start monitoring tasks
            await self._start_monitoring()
            
            self.logger.info("Monitoring integration initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize monitoring integration: {e}")
            raise
    
    async def cleanup(self) -> None:
        """Clean up monitoring resources."""
        try:
            self.logger.info("Cleaning up monitoring integration")
            
            # Stop monitoring
            await self._stop_monitoring()
            
            # Clear data
            with self._lock:
                self.monitored_components.clear()
                self.component_health.clear()
                self.system_metrics_history.clear()
                self.alert_callbacks.clear()
            
            self.logger.info("Monitoring integration cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during monitoring cleanup: {e}")
    
    def add_monitored_component(self, component_name: str, component: Any) -> None:
        """Add a component to monitoring."""
        with self._lock:
            self.monitored_components[component_name] = component
            
            # Initialize health status
            self.component_health[component_name] = ComponentHealth(
                component_name=component_name,
                status=HealthStatus.UNKNOWN,
                last_check=datetime.now(),
                message="Component added to monitoring"
            )
            
            self.logger.info(f"Added component to monitoring: {component_name}")
    
    def remove_monitored_component(self, component_name: str) -> None:
        """Remove a component from monitoring."""
        with self._lock:
            self.monitored_components.pop(component_name, None)
            self.component_health.pop(component_name, None)
            
            self.logger.info(f"Removed component from monitoring: {component_name}")
    
    def add_alert_callback(self, callback: Callable[[str, HealthStatus, str], None]) -> None:
        """Add an alert callback."""
        self.alert_callbacks.append(callback)
    
    def remove_alert_callback(self, callback: Callable[[str, HealthStatus, str], None]) -> None:
        """Remove an alert callback."""
        if callback in self.alert_callbacks:
            self.alert_callbacks.remove(callback)
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk usage (root partition)
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                disk_percent=disk_percent,
                network_io=network_io,
                process_count=process_count
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io={},
                process_count=0
            )
    
    def get_component_health(self, component_name: Optional[str] = None) -> Dict[str, ComponentHealth]:
        """Get health status of components."""
        with self._lock:
            if component_name:
                health = self.component_health.get(component_name)
                return {component_name: health} if health else {}
            else:
                return dict(self.component_health)
    
    def get_platform_health_summary(self) -> Dict[str, Any]:
        """Get overall platform health summary."""
        with self._lock:
            component_statuses = [health.status for health in self.component_health.values()]
            
            # Determine overall health
            if not component_statuses:
                overall_status = HealthStatus.UNKNOWN
            elif any(status == HealthStatus.CRITICAL for status in component_statuses):
                overall_status = HealthStatus.CRITICAL
            elif any(status == HealthStatus.WARNING for status in component_statuses):
                overall_status = HealthStatus.WARNING
            else:
                overall_status = HealthStatus.HEALTHY
            
            # Count components by status
            status_counts = {}
            for status in HealthStatus:
                status_counts[status.value] = sum(1 for s in component_statuses if s == status)
            
            return {
                "overall_status": overall_status.value,
                "total_components": len(self.component_health),
                "status_counts": status_counts,
                "last_update": datetime.now().isoformat(),
                "monitoring_active": self.monitoring_active
            }
    
    def get_metrics_history(self, hours: int = 1) -> List[Dict[str, Any]]:
        """Get system metrics history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_metrics = [
                metrics.to_dict() 
                for metrics in self.system_metrics_history 
                if metrics.timestamp >= cutoff_time
            ]
        
        return recent_metrics
    
    def get_monitoring_statistics(self) -> Dict[str, Any]:
        """Get monitoring system statistics."""
        with self._lock:
            return {
                "monitored_components": len(self.monitored_components),
                "monitoring_active": self.monitoring_active,
                "monitoring_interval_seconds": self.monitoring_interval,
                "health_check_interval_seconds": self.health_check_interval,
                "metrics_history_count": len(self.system_metrics_history),
                "alert_callbacks": len(self.alert_callbacks),
                "alert_thresholds": self.alert_thresholds
            }
    
    async def _start_monitoring(self) -> None:
        """Start monitoring tasks."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        
        # Start system metrics collection
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start component health checks
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        
        self.logger.info("Monitoring tasks started")
    
    async def _stop_monitoring(self) -> None:
        """Stop monitoring tasks."""
        self.monitoring_active = False
        
        # Cancel tasks
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        
        if self.health_check_task:
            self.health_check_task.cancel()
            try:
                await self.health_check_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Monitoring tasks stopped")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop for system metrics."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                metrics = self.get_system_metrics()
                
                # Store metrics
                with self._lock:
                    self.system_metrics_history.append(metrics)
                    
                    # Clean up old metrics
                    cutoff_time = datetime.now() - timedelta(hours=self.metrics_retention_hours)
                    self.system_metrics_history = [
                        m for m in self.system_metrics_history 
                        if m.timestamp >= cutoff_time
                    ]
                
                # Check for alerts
                await self._check_system_alerts(metrics)
                
                # Wait for next interval
                await asyncio.sleep(self.monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _health_check_loop(self) -> None:
        """Health check loop for components."""
        while self.monitoring_active:
            try:
                # Check health of all monitored components
                await self._check_component_health()
                
                # Wait for next interval
                await asyncio.sleep(self.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _check_component_health(self) -> None:
        """Check health of all monitored components."""
        with self._lock:
            components_to_check = dict(self.monitored_components)
        
        for component_name, component in components_to_check.items():
            try:
                health = await self._check_single_component_health(component_name, component)
                
                with self._lock:
                    old_health = self.component_health.get(component_name)
                    self.component_health[component_name] = health
                    
                    # Check for status changes
                    if old_health and old_health.status != health.status:
                        await self._send_health_alert(component_name, health.status, health.message or "Status changed")
                
            except Exception as e:
                self.logger.error(f"Failed to check health of {component_name}: {e}")
                
                # Mark as critical
                health = ComponentHealth(
                    component_name=component_name,
                    status=HealthStatus.CRITICAL,
                    last_check=datetime.now(),
                    message=f"Health check failed: {str(e)}"
                )
                
                with self._lock:
                    self.component_health[component_name] = health
    
    async def _check_single_component_health(self, component_name: str, component: Any) -> ComponentHealth:
        """Check health of a single component."""
        try:
            # Try to call health check method if available
            if hasattr(component, 'get_health_status'):
                status_info = component.get_health_status()
                
                if isinstance(status_info, dict):
                    status = HealthStatus(status_info.get("status", "unknown"))
                    message = status_info.get("message", "")
                    metrics = status_info.get("metrics", {})
                else:
                    status = HealthStatus.HEALTHY
                    message = "Health check successful"
                    metrics = {}
            
            # Check if component is responsive
            elif hasattr(component, 'ping') or hasattr(component, 'is_alive'):
                ping_method = getattr(component, 'ping', None) or getattr(component, 'is_alive', None)
                
                if callable(ping_method):
                    result = ping_method()
                    if asyncio.iscoroutine(result):
                        result = await result
                    
                    if result:
                        status = HealthStatus.HEALTHY
                        message = "Component responsive"
                    else:
                        status = HealthStatus.WARNING
                        message = "Component not responsive"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Component exists"
                
                metrics = {}
            
            # Default case - component exists
            else:
                status = HealthStatus.HEALTHY
                message = "Component available"
                metrics = {}
            
            return ComponentHealth(
                component_name=component_name,
                status=status,
                last_check=datetime.now(),
                message=message,
                metrics=metrics
            )
            
        except Exception as e:
            return ComponentHealth(
                component_name=component_name,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                message=f"Health check error: {str(e)}"
            )
    
    async def _check_system_alerts(self, metrics: SystemMetrics) -> None:
        """Check system metrics against alert thresholds."""
        alerts = []
        
        # CPU alert
        if metrics.cpu_percent > self.alert_thresholds.get("cpu_percent", 80):
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        # Memory alert
        if metrics.memory_percent > self.alert_thresholds.get("memory_percent", 85):
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        # Disk alert
        if metrics.disk_percent > self.alert_thresholds.get("disk_percent", 90):
            alerts.append(f"High disk usage: {metrics.disk_percent:.1f}%")
        
        # Send alerts
        for alert_message in alerts:
            await self._send_health_alert("system", HealthStatus.WARNING, alert_message)
    
    async def _send_health_alert(self, component_name: str, status: HealthStatus, message: str) -> None:
        """Send health alert to callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(component_name, status, message)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")