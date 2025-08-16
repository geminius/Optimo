"""
Unit tests for the MonitoringService.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import time
import threading

from src.services.notification_service import NotificationService, AlertSeverity
from src.services.monitoring_service import (
    MonitoringService,
    PerformanceMetrics,
    HealthCheck,
    HealthStatus
)


class TestMonitoringService(unittest.TestCase):
    """Test cases for MonitoringService."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.notification_service = NotificationService()
        self.monitoring_service = MonitoringService(self.notification_service)
    
    def test_initialization(self):
        """Test service initialization."""
        self.assertIsInstance(self.monitoring_service, MonitoringService)
        self.assertFalse(self.monitoring_service._monitoring_active)
        self.assertEqual(len(self.monitoring_service._metrics_history), 0)
        self.assertEqual(len(self.monitoring_service._health_checks), 0)
        
        # Verify default health checks are registered
        self.assertIn("system_resources", self.monitoring_service._health_check_callbacks)
        self.assertIn("disk_space", self.monitoring_service._health_check_callbacks)
        self.assertIn("memory_usage", self.monitoring_service._health_check_callbacks)
    
    @patch('src.services.monitoring_service.psutil')
    def test_collect_performance_metrics(self, mock_psutil):
        """Test performance metrics collection."""
        # Mock psutil responses
        mock_psutil.cpu_percent.return_value = 45.5
        
        mock_memory = Mock()
        mock_memory.percent = 60.2
        mock_memory.used = 8 * 1024 * 1024 * 1024  # 8GB
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.total = 1000 * 1024 * 1024 * 1024  # 1TB
        mock_disk.used = 500 * 1024 * 1024 * 1024   # 500GB
        mock_disk.free = 500 * 1024 * 1024 * 1024   # 500GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        # Collect metrics
        metrics = self.monitoring_service._collect_performance_metrics()
        
        # Verify metrics
        self.assertIsInstance(metrics, PerformanceMetrics)
        self.assertEqual(metrics.cpu_usage_percent, 45.5)
        self.assertEqual(metrics.memory_usage_percent, 60.2)
        self.assertAlmostEqual(metrics.memory_used_mb, 8192, places=0)
        self.assertAlmostEqual(metrics.memory_available_mb, 4096, places=0)
        self.assertEqual(metrics.disk_usage_percent, 50.0)
        self.assertAlmostEqual(metrics.disk_free_gb, 500, places=0)
        self.assertIsInstance(metrics.timestamp, datetime)
    
    @patch('src.services.monitoring_service.psutil')
    def test_check_performance_thresholds(self, mock_psutil):
        """Test performance threshold checking."""
        # Create metrics that exceed thresholds
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage_percent=96.0,  # Above critical threshold (95%)
            memory_usage_percent=85.0,  # Above warning threshold (80%)
            memory_used_mb=8192,
            memory_available_mb=1024,
            disk_usage_percent=90.0,  # Above warning threshold (85%)
            disk_free_gb=100,
            active_sessions=2
        )
        
        # Check thresholds
        self.monitoring_service._check_performance_thresholds(metrics)
        
        # Verify alerts were created
        alerts = self.notification_service.get_alerts()
        self.assertEqual(len(alerts), 3)
        
        # Check alert severities
        alert_titles = [alert.title for alert in alerts]
        self.assertIn("Critical CPU Usage", alert_titles)
        self.assertIn("High Memory Usage", alert_titles)
        self.assertIn("High Disk Usage", alert_titles)
    
    def test_register_health_check(self):
        """Test registering custom health checks."""
        def custom_health_check():
            return HealthCheck(
                component="custom_component",
                status=HealthStatus.HEALTHY,
                message="All good",
                timestamp=datetime.now()
            )
        
        # Register health check
        self.monitoring_service.register_health_check("custom_component", custom_health_check)
        
        # Verify it was registered
        self.assertIn("custom_component", self.monitoring_service._health_check_callbacks)
    
    @patch('src.services.monitoring_service.psutil')
    def test_system_resources_health_check(self, mock_psutil):
        """Test system resources health check."""
        # Test healthy status
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_psutil.virtual_memory.return_value = mock_memory
        
        health_check = self.monitoring_service._check_system_resources()
        self.assertEqual(health_check.status, HealthStatus.HEALTHY)
        self.assertEqual(health_check.component, "system_resources")
        
        # Test warning status
        mock_psutil.cpu_percent.return_value = 80.0
        health_check = self.monitoring_service._check_system_resources()
        self.assertEqual(health_check.status, HealthStatus.WARNING)
        
        # Test critical status
        mock_psutil.cpu_percent.return_value = 95.0
        health_check = self.monitoring_service._check_system_resources()
        self.assertEqual(health_check.status, HealthStatus.CRITICAL)
    
    @patch('src.services.monitoring_service.psutil')
    def test_disk_space_health_check(self, mock_psutil):
        """Test disk space health check."""
        # Test healthy status
        mock_disk = Mock()
        mock_disk.total = 1000 * 1024 * 1024 * 1024  # 1TB
        mock_disk.used = 500 * 1024 * 1024 * 1024   # 500GB (50%)
        mock_disk.free = 500 * 1024 * 1024 * 1024   # 500GB
        mock_psutil.disk_usage.return_value = mock_disk
        
        health_check = self.monitoring_service._check_disk_space()
        self.assertEqual(health_check.status, HealthStatus.HEALTHY)
        
        # Test warning status
        mock_disk.used = 900 * 1024 * 1024 * 1024   # 900GB (90%)
        mock_disk.free = 100 * 1024 * 1024 * 1024   # 100GB
        
        health_check = self.monitoring_service._check_disk_space()
        self.assertEqual(health_check.status, HealthStatus.WARNING)
        
        # Test critical status
        mock_disk.used = 980 * 1024 * 1024 * 1024   # 980GB (98%)
        mock_disk.free = 20 * 1024 * 1024 * 1024    # 20GB
        
        health_check = self.monitoring_service._check_disk_space()
        self.assertEqual(health_check.status, HealthStatus.CRITICAL)
    
    @patch('src.services.monitoring_service.psutil')
    def test_memory_usage_health_check(self, mock_psutil):
        """Test memory usage health check."""
        # Test healthy status
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_memory.available = 4 * 1024 * 1024 * 1024  # 4GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        health_check = self.monitoring_service._check_memory_usage()
        self.assertEqual(health_check.status, HealthStatus.HEALTHY)
        
        # Test warning status
        mock_memory.percent = 85.0
        mock_memory.available = 1 * 1024 * 1024 * 1024  # 1GB
        
        health_check = self.monitoring_service._check_memory_usage()
        self.assertEqual(health_check.status, HealthStatus.WARNING)
        
        # Test critical status
        mock_memory.percent = 98.0
        mock_memory.available = 0.2 * 1024 * 1024 * 1024  # 200MB
        
        health_check = self.monitoring_service._check_memory_usage()
        self.assertEqual(health_check.status, HealthStatus.CRITICAL)
    
    def test_health_check_error_handling(self):
        """Test health check error handling."""
        def failing_health_check():
            raise Exception("Test error")
        
        # Register failing health check
        self.monitoring_service.register_health_check("failing_component", failing_health_check)
        
        # Run health checks
        self.monitoring_service._run_health_checks()
        
        # Verify error was handled
        health_status = self.monitoring_service.get_health_status()
        self.assertIn("failing_component", health_status)
        self.assertEqual(health_status["failing_component"].status, HealthStatus.UNKNOWN)
    
    def test_overall_health_calculation(self):
        """Test overall health status calculation."""
        # No health checks - should be unknown
        overall_health = self.monitoring_service.get_overall_health()
        self.assertEqual(overall_health, HealthStatus.UNKNOWN)
        
        # Add healthy check
        self.monitoring_service._health_checks["component1"] = HealthCheck(
            component="component1",
            status=HealthStatus.HEALTHY,
            message="All good",
            timestamp=datetime.now()
        )
        overall_health = self.monitoring_service.get_overall_health()
        self.assertEqual(overall_health, HealthStatus.HEALTHY)
        
        # Add warning check - overall should be warning
        self.monitoring_service._health_checks["component2"] = HealthCheck(
            component="component2",
            status=HealthStatus.WARNING,
            message="Warning",
            timestamp=datetime.now()
        )
        overall_health = self.monitoring_service.get_overall_health()
        self.assertEqual(overall_health, HealthStatus.WARNING)
        
        # Add critical check - overall should be critical
        self.monitoring_service._health_checks["component3"] = HealthCheck(
            component="component3",
            status=HealthStatus.CRITICAL,
            message="Critical",
            timestamp=datetime.now()
        )
        overall_health = self.monitoring_service.get_overall_health()
        self.assertEqual(overall_health, HealthStatus.CRITICAL)
    
    def test_metrics_history_management(self):
        """Test metrics history storage and retrieval."""
        # Add some test metrics
        for i in range(5):
            metrics = PerformanceMetrics(
                timestamp=datetime.now() - timedelta(minutes=i),
                cpu_usage_percent=50.0 + i,
                memory_usage_percent=60.0 + i,
                memory_used_mb=8192,
                memory_available_mb=4096,
                disk_usage_percent=70.0,
                disk_free_gb=500,
                active_sessions=1
            )
            self.monitoring_service._metrics_history.append(metrics)
        
        # Test getting current metrics
        current = self.monitoring_service.get_current_metrics()
        self.assertIsNotNone(current)
        self.assertEqual(current.cpu_usage_percent, 54.0)  # Last added (most recent in list)
        
        # Test getting history
        history = self.monitoring_service.get_metrics_history(hours=1)
        self.assertEqual(len(history), 5)
        
        # Test history with limit
        limited_history = self.monitoring_service.get_metrics_history(hours=1, limit=3)
        self.assertEqual(len(limited_history), 3)
    
    def test_threshold_updates(self):
        """Test updating performance thresholds."""
        # Update thresholds
        self.monitoring_service.set_thresholds(
            cpu_warning=70.0,
            cpu_critical=90.0,
            memory_warning=75.0,
            memory_critical=90.0
        )
        
        # Verify thresholds were updated
        self.assertEqual(self.monitoring_service._cpu_warning_threshold, 70.0)
        self.assertEqual(self.monitoring_service._cpu_critical_threshold, 90.0)
        self.assertEqual(self.monitoring_service._memory_warning_threshold, 75.0)
        self.assertEqual(self.monitoring_service._memory_critical_threshold, 90.0)
        
        # Verify notification was sent
        notifications = self.notification_service.get_notifications()
        self.assertTrue(any("Thresholds Updated" in n.title for n in notifications))
    
    @patch('src.services.monitoring_service.time.sleep')
    @patch('src.services.monitoring_service.psutil')
    def test_monitoring_loop_start_stop(self, mock_psutil, mock_sleep):
        """Test starting and stopping the monitoring loop."""
        # Mock psutil for metrics collection
        mock_psutil.cpu_percent.return_value = 50.0
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_memory.used = 8 * 1024 * 1024 * 1024
        mock_memory.available = 4 * 1024 * 1024 * 1024
        mock_psutil.virtual_memory.return_value = mock_memory
        
        mock_disk = Mock()
        mock_disk.total = 1000 * 1024 * 1024 * 1024
        mock_disk.used = 500 * 1024 * 1024 * 1024
        mock_disk.free = 500 * 1024 * 1024 * 1024
        mock_psutil.disk_usage.return_value = mock_disk
        
        # Mock sleep to prevent actual waiting
        mock_sleep.side_effect = lambda x: None
        
        # Start monitoring
        self.monitoring_service.start_monitoring(interval=1)
        self.assertTrue(self.monitoring_service._monitoring_active)
        self.assertIsNotNone(self.monitoring_service._monitoring_thread)
        
        # Give it a moment to start
        time.sleep(0.1)
        
        # Stop monitoring
        self.monitoring_service.stop_monitoring()
        self.assertFalse(self.monitoring_service._monitoring_active)
        
        # Verify notification was sent
        notifications = self.notification_service.get_notifications()
        start_notifications = [n for n in notifications if "Monitoring Started" in n.title]
        stop_notifications = [n for n in notifications if "Monitoring Stopped" in n.title]
        self.assertEqual(len(start_notifications), 1)
        self.assertEqual(len(stop_notifications), 1)
    
    def test_performance_metrics_serialization(self):
        """Test PerformanceMetrics serialization."""
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_usage_percent=75.5,
            memory_usage_percent=80.2,
            memory_used_mb=8192,
            memory_available_mb=2048,
            disk_usage_percent=65.0,
            disk_free_gb=350.5,
            active_sessions=3,
            optimization_queue_size=2
        )
        
        # Test serialization
        metrics_dict = metrics.to_dict()
        self.assertIsInstance(metrics_dict, dict)
        self.assertEqual(metrics_dict['cpu_usage_percent'], 75.5)
        self.assertEqual(metrics_dict['memory_usage_percent'], 80.2)
        self.assertEqual(metrics_dict['active_sessions'], 3)
        self.assertEqual(metrics_dict['optimization_queue_size'], 2)
        self.assertIn('timestamp', metrics_dict)
    
    def test_health_check_serialization(self):
        """Test HealthCheck serialization."""
        health_check = HealthCheck(
            component="test_component",
            status=HealthStatus.WARNING,
            message="Test warning message",
            timestamp=datetime.now(),
            details={"cpu": 85.0, "memory": 75.0}
        )
        
        # Test serialization
        health_dict = health_check.to_dict()
        self.assertIsInstance(health_dict, dict)
        self.assertEqual(health_dict['component'], "test_component")
        self.assertEqual(health_dict['status'], "warning")
        self.assertEqual(health_dict['message'], "Test warning message")
        self.assertEqual(health_dict['details']['cpu'], 85.0)
        self.assertIn('timestamp', health_dict)


if __name__ == '__main__':
    unittest.main()