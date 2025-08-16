"""
Unit tests for the NotificationService.
"""

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import time
import threading

from src.services.notification_service import (
    NotificationService,
    NotificationType,
    AlertSeverity,
    Notification,
    Alert,
    ProgressUpdate
)


class TestNotificationService(unittest.TestCase):
    """Test cases for NotificationService."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = NotificationService()
    
    def test_initialization(self):
        """Test service initialization."""
        self.assertIsInstance(self.service, NotificationService)
        self.assertEqual(len(self.service._notifications), 0)
        self.assertEqual(len(self.service._alerts), 0)
        self.assertEqual(len(self.service._progress_sessions), 0)
    
    def test_send_notification(self):
        """Test sending notifications."""
        # Send a notification
        notification_id = self.service.send_notification(
            NotificationType.INFO,
            "Test Title",
            "Test message",
            session_id="test_session",
            metadata={"key": "value"}
        )
        
        # Verify notification was created
        self.assertIsNotNone(notification_id)
        self.assertEqual(len(self.service._notifications), 1)
        
        notification = self.service._notifications[0]
        self.assertEqual(notification.type, NotificationType.INFO)
        self.assertEqual(notification.title, "Test Title")
        self.assertEqual(notification.message, "Test message")
        self.assertEqual(notification.session_id, "test_session")
        self.assertEqual(notification.metadata["key"], "value")
        self.assertIsInstance(notification.timestamp, datetime)
    
    def test_create_alert(self):
        """Test creating alerts."""
        # Create an alert
        alert_id = self.service.create_alert(
            AlertSeverity.HIGH,
            "Test Alert",
            "Test alert description",
            session_id="test_session",
            metadata={"error_code": 500}
        )
        
        # Verify alert was created
        self.assertIsNotNone(alert_id)
        self.assertEqual(len(self.service._alerts), 1)
        
        alert = self.service._alerts[0]
        self.assertEqual(alert.severity, AlertSeverity.HIGH)
        self.assertEqual(alert.title, "Test Alert")
        self.assertEqual(alert.description, "Test alert description")
        self.assertEqual(alert.session_id, "test_session")
        self.assertEqual(alert.metadata["error_code"], 500)
        self.assertFalse(alert.resolved)
        self.assertIsInstance(alert.timestamp, datetime)
    
    def test_resolve_alert(self):
        """Test resolving alerts."""
        # Create an alert
        alert_id = self.service.create_alert(
            AlertSeverity.MEDIUM,
            "Test Alert",
            "Test description"
        )
        
        # Resolve the alert
        result = self.service.resolve_alert(alert_id)
        self.assertTrue(result)
        
        # Verify alert is resolved
        alert = self.service._alerts[0]
        self.assertTrue(alert.resolved)
        
        # Test resolving non-existent alert
        result = self.service.resolve_alert("non_existent")
        self.assertFalse(result)
    
    def test_progress_tracking(self):
        """Test progress tracking functionality."""
        session_id = "test_session"
        total_steps = 5
        
        # Start progress tracking
        self.service.start_progress_tracking(session_id, total_steps, "Initial step")
        
        # Verify progress was initialized
        self.assertIn(session_id, self.service._progress_sessions)
        progress = self.service.get_progress(session_id)
        self.assertIsNotNone(progress)
        self.assertEqual(progress.session_id, session_id)
        self.assertEqual(progress.total_steps, total_steps)
        self.assertEqual(progress.current_step, 0)
        self.assertEqual(progress.progress_percentage, 0.0)
        
        # Update progress
        self.service.update_progress(session_id, 2, "Middle step")
        progress = self.service.get_progress(session_id)
        self.assertEqual(progress.current_step, 2)
        self.assertEqual(progress.progress_percentage, 40.0)
        self.assertEqual(progress.step_name, "Middle step")
        self.assertIsNotNone(progress.estimated_completion)
        
        # Complete progress tracking
        self.service.complete_progress_tracking(session_id)
        progress = self.service.get_progress(session_id)
        self.assertIsNone(progress)  # Should be cleaned up
    
    def test_progress_time_estimates(self):
        """Test progress time estimation calculations."""
        session_id = "test_session"
        
        # Start progress tracking
        self.service.start_progress_tracking(session_id, 4, "Starting")
        
        # Simulate some time passing and update progress
        time.sleep(0.1)  # Small delay for testing
        self.service.update_progress(session_id, 1, "Step 1")
        
        progress = self.service.get_progress(session_id)
        self.assertIsNotNone(progress.elapsed_time)
        self.assertIsNotNone(progress.remaining_time)
        self.assertIsNotNone(progress.estimated_completion)
        
        # Clean up
        self.service.complete_progress_tracking(session_id)
    
    def test_subscription_system(self):
        """Test event subscription system."""
        # Set up mock callbacks
        notification_callback = Mock()
        alert_callback = Mock()
        progress_callback = Mock()
        
        # Subscribe to events
        self.service.subscribe('notification', notification_callback)
        self.service.subscribe('alert', alert_callback)
        self.service.subscribe('progress', progress_callback)
        
        # Send notification
        self.service.send_notification(NotificationType.INFO, "Test", "Message")
        notification_callback.assert_called_once()
        
        # Create alert
        self.service.create_alert(AlertSeverity.LOW, "Test", "Description")
        alert_callback.assert_called_once()
        
        # Start progress tracking
        self.service.start_progress_tracking("test", 3, "Starting")
        progress_callback.assert_called_once()
        
        # Test unsubscribe
        self.service.unsubscribe('notification', notification_callback)
        self.service.send_notification(NotificationType.INFO, "Test2", "Message2")
        # Should still be called only once (not twice)
        notification_callback.assert_called_once()
    
    def test_get_notifications_filtering(self):
        """Test notification filtering and retrieval."""
        # Create notifications with different types and sessions
        self.service.send_notification(
            NotificationType.INFO, "Info 1", "Message 1", "session1"
        )
        self.service.send_notification(
            NotificationType.ERROR, "Error 1", "Message 2", "session1"
        )
        self.service.send_notification(
            NotificationType.INFO, "Info 2", "Message 3", "session2"
        )
        
        # Test filtering by session
        session1_notifications = self.service.get_notifications(session_id="session1")
        self.assertEqual(len(session1_notifications), 2)
        
        # Test filtering by type
        info_notifications = self.service.get_notifications(type_filter=NotificationType.INFO)
        self.assertEqual(len(info_notifications), 2)
        
        # Test limit
        limited_notifications = self.service.get_notifications(limit=1)
        self.assertEqual(len(limited_notifications), 1)
        
        # Test combined filters
        filtered_notifications = self.service.get_notifications(
            session_id="session1",
            type_filter=NotificationType.INFO
        )
        self.assertEqual(len(filtered_notifications), 1)
    
    def test_get_alerts_filtering(self):
        """Test alert filtering and retrieval."""
        # Create alerts with different severities and sessions
        alert1_id = self.service.create_alert(
            AlertSeverity.LOW, "Alert 1", "Description 1", "session1"
        )
        self.service.create_alert(
            AlertSeverity.HIGH, "Alert 2", "Description 2", "session1"
        )
        self.service.create_alert(
            AlertSeverity.LOW, "Alert 3", "Description 3", "session2"
        )
        
        # Resolve one alert
        self.service.resolve_alert(alert1_id)
        
        # Test filtering by session
        session1_alerts = self.service.get_alerts(session_id="session1")
        self.assertEqual(len(session1_alerts), 2)
        
        # Test filtering by severity
        low_alerts = self.service.get_alerts(severity_filter=AlertSeverity.LOW)
        self.assertEqual(len(low_alerts), 2)
        
        # Test filtering by resolved status
        unresolved_alerts = self.service.get_alerts(resolved_filter=False)
        self.assertEqual(len(unresolved_alerts), 2)
        
        resolved_alerts = self.service.get_alerts(resolved_filter=True)
        self.assertEqual(len(resolved_alerts), 1)
    
    def test_clear_notifications_and_alerts(self):
        """Test clearing notifications and alerts."""
        # Create some notifications and alerts
        self.service.send_notification(NotificationType.INFO, "Test", "Message", "session1")
        self.service.send_notification(NotificationType.INFO, "Test", "Message", "session2")
        self.service.create_alert(AlertSeverity.LOW, "Alert", "Description", "session1")
        self.service.create_alert(AlertSeverity.LOW, "Alert", "Description", "session2")
        
        # Clear notifications for specific session
        self.service.clear_notifications("session1")
        notifications = self.service.get_notifications()
        self.assertEqual(len(notifications), 1)
        self.assertEqual(notifications[0].session_id, "session2")
        
        # Clear all alerts
        self.service.clear_alerts()
        alerts = self.service.get_alerts()
        self.assertEqual(len(alerts), 0)
    
    def test_system_status(self):
        """Test system status reporting."""
        # Create some test data
        self.service.send_notification(NotificationType.INFO, "Test", "Message")
        self.service.create_alert(AlertSeverity.HIGH, "Alert", "Description")
        self.service.start_progress_tracking("session1", 5, "Starting")
        
        # Get system status
        status = self.service.get_system_status()
        
        # Verify status contains expected information
        self.assertIn('active_sessions', status)
        self.assertIn('total_notifications', status)
        self.assertIn('total_alerts', status)
        self.assertIn('unresolved_alerts', status)
        self.assertIn('alert_counts_by_severity', status)
        self.assertIn('timestamp', status)
        
        self.assertEqual(status['active_sessions'], 1)
        self.assertEqual(status['total_notifications'], 1)
        self.assertEqual(status['total_alerts'], 1)
        self.assertEqual(status['unresolved_alerts'], 1)
    
    def test_invalid_subscription(self):
        """Test handling of invalid subscription types."""
        with self.assertRaises(ValueError):
            self.service.subscribe('invalid_type', Mock())
    
    def test_progress_update_nonexistent_session(self):
        """Test updating progress for non-existent session."""
        # This should not raise an exception, just log a warning
        self.service.update_progress("nonexistent", 1, "Step")
        
        # Verify no progress session was created
        progress = self.service.get_progress("nonexistent")
        self.assertIsNone(progress)
    
    def test_concurrent_access(self):
        """Test thread safety of the service."""
        def send_notifications():
            for i in range(10):
                self.service.send_notification(
                    NotificationType.INFO,
                    f"Test {i}",
                    f"Message {i}"
                )
        
        def create_alerts():
            for i in range(10):
                self.service.create_alert(
                    AlertSeverity.LOW,
                    f"Alert {i}",
                    f"Description {i}"
                )
        
        # Run concurrent operations
        thread1 = threading.Thread(target=send_notifications)
        thread2 = threading.Thread(target=create_alerts)
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Verify all notifications and alerts were created
        self.assertEqual(len(self.service._notifications), 10)
        self.assertEqual(len(self.service._alerts), 10)


if __name__ == '__main__':
    unittest.main()